"""pepar — 논문·웹페이지 번역 웹앱."""

import html as htmllib
import json
import os
import queue
import re
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for
from PIL import Image

import ocr
import translator

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)
SOURCES = CACHE / "sources.json"  # key → {kind, url[, name]}. web 문서는 key(해시)만으로 URL을 알 수 없어서 기록해 둔다.
IMAGES = CACHE / "images"  # 업로드한 원본 이미지 i-<hash>.<png|jpg>. 모델별 번역이 같이 쓴다
IMAGES.mkdir(exist_ok=True)
KEY_RE = re.compile(r"^(\d{4}\.\d{4,5}(v\d+)?|[wi]-[0-9a-f]{12})$")
MODEL_RE = re.compile(r"^[\w.\-]+$")
ACTIVE = ("queued", "fetching", "reading", "translating")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 업로드 상한 (이미지 자체 한도는 ocr.MAX_UPLOAD)

# llama-server가 parallel=1이라 번역은 한 번에 하나씩 순서대로 처리한다.
jobs: dict[str, dict] = {}  # "model/key" → 상태
jobs_lock = threading.Lock()
work: "queue.Queue[tuple[str, str]]" = queue.Queue()
sources_lock = threading.Lock()
FAILED = CACHE / "failed_jobs.json"  # 실패한 작업 기록. 재시작해도 첫 화면에 실패 사실이 남도록
FAILED_KEEP = 24 * 3600
# 블록당 평균 번역 시간(초). 대기 중인 작업의 시작 시각을 추정하는 데 쓴다. 끝난 작업마다 갱신.
sec_per_block = 7.0


# ---------------------------------------------------------------- 저장소
# cache/<model>/<key>.html        번역 결과
# cache/<model>/<key>.meta.json   번역 시각·제목·URL·블록 수 등
# cache/<model>/<key>.partial.json 이어서 번역하기 위한 체크포인트 (이미지는 OCR 결과도)
# cache/images/<key>.<png|jpg>    업로드한 이미지


def load_sources() -> dict:
    try:
        return json.loads(SOURCES.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def remember_source(src: translator.Source):
    with sources_lock:
        data = load_sources()
        if data.get(src.key, {}).get("url") != src.url:
            data[src.key] = {"kind": src.kind, "url": src.url}
            SOURCES.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def source_for(key: str) -> translator.Source | None:
    if key.startswith("w-"):
        info = load_sources().get(key)
        return translator.Source("web", key, info["url"]) if info else None
    if key.startswith("i-"):
        return translator.Source("image", key, f"/image/{key}") if image_path(key) else None
    return translator.Source("arxiv", key, key)


def source_url(key: str) -> str:
    src = source_for(key)
    if not src:
        return ""
    return src.url if src.kind in ("web", "image") else f"https://arxiv.org/abs/{key}"


def image_path(key: str) -> Path | None:
    found = list(IMAGES.glob(f"{key}.*"))
    return found[0] if found else None


def image_name(key: str) -> str:
    return load_sources().get(key, {}).get("name") or key


def model_dir(model: str) -> Path:
    d = CACHE / model
    d.mkdir(exist_ok=True)
    return d


def cached_path(model: str, key: str) -> Path | None:
    """arXiv는 버전 지정 시 그 버전, 아니면 캐시된 것 중 최신 버전."""
    d = CACHE / model
    if key.startswith(("w-", "i-")) or "v" in key:
        p = d / f"{key}.html"
        return p if p.exists() else None
    found = sorted(d.glob(f"{key}v*.html"), key=lambda p: int(p.stem.split("v")[-1]))
    return found[-1] if found else None


def checkpoint_path(model: str, key: str) -> Path:
    return model_dir(model) / f"{key}.partial.json"


def migrate_old_cache():
    """모델별 폴더가 생기기 전의 캐시(cache/<key>.html)를 기본 모델 폴더로 옮긴다."""
    for pattern in ("*.html", "*.partial.json"):
        for p in CACHE.glob(pattern):
            p.rename(model_dir(translator.LLM_MODEL) / p.name)


migrate_old_cache()


def doc_info(p: Path, sources: dict) -> dict:
    """번역 결과 파일 하나의 목록용 정보. meta.json이 없는 예전 결과는 파일에서 추정한다."""
    key, model = p.stem, p.parent.name
    try:
        meta = json.loads(p.with_suffix(".meta.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        meta = {}
    title = meta.get("title")
    if not title:
        m = re.search(r"<title>(.*?)</title>", p.read_text(encoding="utf-8", errors="ignore")[:6000], re.S)
        title = htmllib.unescape(m.group(1).strip()).removeprefix("[번역] ") if m else key
    url = meta.get("url") or (
        sources.get(key, {}).get("url", "") if key.startswith("w-") else f"https://arxiv.org/abs/{key}"
    )
    return {
        "model": model,
        "key": key,
        "title": title,
        "url": url,
        "where": "이미지" if key.startswith("i-") else "arXiv" if not key.startswith("w-") else (urlparse(url).hostname or "web"),
        "translated_at": meta.get("translated_at") or datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
        "blocks": meta.get("blocks"),
        "failed": meta.get("failed"),
        "seconds": meta.get("seconds"),
    }


# 본문 검색용: 파일별 추출 텍스트 캐시 {path: (mtime, text)}
_text_cache: dict[str, tuple[float, str]] = {}


def doc_text(p: Path) -> str:
    mtime = p.stat().st_mtime
    hit = _text_cache.get(str(p))
    if hit and hit[0] == mtime:
        return hit[1]
    raw = p.read_text(encoding="utf-8", errors="ignore")
    raw = re.sub(r"<head\b.*?</head>", " ", raw, flags=re.S | re.I)
    raw = re.sub(r"<(script|style)\b.*?</\1>", " ", raw, flags=re.S | re.I)
    raw = re.sub(r'<div class="pepar-bar">.*', " ", raw, flags=re.S)  # 번역 페이지 하단 버튼
    raw = re.sub(r"<[^>]+>", " ", raw)  # 태그와 함께 data-pepar-alt(원문)도 빠진다 → 번역문 위주
    text = " ".join(htmllib.unescape(raw).split())
    _text_cache[str(p)] = (mtime, text)
    return text


# ---------------------------------------------------------------- 작업 큐


def worker():
    global sec_per_block
    while True:
        model, key = work.get()
        job = jobs[f"{model}/{key}"]

        def progress(stage, done, total, **info):
            job.update(stage=stage, done=done, total=total, **info)
            if stage == "translating" and "started" not in job:
                job["started"] = time.time()
                job["resumed"] = done  # 체크포인트에서 이어받은 블록 수 (남은 시간 계산에서 제외)

        try:
            src = source_for(key)
            if src is None:
                raise translator.PeparError("원본 URL 기록을 찾을 수 없습니다. 첫 화면에서 다시 입력해 주세요.")
            checkpoint = checkpoint_path(model, key)
            t0 = time.time()
            retry_url = f"/view/{model}/{key}?retry_failed=1"
            if src.kind == "image":
                resolved, title = key, image_name(key)
                job["title"] = title
                html, failed = translate_image(model, key, title, progress, checkpoint, retry_url)
            else:
                resolved, title, html, failed = translator.run(
                    src, model, progress, checkpoint=checkpoint, retry_url=retry_url
                )
            out = model_dir(model) / f"{resolved}.html"
            out.write_text(html, encoding="utf-8")
            out.with_suffix(".meta.json").write_text(
                json.dumps(
                    {
                        "title": title,
                        "url": source_url(key),
                        "model": model,
                        "translated_at": datetime.now().isoformat(timespec="seconds"),
                        "blocks": job["total"],
                        "failed": failed,
                        "seconds": round(time.time() - t0),
                    },
                    ensure_ascii=False,
                    indent=1,
                ),
                encoding="utf-8",
            )
            if not failed:
                checkpoint.unlink(missing_ok=True)
            job.update(stage="done", failed=failed, key_resolved=resolved)
            rate = job_rate(job)
            if rate:
                sec_per_block = 0.7 * sec_per_block + 0.3 * rate
        except translator.PeparError as e:
            job.update(stage="error", message=str(e))
        except Exception as e:  # noqa: BLE001 - 사용자에게 그대로 보여준다
            traceback.print_exc()
            job.update(stage="error", message=f"{type(e).__name__}: {e}")
        job["finished_at"] = time.time()
        save_failed()


def translate_image(model, key, title, progress, checkpoint, retry_url) -> tuple[str, int]:
    """(결과 HTML, 번역 실패 라벨 수). 결과 페이지는 원본 이미지 위에 번역문 라벨을 덮는다.

    라벨은 <key>.labels.json에도 저장해, 보기 화면은 매번 최신 템플릿으로 그린다 (.html은 목록·검색용).
    """
    path = image_path(key)
    labels, failed = ocr.run(model, path.read_bytes(), progress, checkpoint=checkpoint)
    with Image.open(path) as img:
        width, height = img.size
    data = {"title": title, "labels": labels, "failed": failed, "width": width, "height": height}
    (model_dir(model) / f"{key}.labels.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with app.app_context():
        return render_image(model, key, data), failed


def render_image(model: str, key: str, data: dict) -> str:
    return render_template("image_view.html", model=model, key=key, retry_url=f"/view/{model}/{key}?retry_failed=1", **data)


def save_failed():
    """실패 상태인 작업들을 파일에 기록한다 (성공하거나 닫으면 빠진다)."""
    now = time.time()
    failed = [
        {k: v for k, v in j.items() if k in ("model", "key", "url", "title", "stage", "message", "queued_at", "finished_at")}
        for j in list(jobs.values())
        if j["stage"] == "error" and now - j.get("finished_at", now) < FAILED_KEEP
    ]
    FAILED.write_text(json.dumps(failed, ensure_ascii=False, indent=1), encoding="utf-8")


def load_failed():
    try:
        failed = json.loads(FAILED.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return
    now = time.time()
    for j in failed:
        if now - j.get("finished_at", 0) < FAILED_KEEP:
            jobs.setdefault(f"{j['model']}/{j['key']}", {"done": 0, "total": 0, **j})


load_failed()
threading.Thread(target=worker, daemon=True).start()


def start_job(model: str, key: str):
    with jobs_lock:
        job = jobs.get(f"{model}/{key}")
        if job and job["stage"] in ACTIVE:
            return
        jobs[f"{model}/{key}"] = {
            "model": model,
            "key": key,
            "url": source_url(key),
            "stage": "queued",
            "done": 0,
            "total": 0,
            "message": "",
            "queued_at": time.time(),
        }
        work.put((model, key))


def job_rate(job: dict) -> float | None:
    """이 작업의 블록당 번역 시간(초). 체크포인트에서 이어받은 블록은 뺀다."""
    done_here = job["done"] - job.get("resumed", 0)
    if job.get("started") and done_here > 0:
        return (time.time() - job["started"]) / done_here
    return None


def job_eta(job: dict) -> int | None:
    """남은 시간(초). 블록 수를 아직 모르면(원문을 가져오기 전) None."""
    if not job["total"]:
        return None
    rate = job_rate(job) or sec_per_block
    return round(rate * (job["total"] - job["done"]))


def job_view(job: dict) -> dict:
    """API로 내보낼 작업 상태. 대기 중이면 앞선 작업들로 시작 시각을 추정한다."""
    data = {k: v for k, v in job.items() if k not in ("queued_at", "started", "finished_at")}
    if job["stage"] == "queued":
        ahead = [j for j in jobs.values() if j["stage"] in ACTIVE and j["queued_at"] < job["queued_at"]]
        etas = [job_eta(j) for j in ahead]
        data["ahead"] = len(ahead)
        data["wait"] = sum(e for e in etas if e is not None)
        data["wait_partial"] = any(e is None for e in etas)  # 블록 수를 모르는 작업이 있어 실제로는 더 걸림
    elif job["stage"] == "translating" and job_rate(job):
        data["eta"] = job_eta(job)
    return data


# ---------------------------------------------------------------- 페이지


def get_models() -> tuple[list[dict], str]:
    """(모델 목록, 오류 메시지). 라우터에 연결이 안 되면 기본 모델만."""
    try:
        return translator.list_models(), ""
    except Exception as e:  # noqa: BLE001
        return [{"id": translator.LLM_MODEL, "loaded": False}], f"LLM 서버에서 모델 목록을 가져오지 못했습니다: {e}"


@app.get("/")
def index():
    models, models_error = get_models()
    return render_template("index.html", models=models, models_error=models_error, default_model=translator.LLM_MODEL)


@app.get("/open")
def open_link():
    model = request.args.get("model") or translator.LLM_MODEL
    if not MODEL_RE.match(model):
        abort(400)
    try:
        src = translator.parse_source(request.args.get("url", ""))
    except translator.PeparError as e:
        return render_template("progress.html", error=str(e), model=model, key="", src_url=""), 400
    remember_source(src)
    return redirect(url_for("view", model=model, key=src.key))


@app.post("/upload")
def upload():
    """이미지를 받아 저장하고 번역 화면으로 보낸다. 같은 이미지는 같은 key가 되어 캐시를 다시 쓴다."""
    model = request.form.get("model") or translator.LLM_MODEL
    if not MODEL_RE.match(model):
        abort(400)
    f = request.files.get("image")

    def fail(msg):
        return render_template("progress.html", error=msg, model=model, key="", src_url=""), 400

    if not f or not f.filename:
        return fail("이미지 파일을 골라 주세요.")
    try:
        models = {m["id"]: m for m in translator.list_models()}
    except Exception:  # noqa: BLE001 - 목록을 못 가져오면 확인 없이 진행하고, 안 되면 작업이 실패로 알려준다
        models = {}
    if model in models and not models[model]["vision"]:
        vision = ", ".join(m for m in models if models[m]["vision"])
        return fail(f"{model}은(는) 이미지를 읽지 못하는 모델입니다. 이미지를 지원하는 모델: {vision}")
    try:
        data, ext, _, _ = ocr.normalize(f.read(MAX_UPLOAD_READ))
    except translator.PeparError as e:
        return fail(str(e))
    key = ocr.image_key(data)
    if not image_path(key):
        (IMAGES / f"{key}.{ext}").write_bytes(data)
    with sources_lock:
        sources = load_sources()
        sources[key] = {"kind": "image", "url": f"/image/{key}", "name": Path(f.filename).name[:200]}
        SOURCES.write_text(json.dumps(sources, ensure_ascii=False, indent=1), encoding="utf-8")
    return redirect(url_for("view", model=model, key=key))


MAX_UPLOAD_READ = ocr.MAX_UPLOAD + 1  # 한도를 넘었는지 알 수 있을 만큼만 읽는다


@app.get("/image/<key>")
def image(key):
    if not KEY_RE.match(key) or not key.startswith("i-"):
        abort(404)
    path = image_path(key)
    if not path:
        abort(404)
    return send_file(path, max_age=86400)


@app.get("/view/<model>/<key>")
def view(model, key):
    if not MODEL_RE.match(model) or not KEY_RE.match(key):
        abort(404)
    job = jobs.get(f"{model}/{key}")
    active = job is not None and job["stage"] in ACTIVE
    if request.args.get("retranslate") == "1":  # 전부 새로 번역
        if not active:
            checkpoint_path(model, key).unlink(missing_ok=True)
            start_job(model, key)
        return redirect(url_for("view", model=model, key=key))
    if request.args.get("retry_failed") == "1":  # 체크포인트를 두고 다시 돌리면 빠진 블록만 번역한다
        if not active:
            start_job(model, key)
        return redirect(url_for("view", model=model, key=key))
    if not active:
        labels = CACHE / model / f"{key}.labels.json"
        if key.startswith("i-") and labels.exists():
            return render_image(model, key, json.loads(labels.read_text(encoding="utf-8")))
        path = cached_path(model, key)
        if path:
            return path.read_text(encoding="utf-8")
        start_job(model, key)  # 처음이거나, 지난번에 실패했으면 이어서 다시 시도
    title = image_name(key) if key.startswith("i-") else ""
    return render_template("progress.html", error="", model=model, key=key, src_url=source_url(key), title=title)


@app.get("/paper/<key>")
def old_paper_link(key):
    return redirect(url_for("view", model=translator.LLM_MODEL, key=key))


# ---------------------------------------------------------------- API


@app.get("/api/docs")
def api_docs():
    """번역한 문서 전체, 최근 번역 순."""
    sources = load_sources()
    docs = [doc_info(p, sources) for p in CACHE.glob("*/*.html") if p.parent != IMAGES]
    docs.sort(key=lambda d: d["translated_at"], reverse=True)
    return jsonify(docs)


@app.delete("/api/docs/<model>/<key>")
def api_delete_doc(model, key):
    """번역 결과와 그 기록(meta, 체크포인트)을 지운다. 원본 URL 기록(sources.json)은 다른 모델이 쓸 수 있어 남긴다."""
    if not MODEL_RE.match(model) or not KEY_RE.match(key):
        abort(404)
    job = jobs.get(f"{model}/{key}")
    if job and job["stage"] in ACTIVE:
        return jsonify(error="번역 중인 문서는 지울 수 없습니다."), 409
    d = CACHE / model
    removed = 0
    for name in (f"{key}.html", f"{key}.meta.json", f"{key}.partial.json", f"{key}.labels.json"):
        p = d / name
        if p.exists():
            p.unlink()
            removed += 1
    _text_cache.pop(str(d / f"{key}.html"), None)
    jobs.pop(f"{model}/{key}", None)
    if d.exists() and not any(d.iterdir()):
        d.rmdir()
    others = [p for p in CACHE.glob(f"*/{key}.*") if p.parent != IMAGES]
    if key.startswith("i-") and not others and (p := image_path(key)):
        p.unlink()  # 이 이미지를 번역한 모델이 더 없으면 원본도 지운다
    if not removed:
        return jsonify(error="문서를 찾지 못했습니다."), 404
    return jsonify(ok=True)


@app.get("/api/search")
def api_search():
    """번역된 본문 전문 검색. [{model, key, snippet}]"""
    q = " ".join(request.args.get("q", "").split())
    if len(q) < 2:
        return jsonify([])
    pat = re.compile(re.escape(q), re.I)
    out = []
    for p in CACHE.glob("*/*.html"):
        text = doc_text(p)
        m = pat.search(text)
        if m:
            a, b = max(0, m.start() - 60), min(len(text), m.end() + 80)
            out.append({
                "model": p.parent.name,
                "key": p.stem,
                "snippet": ("…" if a else "") + text[a:b] + ("…" if b < len(text) else ""),
                "count": len(pat.findall(text)),
            })
    return jsonify(out)


@app.get("/api/jobs")
def api_jobs():
    """진행 중·대기 중인 작업과, 최근 24시간 안에 실패한(닫지 않은) 작업."""
    now = time.time()
    out = [
        job_view(j)
        for j in sorted(jobs.values(), key=lambda j: j["queued_at"])
        if j["stage"] in ACTIVE or (j["stage"] == "error" and now - j.get("finished_at", now) < FAILED_KEEP)
    ]
    return jsonify(out)


@app.delete("/api/jobs/<model>/<key>")
def dismiss_job(model, key):
    """실패한 작업 카드를 닫는다."""
    job = jobs.get(f"{model}/{key}")
    if not job or job["stage"] != "error":
        return jsonify(error="닫을 수 있는 실패 작업이 없습니다."), 404
    jobs.pop(f"{model}/{key}")
    save_failed()
    return jsonify(ok=True)


@app.get("/api/jobs/<model>/<key>")
def job_status(model, key):
    job = jobs.get(f"{model}/{key}")
    if not job:
        return jsonify(stage="done" if cached_path(model, key) else "unknown")
    return jsonify(job_view(job))


if __name__ == "__main__":
    from werkzeug.serving import make_server

    # 쉼표로 여러 주소에 동시에 바인딩 (localhost + Tailscale)
    hosts = os.environ.get("PEPAR_HOST", "127.0.0.1,100.92.150.128").split(",")
    port = int(os.environ.get("PEPAR_PORT", "8765"))
    servers = [make_server(h.strip(), port, app, threaded=True) for h in hosts]
    for s in servers[1:]:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    print("pepar listening on", ", ".join(f"http://{h.strip()}:{port}" for h in hosts), flush=True)
    servers[0].serve_forever()
