"""pepar — 논문·웹페이지 번역 웹앱."""

import hashlib
import hmac
import html as htmllib
import ipaddress
import json
import os
import queue
import re
import secrets
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for
from PIL import Image
from werkzeug.exceptions import HTTPException

import ocr
import translator
from config import settings

CACHE = settings["cache_dir"]
CACHE.mkdir(parents=True, exist_ok=True)
SOURCES = CACHE / "sources.json"  # web key는 해시라 URL을 따로 기록
IMAGES = CACHE / "images"
IMAGES.mkdir(exist_ok=True)
KEY_RE = re.compile(r"^(\d{4}\.\d{4,5}(v\d+)?|[wi]-[0-9a-f]{12})$")
MODEL_RE = re.compile(r"^[\w.\-]+$")
ACTIVE = ("queued", "fetching", "reading", "translating")


class Cancelled(Exception):
    pass


app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

# llama-server가 parallel=1이라 번역은 한 번에 하나씩
jobs: dict[str, dict] = {}  # "model/key" → 상태
jobs_lock = threading.Lock()
work: "queue.Queue[dict]" = queue.Queue()
sources_lock = threading.Lock()
sessions_lock = threading.Lock()
FAILED = CACHE / "failed_jobs.json"
FAILED_KEEP = 24 * 3600
# 블록당 평균 번역 시간(초), 대기 시간 추정용
sec_per_block = 7.0


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


def doc_info(p: Path, sources: dict) -> dict:
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


_text_cache: dict[str, tuple[float, str]] = {}


def doc_text(p: Path) -> str:
    mtime = p.stat().st_mtime
    hit = _text_cache.get(str(p))
    if hit and hit[0] == mtime:
        return hit[1]
    raw = p.read_text(encoding="utf-8", errors="ignore")
    raw = re.sub(r"<head\b.*?</head>", " ", raw, flags=re.S | re.I)
    raw = re.sub(r"<(script|style)\b.*?</\1>", " ", raw, flags=re.S | re.I)
    raw = re.sub(r'<div class="pepar-bar">.*', " ", raw, flags=re.S)
    raw = re.sub(r"<[^>]+>", " ", raw)  # data-pepar-alt(원문)도 함께 빠진다
    text = " ".join(htmllib.unescape(raw).split())
    _text_cache[str(p)] = (mtime, text)
    return text




def worker():
    global sec_per_block
    while True:
        job = work.get()
        if job.get("cancel"):
            continue
        model, key = job["model"], job["key"]
        llm = job.get("llm") or model

        def progress(stage, done, total, **info):
            if job.get("cancel"):
                raise Cancelled
            job.update(stage=stage, done=done, total=total, **info)
            if stage == "translating" and "started" not in job:
                job["started"] = time.time()
                job["resumed"] = done  # 체크포인트에서 이어받은 블록 수

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
                html, failed = translate_image(model, key, title, progress, checkpoint, retry_url, llm)
            else:
                resolved, title, html, failed = translator.run(
                    src, llm, progress, checkpoint=checkpoint, retry_url=retry_url,
                    label=model if llm == model else f"{model} + {llm}",
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
        except Cancelled:
            with jobs_lock:
                if jobs.get(f"{model}/{key}") is job:
                    del jobs[f"{model}/{key}"]
        except translator.PeparError as e:
            job.update(stage="error", message=str(e))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            job.update(stage="error", message=f"예기치 못한 오류가 발생했습니다 ({type(e).__name__}). 서버 로그를 확인해 주세요.")
        job["finished_at"] = time.time()
        save_failed()


def translate_image(model, key, title, progress, checkpoint, retry_url, llm) -> tuple[str, int]:
    """라벨은 <key>.labels.json에도 저장한다 (보기 화면은 매번 최신 템플릿으로 그림)."""
    path = image_path(key)
    labels, failed = ocr.run(llm, path.read_bytes(), progress, checkpoint=checkpoint)
    with Image.open(path) as img:
        width, height = img.size
    data = {"title": title, "labels": labels, "failed": failed, "width": width, "height": height}
    (model_dir(model) / f"{key}.labels.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with app.app_context():
        return render_image(model, key, data), failed


def render_image(model: str, key: str, data: dict) -> str:
    return render_template("image_view.html", model=model, key=key, retry_url=f"/view/{model}/{key}?retry_failed=1", **data)


def save_failed():
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


def start_job(model: str, key: str, llm: str | None = None):
    with jobs_lock:
        job = jobs.get(f"{model}/{key}")
        if job and job["stage"] in ACTIVE:
            return
        job = jobs[f"{model}/{key}"] = {
            "model": model,
            "key": key,
            "url": source_url(key),
            "stage": "queued",
            "done": 0,
            "total": 0,
            "message": "",
            "queued_at": time.time(),
        }
        if llm and llm != model:
            job["llm"] = llm
        work.put(job)


def job_rate(job: dict) -> float | None:
    done_here = job["done"] - job.get("resumed", 0)
    if job.get("started") and done_here > 0:
        return (time.time() - job["started"]) / done_here
    return None


def job_eta(job: dict) -> int | None:
    if not job["total"]:
        return None
    rate = job_rate(job) or sec_per_block
    return round(rate * (job["total"] - job["done"]))


def job_view(job: dict) -> dict:
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




def get_models() -> tuple[list[dict], str]:
    try:
        return translator.list_models(), ""
    except Exception as e:  # noqa: BLE001
        print(f"[pepar] 모델 목록 실패: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return [], f"LLM 서버({translator.LLM_URL})에 연결할 수 없습니다. 서버가 실행 중인지, config.toml의 llm_url이 맞는지 확인해 주세요."


def pick_model(requested: str | None) -> str:
    if requested:
        return requested
    model = translator.pick_default_model(get_models()[0])
    if not model:
        raise translator.PeparError("사용할 수 있는 모델이 없습니다. LLM 서버 주소(config.toml의 llm_url)와 실행 상태를 확인해 주세요.")
    return model


PASSWORD = settings["password"]


LOGIN_FAILS = 5  # 이만큼 틀리면 그 IP는 LOGIN_WINDOW초 동안 로그인할 수 없다
LOGIN_WINDOW = 15 * 60
SESSION_LIFETIME = 30 * 86400
TRUSTED_PROXIES = [ipaddress.ip_network(p) for p in settings["trusted_proxies"]]
SESSIONS = CACHE / "sessions.json"
failed_logins: dict[str, list[float]] = {}
login_lock = threading.Lock()
_sessions: dict | None = None  # {"secret", "fp", "sessions": {sha256(token): 만료 시각}}


def from_trusted_proxy() -> bool:
    try:
        peer = ipaddress.ip_address(request.remote_addr or "")
    except ValueError:
        return False
    return any(peer in net for net in TRUSTED_PROXIES)


def client_ip() -> str:
    """신뢰하는 프록시(Cloudflare Tunnel 등)가 보낸 CF-Connecting-IP만 믿는다. IPv6는 /64 단위로 묶는다."""
    ip = request.remote_addr or ""
    if from_trusted_proxy():
        ip = request.headers.get("CF-Connecting-IP", "").strip() or ip
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return request.remote_addr or ""
    return str(ipaddress.ip_network(f"{addr}/64", strict=False)) if addr.version == 6 else str(addr)


def _sessions_state() -> dict:
    """세션 저장소. 비밀번호가 바뀌었으면 기존 세션을 모두 버린다. 호출할 때 sessions_lock을 잡고 있어야 한다."""
    global _sessions
    if _sessions is None:
        try:
            _sessions = json.loads(SESSIONS.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            _sessions = {}
        _sessions.setdefault("secret", secrets.token_hex(32))
    fp = hmac.new(_sessions["secret"].encode(), PASSWORD.encode(), "sha256").hexdigest()
    if _sessions.get("fp") != fp:
        _sessions.update(fp=fp, sessions={})
    return _sessions


def _save_sessions():
    tmp = SESSIONS.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(_sessions, f)
    os.replace(tmp, SESSIONS)


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def start_session() -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    with sessions_lock:
        state = _sessions_state()
        live = {h: exp for h, exp in state["sessions"].items() if exp > now}
        state["sessions"] = dict(sorted(live.items(), key=lambda kv: kv[1])[-199:])
        state["sessions"][_digest(token)] = now + SESSION_LIFETIME
        _save_sessions()
    return token


def end_session(token: str):
    with sessions_lock:
        if _sessions_state()["sessions"].pop(_digest(token), None) is not None:
            _save_sessions()


def logged_in() -> bool:
    token = request.cookies.get("pepar_auth", "")
    if not token:
        return False
    with sessions_lock:
        return _sessions_state()["sessions"].get(_digest(token), 0) > time.time()


def lockout_seconds(ip: str) -> int:
    now = time.time()
    with login_lock:
        recent = [t for t in failed_logins.get(ip, []) if now - t < LOGIN_WINDOW]
        if recent:
            failed_logins[ip] = recent
        else:
            failed_logins.pop(ip, None)
        return int(recent[0] + LOGIN_WINDOW - now) + 1 if len(recent) >= LOGIN_FAILS else 0


def safe_next(path: str) -> str:
    return path if path.startswith("/") and not path.startswith("//") and "\\" not in path else "/"


@app.before_request
def require_login():
    if not PASSWORD or request.endpoint in ("login", "logout") or logged_in():
        return None
    if request.path.startswith("/api/"):
        return jsonify(error="로그인이 필요합니다."), 401
    return redirect(url_for("login", next=request.full_path.rstrip("?")))


@app.route("/login", methods=["GET", "POST"])
def login():
    nxt = safe_next(request.values.get("next", ""))
    if not PASSWORD or (request.method == "GET" and logged_in()):
        return redirect(nxt)
    if request.method == "GET":
        return render_template("login.html", error="", next=nxt)
    ip = client_ip()
    wait = lockout_seconds(ip)
    if wait:
        return render_template("login.html", error=f"시도가 너무 많습니다. {wait // 60 + 1}분 뒤에 다시 시도하세요.", next=nxt), 429
    if hmac.compare_digest(request.form.get("password", "").encode(), PASSWORD.encode()):
        failed_logins.pop(ip, None)
        resp = redirect(nxt)
        forwarded_https = request.headers.get("X-Forwarded-Proto", "").lower() == "https" or '"https"' in request.headers.get("CF-Visitor", "")
        secure = request.is_secure or (from_trusted_proxy() and forwarded_https)
        resp.set_cookie("pepar_auth", start_session(), max_age=SESSION_LIFETIME, httponly=True, samesite="Strict", secure=secure)
        return resp
    with login_lock:
        failed_logins.setdefault(ip, []).append(time.time())
    time.sleep(1)
    return render_template("login.html", error="비밀번호가 틀렸습니다.", next=nxt), 401


@app.after_request
def no_shared_cache(resp):
    if PASSWORD:
        resp.headers["Cache-Control"] = "private, no-store"
    return resp


@app.get("/logout")
def logout():
    end_session(request.cookies.get("pepar_auth", ""))
    resp = redirect(url_for("login"))
    resp.delete_cookie("pepar_auth")
    return resp


HTTP_ERRORS = {
    400: "잘못된 요청입니다.",
    404: "찾을 수 없는 페이지입니다. 번역 기록이 삭제되었을 수 있습니다.",
    405: "허용되지 않는 요청입니다.",
    413: f"업로드한 파일이 너무 큽니다 (최대 {app.config['MAX_CONTENT_LENGTH'] // 1024 // 1024}MB).",
    500: "서버에서 오류가 발생했습니다. 서버 로그를 확인해 주세요.",
}


@app.errorhandler(HTTPException)
def http_error(e: HTTPException):
    msg = HTTP_ERRORS.get(e.code, "요청을 처리하지 못했습니다.")
    if request.path.startswith("/api/"):
        return jsonify(error=msg), e.code
    return render_template("progress.html", error=msg, model="", key="", src_url=""), e.code


@app.get("/")
def index():
    models, models_error = get_models()
    return render_template("index.html", models=models, models_error=models_error, default_model=translator.pick_default_model(models), auth=bool(PASSWORD))


@app.get("/open")
def open_link():
    try:
        model = pick_model(request.args.get("model"))
        if not MODEL_RE.match(model):
            abort(400)
        src = translator.parse_source(request.args.get("url", ""))
    except translator.PeparError as e:
        return render_template("progress.html", error=str(e), model="", key="", src_url=""), 400
    remember_source(src)
    return redirect(url_for("view", model=model, key=src.key))


@app.post("/upload")
def upload():
    """같은 이미지는 같은 key라 캐시를 다시 쓴다."""
    def fail(msg):
        return render_template("progress.html", error=msg, model=model, key="", src_url=""), 400

    model = ""
    try:
        model = pick_model(request.form.get("model"))
    except translator.PeparError as e:
        return fail(str(e))
    if not MODEL_RE.match(model):
        abort(400)
    f = request.files.get("image")

    if not f or not f.filename:
        return fail("이미지 파일을 골라 주세요.")
    try:
        models = {m["id"]: m for m in translator.list_models()}
    except Exception:  # noqa: BLE001 - 목록 없이 진행
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


MAX_UPLOAD_READ = ocr.MAX_UPLOAD + 1  # 한도 초과를 알 수 있도록 1바이트 더 읽는다


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
    if request.args.get("retranslate") == "1":
        if not active:
            checkpoint_path(model, key).unlink(missing_ok=True)
            start_job(model, key)
        return redirect(url_for("view", model=model, key=key))
    if request.args.get("retry_failed") == "1":  # 체크포인트를 두고 다시 돌려 빠진 블록만 번역
        llm = request.args.get("with") or None  # 빠진 블록을 번역할 모델
        if llm and not MODEL_RE.match(llm):
            abort(400)
        if not active:
            start_job(model, key, llm)
        return redirect(url_for("view", model=model, key=key))
    if not active:
        labels = CACHE / model / f"{key}.labels.json"
        if key.startswith("i-") and labels.exists():
            return render_image(model, key, json.loads(labels.read_text(encoding="utf-8")))
        path = cached_path(model, key)
        if path:
            return path.read_text(encoding="utf-8")
        start_job(model, key)  # 처음이거나 지난번에 실패했으면 이어서 시도
    title = image_name(key) if key.startswith("i-") else ""
    return render_template("progress.html", error="", model=model, key=key, src_url=source_url(key), title=title)


@app.get("/paper/<key>")
def old_paper_link(key):
    try:
        model = pick_model(None)
    except translator.PeparError as e:
        return render_template("progress.html", error=str(e), model="", key="", src_url=""), 503
    return redirect(url_for("view", model=model, key=key))




@app.get("/api/docs")
def api_docs():
    sources = load_sources()
    docs = [doc_info(p, sources) for p in CACHE.glob("*/*.html") if p.parent != IMAGES]
    docs.sort(key=lambda d: d["translated_at"], reverse=True)
    return jsonify(docs)


@app.delete("/api/docs/<model>/<key>")
def api_delete_doc(model, key):
    """번역 결과와 기록을 지운다. sources.json은 다른 모델이 쓸 수 있어 남긴다."""
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
    """진행·대기 중인 작업과 최근 24시간 안의 실패 작업."""
    now = time.time()
    out = [
        job_view(j)
        for j in sorted(jobs.values(), key=lambda j: j["queued_at"])
        if j["stage"] in ACTIVE or (j["stage"] == "error" and now - j.get("finished_at", now) < FAILED_KEEP)
    ]
    return jsonify(out)


@app.delete("/api/jobs/<model>/<key>")
def close_job(model, key):
    """실패한 작업 카드를 닫거나, 진행 중인 작업을 취소한다."""
    with jobs_lock:
        job = jobs.get(f"{model}/{key}")
        if not job:
            return jsonify(error="해당 작업이 없습니다."), 404
        if job["stage"] == "error":
            del jobs[f"{model}/{key}"]
            save_failed()
        else:
            job["cancel"] = True
            if job["stage"] == "queued":
                del jobs[f"{model}/{key}"]
            else:
                job["cancelling"] = True
    return jsonify(ok=True)


@app.get("/api/jobs/<model>/<key>")
def job_status(model, key):
    job = jobs.get(f"{model}/{key}")
    if not job:
        return jsonify(stage="done" if cached_path(model, key) else "unknown")
    return jsonify(job_view(job))


if __name__ == "__main__":
    from werkzeug.serving import make_server

    hosts = settings["host"]
    port = settings["port"]
    servers = [make_server(h, port, app, threaded=True) for h in hosts]
    for s in servers[1:]:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    print("pepar listening on", ", ".join(f"http://{h}:{port}" for h in hosts), flush=True)
    servers[0].serve_forever()
