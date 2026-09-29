"""arXiv 논문이나 일반 웹페이지를 가져와 블록 단위로 로컬 LLM에 번역시키고, 번역된 HTML을 만든다."""

import copy
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

LLM_URL = os.environ.get("PEPAR_LLM_URL", "http://127.0.0.1:9090/v1/chat/completions")
LLM_MODEL = os.environ.get("PEPAR_MODEL", "gemma4-12b")
BATCH_CHARS = int(os.environ.get("PEPAR_BATCH_CHARS", "2500"))
BATCH_ITEMS = 12
MAX_BLOCKS = 3000

ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")
BARE_ARXIV_ID = re.compile(r"^(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)$", re.I)
# LLM에 보내는 자리표시 태그 (모델이 HTML에 익숙해서 괄호 기호보다 잘 지킨다)
#   <x3/>        통째로 보존한 요소 (수식, 인용, 코드, 이미지 등)
#   <t3>…</t3>   링크·강조. 모델이 번역문의 알맞은 낱말을 감싸도록 한다
# 그룹: 1=닫는 태그면 "/", 2=종류(t|x), 3=번호
TOKEN = re.compile(r"<(/?)([tx])(\d+)\s*/?>")
OLD_TOKEN = re.compile(r"⟦(/?)(\d+)⟧")  # 예전 체크포인트 형식

UA_PEPAR = {"User-Agent": "pepar/0.1 (personal paper translator)"}
UA_BROWSER = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


class PeparError(Exception):
    pass


class LLMUnavailable(PeparError):
    pass


# ---------------------------------------------------------------- 입력 판별


@dataclass
class Source:
    kind: str  # "arxiv" | "web" | "image"
    key: str  # 캐시/URL에 쓰는 식별자: "2412.13742" 또는 "w-<hash>"
    url: str  # web이면 원본 URL, arxiv면 논문 ID


def parse_source(text: str) -> Source:
    text = text.strip()
    if not text:
        raise PeparError("URL이나 arXiv ID를 입력해 주세요.")
    m = BARE_ARXIV_ID.match(text)
    if m:
        return Source("arxiv", m.group(1), m.group(1))
    if not re.match(r"^https?://", text, re.I):
        if " " in text or "." not in text:
            raise PeparError("URL이나 arXiv ID를 인식하지 못했습니다.")
        text = "https://" + text
    host = urlparse(text).hostname or ""
    if host == "arxiv.org" or host.endswith(".arxiv.org"):
        m = ARXIV_ID.search(urlparse(text).path)
        if not m:
            raise PeparError("arXiv 링크에서 논문 ID를 찾지 못했습니다.")
        arxiv_id = m.group(1) + (m.group(2) or "")
        return Source("arxiv", arxiv_id, arxiv_id)
    url = urldefrag(text)[0]
    return Source("web", "w-" + hashlib.sha1(url.encode()).hexdigest()[:12], url)


# ---------------------------------------------------------------- 문서 불러오기


@dataclass
class Doc:
    soup: BeautifulSoup
    url: str  # 상대 경로 해석 기준
    key: str  # arXiv면 버전 포함 ID로 확정됨
    kind: str
    title: str
    context: str
    items: list = field(default_factory=list)
    keep_scripts: bool = False  # web: 사이트 스크립트를 살려 두고 번역은 브라우저에서 다시 입힌다


ARXIV_BLOCKS = "article.ltx_document p.ltx_p, article.ltx_document .ltx_title, article.ltx_document figcaption"
ARXIV_ATOMIC = "math, cite, a, .ltx_tag, .ltx_note, .ltx_pubnotes, img, svg"

WEB_BLOCKS = "p, h1, h2, h3, h4, h5, h6, li, dt, dd, figcaption, blockquote, td, th, caption, summary, div"
WEB_ATOMIC = "pre, code, kbd, samp, var, math, img, svg, picture, video, audio, iframe, sup, br, input, select, textarea, button"
WEB_PAIRS = "a, em, strong, b, i, u, mark"
SKIP_ROLES = ["navigation", "contentinfo", "search", "banner"]
WEB_SKIP = ["nav", "footer", "aside", "form", "pre", "code", "button", "template", "svg", "math", "select", "textarea"]


def _outermost(nodes):
    chosen = set()
    for n in nodes:
        if not any(id(p) in chosen for p in n.parents):
            chosen.add(id(n))
            yield n


def _text(el) -> str:
    return " ".join(el.get_text(" ").split())


def load_arxiv(src: Source) -> Doc:
    r = requests.get(f"https://arxiv.org/html/{src.url}", headers=UA_PEPAR, timeout=60)
    if r.status_code == 404:
        raise PeparError("이 논문은 arXiv HTML 버전이 없습니다 (LaTeX → HTML 변환 실패 논문).")
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "lxml")
    if not soup.select_one("article.ltx_document"):
        raise PeparError("arXiv HTML 논문 본문을 찾지 못했습니다.")
    m = re.search(rf"arXiv:({re.escape(src.url.split('v')[0])}v\d+)", soup.get_text(" "))
    resolved = m.group(1) if m else src.url

    parts, title = [], ""
    t = soup.select_one("h1.ltx_title_document")
    if t:
        t = copy.copy(t)
        for n in t.select(".ltx_pubnotes, .ltx_note"):
            n.decompose()
        title = _text(t)
        parts.append("Title: " + title)
    abstract = soup.select_one(".ltx_abstract")
    if abstract:
        parts.append("Abstract: " + _text(abstract).removeprefix("Abstract").strip())

    doc = Doc(soup, r.url, resolved, "arxiv", title, "\n".join(parts))
    for el in _outermost(soup.select(ARXIV_BLOCKS)):
        if el.find_parent(class_=re.compile(r"ltx_(bibliography|tabular|equation)")):
            continue
        _add_item(doc, el, ARXIV_ATOMIC, None)
    return doc


def load_web(src: Source, keep_scripts: bool = True) -> Doc:
    try:
        r = requests.get(src.url, headers=UA_BROWSER, timeout=60)
    except requests.RequestException as e:
        raise PeparError(f"페이지를 가져오지 못했습니다: {e}") from e
    if r.status_code in (401, 403, 429, 503) and (r.headers.get("cf-mitigated") or "cloudflare" in r.headers.get("server", "").lower()):
        raise PeparError(f"사이트의 봇 차단(Cloudflare)에 막혀 페이지를 가져올 수 없습니다 (HTTP {r.status_code}).")
    if r.status_code in (401, 403):
        raise PeparError(f"사이트가 접근을 거부했습니다 (HTTP {r.status_code}). 로그인이 필요하거나 봇을 차단하는 사이트일 수 있습니다.")
    if r.status_code >= 400:
        raise PeparError(f"페이지를 가져오지 못했습니다 (HTTP {r.status_code}).")
    ctype = r.headers.get("content-type", "")
    if "html" not in ctype and "xml" not in ctype:
        raise PeparError(f"HTML 페이지가 아닙니다 ({ctype.split(';')[0] or '알 수 없는 형식'}). PDF 등은 아직 지원하지 않습니다.")
    soup = BeautifulSoup(r.content, "lxml")
    if not soup.body:
        raise PeparError("페이지 본문이 비어 있습니다.")

    base = r.url
    if soup.base and soup.base.get("href"):
        base = urljoin(r.url, soup.base["href"])
        soup.base.decompose()
    if not keep_scripts:
        # 정적 모드: 스크립트가 다시 렌더링하면서 번역을 원문으로 덮어쓰지 않도록 제거한다.
        # 대신 스크립트로 그리는 영역(상품 목록, 탭 등)은 비거나 동작하지 않는다.
        for t in soup(["script", "noscript"]):
            t.decompose()
    for meta in soup.find_all("meta", attrs={"http-equiv": re.compile("content-security-policy", re.I)}):
        meta.decompose()
    # lazy-load 이미지는 스크립트 없이도 보이게
    for img in soup.find_all("img"):
        lazy = img.get("data-src") or img.get("data-lazy-src")
        if lazy and (not img.get("src") or img["src"].startswith("data:")):
            img["src"] = lazy

    title = _text(soup.title) if soup.title else ""
    desc = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
    root = soup.select_one("main") or soup.select_one('[role="main"]') or soup.select_one("article") or soup.body
    parts = [f"Title: {title}"] if title else []
    if desc and desc.get("content"):
        parts.append("Description: " + desc["content"].strip())
    parts.append("Beginning of the page: " + _text(root)[:1200])

    doc = Doc(soup, base, src.key, "web", title, "\n".join(parts), keep_scripts=keep_scripts)
    blocks = root.select(WEB_BLOCKS)
    block_ids = {id(b) for b in blocks}
    # 가장 안쪽 블록만 번역 단위로 쓴다 (블록 안에 블록이 있으면 안쪽 것을 택함)
    has_inner = {id(p) for b in blocks for p in b.parents if id(p) in block_ids}
    for el in blocks:
        if id(el) in has_inner:
            continue
        if el.find_parent(WEB_SKIP) or el.find_parent(attrs={"role": SKIP_ROLES}) or el.get("aria-hidden") == "true":
            continue
        # 다른 언어판으로 가는 링크(위키백과의 언어 목록 등)는 번역하지 않는다
        lang_link = el.find(lambda t: t.name == "a" and (t.get("hreflang") or t.get("lang")))
        if lang_link and _text(lang_link) == _text(el):
            continue
        for unit in _split_at_br(soup, el):
            _add_item(doc, unit, WEB_ATOMIC, WEB_PAIRS)
        if len(doc.items) > MAX_BLOCKS:
            raise PeparError(f"페이지가 너무 깁니다 (번역 블록 {MAX_BLOCKS}개 초과).")
    if not doc.items:
        raise PeparError("번역할 본문을 찾지 못했습니다. 자바스크립트로만 그려지는 페이지일 수 있습니다.")
    return doc


SEG_CLASS = "pepar-seg"


def _split_at_br(soup, el):
    """<br>로 줄을 나눈 블록은 줄 구간마다 따로 번역한다.

    <br>을 토큰으로 바꿔 통째로 보내면, 모델이 줄 경계에서 응답을 여러 항목으로 쪼개
    토큰 검사에 계속 실패한다 (예: 문단을 <br><br>로만 구분한 긴 상품 설명).
    직계 자식 <br> 사이의 구간을 <span class="pepar-seg">로 감싸 각각 번역 단위로 쓴다.
    """
    if not el.find("br", recursive=False):
        return [el]
    runs, cur = [], []
    for node in list(el.contents):
        if getattr(node, "name", None) == "br":
            runs.append(cur)
            cur = []
        else:
            cur.append(node)
    runs.append(cur)
    units = []
    for run in runs:
        if not any((n.get_text() if hasattr(n, "get_text") else str(n)).strip() for n in run):
            continue
        span = soup.new_tag("span", attrs={"class": SEG_CLASS})
        run[0].insert_before(span)
        for n in run:
            span.append(n.extract())
        units.append(span)
    return units


def load(src: Source, keep_scripts: bool = True) -> Doc:
    return load_arxiv(src) if src.kind == "arxiv" else load_web(src, keep_scripts)


# ---------------------------------------------------------------- 추출


def protect(el, atomic: str, pairs: str | None) -> tuple[str, list[str], dict[int, str]]:
    """보존할 요소를 <xN/>로, 링크·강조는 <tN>…</tN>로 치환한다. el을 변경한다.

    반환: (LLM에 보낼 텍스트, 원본 조각/여는 태그 목록, {n: 닫는 태그})
    """
    saved: list[str] = []
    closers: dict[int, str] = {}
    for node in list(_outermost(el.select(atomic))):
        saved.append(str(node))
        node.replace_with(f"<x{len(saved) - 1}/>")
    if pairs:
        for node in el.select(pairs):
            n = len(saved)
            shell = copy.copy(node)
            shell.clear()
            s = str(shell)
            saved.append(s[: s.rindex("</")] if "</" in s else s)
            closers[n] = f"</{node.name}>"
            node.insert_before(f"<t{n}>")
            node.insert_after(f"</t{n}>")
            node.unwrap()
    return " ".join(el.get_text().split()), saved, closers


# 번역할 가치가 있는지: 토큰을 뺀 텍스트에 한글이 아닌 글자가 2자 이상
WORTH = re.compile(r"[^\W\d_가-힣ㄱ-ㅎㅏ-ㅣ]{2}")


def _add_item(doc: Doc, el, atomic: str, pairs: str | None):
    orig = el.decode_contents()
    text, saved, closers = protect(el, atomic, pairs)
    if not WORTH.search(TOKEN.sub("", text)):
        el.clear()
        el.append(BeautifulSoup(orig, "html.parser"))
        return
    doc.items.append({"el": el, "orig": orig, "text": text, "saved": saved, "closers": closers})


def batches(items):
    cur, size = [], 0
    for i, it in enumerate(items):
        if cur and (size + len(it["text"]) > BATCH_CHARS or len(cur) >= BATCH_ITEMS):
            yield cur
            cur, size = [], 0
        cur.append(i)
        size += len(it["text"])
    if cur:
        yield cur


# ---------------------------------------------------------------- LLM

SYSTEM = """You are a professional translator into Korean.

Document context (use it for consistent terminology only; do not translate it):
{context}

Rules:
1. Translate each item's "text" into natural Korean. {style}
2. Self-closing tags like <x0/>, <x1/> stand for formulas, citations, code, images or other elements that must be kept. Copy every such tag exactly once and unchanged, placed where it fits Korean grammar.
3. A tag pair <t0>...</t0> marks a link or emphasis. Keep both tags and put them around the Korean words that correspond to the original span.
4. Never add, drop or renumber tags.
5. Keep product/model/method names, acronyms, dataset and metric names, code identifiers in their original form. A technical term may be translated with the original in parentheses the first time it appears in an item.
6. Output ONLY a JSON array of objects {{"id": <same id>, "text": "<Korean>"}} covering every input id. No commentary."""

STYLE = {
    "arxiv": "This is an academic paper: use the written plain style (~한다, ~이다), not 합니다체.",
    "web": "Use the written plain style (~한다, ~이다) for prose; translate short UI labels and headings concisely.",
    "image": "These are text labels read from an image (a diagram, figure, screenshot, sign or photo). Each Korean label is "
    "drawn over the original in the same box, so keep it about as short as the original; use noun phrases for labels. "
    "Unlike rule 5, translate common technical terms into Korean (e.g. 'hidden state' → '은닉 상태', 'Input Encoder' → "
    "'입력 인코더') and do not add the original in parentheses; keep only proper names, acronyms, symbols and code as they are.",
}


def list_models() -> list[dict]:
    """라우터에 등록된 모델 목록: [{id, loaded, vision}]."""
    url = LLM_URL.rsplit("/chat/completions", 1)[0] + "/models"
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    out = []
    for m in r.json().get("data", []):
        if m.get("id") == "default":
            continue
        out.append({
            "id": m["id"],
            "loaded": (m.get("status") or {}).get("value") in ("loaded", "sleeping"),
            "vision": "image" in (m.get("architecture") or {}).get("input_modalities", []),
        })
    return out


def _post(body: dict, attempts: int = 8, wait: int = 20) -> dict:
    """LLM 서버가 죽었거나 모델을 다시 올리는 중이면 잠시 기다렸다 재시도하고, 끝내 안 되면 작업을 중단한다."""
    last = ""
    for n in range(attempts):
        try:
            r = requests.post(LLM_URL, json=body, timeout=900)
            if r.status_code < 500:
                r.raise_for_status()
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except (requests.ConnectionError, requests.Timeout) as e:
            last = f"{type(e).__name__}: {e}"
        print(f"[pepar] LLM 서버 응답 없음 ({n + 1}/{attempts}), {wait}초 후 재시도: {last}", file=sys.stderr, flush=True)
        time.sleep(wait)
    raise LLMUnavailable(f"LLM 서버({LLM_URL})에 연결할 수 없습니다: {last}")


# 응답을 이 스키마로 강제한다 (llama-server가 문법 수준에서 막아서, 따옴표 때문에 JSON이 깨지지 않는다)
RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "text": {"type": "string"}},
        "required": ["id", "text"],
    },
}


def _call(model: str, system: str, payload: list[dict], temperature: float = 0.2) -> list[dict]:
    body = {
        "model": model,
        "temperature": temperature,
        "response_format": {"type": "json_schema", "json_schema": {"name": "translations", "schema": RESPONSE_SCHEMA}},
        "max_tokens": 8192,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    }
    content = _post(body)["choices"][0]["message"]["content"]
    i, j = content.find("["), content.rfind("]")
    if i < 0 or j < i:
        raise ValueError("JSON 배열 없음")
    return json.loads(content[i : j + 1])


def _tokens(s: str, pairs: bool = True) -> Counter:
    return Counter(f"{m.group(1)}{m.group(2)}{m.group(3)}" for m in TOKEN.finditer(s) if pairs or m.group(2) == "x")


def _valid(src_text: str, translated, pairs: bool = True) -> bool:
    if not isinstance(translated, str) or not translated.strip():
        return False
    return _tokens(translated, pairs) == _tokens(src_text, pairs)


def _strip_pairs(s: str) -> str:
    return TOKEN.sub(lambda m: m.group(0) if m.group(2) == "x" else "", s)


def _one(model: str, system: str, text: str, temperature: float) -> str | None:
    result = _call(model, system, [{"id": 0, "text": text}], temperature)
    return result[0].get("text") if result and isinstance(result[0], dict) else None


def translate_batch(model: str, system: str, items: list[dict], idxs: list[int]) -> dict[int, str]:
    """{item index: 번역문}. 실패한 항목은 빠진다."""
    out = {}
    try:
        result = _call(model, system, [{"id": k, "text": items[i]["text"]} for k, i in enumerate(idxs)])
        by_id = {r.get("id"): r.get("text") for r in result if isinstance(r, dict)}
        for k, i in enumerate(idxs):
            if _valid(items[i]["text"], by_id.get(k)):
                out[i] = by_id[k]
    except (requests.HTTPError, ValueError, KeyError) as e:
        print(f"[pepar] 배치 실패, 항목별로 재시도: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    # 실패한 항목은 하나씩: 같은 온도 → 온도를 올려서 → 링크·강조 태그를 빼고(링크는 잃지만 번역은 살린다)
    for i in idxs:
        if i in out:
            continue
        text = items[i]["text"]
        attempts = [(text, 0.2, True), (text, 0.7, True)]
        if len(idxs) == 1:
            attempts = attempts[1:]  # 배치가 이미 이 항목 하나였으면 같은 조건 재시도는 건너뜀
        if _strip_pairs(text) != text:
            attempts.append((_strip_pairs(text), 0.2, False))
        for src_text, temp, pairs in attempts:
            try:
                t = _one(model, system, src_text, temp)
            except (requests.HTTPError, ValueError, KeyError, IndexError) as e:
                print(f"[pepar] 항목 {i} 재시도 오류: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                continue
            if _valid(src_text, t, pairs):
                out[i] = t
                if not pairs:
                    print(f"[pepar] 항목 {i}: 링크·강조 표시를 빼고 번역함", file=sys.stderr, flush=True)
                break
        else:
            print(f"[pepar] 항목 {i} 번역 실패, 원문 유지: {text[:80]!r}", file=sys.stderr, flush=True)
    return out


# ---------------------------------------------------------------- 결과 HTML

PEPAR_HEAD = """
<style>
  .pepar-bar { position: fixed; right: 16px; bottom: 16px; z-index: 2147483647; display: flex; gap: 8px;
    align-items: center; font: 14px/1.2 system-ui, sans-serif; }
  .pepar-bar a, .pepar-bar button { padding: 8px 12px; border-radius: 8px; border: 1px solid #8884;
    background: #1f6feb; color: #fff; cursor: pointer; text-decoration: none; font: inherit; }
  .pepar-bar a { background: #444; }
  .pepar-bar .pepar-model { padding: 4px 8px; border-radius: 6px; background: #0008; color: #fff; font-size: 12px; }
  .pepar-bar a.pepar-retry { background: #d29922; }
  .pepar-untranslated { border-left: 3px dotted #d29922; padding-left: 6px; }
</style>
"""

PEPAR_SCRIPT = """
(() => {
  const cfg = JSON.parse(document.getElementById('pepar-config').textContent);
  const btn = document.getElementById('pepar-toggle');
  const norm = s => s.replace(/\\s+/g, ' ').trim();
  let showingOrig = false;

  // 원문/번역 전환: 번역된 블록마다 data-pepar-alt에 반대쪽 HTML을 들고 있다
  btn.addEventListener('click', () => {
    document.querySelectorAll('[data-pepar-alt]').forEach(el => {
      const cur = el.innerHTML;
      el.innerHTML = el.dataset.peparAlt;
      el.dataset.peparAlt = cur;
    });
    showingOrig = !showingOrig;
    btn.textContent = showingOrig ? '번역 보기' : '원문 보기';
  });

  if (!cfg.map) return;

  // 스크립트 유지 모드: 사이트 스크립트가 블록을 원문으로 다시 그리면, 원문 텍스트로 번역을 찾아 다시 입힌다
  function reapply() {
    if (showingOrig) return;
    for (const el of document.querySelectorAll(cfg.blocks)) {
      if (el.closest('.pepar-bar') || el.querySelector(cfg.blocks)) continue;
      const ko = cfg.map[norm(el.textContent)];
      if (ko !== undefined) {
        el.dataset.peparAlt = el.innerHTML;
        el.innerHTML = ko;
      }
    }
  }
  let timer;
  new MutationObserver(() => { clearTimeout(timer); timer = setTimeout(reapply, 120); })
    .observe(document.documentElement, { childList: true, subtree: true, characterData: true });
  reapply();

  // <base>가 원본 사이트를 가리키므로 페이지 안 이동(#...) 링크는 여기서 스크롤로 처리한다
  document.addEventListener('click', e => {
    const a = e.target.closest && e.target.closest('a[data-pepar-hash]');
    if (!a || e.defaultPrevented) return;
    const id = decodeURIComponent(a.dataset.peparHash.slice(1));
    const target = id ? (document.getElementById(id) || document.getElementsByName(id)[0]) : document.body;
    if (!target) return;
    e.preventDefault();
    target.scrollIntoView();
    history.replaceState(null, '', '#' + id);
  });
})();
"""


def _bar_html(doc: "Doc", model: str, home: str, failed: int, retry_url: str | None) -> str:
    src_label = "arXiv 원문" if doc.kind == "arxiv" else "원문 페이지"
    retry = ""
    if failed and retry_url:
        retry = f'<a class="pepar-retry" href="{html_escape(retry_url)}" title="번역에 실패한 블록만 다시 번역">실패 {failed}블록 재번역</a>'
    return (
        '<div class="pepar-bar">'
        f'<span class="pepar-model">{html_escape(model)}</span>'
        f"{retry}"
        f'<a href="{html_escape(home)}">PEPAR</a>'
        f'<a href="{html_escape(doc.url)}" target="_blank" rel="noopener">{src_label}</a>'
        '<button id="pepar-toggle" type="button">원문 보기</button>'
        "</div>"
    )

SKIP_URL = ("#", "data:", "javascript:", "mailto:", "tel:")


def _absolutize(soup, base: str):
    for tag in soup.find_all(True):
        for attr in ("src", "href", "data", "poster", "action"):
            v = tag.get(attr)
            if isinstance(v, str) and v and not v.startswith(SKIP_URL):
                tag[attr] = urljoin(base, v)
        srcset = tag.get("srcset")
        if isinstance(srcset, str) and srcset:
            parts = []
            for c in srcset.split(","):
                bits = c.strip().split(None, 1)
                if bits:
                    bits[0] = urljoin(base, bits[0])
                    parts.append(" ".join(bits))
            tag["srcset"] = ", ".join(parts)


def _restore(translated: str, it: dict) -> str:
    """번역문의 자리표시 태그를 원래 요소·태그로 바꾸고, 나머지 글자는 이스케이프한 HTML."""
    out, pos = [], 0
    for m in TOKEN.finditer(translated):
        out.append(html_escape(translated[pos : m.start()]))
        n = int(m.group(3))
        if m.group(1):
            out.append(it["closers"].get(n, ""))
        elif n < len(it["saved"]):
            out.append(it["saved"][n])
        pos = m.end()
    out.append(html_escape(translated[pos:]))
    return "".join(out)


def _migrate_checkpoint(saved: dict) -> dict:
    """⟦n⟧ 형식으로 저장된 예전 체크포인트를 태그 형식으로 바꾼다. 쌍 여부는 원문(키)에서 판단."""
    def conv(s: str, pairs: set) -> str:
        return OLD_TOKEN.sub(lambda m: f"</t{m.group(2)}>" if m.group(1) else (f"<t{m.group(2)}>" if m.group(2) in pairs else f"<x{m.group(2)}/>"), s)
    out = {}
    for k, v in saved.items():
        if "⟦" in k or "⟦" in v:
            pairs = {m.group(2) for m in OLD_TOKEN.finditer(k) if m.group(1)}
            k, v = conv(k, pairs), conv(v, pairs)
        out[k] = v
    return out


def html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build(doc: Doc, translations: dict[int, str], model: str, home: str, retry_url: str | None = None) -> str:
    soup = doc.soup
    reapply_map = {}  # 원문 텍스트 → 번역 HTML (스크립트 유지 모드에서 브라우저가 다시 입힐 때 씀)
    for i, it in enumerate(doc.items):
        el = it["el"]
        el.clear()
        if i in translations:
            ko = _restore(translations[i], it)
            el.append(BeautifulSoup(ko, "html.parser"))
            el["data-pepar-alt"] = it["orig"]
            if doc.keep_scripts:
                orig_text = " ".join(BeautifulSoup(it["orig"], "html.parser").get_text().split())
                reapply_map[orig_text] = el.decode_contents()
        else:
            el.append(BeautifulSoup(it["orig"], "html.parser"))
            el["class"] = el.get("class", []) + ["pepar-untranslated"]
    if doc.keep_scripts:
        for a in soup.select('a[href^="#"]'):
            a["data-pepar-hash"] = a["href"]
    _absolutize(soup, doc.url)
    if soup.html:
        soup.html["lang"] = "ko"
    head = soup.head
    if head is None:
        head = soup.new_tag("head")
        soup.html.insert(0, head)
    for meta in head.find_all("meta", charset=True):
        meta.decompose()
    head.insert(0, BeautifulSoup('<meta charset="utf-8">', "html.parser"))
    if doc.keep_scripts:
        # 사이트 스크립트가 상대 경로로 부르는 파일·API가 원본 사이트로 가도록
        base = soup.new_tag("base", href=doc.url)
        head.insert(1, base)
    if soup.title and soup.title.string:
        soup.title.string = "[번역] " + soup.title.string
    head.append(BeautifulSoup(PEPAR_HEAD, "html.parser"))
    failed = len(doc.items) - len(translations)
    config = {"map": reapply_map if doc.keep_scripts else None, "blocks": f"{WEB_BLOCKS}, span.{SEG_CLASS}"}
    config_json = json.dumps(config, ensure_ascii=False).replace("</", "<\\/")
    soup.body.append(BeautifulSoup(_bar_html(doc, model, home, failed, retry_url), "html.parser"))
    soup.body.append(BeautifulSoup(
        f'<script type="application/json" id="pepar-config">{config_json}</script><script>{PEPAR_SCRIPT}</script>',
        "html.parser",
    ))
    return str(soup)


# ---------------------------------------------------------------- 전체 파이프라인


def run(
    src: Source,
    model: str,
    on_progress=lambda stage, done, total, **info: None,
    home: str = "/",
    checkpoint=None,
    keep_scripts: bool = True,
    retry_url: str | None = None,
):
    """(확정된 key, 제목, 번역된 HTML, 번역 실패 블록 수)를 반환.

    on_progress(stage, done, total, **info): 진행 상황 콜백. 원문을 불러온 뒤 한 번 title=문서 제목을 함께 넘긴다.
    checkpoint: 묶음마다 {원문: 번역문}을 저장할 JSON 경로. 작업이 중간에 끊겨도
    다음 실행에서 이미 번역한 블록은 건너뛴다.
    """
    on_progress("fetching", 0, 0)
    doc = load(src, keep_scripts)
    items = doc.items
    total = len(items)
    system = SYSTEM.format(context=doc.context, style=STYLE[doc.kind])

    saved = {}
    if checkpoint and checkpoint.exists():
        try:
            saved = _migrate_checkpoint(json.loads(checkpoint.read_text(encoding="utf-8")))
        except ValueError:
            saved = {}
    translations = {i: saved[it["text"]] for i, it in enumerate(items) if it["text"] in saved}
    todo = [i for i in range(total) if i not in translations]

    done = total - len(todo)
    on_progress("translating", done, total, title=doc.title)
    for idxs in batches([items[i] for i in todo]):
        real = [todo[k] for k in idxs]
        new = translate_batch(model, system, items, real)
        translations.update(new)
        if checkpoint:
            saved.update({items[i]["text"]: t for i, t in new.items()})
            checkpoint.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
        done += len(real)
        on_progress("translating", done, total)
    return doc.key, doc.title, build(doc, translations, model, home, retry_url), total - len(translations)
