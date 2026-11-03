"""arXiv 논문이나 일반 웹페이지를 가져와 블록 단위로 로컬 LLM에 번역시키고, 번역된 HTML을 만든다."""

import copy
import hashlib
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

import netguard
from config import settings
from errors import LLMUnavailable, PeparError

LLM_URL = settings["llm_url"]
BATCH_CHARS = 2500
BATCH_ITEMS = 12
MAX_BLOCKS = 3000

ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")
BARE_ARXIV_ID = re.compile(r"^(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)$", re.I)
# LLM에 보내는 자리표시: <x3/> 보존할 요소, <t3>…</t3> 링크·강조
TOKEN = re.compile(r"<(/?)([tx])(\d+)\s*/?>")
OLD_TOKEN = re.compile(r"⟦(/?)(\d+)⟧")

UA_PEPAR = {"User-Agent": "pepar/0.1 (personal paper translator)"}
UA_BROWSER = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}




@dataclass
class Source:
    kind: str  # "arxiv" | "web" | "image" | "text" | "pdf"
    key: str
    url: str


class PdfFound(Exception):
    """가져온 문서가 PDF였다. app이 PDF 번역으로 넘긴다."""

    def __init__(self, data: bytes, url: str):
        super().__init__(url)
        self.data, self.url = data, url


def unwrap_viewer(url: str) -> str:
    """웹 PDF 뷰어로 감싼 주소면 안쪽 PDF 주소를. PDF.js(viewer.html?file=...), Google Docs·Drive 뷰어(?url=...)."""
    u = urlparse(url)
    q = parse_qs(u.query)
    if u.path.endswith("viewer.html") and q.get("file"):
        return urljoin(url, q["file"][0])  # 뷰어 기준 상대 경로일 수 있다
    if u.hostname in ("docs.google.com", "drive.google.com") and ("viewer" in u.path or "gview" in u.path) and q.get("url"):
        return q["url"][0]
    return url


def _web(url: str) -> Source:
    return Source("web", "w-" + hashlib.sha1(url.encode()).hexdigest()[:12], url)


def parse_source(text: str, pdf: bool = False) -> Source:
    """pdf: PDF 탭의 링크 칸. arXiv도 웹(HTML) 버전이 아니라 그 논문의 PDF를 번역한다."""
    text = text.strip()
    if not text:
        raise PeparError("URL이나 arXiv ID를 입력해 주세요.")
    m = BARE_ARXIV_ID.match(text)
    if m:
        return _web(f"https://arxiv.org/pdf/{m.group(1)}") if pdf else Source("arxiv", m.group(1), m.group(1))
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
        return _web(f"https://arxiv.org/pdf/{arxiv_id}") if pdf else Source("arxiv", arxiv_id, arxiv_id)
    url = urldefrag(unwrap_viewer(text))[0]
    if url != urldefrag(text)[0] and not re.match(r"^https?://", url, re.I):
        raise PeparError("PDF 뷰어 주소에서 PDF 파일 주소를 찾지 못했습니다.")
    return _web(url)




@dataclass
class Doc:
    soup: BeautifulSoup
    url: str
    key: str
    kind: str
    title: str
    context: str
    items: list = field(default_factory=list)
    keep_scripts: bool = False
    lang: str = "ko"


ARXIV_BLOCKS = "article.ltx_document p.ltx_p, article.ltx_document .ltx_title, article.ltx_document figcaption"
ARXIV_ATOMIC = "math, cite, a, .ltx_tag, .ltx_note, .ltx_pubnotes, img, svg"

WEB_BLOCKS = "p, h1, h2, h3, h4, h5, h6, li, dt, dd, figcaption, blockquote, td, th, caption, summary, div"
WEB_ATOMIC = "pre, code, kbd, samp, var, math, img, svg, picture, video, audio, iframe, sup, br, input, select, textarea, button"
WEB_PAIRS = "a, em, strong, b, i, u, mark"
SKIP_ROLES = ["navigation", "contentinfo", "search", "banner"]
WEB_SKIP = ["nav", "footer", "aside", "form", "pre", "code", "button", "template", "svg", "math", "select", "textarea"]


def describe_request_error(e: Exception) -> str:
    print(f"[pepar] 요청 실패: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    if isinstance(e, requests.exceptions.Timeout):
        return "응답 시간이 초과되었습니다."
    if isinstance(e, requests.exceptions.SSLError):
        return "보안 연결(SSL)에 실패했습니다."
    if isinstance(e, requests.exceptions.TooManyRedirects):
        return "리다이렉트가 너무 많습니다."
    if isinstance(e, (requests.exceptions.InvalidURL, requests.exceptions.InvalidSchema, requests.exceptions.MissingSchema)):
        return "주소 형식이 올바르지 않습니다."
    if isinstance(e, requests.exceptions.ConnectionError):
        return "서버에 연결할 수 없습니다. 주소가 맞는지, 사이트가 열려 있는지 확인해 주세요."
    return "요청에 실패했습니다."


def _outermost(nodes):
    chosen = set()
    for n in nodes:
        if not any(id(p) in chosen for p in n.parents):
            chosen.add(id(n))
            yield n


def _text(el) -> str:
    return " ".join(el.get_text(" ").split())


def load_arxiv(src: Source, lang: str = "ko") -> Doc:
    try:
        r = requests.get(f"https://arxiv.org/html/{src.url}", headers=UA_PEPAR, timeout=60)
    except requests.RequestException as e:
        raise PeparError(f"arXiv에서 논문을 가져오지 못했습니다: {describe_request_error(e)}") from e
    if r.status_code == 404:
        # HTML 버전이 없는 논문은 PDF로
        try:
            p = requests.get(f"https://arxiv.org/pdf/{src.url}", headers=UA_PEPAR, timeout=120)
        except requests.RequestException as e:
            raise PeparError(f"arXiv에서 논문 PDF를 가져오지 못했습니다: {describe_request_error(e)}") from e
        if p.status_code == 200 and p.content.startswith(b"%PDF-"):
            raise PdfFound(p.content, p.url)
        raise PeparError("arXiv에서 이 논문을 찾지 못했습니다 (HTML·PDF 모두 없음). 논문 ID가 맞는지 확인해 주세요.")
    if r.status_code >= 400:
        raise PeparError(f"arXiv에서 논문을 가져오지 못했습니다 (HTTP {r.status_code}).")
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

    doc = Doc(soup, r.url, resolved, "arxiv", title, "\n".join(parts), lang=lang)
    for el in _outermost(soup.select(ARXIV_BLOCKS)):
        if el.find_parent(class_=re.compile(r"ltx_(bibliography|tabular|equation)")):
            continue
        _add_item(doc, el, ARXIV_ATOMIC, None)
    return doc


def load_web(src: Source, keep_scripts: bool = settings["keep_scripts"], lang: str = "ko") -> Doc:
    try:
        r = netguard.fetch(src.url, UA_BROWSER, settings["allow_private_urls"])
    except requests.RequestException as e:
        raise PeparError(f"페이지를 가져오지 못했습니다: {describe_request_error(e)}") from e
    if r.status_code in (401, 403, 429, 503) and (r.headers.get("cf-mitigated") or "cloudflare" in r.headers.get("server", "").lower()):
        raise PeparError(f"사이트의 봇 차단(Cloudflare)에 막혀 페이지를 가져올 수 없습니다 (HTTP {r.status_code}).")
    if r.status_code in (401, 403):
        raise PeparError(f"사이트가 접근을 거부했습니다 (HTTP {r.status_code}). 로그인이 필요하거나 봇을 차단하는 사이트일 수 있습니다.")
    if r.status_code >= 400:
        raise PeparError(f"페이지를 가져오지 못했습니다 (HTTP {r.status_code}).")
    ctype = r.headers.get("content-type", "")
    if "pdf" in ctype or r.content.startswith(b"%PDF-"):
        raise PdfFound(r.content, r.url)
    if "html" not in ctype and "xml" not in ctype:
        raise PeparError(f"HTML 페이지도 PDF도 아닙니다 ({ctype.split(';')[0] or '알 수 없는 형식'}).")
    soup = BeautifulSoup(r.content, "lxml")
    if not soup.body:
        raise PeparError("페이지 본문이 비어 있습니다.")

    base = r.url
    if soup.base and soup.base.get("href"):
        base = urljoin(r.url, soup.base["href"])
        soup.base.decompose()
    if not keep_scripts:
        # 스크립트가 다시 그리며 번역을 덮어쓰지 않도록 지운다
        for t in soup(["script", "noscript"]):
            t.decompose()
    for meta in soup.find_all("meta", attrs={"http-equiv": re.compile("content-security-policy", re.I)}):
        meta.decompose()
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

    doc = Doc(soup, base, src.key, "web", title, "\n".join(parts), keep_scripts=keep_scripts, lang=lang)
    blocks = root.select(WEB_BLOCKS)
    block_ids = {id(b) for b in blocks}
    has_inner = {id(p) for b in blocks for p in b.parents if id(p) in block_ids}
    for el in blocks:
        if id(el) in has_inner:
            continue
        if el.find_parent(WEB_SKIP) or el.find_parent(attrs={"role": SKIP_ROLES}) or el.get("aria-hidden") == "true":
            continue
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


# ---------------------------------------------------------------- 붙여넣은 텍스트

TEXT_MAX = 200_000
_BULLET = re.compile(r"^\s*([-*•·▪◦‣]|\d+[.)]|[a-zA-Z][.)])\s")
_CJK_END = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7a3]")
TEXT_CSS = """
  :root { --bg: #fafaf9; --fg: #1c1a17; --muted: #6b6760; }
  @media (prefers-color-scheme: dark) { :root { --bg: #161615; --fg: #ecebe8; --muted: #9c9890; } }
  body { margin: 0; background: var(--bg); color: var(--fg); font: 17px/1.75 system-ui, "Pretendard", "Noto Sans KR", sans-serif; }
  main.pepar-text { max-width: 760px; margin: 0 auto; padding: 40px 16px 120px; }
  .para { margin: 0 0 1.1em; } .para p { margin: 0; overflow-wrap: anywhere; }
"""


def _width(line: str) -> int:
    """글줄의 대략적인 폭. 한중일 문자는 라틴 문자 두 개 폭."""
    return len(line) + len(_CJK_END.findall(line))


def _hard_wrapped(lines: list[str]) -> bool:
    """일정한 폭에서 끊긴 줄들인지 (마지막 줄 말고는 가장 긴 줄의 75% 이상)."""
    widths = [_width(line) for line in lines]
    top = max(widths)
    return top >= 30 and all(w >= 0.75 * top for w in widths[:-1])


def text_paragraphs(text: str) -> list[list[str]]:
    """붙여넣은 텍스트 → 문단마다 줄 목록 (복사하며 끊긴 줄은 잇는다)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    paras = []
    for block in re.split(r"\n\s*\n", text):
        lines = [" ".join(line.split()) for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        if len(lines) > 1 and (any(_BULLET.match(line) for line in lines) or not _hard_wrapped(lines)):
            paras.append(lines)
            continue
        joined = lines[0]
        for line in lines[1:]:
            if joined.endswith("-") and line[:1].islower():
                joined = joined[:-1] + line
            else:
                joined += ("" if _CJK_END.match(joined[-1]) and _CJK_END.match(line[0]) else " ") + line
        paras.append([joined])
    return paras


def load_text(src: Source, text: str, lang: str = "ko") -> Doc:
    if len(text) > TEXT_MAX:
        raise PeparError(f"텍스트가 너무 깁니다 (최대 {TEXT_MAX:,}자).")
    paras = text_paragraphs(text)
    if not paras:
        raise PeparError("번역할 텍스트가 비어 있습니다.")
    title = paras[0][0][:80]
    soup = BeautifulSoup(
        f'<!doctype html><html><head><title></title><link rel="icon" href="/favicon.svg" type="image/svg+xml"><style>{TEXT_CSS}</style></head>'
        '<body><main class="pepar-text"></main></body></html>',
        "lxml",
    )
    soup.title.string = title
    main = soup.select_one("main")
    for lines in paras:
        div = soup.new_tag("div", attrs={"class": "para"})
        for line in lines:
            p = soup.new_tag("p")
            p.string = line
            div.append(p)
        main.append(div)
    doc = Doc(soup, src.url, src.key, "text", title, "Beginning of the text: " + text[:1200], lang=lang)
    for p in main.select("p"):
        _add_item(doc, p, "br", None)
        if len(doc.items) > MAX_BLOCKS:
            raise PeparError(f"텍스트가 너무 깁니다 (번역 블록 {MAX_BLOCKS}개 초과).")
    if not doc.items:
        raise PeparError("번역할 글자가 없습니다. 이미 번역할 언어로 된 글일 수 있습니다.")
    return doc


SEG_CLASS = "pepar-seg"


def _split_at_br(soup, el):
    """<br>로 나뉜 블록은 줄 구간마다 따로 번역한다 (통째로 보내면 모델이 줄 경계에서 응답을 쪼개 토큰 검사에 실패한다)."""
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


def load(src: Source, keep_scripts: bool = settings["keep_scripts"], lang: str = "ko", text: str | None = None) -> Doc:
    if src.kind == "text":
        return load_text(src, text or "", lang)
    return load_arxiv(src, lang) if src.kind == "arxiv" else load_web(src, keep_scripts, lang)




def protect(el, atomic: str, pairs: str | None) -> tuple[str, list[str], dict[int, str]]:
    """보존할 요소를 <xN/>로, 링크·강조를 <tN>…</tN>로 치환한다. 반환: (LLM에 보낼 텍스트, 원본 조각/여는 태그 목록, {n: 닫는 태그})."""
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


LANGS = {
    "ko": ("한국어", "Korean"),
    "en": ("English", "English"),
    "ja": ("日本語", "Japanese"),
    "zh": ("中文(简体)", "Simplified Chinese"),
    "es": ("Español", "Spanish"),
    "fr": ("Français", "French"),
    "de": ("Deutsch", "German"),
}
DEFAULT_LANG = "ko"

CUSTOM_LANGS: dict[str, str] = {}
CUSTOM_RE = re.compile(r"^x-[0-9a-f]{8}$")
CUSTOM_MAX = 40


def custom_code(text: str) -> str:
    """같은 표현이면 대소문자·띄어쓰기가 달라도 같은 코드."""
    norm = " ".join(text.lower().split())
    return "x-" + hashlib.sha256(norm.encode()).hexdigest()[:8]


def clean_custom(text: str) -> str:
    """직접 입력한 언어 이름. 제어 문자와 줄바꿈을 없애고 길이를 자른다. 비면 PeparError."""
    text = " ".join("".join(c for c in text if c.isprintable()).split())[:CUSTOM_MAX]
    if not text:
        raise PeparError("번역할 언어를 입력해 주세요.")
    return text


def is_lang(code: str) -> bool:
    return code in LANGS or code in CUSTOM_LANGS


def lang_name(code: str) -> str:
    """화면에 보일 이름."""
    if code in LANGS:
        return LANGS[code][0]
    return CUSTOM_LANGS.get(code, code)

# 번역할 글자가 있는지: 번역할 언어의 문자가 아닌 글자가 2자 이상 (일본어는 가나만 자기 문자)
KANA = "\u3040-\u30ff"
HANZI = "\u3400-\u9fff"
_OWN_SCRIPT = {"ko": "가-힣ㄱ-ㅎㅏ-ㅣ", "ja": KANA, "zh": HANZI}
_WORTH = {lang: re.compile(rf"[^\W\d_{_OWN_SCRIPT.get(lang, '')}]") for lang in LANGS}
_WORTH_ANY = re.compile(r"[^\W\d_]")
_WORTH_JA_TEXT = re.compile(rf"[^\W\d_{KANA}{HANZI}]")
_HAS_KANA = re.compile(rf"[{KANA}]")


def worth(text: str, lang: str = "ko") -> bool:
    pat = _WORTH_JA_TEXT if lang == "ja" and _HAS_KANA.search(text) else _WORTH.get(lang, _WORTH_ANY)
    return len(pat.findall(text)) >= 2


def _add_item(doc: Doc, el, atomic: str, pairs: str | None):
    orig = el.decode_contents()
    text, saved, closers = protect(el, atomic, pairs)
    if not worth(TOKEN.sub("", text), doc.lang):
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



SYSTEM = """You are a professional translator into {lang}.

Document context (use it for consistent terminology only; do not translate it):
{context}

Rules:
1. Translate each item's "text" into natural {lang}. {style}
2. Self-closing tags like <x0/>, <x1/> stand for formulas, citations, code, images or other elements that must be kept. Copy every such tag exactly once and unchanged, placed where it fits {lang} grammar.
3. A tag pair <t0>...</t0> marks a link or emphasis. Keep both tags and put them around the {lang} words that correspond to the original span.
4. Never add, drop or renumber tags.
5. Keep product/model/method names, acronyms, dataset and metric names, code identifiers in their original form. A technical term may be translated with the original in parentheses the first time it appears in an item.
6. If an item is already in {lang}, return it unchanged.
7. Output ONLY a JSON array of objects {{"id": <same id>, "text": "<{lang}>"}} covering every input id. No commentary."""

STYLE = {
    "ko": {
        "pdf": "This is a PDF document (often a paper or report). Each item is a paragraph, heading or label drawn over the "
        "original in its box: use the written plain style (~한다, ~이다) for prose and translate headings and labels concisely.",
        "arxiv": "This is an academic paper: use the written plain style (~한다, ~이다), not 합니다체.",
        "web": "Use the written plain style (~한다, ~이다) for prose; translate short UI labels and headings concisely.",
    },
    "ja": {
        "arxiv": "This is an academic paper: use the written plain style (である調), not です・ます調.",
    },
    "*": {
        "pdf": "This is a PDF document (often a paper or report). Each item is a paragraph, heading or label drawn over the "
        "original in its box: use a formal written register for prose and translate headings and labels concisely.",
        "text": "This is plain text pasted by the user. Keep its tone and register (formal or casual, polite or plain) "
        "and keep list markers and numbering.",
        "arxiv": "This is an academic paper: use a formal academic register.",
        "web": "Use natural written prose; translate short UI labels and headings concisely.",
    },
}
IMAGE_EXAMPLES = {
    "ko": "'hidden state' → '은닉 상태', 'Input Encoder' → '입력 인코더'",
    "ja": "'hidden state' → '隠れ状態', 'Input Encoder' → '入力エンコーダ'",
    "zh": "'hidden state' → '隐藏状态', 'Input Encoder' → '输入编码器'",
}
IMAGE_STYLE = (
    "These are text labels read from an image (a diagram, figure, screenshot, sign or photo). Each translated label is "
    "drawn over the original in the same box, so keep it about as short as the original; use noun phrases for labels. "
    "Unlike rule 5, translate common technical terms{examples} and do not add the original in parentheses; "
    "keep only proper names, acronyms, symbols and code as they are."
)


def system_prompt(context: str, kind: str, lang: str = "ko") -> str:
    if kind == "image":
        ex = IMAGE_EXAMPLES.get(lang)
        style = IMAGE_STYLE.format(examples=f" (e.g. {ex})" if ex else "")
    else:
        style = STYLE.get(lang, {}).get(kind) or STYLE["*"][kind]
    if lang in LANGS:
        return SYSTEM.format(lang=LANGS[lang][1], context=context, style=style)
    target = CUSTOM_LANGS[lang]
    style = (
        f'The target language was typed by the user as free text: "{target}". Interpret it as a language, dialect, '
        f"script or register and write every translation in it. {style}"
    )
    return SYSTEM.format(lang=f'"{target}"', context=context, style=style)


def list_models() -> list[dict]:
    """[{id, loaded, vision}]. vision은 모르면 None (llama-server 실행 방식마다 대답 형식이 다르다)."""
    r = requests.get(LLM_URL.rsplit("/chat/completions", 1)[0] + "/models", timeout=5)
    r.raise_for_status()
    body = r.json()
    caps = {x.get("name") or x.get("model"): x["capabilities"]
            for x in body.get("models") or [] if isinstance(x, dict) and isinstance(x.get("capabilities"), list)}
    out = []
    for m in body.get("data", []):
        if m.get("id") == "default":
            continue
        modalities = (m.get("architecture") or {}).get("input_modalities")
        if modalities is not None:
            vision = "image" in modalities
        elif m["id"] in caps:
            vision = "multimodal" in caps[m["id"]]
        else:
            vision = None
        status = m.get("status")
        out.append({
            "id": m["id"],
            "loaded": status is None or status.get("value") in ("loaded", "sleeping"),
            "vision": vision,
        })
    return out


def pick_default_model(models: list[dict]) -> str:
    for m in models:
        if m["loaded"]:
            return m["id"]
    return models[0]["id"] if models else ""


class Cancelled(Exception):
    """작업이 취소되었다."""


should_stop = lambda: False  # noqa: E731


def _post(body: dict, attempts: int = 8, wait: int = 20) -> dict:
    """LLM에 보낸다. 서버가 안 되면 기다렸다 다시 하고, 기다리는 동안에도 취소를 본다."""
    last = detail = ""
    for n in range(attempts):
        if should_stop():
            raise Cancelled
        try:
            r = requests.post(LLM_URL, json=body, timeout=900)
            if r.status_code < 500:
                r.raise_for_status()
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            detail = f"HTTP {r.status_code}"
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as e:
            # 응답이 중간에 끊긴 것(LLM 서버가 죽었다 다시 뜰 때)도 다시 시도
            last = f"{type(e).__name__}: {e}"
            detail = ("응답 시간 초과" if isinstance(e, requests.Timeout)
                      else "응답이 중간에 끊김" if isinstance(e, requests.exceptions.ChunkedEncodingError) else "연결 실패")
        print(f"[pepar] LLM 서버 응답 없음 ({n + 1}/{attempts}), {wait}초 후 재시도: {last}", file=sys.stderr, flush=True)
        until = time.time() + wait
        while time.time() < until:
            if should_stop():
                raise Cancelled
            time.sleep(min(0.5, max(0, until - time.time())))
    raise LLMUnavailable(
        f"LLM 서버({LLM_URL})가 응답하지 않습니다 ({detail}). 서버가 실행 중인지 확인한 뒤 다시 시도해 주세요. 이미 번역한 부분은 이어서 진행됩니다."
    )


# 스키마로 강제해 JSON이 깨지지 않게
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
    """{항목 index: 번역문}. 실패한 항목은 빠진다."""
    out = {}
    try:
        result = _call(model, system, [{"id": k, "text": items[i]["text"]} for k, i in enumerate(idxs)])
        by_id = {r.get("id"): r.get("text") for r in result if isinstance(r, dict)}
        for k, i in enumerate(idxs):
            if _valid(items[i]["text"], by_id.get(k)):
                out[i] = by_id[k]
    except (requests.HTTPError, ValueError, KeyError) as e:
        print(f"[pepar] 배치 실패, 항목별로 재시도: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    # 실패한 항목은 하나씩: 같은 온도 → 높은 온도 → 링크·강조 태그를 빼고
    for i in idxs:
        if i in out:
            continue
        text = items[i]["text"]
        attempts = [(text, 0.2, True), (text, 0.7, True)]
        if len(idxs) == 1:
            attempts = attempts[1:]
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

  // 텍스트 번역: 지금 보이는 쪽(번역 또는 원문)을 문단 구분을 살려 복사한다
  const copyBtn = document.getElementById('pepar-copy');
  if (copyBtn) copyBtn.addEventListener('click', async () => {
    const text = [...document.querySelectorAll('main.pepar-text .para')]
      .map(d => [...d.querySelectorAll('p')].map(p => p.textContent).join('\\n')).join('\\n\\n');
    let ok = false;
    try { await navigator.clipboard.writeText(text); ok = true; } catch {}
    if (!ok) {  // http로 접속하면 clipboard API를 쓸 수 없다
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.append(ta);
      ta.select();
      try { ok = document.execCommand('copy'); } catch {}
      ta.remove();
    }
    const label = copyBtn.textContent;
    copyBtn.textContent = ok ? '복사됨' : '복사 실패';
    setTimeout(() => { copyBtn.textContent = label; }, 1500);
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
    src_label = {"arxiv": "arXiv 원문", "text": "원문 텍스트"}.get(doc.kind, "원문 페이지")
    copy = '<button id="pepar-copy" type="button">복사</button>' if doc.kind == "text" else ""
    retry = ""
    if failed and retry_url:
        retry = f'<a class="pepar-retry" href="{html_escape(retry_url)}" title="번역에 실패한 블록만 다시 번역">실패 {failed}블록 재번역</a>'
    return (
        '<div class="pepar-bar">'
        f'<span class="pepar-model">{html_escape(model)}</span>'
        f"{retry}"
        f'<a href="{html_escape(home)}">PEPAR</a>'
        f'<a href="{html_escape(doc.url)}" target="_blank" rel="noopener">{src_label}</a>'
        f"{copy}"
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
    """자리표시 태그를 원래 요소로 복원하고 나머지는 이스케이프한다."""
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
    """예전 ⟦n⟧ 체크포인트를 태그 형식으로 바꾼다."""
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
    reapply_map = {}
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
        soup.html["lang"] = doc.lang
    head = soup.head
    if head is None:
        head = soup.new_tag("head")
        soup.html.insert(0, head)
    for meta in head.find_all("meta", charset=True):
        meta.decompose()
    head.insert(0, BeautifulSoup('<meta charset="utf-8">', "html.parser"))
    if doc.keep_scripts:
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




def run(
    src: Source,
    model: str,
    on_progress=lambda stage, done, total, **info: None,
    home: str = "/",
    checkpoint=None,
    keep_scripts: bool = settings["keep_scripts"],
    retry_url: str | None = None,
    label: str | None = None,
    lang: str = "ko",
    text: str | None = None,
):
    """(key, 제목, 번역 HTML, 실패 블록 수). checkpoint에는 묶음마다 {원문: 번역문}을 저장해 끊겨도 이어서 번역한다. label은 결과 페이지에 표시할 모델 이름."""
    on_progress("fetching", 0, 0)
    doc = load(src, keep_scripts, lang, text)
    items = doc.items
    total = len(items)
    system = system_prompt(doc.context, doc.kind, lang)

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
    return doc.key, doc.title, build(doc, translations, label or model, home, retry_url), total - len(translations)
