"""PDF: 쪽마다 그림으로 그리고, 텍스트 레이어에서 줄과 위치를 꺼내 문단으로 묶은 뒤 번역한다.

보기 화면은 이미지 번역과 같다: 쪽 그림 위의 문단 자리에 번역문을 덮는다.
텍스트 레이어가 있으면 글자와 위치를 PDF에서 그대로 가져오므로 OCR처럼 잘못 읽거나 어긋나지 않는다.
텍스트 레이어가 없는 쪽(스캔본)만 그림을 OCR로 읽는다.
"""

import ctypes
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw
from PIL import Image

import ocr
import translator

MAX_PAGES = 80
RENDER_WIDTH = 1400  # 쪽 그림의 가로 픽셀
MIN_CHARS = 20  # 텍스트 레이어의 글자가 이보다 적은 쪽은 스캔본으로 보고 OCR
# 같은 줄로 잇는 가로 간격의 한도 (글꼴 크기 배수). 굵은 머리말 뒤 빈칸이 1.1배쯤, 두 단 논문의 단 사이가 1.7배쯤이다
JOIN_GAP = 1.2


def pdf_key(data: bytes) -> str:
    return "p-" + hashlib.sha256(data).hexdigest()[:12]


def check(data: bytes) -> int:
    """PDF로 열리는지, 쪽 수가 한도 안인지 확인하고 쪽 수를 돌려준다."""
    if not data.startswith(b"%PDF-"):
        raise translator.PeparError("PDF 파일이 아닙니다.")
    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as e:
        msg = "암호가 걸린 PDF는 열 수 없습니다." if "password" in str(e).lower() else f"PDF를 열 수 없습니다: {e}"
        raise translator.PeparError(msg) from e
    try:
        n = len(pdf)
    finally:
        pdf.close()
    if n == 0:
        raise translator.PeparError("쪽이 없는 PDF입니다.")
    if n > MAX_PAGES:
        raise translator.PeparError(f"PDF가 너무 깁니다 ({n}쪽, 최대 {MAX_PAGES}쪽).")
    return n


# ---------------------------------------------------------------- 텍스트 레이어

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f￾￿]")


def _clean(text: str) -> str:
    """합자(ﬁ)를 풀고 제어 문자·줄바꿈을 없앤다. 줄 끝 하이픈 표시(\\x02)는 pdfium이 넣는다."""
    text = unicodedata.normalize("NFKC", ocr.clean(text)).replace("\x02", "-")
    return " ".join(_CONTROL.sub("", text).split())


# ---------------------------------------------------------------- 수식
# 수식은 글꼴로 알아본다: LaTeX 수식 글꼴(CMMI 이탤릭, CMSY 기호, CMEX 큰 기호, MSAM·MSBM 칠판체 등)과
# 이름에 Math가 들어간 글꼴. 바로 선 수식 글자(CMR)는 본문 글꼴이 CM이 아닐 때만 수식으로 본다
# (CM 본문 논문에서는 CMR이 본문이다).
MATH_FONT = re.compile(r"^(CMMI|CMSY|CMEX|CMBSY|MSAM|MSBM|EUFM|EUSM|EURM|RSFS|STMARY|WASY)|MATH", re.I)
MATH_ITALIC = re.compile(r"^CMMI|MATH.*ITAL|ITAL.*MATH", re.I)
UPRIGHT_MATH = re.compile(r"^CMR", re.I)
# 줄 글 안에서 수식 자리 표시 (문단으로 묶은 뒤 <x0/> 같은 번역 표시로 바꾼다)
MARK = re.compile("\ue000(\\d+)\ue001")


def _font(tp, i: int) -> str:
    buf = ctypes.create_string_buffer(128)
    flags = ctypes.c_int()
    pdfium_raw.FPDFText_GetFontInfo(tp.raw, i, buf, 128, ctypes.byref(flags))
    return buf.value.decode("latin-1").split("+")[-1]  # 부분 글꼴 표시(ABCDEF+)는 뺀다


_DOUBLE_STRUCK = {"C": "ℂ", "H": "ℍ", "N": "ℕ", "P": "ℙ", "Q": "ℚ", "R": "ℝ", "Z": "ℤ"}


def _math_char(ch: str, font: str) -> str:
    """칠판체 글꼴(MSBM)은 보통 글자 코드로 나온다: ℝ 같은 글자로 바꾼다."""
    if font.upper().startswith("MSBM") and "A" <= ch <= "Z":
        return _DOUBLE_STRUCK.get(ch) or chr(0x1D538 + ord(ch) - ord("A"))
    return ch


def _runs(chars: list[tuple[str, str]]) -> list[list[str]]:
    """[(글자, 모양)] → 모양이 같은 글자끼리 묶은 [[글, 모양]]. 모양: i 이탤릭, ^ 위첨자, _ 아래첨자.
    위·아래첨자 앞의 띄어쓰기는 뺀다 (pdfium이 첨자마다 끼워 넣어 W Q i처럼 벌어진다)."""
    chars = [(ch, st) for k, (ch, st) in enumerate(chars)
             if not (ch == " " and k + 1 < len(chars) and chars[k + 1][1][-1:] in ("^", "_"))]
    out = []
    for ch, style in chars:
        if out and out[-1][1] == style:
            out[-1][0] += ch
        else:
            out.append([ch, style])
    if out:
        out[0][0] = out[0][0].lstrip()
        out[-1][0] = out[-1][0].rstrip()
    for r in out:  # 안전장치: 혹시 남은 반쪽 글자(짝 없는 서로게이트)는 뺀다
        r[0] = ocr.clean(r[0])
    return [r for r in out if r[0]]


def _plain(runs: list[list[str]]) -> str:
    return "".join(t for t, _ in runs)


def _line_text(chars: list[dict], cm_body: bool, maths: list) -> str:
    """줄의 글자들 → 글. 수식 글자가 이어진 부분은 maths에 [[글, 모양]]으로 넣고 그 자리에 표시를 둔다.
    위·아래첨자는 글자 크기(본문의 85% 미만)와 높이로 가린다."""
    is_math = [bool(MATH_FONT.search(c["font"]) or (not cm_body and UPRIGHT_MATH.match(c["font"]))) for c in chars]
    base = sorted(c["size"] for c in chars)[len(chars) // 2]  # 줄의 보통 글자 크기 (수식뿐인 줄도 있어 전체에서)
    full = [c["y2"] for c in chars if c["size"] >= 0.9 * base] or [c["y2"] for c in chars]
    baseline = sorted(full)[len(full) // 2]  # 글자 아래쪽(기준선) 높이

    def position(c) -> str:
        if c["size"] < 0.85 * base:
            if c["y2"] < baseline - 0.2 * base:
                return "^"
            if c["y2"] > baseline + 0.1 * base:
                return "_"
        return ""

    # 본문 글꼴로 쓴 수식 조각도 수식에 붙인다: 수식 바로 뒤의 작은 위·아래첨자(d_model의 model),
    # 수식 글자 사이에 띄어쓰기 없이 낀 기호(step_num의 _)
    for k, c in enumerate(chars):
        if is_math[k] or k == 0 or not is_math[k - 1]:
            continue
        glued = (not c["space"] and k + 1 < len(chars) and is_math[k + 1] and not chars[k + 1]["space"]
                 and not c["ch"].isalnum())
        if position(c) or glued:  # 첨자는 앞에 띄어쓰기가 있어도 (pdfium이 끼워 넣는다)
            is_math[k] = True
    out, run = [], None

    def end_run():
        runs = _runs(run)
        if runs:
            maths.append(runs)
            out.append(f"\ue000{len(maths) - 1}\ue001")

    for c, m in zip(chars, is_math):
        if not m:
            if run is not None:
                end_run()
                run = None
            out.append((" " if c["space"] and out else "") + c["ch"])
            continue
        style = ("i" if MATH_ITALIC.search(c["font"]) and c["ch"].isalpha() else "") + position(c)
        if run is None:
            if c["space"] and out:
                out.append(" ")
            run = []
        elif c["space"]:
            run.append((" ", ""))
        run.append((_math_char(c["ch"], c["font"]), style))
    if run is not None:
        end_run()
    return "".join(out)


def finish(labels: list[dict], maths: list) -> list[dict]:
    """문단으로 묶은 뒤: 수식 표시를 번역용 <x0/>, <x1/>…로 바꾼다.
    text: 수식을 풀어 쓴 글 (목록·검색용), src: 번역할 글 (수식 자리는 표시), math: 표시마다 수식 [[글, 모양]].
    수식뿐인 문단(본문 글자가 2자 미만)은 keep: 번역하지 않고 원본을 그대로 둔다."""
    for lb in labels:
        idx = [int(n) for n in MARK.findall(lb["text"])]
        if not idx:
            continue
        k = iter(range(len(idx)))
        src = MARK.sub(lambda m: f"<x{next(k)}/>", lb["text"])
        lb["math"] = [maths[n] for n in idx]
        lb["text"] = MARK.sub(lambda m: _plain(maths[int(m.group(1))]), lb["text"])
        lb["src"] = src
        if len(re.findall(r"[^\W\d_]", translator.TOKEN.sub("", src))) < 2:
            lb["keep"] = True
    return labels


def page_lines(page) -> tuple[list[dict], list]:
    """쪽의 글줄 [{text, box, size}]과 수식 목록. box는 쪽 크기에 대한 0~1 비율 [x1, y1, x2, y2] (왼쪽 위 기준).
    text 안의 수식 자리에는 표시(\\ue000번호\\ue001)가 있다 → finish()로 문단마다 정리한다.

    글자 하나씩 위치를 받아 줄을 만든다 (pdfium이 주는 줄 조각은 링크 글자 둘레에서 엉뚱하게 끊긴다).
    pdfium은 줄이 끝났다고 보면 \\r\\n을 끼워 넣는데, 수식의 위·아래첨자마다 끼워 넣어서 믿지 않는다.
    왼쪽으로 돌아가거나, 높이가 바뀌거나, 단 사이처럼 멀리 건너뛰면 새 줄로 본다. 돌아간 글자(arXiv 여백의 세로 도장)는 뺀다.
    """
    W, H = page.get_size()
    tp = page.get_textpage()
    lines, cur = [], None
    fonts = Counter()

    def flush():
        if cur and cur["chars"]:
            lines.append(cur)

    try:
        n_chars = tp.count_chars()
        pair_tail = False
        for i in range(n_chars):
            if pair_tail:  # 바로 앞 글자와 합친 뒤쪽 반쪽
                pair_tail = False
                continue
            code = pdfium_raw.FPDFText_GetUnicode(tp.raw, i)
            boxes = [i]
            # 기본 범위 밖의 글자(수학 영숫자 𝑥, 𝒪 등)는 반쪽 둘(서로게이트 쌍)로 나온다: 하나로 합친다.
            # 짝이 없는 반쪽은 버린다 (그대로 두면 저장할 때 UTF-8로 쓰지 못해 작업이 실패한다)
            if 0xD800 <= code <= 0xDBFF and i + 1 < n_chars:
                low = pdfium_raw.FPDFText_GetUnicode(tp.raw, i + 1)
                if 0xDC00 <= low <= 0xDFFF:
                    code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                    boxes.append(i + 1)
                    pair_tail = True
            if 0xD800 <= code <= 0xDFFF:
                continue
            ch = chr(code) if code else ""
            if abs(math.sin(pdfium_raw.FPDFText_GetCharAngle(tp.raw, i))) > 0.1:
                continue
            if not ch or ch.isspace():  # \r\n도 띄어쓰기로만 본다 (줄 끝은 위치로)
                if cur:
                    cur["space"] = True
                continue
            # 합친 글자는 두 반쪽 중 크기가 있는 쪽 상자를 쓴다
            left, bottom, right, top = next((b for b in (tp.get_charbox(k) for k in boxes) if b[2] > b[0] or b[3] > b[1]),
                                            tp.get_charbox(i))
            if right <= left and top <= bottom:
                continue
            x1, y1, x2, y2 = left, H - top, right, H - bottom
            # 거리 기준은 글꼴 크기로 (글자 상자 높이는 x 높이 정도라 너무 작다)
            fs = pdfium_raw.FPDFText_GetFontSize(tp.raw, i) or (y2 - y1)
            font = _font(tp, i)
            fonts[font] += 1
            if cur:
                lh = cur["h"]
                mid = (y1 + y2) / 2
                same_row = cur["y1"] - 0.3 * lh <= mid <= cur["y2"] + 0.3 * lh
                gap = x1 - cur["x2"]
                # 위첨자와 아래첨자가 같은 자리에 겹쳐 있으면(d_model^−0.5) 왼쪽으로 돌아가도 같은 줄이다
                stacked = fs < 0.85 * lh and x1 > cur["x1"] and cur["y1"] - 0.6 * lh <= mid <= cur["y2"] + 0.6 * lh
                if not stacked and (not same_row or gap < -lh or gap > JOIN_GAP * lh):
                    flush()
                    cur = None
            if cur is None:
                cur = {"chars": [], "x1": x1, "y1": y1, "x2": x2, "y2": y2, "h": fs, "space": False}
            cur["chars"].append({"ch": ch, "y2": y2, "size": fs, "font": font, "space": cur["space"] and bool(cur["chars"])})
            cur["space"] = False
            cur["x1"], cur["y1"] = min(cur["x1"], x1), min(cur["y1"], y1)
            cur["x2"], cur["y2"] = max(cur["x2"], x2), max(cur["y2"], y2)
            cur["h"] = max(cur["h"], fs)
        flush()
    finally:
        tp.close()
    # 같은 높이에서 바로 옆에 있는 줄은 잇는다: 굵은 글씨의 머리말("Encoder:")처럼 문서 순서상 줄 뒤에 나오는 글자가 있다
    merged = []
    for ln in sorted(lines, key=lambda ln: ln["x1"]):
        for m in merged:
            lh = max(m["h"], ln["h"])
            overlap = min(m["y2"], ln["y2"]) - max(m["y1"], ln["y1"])
            if overlap >= 0.5 * min(m["y2"] - m["y1"], ln["y2"] - ln["y1"]) and -0.5 * lh <= ln["x1"] - m["x2"] <= JOIN_GAP * lh:
                ln["chars"][0]["space"] = True
                m["chars"] += ln["chars"]
                m["x2"], m["y1"], m["y2"] = max(m["x2"], ln["x2"]), min(m["y1"], ln["y1"]), max(m["y2"], ln["y2"])
                m["h"] = max(m["h"], ln["h"])
                break
        else:
            merged.append(ln)
    lines = sorted(merged, key=lambda ln: (ln["y1"], ln["x1"]))
    cm_body = bool(fonts) and bool(UPRIGHT_MATH.match(fonts.most_common(1)[0][0]))  # 본문이 CM 글꼴인 논문
    maths: list = []
    out = []
    for ln in lines:
        text = _clean(_line_text(ln["chars"], cm_body, maths))
        if text:
            out.append({"text": text, "size": round(ln["h"], 1),  # 글꼴 크기 (제목 찾기에 쓴다. 문단으로 묶으면 빠진다)
                        "box": [round(ln["x1"] / W, 4), round(ln["y1"] / H, 4), round(ln["x2"] / W, 4), round(ln["y2"] / H, 4)]})
    return out, maths


def title(pdf_path: Path) -> str:
    """문서 제목: PDF 정보의 제목, 없으면 첫 쪽 위쪽 절반에서 글꼴이 가장 큰 줄들 (80자까지)."""
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        meta = _clean(pdf.get_metadata_dict().get("Title", ""))
        if len(meta) > 3 and meta.lower() not in ("untitled", "microsoft word") and not meta.lower().endswith((".pdf", ".dvi", ".doc", ".docx")):
            return meta[:80]
        page = pdf[0]
        lines, maths = page_lines(page)
        top = [{**ln, "text": MARK.sub(lambda m: _plain(maths[int(m.group(1))]), ln["text"])} for ln in lines if ln["box"][1] < 0.5]
        page.close()
        if not top:
            return ""
        biggest = max(ln["size"] for ln in top)
        return " ".join(ln["text"] for ln in top if ln["size"] >= biggest * 0.95)[:80]
    finally:
        pdf.close()


# ---------------------------------------------------------------- 전체 파이프라인


def render(pdf_path: Path, out_dir: Path) -> list[tuple[int, int]]:
    """쪽마다 <out_dir>/<n>.png (1부터)를 그린다. 이미 있으면 그대로 둔다. [(가로, 세로)] 픽셀."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(pdf_path)
    sizes = []
    try:
        for n, page in enumerate(pdf, 1):
            out = out_dir / f"{n}.png"
            if not out.exists():
                w, _ = page.get_size()
                img = page.render(scale=min(4.0, RENDER_WIDTH / w)).to_pil()
                tmp = out.with_suffix(".tmp")
                img.save(tmp, "PNG", optimize=True)
                tmp.replace(out)
            with Image.open(out) as im:
                sizes.append(im.size)
            page.close()
    finally:
        pdf.close()
    return sizes


def read(model: str, pdf_path: Path, pages_dir: Path, on_progress) -> list[dict]:
    """쪽마다 {"labels": [...], "ocr": bool, "error"?}. 텍스트 레이어가 있는 쪽은 줄을 문단으로 묶고, 없는 쪽은 OCR."""
    sizes = render(pdf_path, pages_dir)
    pdf = pdfium.PdfDocument(pdf_path)
    pages = []
    try:
        for n, page in enumerate(pdf, 1):
            on_progress("reading", n - 1, len(sizes))
            W, H = page.get_size()
            tp = page.get_textpage()
            chars = tp.count_chars()
            tp.close()
            if chars >= MIN_CHARS:
                lines, maths = page_lines(page)
                pages.append({"labels": finish(ocr._blocks(lines, W, H, paragraphs=True), maths), "ocr": False})
                pages[-1]["pt"] = [round(W, 2), round(H, 2)]  # 쪽 크기(포인트). 번역본을 PDF로 저장할 때 원본과 같은 크기로
            else:
                try:
                    labels = ocr.read_image(model, (pages_dir / f"{n}.png").read_bytes(), lambda *a: None)
                    pages.append({"labels": labels, "ocr": True, "pt": [round(W, 2), round(H, 2)]})
                except translator.PeparError:
                    raise
                except Exception as e:  # noqa: BLE001 - 비전이 안 되는 모델 등. 그 쪽만 원문으로 둔다
                    print(f"[pepar] PDF {n}쪽 OCR 실패: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                    pages.append({"labels": [], "ocr": True, "error": f"{n}쪽은 글자 레이어가 없는 스캔 쪽인데, 이미지를 읽지 못했습니다 (이미지를 읽는 모델이 필요합니다)."})
            page.close()
    finally:
        pdf.close()
    for pg, (w, h) in zip(pages, sizes):
        pg["width"], pg["height"] = w, h
    return pages


def run(model: str, pdf_path: Path, pages_dir: Path, on_progress, checkpoint=None, lang: str = "ko") -> tuple[list[dict], int]:
    """(쪽 목록, 번역 실패 라벨 수). checkpoint에 읽은 결과와 번역문을 저장해, 실패분만 다시 번역할 때 다시 읽지 않는다."""
    saved = {}
    if checkpoint and checkpoint.exists():
        try:
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        except ValueError:
            saved = {}
    pages = saved.get("pages")
    if pages is None:
        pages = read(model, pdf_path, pages_dir, on_progress)
        saved = {"pages": pages}
        if checkpoint:
            checkpoint.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
    labels = [lb for pg in pages for lb in pg["labels"]]  # 같은 dict라 번역문이 쪽 목록에 그대로 들어간다
    if not labels:
        raise translator.PeparError("PDF에서 글자를 찾지 못했습니다.")

    def save():
        if checkpoint:
            checkpoint.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")

    failed = ocr.translate_labels(model, labels, on_progress, lang, "pdf", save)
    return pages, failed
