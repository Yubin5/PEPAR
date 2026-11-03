"""PDF: 쪽 그림 위의 문단 자리에 번역문을 덮는다. 글자와 위치는 텍스트 레이어에서, 스캔 쪽만 OCR."""

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
RENDER_WIDTH = 1400
MIN_CHARS = 20
# 같은 줄로 잇는 가로 간격 한도 (글꼴 크기 배수)
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
# 수식은 글꼴로 알아본다 (CMR은 본문이 CM이 아닐 때만)
MATH_FONT = re.compile(r"^(CMMI|CMSY|CMEX|CMBSY|MSAM|MSBM|EUFM|EUSM|EURM|RSFS|STMARY|WASY)|MATH", re.I)
MATH_ITALIC = re.compile(r"^CMMI|MATH.*ITAL|ITAL.*MATH", re.I)
UPRIGHT_MATH = re.compile(r"^CMR", re.I)
MARK = re.compile("\ue000(\\d+)\ue001")


def _font(tp, i: int) -> str:
    buf = ctypes.create_string_buffer(128)
    flags = ctypes.c_int()
    pdfium_raw.FPDFText_GetFontInfo(tp.raw, i, buf, 128, ctypes.byref(flags))
    return buf.value.decode("latin-1").split("+")[-1]


_DOUBLE_STRUCK = {"C": "ℂ", "H": "ℍ", "N": "ℕ", "P": "ℙ", "Q": "ℚ", "R": "ℝ", "Z": "ℤ"}


def _math_char(ch: str, font: str) -> str:
    """칠판체 글꼴(MSBM)은 보통 글자 코드로 나온다: ℝ 같은 글자로 바꾼다."""
    if font.upper().startswith("MSBM") and "A" <= ch <= "Z":
        return _DOUBLE_STRUCK.get(ch) or chr(0x1D538 + ord(ch) - ord("A"))
    return ch


def _runs(chars: list[tuple[str, str]]) -> list[list[str]]:
    """[(글자, 모양)] → [[글, 모양]] (i 이탤릭, ^ 위첨자, _ 아래첨자). 첨자 앞 띄어쓰기는 뺀다."""
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
    for r in out:
        r[0] = ocr.clean(r[0])
    return [r for r in out if r[0]]


def _plain(runs: list[list[str]]) -> str:
    return "".join(t for t, _ in runs)


def _line_text(chars: list[dict], cm_body: bool, maths: list) -> str:
    """줄의 글자들 → 글. 수식 부분은 maths에 넣고 그 자리에 표시를 둔다."""
    is_math = [bool(MATH_FONT.search(c["font"]) or (not cm_body and UPRIGHT_MATH.match(c["font"]))) for c in chars]
    base = sorted(c["size"] for c in chars)[len(chars) // 2]
    full = [c["y2"] for c in chars if c["size"] >= 0.9 * base] or [c["y2"] for c in chars]
    baseline = sorted(full)[len(full) // 2]

    def position(c) -> str:
        if c["size"] < 0.85 * base:
            if c["y2"] < baseline - 0.2 * base:
                return "^"
            if c["y2"] > baseline + 0.1 * base:
                return "_"
        return ""

    # 본문 글꼴로 쓴 수식 조각(d_model의 model, step_num의 _)도 수식에 붙인다
    for k, c in enumerate(chars):
        if is_math[k] or k == 0 or not is_math[k - 1]:
            continue
        glued = (not c["space"] and k + 1 < len(chars) and is_math[k + 1] and not chars[k + 1]["space"]
                 and not c["ch"].isalnum())
        if position(c) or glued:
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
    """수식 표시를 <x0/>…로 바꾼다 (text: 풀어 쓴 글, src: 번역할 글, math: 수식, keep: 수식뿐인 문단)."""
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
    """쪽의 글줄 [{text, box, size}]과 수식 목록. box는 쪽 크기에 대한 0~1 비율 [x1, y1, x2, y2]."""
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
            if pair_tail:
                pair_tail = False
                continue
            code = pdfium_raw.FPDFText_GetUnicode(tp.raw, i)
            boxes = [i]
            # 기본 범위 밖 글자(𝑥 등)는 반쪽 둘로 나온다: 합치고, 짝 없는 반쪽은 버린다
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
            if not ch or ch.isspace():  # pdfium의 \r\n은 첨자마다 끼어 있어 줄 끝으로 믿지 않는다
                if cur:
                    cur["space"] = True
                continue
            left, bottom, right, top = next((b for b in (tp.get_charbox(k) for k in boxes) if b[2] > b[0] or b[3] > b[1]),
                                            tp.get_charbox(i))
            if right <= left and top <= bottom:
                continue
            x1, y1, x2, y2 = left, H - top, right, H - bottom
            fs = pdfium_raw.FPDFText_GetFontSize(tp.raw, i) or (y2 - y1)
            font = _font(tp, i)
            fonts[font] += 1
            if cur:
                lh = cur["h"]
                mid = (y1 + y2) / 2
                same_row = cur["y1"] - 0.3 * lh <= mid <= cur["y2"] + 0.3 * lh
                gap = x1 - cur["x2"]
                # 겹친 위·아래첨자(d_model^−0.5)는 왼쪽으로 돌아가도 같은 줄
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
    # 같은 높이 바로 옆 줄은 잇는다 (문서 순서상 뒤에 나오는 굵은 머리말)
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
    cm_body = bool(fonts) and bool(UPRIGHT_MATH.match(fonts.most_common(1)[0][0]))
    maths: list = []
    out = []
    for ln in lines:
        text = _clean(_line_text(ln["chars"], cm_body, maths))
        if text:
            out.append({"text": text, "size": round(ln["h"], 1),
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
                pages[-1]["pt"] = [round(W, 2), round(H, 2)]
            else:
                try:
                    labels = ocr.read_image(model, (pages_dir / f"{n}.png").read_bytes(), lambda *a: None)
                    pages.append({"labels": labels, "ocr": True, "pt": [round(W, 2), round(H, 2)]})
                except translator.PeparError:
                    raise
                except Exception as e:  # noqa: BLE001
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
    labels = [lb for pg in pages for lb in pg["labels"]]
    if not labels:
        raise translator.PeparError("PDF에서 글자를 찾지 못했습니다.")

    def save():
        if checkpoint:
            checkpoint.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")

    failed = ocr.translate_labels(model, labels, on_progress, lang, "pdf", save)
    return pages, failed
