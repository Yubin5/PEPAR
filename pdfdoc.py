"""PDF: 쪽마다 그림으로 그리고, 텍스트 레이어에서 줄과 위치를 꺼내 문단으로 묶은 뒤 번역한다.

보기 화면은 이미지 번역과 같다: 쪽 그림 위의 문단 자리에 번역문을 덮는다.
텍스트 레이어가 있으면 글자와 위치를 PDF에서 그대로 가져오므로 OCR처럼 잘못 읽거나 어긋나지 않는다.
텍스트 레이어가 없는 쪽(스캔본)만 그림을 OCR로 읽는다.
"""

import hashlib
import json
import math
import re
import sys
import unicodedata
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


def page_lines(page) -> list[dict]:
    """쪽의 글줄 [{text, box}]. box는 쪽 크기에 대한 0~1 비율 [x1, y1, x2, y2] (왼쪽 위 기준).

    글자 하나씩 위치를 받아 줄을 만든다 (pdfium이 주는 줄 조각은 링크 글자 둘레에서 엉뚱하게 끊긴다).
    pdfium은 글자를 문서 순서로 주고, 줄이 끝났다고 보면 \r\n을 끼워 넣는다. 그 밖에도 왼쪽으로 돌아가거나,
    높이가 바뀌거나, 단 사이처럼 멀리 건너뛰면 새 줄로 본다. 돌아간 글자(arXiv 여백의 세로 도장)는 뺀다.
    """
    W, H = page.get_size()
    tp = page.get_textpage()
    lines, cur = [], None

    def flush():
        if cur and cur["text"].strip():
            lines.append(cur)

    try:
        for i in range(tp.count_chars()):
            code = pdfium_raw.FPDFText_GetUnicode(tp.raw, i)
            ch = chr(code) if code else ""
            if ch in ("\r", "\n"):
                flush()
                cur = None
                continue
            if abs(math.sin(pdfium_raw.FPDFText_GetCharAngle(tp.raw, i))) > 0.1:
                continue
            if not ch or ch.isspace():
                if cur:
                    cur["space"] = True
                continue
            left, bottom, right, top = tp.get_charbox(i)
            if right <= left and top <= bottom:
                continue
            x1, y1, x2, y2 = left, H - top, right, H - bottom
            # 거리 기준은 글꼴 크기로 (글자 상자 높이는 x 높이 정도라 너무 작다)
            fs = pdfium_raw.FPDFText_GetFontSize(tp.raw, i) or (y2 - y1)
            if cur:
                lh = cur["h"]
                mid = (y1 + y2) / 2
                same_row = cur["y1"] - 0.3 * lh <= mid <= cur["y2"] + 0.3 * lh
                gap = x1 - cur["x2"]
                if not same_row or gap < -lh or gap > JOIN_GAP * lh:
                    flush()
                    cur = None
            if cur is None:
                cur = {"text": ch, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "h": fs, "space": False}
                continue
            cur["text"] += (" " if cur["space"] else "") + ch
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
                m["text"] += " " + ln["text"]
                m["x2"], m["y1"], m["y2"] = max(m["x2"], ln["x2"]), min(m["y1"], ln["y1"]), max(m["y2"], ln["y2"])
                m["h"] = max(m["h"], ln["h"])
                break
        else:
            merged.append(ln)
    lines = sorted(merged, key=lambda ln: (ln["y1"], ln["x1"]))
    return [
        {"text": t, "box": [round(ln["x1"] / W, 4), round(ln["y1"] / H, 4), round(ln["x2"] / W, 4), round(ln["y2"] / H, 4)],
         "size": round(ln["h"], 1)}  # 글꼴 크기 (제목 찾기에 쓴다. 문단으로 묶으면 빠진다)
        for ln in lines
        if (t := _clean(ln["text"]))
    ]


def title(pdf_path: Path) -> str:
    """문서 제목: PDF 정보의 제목, 없으면 첫 쪽 위쪽 절반에서 글꼴이 가장 큰 줄들 (80자까지)."""
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        meta = _clean(pdf.get_metadata_dict().get("Title", ""))
        if len(meta) > 3 and meta.lower() not in ("untitled", "microsoft word") and not meta.lower().endswith((".pdf", ".dvi", ".doc", ".docx")):
            return meta[:80]
        page = pdf[0]
        top = [ln for ln in page_lines(page) if ln["box"][1] < 0.5]
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
                pages.append({"labels": ocr._blocks(page_lines(page), W, H, paragraphs=True), "ocr": False})
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
