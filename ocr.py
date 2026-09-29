"""이미지 속 글자를 로컬 LLM(비전)으로 위치와 함께 읽고, 라벨마다 번역한다.

결과 페이지는 원본 이미지 위의 각 라벨 자리에 번역문을 덮어 보여준다.
"""

import base64
import hashlib
import io
import json
import re
from collections import Counter

from PIL import Image, ImageOps

import translator

MAX_SIDE = 2048  # 이보다 큰 이미지는 줄인다. 모델 쪽 비전 인코더가 어차피 축소하므로 잃는 게 없다
MAX_UPLOAD = 20 * 1024 * 1024

OCR_PROMPT = (
    "Detect every separate text label in the image (words, lines, captions, labels in diagrams, "
    "text on signs or UI). Group words that belong to the same line or phrase into one label. "
    'Return a JSON array, one object per label: {"text": ..., "box_2d": [ymin, xmin, ymax, xmax]} '
    "with coordinates normalized to 0-1000. Transcribe the text exactly."
)

# 모델 출력을 이 스키마로 강제한다. 스키마 없이 받으면 12b가 끝을 못 내고 쓰레기 토큰을 이어 붙인다
OCR_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "box_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
        },
        "required": ["text", "box_2d"],
    },
}


# ---------------------------------------------------------------- 업로드


def normalize(data: bytes) -> tuple[bytes, str, int, int]:
    """업로드된 이미지를 (저장할 바이트, 확장자, 너비, 높이)로.

    EXIF 회전을 픽셀에 적용해 둔다. 브라우저는 EXIF대로 돌려 보여주는데 모델은 무시하므로,
    그대로 두면 폰 사진에서 라벨 위치가 어긋난다.
    """
    if len(data) > MAX_UPLOAD:
        raise translator.PeparError(f"이미지가 너무 큽니다 (최대 {MAX_UPLOAD // 1024 // 1024}MB).")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:  # noqa: BLE001 - Pillow는 형식마다 다른 예외를 낸다
        raise translator.PeparError(f"이미지를 열 수 없습니다: {e}") from e
    photo = (img.format or "").upper() in ("JPEG", "MPO")  # 사진은 JPEG으로, 나머지(스크린샷·도표)는 PNG로
    img = ImageOps.exif_transpose(img)
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    if photo:
        img.convert("RGB").save(buf, "JPEG", quality=92)
        ext = "jpg"
    else:
        img.convert("RGBA" if "A" in img.getbands() or img.mode == "P" else "RGB").save(buf, "PNG", optimize=True)
        ext = "png"
    return buf.getvalue(), ext, img.width, img.height


def image_key(data: bytes) -> str:
    return "i-" + hashlib.sha256(data).hexdigest()[:12]


# ---------------------------------------------------------------- OCR


def _square(img: Image.Image) -> tuple[bytes, float, float]:
    """이미지 오른쪽·아래를 테두리 색으로 채워 정사각형으로 만든다. (PNG 바이트, 가로 배율, 세로 배율).

    12b는 가로나 세로로 긴 이미지에서 세로 좌표가 수십 px씩 틀린다 (900x520에서 최대 55px).
    정사각형으로 채워 보내면 4px 이내로 맞는다. 배율은 정사각형 기준 좌표를 원본 기준으로 바꾸는 데 쓴다.
    """
    w, h = img.size
    side = max(w, h)
    edge = [img.getpixel((x, y)) for x in range(0, w, max(1, w // 50)) for y in (0, h - 1)]
    edge += [img.getpixel((x, y)) for y in range(0, h, max(1, h // 50)) for x in (0, w - 1)]
    fill = tuple(sorted(c[i] for c in edge)[len(edge) // 2] for i in range(3))
    sq = Image.new("RGB", (side, side), fill)
    sq.paste(img, (0, 0))
    buf = io.BytesIO()
    sq.save(buf, "PNG")
    return buf.getvalue(), side / w, side / h


def extract(model: str, img: Image.Image) -> list[dict]:
    """[{text, box: [x1, y1, x2, y2]}]. box는 이미지 크기에 대한 0~1 비율."""
    square, sx, sy = _square(img)
    body = {
        "model": model,
        "temperature": 0,
        # 정상 조각은 800토큰 안쪽이다. 게임 화면처럼 반복에 빠지면 8192까지 8분을 쓰므로 일찍 끊는다
        # (잘려도 완성된 라벨까지는 아래에서 살린다)
        "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": {"name": "labels", "schema": OCR_SCHEMA}},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(square).decode()}"}},
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ],
    }
    r = translator._post(body)
    content = r["choices"][0]["message"]["content"]
    try:
        raw = json.loads(content[content.find("[") : content.rfind("]") + 1])
    except ValueError:
        # 길이 제한에 걸려 배열이 닫히지 않았으면 완성된 객체까지만 살린다
        raw = [json.loads(m) for m in re.findall(r'\{[^{}]*"box_2d"[^{}]*\}', content)]
    labels = []
    for o in raw:
        text = " ".join(clean(str(o.get("text", ""))).split())
        box = o.get("box_2d")
        if not text or not isinstance(box, list) or len(box) != 4:
            continue
        y1, x1, y2, x2 = (min(max(float(v), 0), 1000) / 1000 for v in box)
        x1, x2, y1, y2 = min(x1 * sx, 1), min(x2 * sx, 1), min(y1 * sy, 1), min(y2 * sy, 1)
        if x2 <= x1 or y2 <= y1:  # 채운 부분에만 걸친 박스도 여기서 빠진다
            continue
        labels.append({"text": text, "box": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]})
    return labels


# ---------------------------------------------------------------- 큰 이미지 나눠 읽기
# 큰 캡처를 통째로 보내면 모델이 줄여서 보느라 작은 글자를 잘못 읽거나 지어낸다
# (2048x963 쇼핑몰 캡처에서 제품명을 틀리고, 없는 줄을 15번 되풀이했다).
# 그래서 여백(빈 줄)을 따라 조각으로 나눠 따로 읽는다. 여백에서 자르므로 글자가 잘리지 않는다.
# 여백이 없을 때만 겹치게 억지로 자르고, 그 경계에 걸린 라벨은 버린다 (옆 조각에 온전히 들어 있다).

TILE = 1024  # 가로세로가 이보다 큰 영역은 나눈다
MIN_GAP = 6  # 이만큼 이상 이어진 빈 줄에서만 자른다 (낱말 사이 빈칸보다 넓게)
OVERLAP = 160  # 억지로 자를 때 두 조각이 겹치는 폭. 글자 한 줄 높이보다 넉넉하게
EDGE = 8  # 억지로 자른 경계에서 이만큼 안쪽까지 닿은 라벨은 잘린 것으로 본다


def _uniform(hist: list[int], n: int) -> bool:
    """한 줄의 밝기 히스토그램. 거의 모든 픽셀이 그 줄의 대표 밝기 근처면 빈 줄 (여백이나 구분선)."""
    acc = 0
    for med, c in enumerate(hist):
        acc += c
        if acc * 2 >= n:
            break
    near = sum(hist[max(0, med - 24) : med + 25])
    return n - near <= max(1, n * 0.003)


def _blank(gray: Image.Image, box, axis: int) -> list[bool]:
    """box 안의 줄마다 빈 줄인지. axis=0이면 가로줄(행), 1이면 세로줄(열)."""
    x0, y0, x1, y1 = box
    if axis == 0:
        return [_uniform(gray.crop((x0, y, x1, y + 1)).histogram(), x1 - x0) for y in range(y0, y1)]
    return [_uniform(gray.crop((x, y0, x + 1, y1)).histogram(), y1 - y0) for x in range(x0, x1)]


def _runs(flags: list[bool]) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, f in enumerate(flags + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i))
            start = None
    return runs


def _cut(flags: list[bool]) -> int | None:
    """가운데에 가장 가까운 넓은 빈 줄의 중심. 없으면 None."""
    n, best = len(flags), None
    for a, b in _runs(flags):
        if a == 0 or b == n or b - a < MIN_GAP:
            continue  # 가장자리 여백은 자를 곳이 아니다
        c = (a + b) // 2
        if best is None or min(c, n - c) > min(best, n - best):
            best = c
    return best


def _trim(gray: Image.Image, box, hard):
    """가장자리 여백을 잘라낸 (box, hard). 내용이 없으면 None. 여백이 잘린 쪽은 억지 경계가 아니게 된다."""
    x0, y0, x1, y1 = box
    rows = [i for i, b in enumerate(_blank(gray, box, 0)) if not b]
    if not rows:
        return None
    ny0, ny1 = y0 + rows[0], y0 + rows[-1] + 1
    cols = [i for i, b in enumerate(_blank(gray, (x0, ny0, x1, ny1), 1)) if not b]
    nx0, nx1 = x0 + cols[0], x0 + cols[-1] + 1
    pad = 4  # 글자 바로 옆에서 자르지 않도록
    nb = (max(x0, nx0 - pad), max(y0, ny0 - pad), min(x1, nx1 + pad), min(y1, ny1 + pad))
    nh = tuple(h and nb[i] == box[i] for i, h in enumerate(hard))
    return nb, nh


def _tiles(gray: Image.Image, box, hard=(False,) * 4, out=None) -> list:
    """[(box, hard)]. box=(x0, y0, x1, y1) 픽셀, hard=(왼, 위, 오른, 아래) 억지로 자른 경계인지."""
    out = [] if out is None else out
    trimmed = _trim(gray, box, hard)
    if trimmed is None:
        return out
    box, hard = trimmed
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w <= TILE and h <= TILE:
        out.append((box, hard))
        return out
    # 너무 긴 쪽부터 빈 줄을 찾고, 없으면 다른 쪽이라도 (위쪽 메뉴 줄을 떼어내면 그 아래에서 세로 여백이 생긴다)
    for axis in sorted((0, 1), key=lambda a: -(h if a == 0 else w)):
        c = _cut(_blank(gray, box, axis))
        if c is None:
            continue
        if axis == 0:
            _tiles(gray, (x0, y0, x1, y0 + c), hard[:3] + (False,), out)
            _tiles(gray, (x0, y0 + c, x1, y1), hard[:1] + (False,) + hard[2:], out)
        else:
            _tiles(gray, (x0, y0, x0 + c, y1), hard[:2] + (False,) + hard[3:], out)
            _tiles(gray, (x0 + c, y0, x1, y1), (False,) + hard[1:], out)
        return out
    # 빈 줄이 없다: 긴 쪽을 반으로, 겹치게 자른다
    if h >= w:
        m = y0 + h // 2
        _tiles(gray, (x0, y0, x1, m + OVERLAP // 2), hard[:3] + (True,), out)
        _tiles(gray, (x0, m - OVERLAP // 2, x1, y1), hard[:1] + (True,) + hard[2:], out)
    else:
        m = x0 + w // 2
        _tiles(gray, (x0, y0, m + OVERLAP // 2, y1), hard[:2] + (True,) + hard[3:], out)
        _tiles(gray, (m - OVERLAP // 2, y0, x1, y1), (True,) + hard[1:], out)
    return out


def _dedupe(labels: list[dict]) -> list[dict]:
    """겹치게 자른 곳에서 두 조각이 같은 글자를 읽었으면 큰 쪽만 남긴다."""
    def area(b):
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    kept = []
    for lb in sorted(labels, key=lambda lb: -area(lb["box"])):
        b = lb["box"]
        dup = False
        for k in kept:
            if k["tile"] == lb["tile"]:
                continue
            kb = k["box"]
            inter = area([max(b[0], kb[0]), max(b[1], kb[1]), min(b[2], kb[2]), min(b[3], kb[3])])
            if inter > 0.6 * area(b):
                dup = True
                break
        if not dup:
            kept.append(lb)
    kept.sort(key=lambda lb: lb["order"])
    return kept


REPEAT = 3  # 한 조각에서 같은 글자가 이만큼 나오면 모델이 지어낸 것으로 본다


def _drop_repeats(labels: list[dict]) -> list[dict]:
    """글자가 빽빽하거나 사진이 있으면 12b가 같은 글자를 되풀이해 지어낸다.
    (쇼핑몰 캡처에서 제품명 15번, 뉴스 사진 위에 'CDU-CSU' 5번) 그런 글자는 통째로 버린다."""
    count = Counter((lb.get("tile", 0), lb["text"]) for lb in labels)
    return [lb for lb in labels if count[(lb.get("tile", 0), lb["text"])] < REPEAT]


SURROGATE = re.compile(r"[\ud800-\udfff]")


def clean(s: str) -> str:
    """모델이 이모지를 반쪽만(짝 없는 서로게이트) 내놓기도 한다. 그대로 두면 UTF-8로 저장할 때 터진다."""
    return SURROGATE.sub("", s)


CJK = re.compile(r"[぀-ヿ㐀-鿿가-힣]")


def _join(a: str, b: str) -> str:
    if a.endswith("-") and b[:1].islower():
        return a[:-1] + b  # 줄 끝에서 끊긴 낱말
    return a + ("" if CJK.match(a[-1:]) and CJK.match(b[:1]) else " ") + b


def _blocks(labels: list[dict], W: int, H: int) -> list[dict]:
    """모델은 여러 줄 문단을 줄마다 따로 돌려준다. 그대로 번역하면 문장이 줄마다 끊기고,
    줄 간격이 촘촘해 덮개끼리 겹친다. 바로 아래에 붙어 있고 글자 크기가 비슷하며
    왼쪽이나 가운데가 맞는 줄은 한 문단으로 합친다. lines는 합친 줄 수 (보기 화면이 글자 크기를 정하는 데 쓴다)."""
    px = lambda b: (b[0] * W, b[1] * H, b[2] * W, b[3] * H)  # noqa: E731
    blocks = []
    for lb in sorted(labels, key=lambda lb: (lb["box"][1], lb["box"][0])):
        x1, y1, x2, y2 = px(lb["box"])
        h = y2 - y1
        target = None
        for bl in reversed(blocks):
            lx1, ly1, lx2, ly2 = bl["last"]
            lh = ly2 - ly1
            # 줄 간격이 줄 높이의 0.6배 이내이고 글자 크기가 거의 같을 때만 (제목 아래 별점 줄 같은 건 따로 둔다)
            # 크기는 여유 있게 본다 (밈의 두 줄 자막을 모델이 68px, 88px로 돌려주기도 한다)
            if not (ly1 < y1 and y1 - ly2 < 0.6 * lh and 0.7 < h / lh < 1.4):
                continue
            bx1, _, bx2, _ = px(bl["box"])
            aligned = abs(x1 - bx1) < 1.5 * lh or abs((x1 + x2) / 2 - (bx1 + bx2) / 2) < 1.5 * lh
            if aligned and min(x2, bx2) > max(x1, bx1):
                target = bl
                break
        if target is None:
            blocks.append({"text": lb["text"], "box": list(lb["box"]), "lines": 1, "last": (x1, y1, x2, y2), "lefts": [x1], "mids": [(x1 + x2) / 2], "h": h})
            continue
        b = target["box"]
        target["text"] = _join(target["text"], lb["text"])
        target["box"] = [min(b[0], lb["box"][0]), min(b[1], lb["box"][1]), max(b[2], lb["box"][2]), max(b[3], lb["box"][3])]
        target["lines"] += 1
        target["last"] = (x1, y1, x2, y2)
        target["lefts"].append(x1)
        target["mids"].append((x1 + x2) / 2)
    for bl in blocks:  # 줄의 왼쪽 끝보다 가운데가 더 잘 맞으면 가운데 정렬 (밈 자막, 제목)
        spread = lambda v: max(v) - min(v)  # noqa: E731
        if bl["lines"] > 1 and spread(bl["lefts"]) > 0.1 * bl["h"] and spread(bl["mids"]) < 0.5 * spread(bl["lefts"]):
            bl["align"] = "center"
    return [{k: v for k, v in bl.items() if k not in ("last", "lefts", "mids", "h")} for bl in blocks]


def read_image(model: str, data: bytes, on_progress) -> list[dict]:
    """이미지 전체의 라벨 [{text, box, lines}]. 크면 나눠 읽고, 줄을 문단으로 합친다."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    W, H = img.size
    if W <= TILE and H <= TILE:
        on_progress("reading", 0, 1)
        return _blocks(_drop_repeats(extract(model, img)), W, H)
    tiles = _tiles(img.convert("L"), (0, 0, W, H))
    labels = []
    for t, ((x0, y0, x1, y1), hard) in enumerate(tiles):
        on_progress("reading", t, len(tiles))
        tw, th = x1 - x0, y1 - y0
        for lb in extract(model, img.crop((x0, y0, x1, y1))):
            bx1, by1, bx2, by2 = lb["box"][0] * tw, lb["box"][1] * th, lb["box"][2] * tw, lb["box"][3] * th
            if (hard[0] and bx1 < EDGE) or (hard[1] and by1 < EDGE) or (hard[2] and bx2 > tw - EDGE) or (hard[3] and by2 > th - EDGE):
                continue  # 억지로 자른 경계에 걸림 → 옆 조각의 온전한 것을 쓴다
            box = [round((x0 + bx1) / W, 4), round((y0 + by1) / H, 4), round((x0 + bx2) / W, 4), round((y0 + by2) / H, 4)]
            labels.append({"text": lb["text"], "box": box, "tile": t, "order": len(labels)})
    labels = _dedupe(_drop_repeats(labels))
    return _blocks([{"text": lb["text"], "box": lb["box"]} for lb in labels], W, H)


# ---------------------------------------------------------------- 번역


def run(model: str, data: bytes, on_progress, checkpoint=None) -> tuple[list[dict], int]:
    """(라벨 목록 [{text, box, ko?}], 번역 실패 라벨 수).

    checkpoint에 OCR 결과와 번역문을 저장해 두어, 실패한 라벨만 다시 번역할 때 OCR을 되풀이하지 않는다.
    """
    saved = {}
    if checkpoint and checkpoint.exists():
        try:
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        except ValueError:
            saved = {}
    labels = saved.get("labels")
    if labels is None:
        labels = read_image(model, data, on_progress)
        saved = {"labels": labels}
        if checkpoint:
            checkpoint.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
    if not labels:
        raise translator.PeparError("이미지에서 글자를 찾지 못했습니다.")

    # 숫자·기호뿐인 라벨은 번역할 게 없으니 원문 그대로 둔다
    todo = [i for i, lb in enumerate(labels) if "ko" not in lb and translator.WORTH.search(lb["text"])]
    items = [{"text": lb["text"]} for lb in labels]
    context = "\n".join(lb["text"] for lb in labels)[:3000]
    system = translator.SYSTEM.format(context=context, style=translator.STYLE["image"])
    total = sum(1 for lb in labels if translator.WORTH.search(lb["text"]))
    done = total - len(todo)
    on_progress("translating", done, total)
    for idxs in translator.batches([items[i] for i in todo]):
        real = [todo[k] for k in idxs]
        for i, t in translator.translate_batch(model, system, items, real).items():
            labels[i]["ko"] = clean(t)
        if checkpoint:
            checkpoint.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
        done += len(real)
        on_progress("translating", done, total)
    failed = sum(1 for lb in labels if "ko" not in lb and translator.WORTH.search(lb["text"]))
    return labels, failed
