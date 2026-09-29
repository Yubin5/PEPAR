"""이미지 속 글자를 로컬 LLM(비전)으로 위치와 함께 읽고, 라벨마다 번역한다.

결과 페이지는 원본 이미지 위의 각 라벨 자리에 번역문을 덮어 보여준다.
"""

import base64
import hashlib
import io
import json
import re

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


def _square(data: bytes) -> tuple[bytes, float, float]:
    """이미지 오른쪽·아래를 테두리 색으로 채워 정사각형으로 만든다. (PNG 바이트, 가로 배율, 세로 배율).

    12b는 가로나 세로로 긴 이미지에서 세로 좌표가 수십 px씩 틀린다 (900x520에서 최대 55px).
    정사각형으로 채워 보내면 4px 이내로 맞는다. 배율은 정사각형 기준 좌표를 원본 기준으로 바꾸는 데 쓴다.
    """
    img = Image.open(io.BytesIO(data)).convert("RGB")
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


def extract(model: str, data: bytes) -> list[dict]:
    """[{text, box: [x1, y1, x2, y2]}]. box는 이미지 크기에 대한 0~1 비율."""
    square, sx, sy = _square(data)
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 8192,
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
        text = " ".join(str(o.get("text", "")).split())
        box = o.get("box_2d")
        if not text or not isinstance(box, list) or len(box) != 4:
            continue
        y1, x1, y2, x2 = (min(max(float(v), 0), 1000) / 1000 for v in box)
        x1, x2, y1, y2 = min(x1 * sx, 1), min(x2 * sx, 1), min(y1 * sy, 1), min(y2 * sy, 1)
        if x2 <= x1 or y2 <= y1:  # 채운 부분에만 걸친 박스도 여기서 빠진다
            continue
        labels.append({"text": text, "box": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]})
    return labels


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
        on_progress("reading", 0, 0)
        labels = extract(model, data)
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
            labels[i]["ko"] = t
        if checkpoint:
            checkpoint.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
        done += len(real)
        on_progress("translating", done, total)
    failed = sum(1 for lb in labels if "ko" not in lb and translator.WORTH.search(lb["text"]))
    return labels, failed
