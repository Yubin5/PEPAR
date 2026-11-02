import json
import time
import unittest

from _app import pepar  # noqa: E402
import pdfdoc  # noqa: E402
import pypdfium2 as pdfium  # noqa: E402
import translator  # noqa: E402


def make_pdf(lines: list[tuple]) -> bytes:
    """글자 레이어가 있는 한 쪽짜리 PDF. lines: [(x, y, 글[, 글꼴, 크기])] (PDF 좌표, 왼쪽 아래 기준).
    글꼴 F1은 Helvetica(본문), F2는 CMMI10 (LaTeX 수식 이탤릭 이름; 수식 알아보기 시험용). 기본 F1 12pt."""
    text = "".join(f"BT /{(l + ('F1', 12))[3]} {(l + ('F1', 12))[4]} Tf {l[0]} {l[1]} Td ({l[2]}) Tj ET\n" for l in lines).encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >>",
        b"<< /Length " + str(len(text)).encode() + b" >>\nstream\n" + text + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /CMMI10 >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for n, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


SAMPLE = make_pdf([
    (72, 720, "A Short Title"),
    (72, 690, "This is the first line of a paragraph that is long enough"),
    (72, 676, "and this line continues the same paragraph."),
])


class PageLinesTest(unittest.TestCase):
    def test_lines_and_paragraphs(self):
        pdf = pdfium.PdfDocument(SAMPLE)
        page = pdf[0]
        lines, maths = pdfdoc.page_lines(page)
        self.assertEqual([ln["text"] for ln in lines][0], "A Short Title")
        self.assertEqual(maths, [])
        W, H = page.get_size()
        blocks = pepar.ocr._blocks(lines, W, H, paragraphs=True)
        self.assertEqual(blocks[1]["text"], "This is the first line of a paragraph that is long enough and this line continues the same paragraph.")
        self.assertEqual(blocks[1]["lines"], 2)

    def test_math_is_protected_or_kept(self):
        # 문장 속 수식(CMMI 글꼴)은 <x0/>로 바꿔 번역하고, 위첨자는 모양으로 남긴다. 수식뿐인 줄은 번역하지 않는다
        pdf = pdfium.PdfDocument(make_pdf([
            (72, 700, "Let the matrix"), (152, 700, "W", "F2", 12), (164, 705, "Q", "F2", 8),
            (175, 700, "be learned during training of the model."),
            (72, 650, "y", "F2", 12), (84, 650, "=", "F2", 12), (96, 650, "x", "F2", 12),
        ]))
        page = pdf[0]
        W, H = page.get_size()
        lines, maths = pdfdoc.page_lines(page)
        labels = pdfdoc.finish(pepar.ocr._blocks(lines, W, H, paragraphs=True), maths)
        prose, eq = labels
        self.assertEqual(prose["src"], "Let the matrix <x0/> be learned during training of the model.")
        self.assertEqual(prose["math"], [[["W", "i"], ["Q", "i^"]]])
        self.assertNotIn("keep", prose)
        self.assertTrue(eq["keep"])
        # 번역문의 수식 자리는 HTML로 다시 그린다
        self.assertEqual(str(pepar.rich_text("행렬 <x0/>를 학습한다", prose["math"])),
                         '행렬 <span class="m"><i>W</i><sup><i>Q</i></sup></span>를 학습한다')

    def test_check_rejects_non_pdf(self):
        with self.assertRaises(translator.PeparError):
            pdfdoc.check(b"hello")


class PdfUploadTest(unittest.TestCase):
    def setUp(self):
        self.client = pepar.app.test_client()
        self._orig = translator.translate_batch
        translator.translate_batch = lambda model, system, items, idxs: {i: "번역 " + items[i]["text"] for i in idxs}
        pepar.jobs.clear()

    def tearDown(self):
        translator.translate_batch = self._orig

    def wait(self):
        end = time.time() + 10
        while time.time() < end and any(j["stage"] in pepar.ACTIVE for j in pepar.jobs.values()):
            time.sleep(0.05)

    def test_upload_translate_view_delete(self):
        import io
        r = self.client.post("/upload", data={"model": "a", "pdf": (io.BytesIO(SAMPLE), "paper.pdf")})  # PDF 탭의 파일 칸
        key = r.headers["Location"].rsplit("/", 1)[1]
        self.assertTrue(key.startswith("p-"))
        self.client.get(f"/view/a/{key}")
        self.wait()
        data = json.loads((pepar.model_dir("a") / f"{key}.pages.json").read_text())
        self.assertEqual(data["failed"], 0)
        self.assertEqual(data["pages"][0]["labels"][0]["ko"], "번역 A Short Title")
        html = self.client.get(f"/view/a/{key}").get_data(as_text=True)
        self.assertIn(f"/pdf/{key}/1.png", html)
        self.assertEqual(self.client.get(f"/pdf/{key}/1.png").status_code, 200)
        self.assertEqual(self.client.get(f"/pdf/{key}").mimetype, "application/pdf")
        doc = next(d for d in self.client.get("/api/docs").get_json() if d["key"] == key)
        self.assertEqual((doc["kind"], doc["title"]), ("pdf", "paper.pdf"))
        self.assertEqual(self.client.delete(f"/api/docs/a/{key}").status_code, 200)
        self.assertFalse(pepar.pdf_path(key).exists())
        self.assertFalse(pepar.pdf_pages_dir(key).exists())


class ArxivPdfFallbackTest(unittest.TestCase):
    def test_no_html_version_falls_back_to_pdf(self):
        class R:
            def __init__(self, status, content=b"", url=""):
                self.status_code, self.content, self.url = status, content, url

        calls = []

        def fake_get(url, **kw):
            calls.append(url)
            return R(404) if "/html/" in url else R(200, SAMPLE, url)

        orig = translator.requests.get
        translator.requests.get = fake_get
        try:
            with self.assertRaises(translator.PdfFound) as cm:
                translator.load_arxiv(translator.Source("arxiv", "2401.00001", "2401.00001"))
        finally:
            translator.requests.get = orig
        self.assertEqual(calls, ["https://arxiv.org/html/2401.00001", "https://arxiv.org/pdf/2401.00001"])
        self.assertTrue(cm.exception.data.startswith(b"%PDF-"))


class ViewerUrlTest(unittest.TestCase):
    def test_pdf_tab_link_field(self):
        r = pepar.app.test_client().get("/open?model=a&url=&pdf_url=https://y.org/b.pdf")
        self.assertEqual(r.status_code, 302)
        self.assertEqual(r.headers["Location"], f"/view/a/{translator.parse_source('https://y.org/b.pdf').key}")

    def test_arxiv_in_pdf_tab_is_translated_as_pdf(self):
        # 문서 탭은 arXiv 웹(HTML) 버전, PDF 탭 링크 칸은 PDF
        self.assertEqual(translator.parse_source("https://arxiv.org/pdf/1706.03762").kind, "arxiv")
        for text in ("https://arxiv.org/pdf/1706.03762", "https://arxiv.org/abs/1706.03762", "1706.03762"):
            src = translator.parse_source(text, pdf=True)
            self.assertEqual((src.kind, src.url), ("web", "https://arxiv.org/pdf/1706.03762"))
        r = pepar.app.test_client().get("/open?model=a&url=&pdf_url=https://arxiv.org/pdf/1706.03762")
        self.assertEqual(r.headers["Location"], f"/view/a/{translator.parse_source('1706.03762', pdf=True).key}")

    def test_unwrap(self):
        self.assertEqual(translator.parse_source("https://x.org/pdfjs/web/viewer.html?file=/files/a.pdf").url, "https://x.org/files/a.pdf")
        self.assertEqual(translator.parse_source("https://docs.google.com/viewer?url=https://y.org/b.pdf&embedded=true").url, "https://y.org/b.pdf")
        self.assertEqual(translator.parse_source("https://y.org/b.pdf").url, "https://y.org/b.pdf")


if __name__ == "__main__":
    unittest.main()
