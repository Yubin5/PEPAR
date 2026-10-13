import io
import sys
import unittest
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from _app import pepar  # noqa: E402  (must come first: points the cache at a temp dir)
import ocr  # noqa: E402
import translator  # noqa: E402


class HttpErrorTest(unittest.TestCase):
    def setUp(self):
        self.client = pepar.app.test_client()

    def test_html_404_is_korean_page(self):
        r = self.client.get("/view/bad model/xx")
        self.assertEqual(r.status_code, 404)
        self.assertIn("찾을 수 없는 페이지", r.get_data(as_text=True))

    def test_api_404_is_json(self):
        r = self.client.delete("/api/docs/some-model/w-000000000000")
        self.assertEqual(r.status_code, 404)
        self.assertIn("error", r.get_json())

    def test_405(self):
        r = self.client.post("/open")
        self.assertEqual(r.status_code, 405)
        self.assertIn("허용되지 않는", r.get_data(as_text=True))

    def test_upload_too_large(self):
        big = io.BytesIO(b"0" * (pepar.app.config["MAX_CONTENT_LENGTH"] + 1))
        r = self.client.post("/upload", data={"image": (big, "a.png")}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 413)
        self.assertIn("너무 큽니다", r.get_data(as_text=True))

    def test_models_error_hides_raw_exception(self):
        models, msg = pepar.get_models()
        self.assertEqual(models, [])
        self.assertIn("llm_url", msg)
        self.assertNotIn("Connection", msg)

    def test_open_without_llm_server_says_so(self):
        r = self.client.get("/open?url=https://example.com")
        self.assertEqual(r.status_code, 400)
        self.assertIn("사용할 수 있는 모델이 없습니다", r.get_data(as_text=True))


class DescribeErrorTest(unittest.TestCase):
    def test_messages(self):
        ex = requests.exceptions
        cases = [
            (ex.ConnectTimeout("x"), "시간이 초과"),
            (ex.SSLError("x"), "SSL"),
            (ex.TooManyRedirects("x"), "리다이렉트"),
            (ex.MissingSchema("x"), "형식"),
            (ex.ConnectionError("HTTPSConnectionPool(host=...)"), "연결할 수 없습니다"),
            (ex.RequestException("x"), "실패"),
        ]
        for e, expected in cases:
            msg = translator.describe_request_error(e)
            self.assertIn(expected, msg, type(e).__name__)
            self.assertNotIn("HTTPSConnectionPool", msg)


class ImageErrorTest(unittest.TestCase):
    def test_garbage_bytes(self):
        with self.assertRaises(translator.PeparError) as cm:
            ocr.normalize(b"not an image")
        self.assertIn("열 수 없습니다", str(cm.exception))
        self.assertNotIn("cannot identify", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
