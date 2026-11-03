import io
import unittest

from _app import pepar  # noqa: E402
import translator  # noqa: E402


class Resp:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


ROUTER = {"data": [
    {"id": "default"},
    {"id": "big", "status": {"value": "loaded"}, "architecture": {"input_modalities": ["text", "image"]}},
    {"id": "txt", "status": {"value": "unloaded"}, "architecture": {"input_modalities": ["text"]}},
]}
SINGLE = {
    "models": [{"name": "one.gguf", "model": "one.gguf", "capabilities": ["completion", "multimodal"]}],
    "object": "list",
    "data": [{"id": "one.gguf", "object": "model", "meta": {"n_ctx": 8192}}],
}


class ListModelsTest(unittest.TestCase):
    def setUp(self):
        self._orig = translator.requests.get

    def tearDown(self):
        translator.requests.get = self._orig

    def models(self, body):
        translator.requests.get = lambda *a, **kw: Resp(body)
        return translator.list_models()

    def test_router_format(self):
        self.assertEqual(self.models(ROUTER), [
            {"id": "big", "loaded": True, "vision": True},
            {"id": "txt", "loaded": False, "vision": False},
        ])

    def test_single_model_format(self):
        self.assertEqual(self.models(SINGLE), [{"id": "one.gguf", "loaded": True, "vision": True}])
        text_only = {**SINGLE, "models": [{"name": "one.gguf", "capabilities": ["completion"]}]}
        self.assertEqual(self.models(text_only)[0]["vision"], False)

    def test_unknown_format(self):
        self.assertIsNone(self.models({"data": [{"id": "x"}]})[0]["vision"])

    def test_upload_not_blocked_when_unknown(self):
        translator.requests.get = lambda *a, **kw: Resp({"data": [{"id": "x"}]})
        r = pepar.app.test_client().post("/upload", data={"model": "x", "image": (io.BytesIO(b"not an image"), "a.png")},
                                         headers={"Accept": "application/json"})
        self.assertEqual(r.status_code, 400)
        self.assertNotIn("이미지를 읽지 못하는 모델", r.get_json()["error"])
        translator.requests.get = lambda *a, **kw: Resp(ROUTER)
        r = pepar.app.test_client().post("/upload", data={"model": "txt", "image": (io.BytesIO(b"x"), "a.png")},
                                         headers={"Accept": "application/json"})
        self.assertIn("이미지를 읽지 못하는 모델", r.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
