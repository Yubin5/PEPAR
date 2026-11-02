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


ROUTER = {"data": [  # 여러 모델을 설정 파일로 돌리는 방식
    {"id": "default"},
    {"id": "big", "status": {"value": "loaded"}, "architecture": {"input_modalities": ["text", "image"]}},
    {"id": "txt", "status": {"value": "unloaded"}, "architecture": {"input_modalities": ["text"]}},
]}
SINGLE = {  # 모델 하나만 띄우는 방식: 이미지 여부는 같은 이름의 models 목록에
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

    def test_docker_falls_back_to_host(self):
        # 컨테이너 안의 127.0.0.1은 컨테이너 자신: 응답이 없으면 바깥 컴퓨터로 보내고, 된 주소를 기억한다
        import requests
        saved = (translator.IN_DOCKER, translator.LLM_URL, translator._llm_ok)
        tried = []

        def fake_get(url, **kw):
            tried.append(url)
            if "127.0.0.1" in url:
                raise requests.ConnectionError("refused")
            return Resp(ROUTER)

        try:
            translator.IN_DOCKER = True
            translator.LLM_URL = translator._llm_ok = "http://127.0.0.1:8080/v1/chat/completions"
            translator.requests.get = fake_get
            self.assertEqual(len(translator.list_models()), 2)
            self.assertEqual(tried, ["http://127.0.0.1:8080/v1/models", "http://host.docker.internal:8080/v1/models"])
            tried.clear()
            translator.list_models()
            self.assertEqual(tried, ["http://host.docker.internal:8080/v1/models"])  # 기억한 주소부터
            translator.IN_DOCKER = False  # 도커 밖에서는 다른 주소로 바꾸지 않는다
            translator._llm_ok = translator.LLM_URL
            with self.assertRaises(requests.ConnectionError):
                translator.list_models()
        finally:
            translator.IN_DOCKER, translator.LLM_URL, translator._llm_ok = saved

    def test_upload_not_blocked_when_unknown(self):
        # 이미지 여부를 알 수 없으면 막지 않는다 (그림이 아닌 데이터라 이미지 열기에서 실패해야 한다)
        translator.requests.get = lambda *a, **kw: Resp({"data": [{"id": "x"}]})
        r = pepar.app.test_client().post("/upload", data={"model": "x", "image": (io.BytesIO(b"not an image"), "a.png")},
                                         headers={"Accept": "application/json"})
        self.assertEqual(r.status_code, 400)
        self.assertNotIn("이미지를 읽지 못하는 모델", r.get_json()["error"])
        # 못 읽는다고 확실하면 막는다
        translator.requests.get = lambda *a, **kw: Resp(ROUTER)
        r = pepar.app.test_client().post("/upload", data={"model": "txt", "image": (io.BytesIO(b"x"), "a.png")},
                                         headers={"Accept": "application/json"})
        self.assertIn("이미지를 읽지 못하는 모델", r.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
