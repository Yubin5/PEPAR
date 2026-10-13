import json
import time
import unittest

from _app import pepar  # noqa: E402
import translator  # noqa: E402

KEY = "2401.00001"


class RetryFailedTest(unittest.TestCase):
    def setUp(self):
        self.client = pepar.app.test_client()
        self.calls = []

        def fake_run(src, model, progress, checkpoint=None, retry_url=None, label=None):
            saved = json.loads(checkpoint.read_text()) if checkpoint and checkpoint.exists() else None
            self.calls.append({"model": model, "label": label, "checkpoint": saved})
            progress("translating", 1, 1)
            return src.key + "v1", "t", "<html></html>", 0

        self._orig = translator.run
        translator.run = fake_run
        for m in ("a", "b"):
            for p in pepar.model_dir(m).glob("*"):
                p.unlink()
        pepar.jobs.clear()

    def tearDown(self):
        translator.run = self._orig

    def wait(self):
        end = time.time() + 5
        while time.time() < end and any(j["stage"] in pepar.ACTIVE for j in pepar.jobs.values()):
            time.sleep(0.02)

    def test_other_model_translates_into_the_same_document(self):
        pepar.checkpoint_path("a", KEY).write_text(json.dumps({"hello": "안녕"}))
        (pepar.model_dir("a") / f"{KEY}v1.html").write_text("old")
        self.assertEqual(self.client.get(f"/view/a/{KEY}?retry_failed=1&with=b").status_code, 302)
        self.wait()
        self.assertEqual(self.calls, [{"model": "b", "label": "a + b", "checkpoint": {"hello": "안녕"}}])
        self.assertEqual((pepar.model_dir("a") / f"{KEY}v1.html").read_text(), "<html></html>")
        self.assertEqual(list(pepar.model_dir("b").glob("*")), [])

    def test_same_model(self):
        pepar.checkpoint_path("a", KEY).write_text(json.dumps({"x": "y"}))
        self.client.get(f"/view/a/{KEY}?retry_failed=1")
        self.wait()
        self.assertEqual(self.calls, [{"model": "a", "label": "a", "checkpoint": {"x": "y"}}])

    def test_bad_model_rejected(self):
        self.assertEqual(self.client.get(f"/view/a/{KEY}?retry_failed=1&with=../x").status_code, 400)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
