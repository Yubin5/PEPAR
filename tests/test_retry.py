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
        self.langs = []
        self.texts = []

        def fake_run(src, model, progress, checkpoint=None, retry_url=None, label=None, lang="ko", text=None):
            saved = json.loads(checkpoint.read_text()) if checkpoint and checkpoint.exists() else None
            self.calls.append({"model": model, "label": label, "checkpoint": saved})
            self.langs.append(lang)
            self.texts.append(text)
            progress("translating", 1, 1)
            return src.key + "v1", "t", "<html></html>", 0

        self._orig = translator.run
        translator.run = fake_run
        for m in ("a", "b", "a@en"):
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

    def test_other_language_has_its_own_slot(self):
        (pepar.model_dir("a") / f"{KEY}v1.html").write_text("ko")
        r = self.client.get(f"/open?model=a&lang=en&url={KEY}")
        self.assertEqual(r.headers["Location"], f"/view/a@en/{KEY}")
        self.client.get(f"/view/a@en/{KEY}")
        self.wait()
        self.assertEqual(self.calls, [{"model": "a", "label": "a · English", "checkpoint": None}])
        self.assertEqual(self.langs, ["en"])
        self.assertEqual((pepar.model_dir("a") / f"{KEY}v1.html").read_text(), "ko")
        self.assertEqual((pepar.model_dir("a@en") / f"{KEY}v1.html").read_text(), "<html></html>")

    def test_korean_keeps_the_plain_model_slot(self):
        r = self.client.get(f"/open?model=a&lang=ko&url={KEY}")
        self.assertEqual(r.headers["Location"], f"/view/a/{KEY}")

    def test_bad_language_or_slot_rejected(self):
        self.assertEqual(self.client.get(f"/open?model=a&lang=xx&url={KEY}").status_code, 400)
        self.assertEqual(self.client.get(f"/open?model=a@en&url={KEY}").status_code, 400)
        self.assertEqual(self.client.get(f"/view/a@en/{KEY}?retry_failed=1&with=b@ja").status_code, 400)

    def test_custom_language(self):
        import translator
        r = self.client.get(f"/open?model=a&lang=other&lang_text=%20Vietnamese%0A&url={KEY}")
        code = translator.custom_code("vietnamese")
        self.assertEqual(r.headers["Location"], f"/view/a@{code}/{KEY}")
        self.assertEqual(translator.lang_name(code), "Vietnamese")
        self.assertEqual(json.loads((pepar.CACHE / "langs.json").read_text())[code], "Vietnamese")
        r2 = self.client.get(f"/open?model=a&lang=other&lang_text=VIETNAMESE&url={KEY}")
        self.assertEqual(r2.headers["Location"], f"/view/a@{code}/{KEY}")
        for p in pepar.model_dir(f"a@{code}").glob("*"):
            p.unlink()
        self.client.get(f"/view/a@{code}/{KEY}")
        self.wait()
        self.assertEqual(self.langs, [code])
        self.assertEqual(self.calls[0]["label"], "a · Vietnamese")
        self.assertIn('"Vietnamese"', translator.system_prompt("c", "web", code))

    def test_custom_language_must_not_be_empty(self):
        r = self.client.get(f"/open?model=a&lang=other&lang_text=%20%20&url={KEY}")
        self.assertEqual(r.status_code, 400)

    def test_unknown_custom_code_fails_the_job(self):
        self.client.get(f"/view/a@x-00000000/{KEY}")
        self.wait()
        self.assertEqual(self.calls, [])
        self.assertEqual(pepar.jobs[f"a@x-00000000/{KEY}"]["stage"], "error")

    def test_pasted_text(self):
        body = "First line of the text\n\nSecond paragraph."
        r = self.client.post("/text", data={"model": "a", "lang": "en", "text": body})
        key = r.headers["Location"].rsplit("/", 1)[1]
        self.assertTrue(key.startswith("t-"))
        self.assertEqual(r.headers["Location"], f"/view/a@en/{key}")
        self.assertEqual(self.client.get(f"/text/{key}").get_data(as_text=True), body)
        r2 = self.client.post("/text", data={"model": "a", "lang": "en", "text": body + "\n"})
        self.assertEqual(r2.headers["Location"], r.headers["Location"])
        self.client.get(f"/view/a@en/{key}")
        self.wait()
        self.assertEqual(self.texts, [body])
        self.assertEqual(self.langs, ["en"])
        doc = next(d for d in self.client.get("/api/docs").get_json() if d["key"].startswith(key))  # 가짜 run은 key에 v1을 붙인다
        self.assertEqual((doc["where"], doc["lang"]), ("텍스트", "en"))
        (pepar.model_dir("a@en") / f"{key}.html").write_text("<html></html>")
        self.assertEqual(self.client.delete(f"/api/docs/a@en/{key}").status_code, 200)
        self.assertFalse(pepar.text_path(key).exists())

    def test_empty_text_rejected(self):
        self.assertEqual(self.client.post("/text", data={"model": "a", "text": "  \n "}).status_code, 400)

    def test_start_from_home_without_leaving(self):
        J = {"Accept": "application/json"}
        for p in pepar.model_dir("a").glob(f"{KEY}*"):
            p.unlink()
        r = self.client.get(f"/open?model=a&url={KEY}", headers=J)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json(), {"status": "started", "model": "a", "key": KEY, "href": f"/view/a/{KEY}"})
        self.wait()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.client.get(f"/open?model=a&url={KEY}", headers=J).get_json()["status"], "done")
        self.assertEqual(len(self.calls), 1)
        r = self.client.post("/text", data={"model": "a", "text": "Hello there.\n\nSecond."}, headers=J)
        self.assertEqual(r.get_json()["status"], "started")
        r = self.client.post("/text", data={"model": "a", "text": "  "}, headers=J)
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", r.get_json())
        self.wait()

    def test_running_job_is_reported(self):
        J = {"Accept": "application/json"}
        pepar.jobs[f"a/{KEY}"] = {"model": "a", "key": KEY, "stage": "translating", "done": 0, "total": 1, "queued_at": 0}
        self.assertEqual(self.client.get(f"/open?model=a&url={KEY}", headers=J).get_json()["status"], "running")
        pepar.jobs.clear()

    def test_bad_model_rejected(self):
        self.assertEqual(self.client.get(f"/view/a/{KEY}?retry_failed=1&with=../x").status_code, 400)
        self.assertEqual(self.calls, [])


class LLMRetryTest(unittest.TestCase):
    def test_response_cut_midway_is_retried(self):
        import requests

        class Ok:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"ok": True}

        calls = []

        def fake_post(*a, **kw):
            calls.append(1)
            if len(calls) == 1:
                raise requests.exceptions.ChunkedEncodingError("Response ended prematurely")
            return Ok()

        orig = translator.requests.post
        translator.requests.post = fake_post
        try:
            self.assertEqual(translator._post({}, attempts=3, wait=0), {"ok": True})
        finally:
            translator.requests.post = orig
        self.assertEqual(len(calls), 2)


    def test_cancel_during_retry_wait_stops_at_once(self):
        class Busy:
            status_code = 503
            text = "busy"

        stop = {"now": False}

        def fake_post(*a, **kw):
            stop["now"] = True  # 첫 시도가 실패한 뒤 사용자가 취소했다
            return Busy()

        orig_post, orig_stop = translator.requests.post, translator.should_stop
        translator.requests.post = fake_post
        translator.should_stop = lambda: stop["now"]
        t = time.time()
        try:
            with self.assertRaises(translator.Cancelled):
                translator._post({}, attempts=3, wait=20)
        finally:
            translator.requests.post, translator.should_stop = orig_post, orig_stop
        self.assertLess(time.time() - t, 2)


if __name__ == "__main__":
    unittest.main()
