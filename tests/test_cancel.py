import threading
import time
import unittest

from _app import pepar  # noqa: E402
import translator  # noqa: E402


def wait_for(cond, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


class CancelTest(unittest.TestCase):
    def setUp(self):
        self.client = pepar.app.test_client()
        self.started = threading.Event()
        self.batches = 0
        self.release = threading.Event()
        test = self

        def slow_run(src, model, progress, checkpoint=None, retry_url=None, label=None):
            for i in range(200):
                progress("translating", i, 200)
                test.batches += 1
                test.started.set()
                test.release.wait(0.02)
            return src.key, "t", "<html></html>", 0

        self._orig = translator.run
        translator.run = slow_run

    def tearDown(self):
        translator.run = self._orig
        self.release.set()
        wait_for(lambda: not [j for j in pepar.jobs.values() if j["stage"] in pepar.ACTIVE], 10)
        pepar.jobs.clear()

    def test_cancel_running_job(self):
        pepar.start_job("m", "2401.00001")
        self.assertTrue(self.started.wait(3))
        r = self.client.delete("/api/jobs/m/2401.00001")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(wait_for(lambda: "m/2401.00001" not in pepar.jobs))
        stopped_at = self.batches
        time.sleep(0.3)
        self.assertEqual(self.batches, stopped_at)
        self.assertLess(stopped_at, 200)

    def test_cancel_queued_job_never_runs(self):
        pepar.start_job("m", "2401.00001")
        self.assertTrue(self.started.wait(3))
        pepar.start_job("m", "2401.00002")
        self.assertEqual(pepar.jobs["m/2401.00002"]["stage"], "queued")
        self.assertEqual(self.client.delete("/api/jobs/m/2401.00002").status_code, 200)
        self.assertNotIn("m/2401.00002", pepar.jobs)
        self.client.delete("/api/jobs/m/2401.00001")
        self.assertTrue(wait_for(lambda: not pepar.jobs))
        time.sleep(0.2)
        self.assertNotIn("m/2401.00002", pepar.jobs)

    def test_requeue_after_cancel_runs_once(self):
        pepar.start_job("m", "2401.00001")
        self.assertTrue(self.started.wait(3))
        pepar.start_job("m", "2401.00003")
        self.client.delete("/api/jobs/m/2401.00003")
        pepar.start_job("m", "2401.00003")
        self.client.delete("/api/jobs/m/2401.00001")
        self.assertTrue(wait_for(lambda: pepar.jobs.get("m/2401.00003", {}).get("stage") == "translating"))
        self.assertEqual(sum(1 for j in pepar.jobs.values() if j["stage"] in pepar.ACTIVE), 1)

    def test_cancel_unknown_job(self):
        self.assertEqual(self.client.delete("/api/jobs/m/2401.99999").status_code, 404)

    def test_dismiss_failed_job_still_works(self):
        pepar.jobs["m/2401.00009"] = {"model": "m", "key": "2401.00009", "stage": "error", "message": "x", "done": 0, "total": 0}
        self.assertEqual(self.client.delete("/api/jobs/m/2401.00009").status_code, 200)
        self.assertNotIn("m/2401.00009", pepar.jobs)


if __name__ == "__main__":
    unittest.main()
