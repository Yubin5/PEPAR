import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import netguard  # noqa: E402
import translator  # noqa: E402
from errors import PeparError  # noqa: E402

PAGE = b"<html><head><title>T</title><script>alert(1)</script></head><body><p>Hello there, this is a long enough paragraph.</p></body></html>"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/page")
            self.end_headers()
        elif self.path == "/loop":
            self.send_response(302)
            self.send_header("Location", "/loop")
            self.end_headers()
        elif self.path == "/big":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"x" * (netguard.MAX_BYTES + 1))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE)

    def log_message(self, *args):
        pass


class NetguardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), Handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_is_public(self):
        for addr in ("127.0.0.1", "10.0.0.1", "192.168.1.1", "172.16.0.1", "169.254.169.254", "100.64.0.1", "0.0.0.0",
                     "::1", "fe80::1", "fc00::1", "::ffff:127.0.0.1", "::ffff:10.0.0.1"):
            self.assertFalse(netguard.is_public(addr), addr)
        for addr in ("93.184.216.34", "8.8.8.8", "2606:4700:4700::1111"):
            self.assertTrue(netguard.is_public(addr), addr)

    def test_check_url_blocks_private_and_odd_schemes(self):
        for url in ("http://127.0.0.1/", "http://localhost/", "http://[::1]/", "http://10.1.2.3:8080/x",
                    "http://169.254.169.254/latest/meta-data/", "http://[::ffff:7f00:1]/", "http://2130706433/"):
            with self.assertRaises(PeparError, msg=url):
                netguard.check_url(url)
        for url in ("file:///etc/passwd", "ftp://example.com/", "http:///x", "gopher://x"):
            with self.assertRaises(PeparError, msg=url):
                netguard.check_url(url)
        netguard.check_url("http://93.184.216.34/")

    def test_fetch_blocks_private_by_default(self):
        with self.assertRaises(PeparError):
            netguard.fetch(self.base + "/page", {}, allow_private=False)

    def test_connection_level_guard(self):
        s = requests.Session()
        s.mount("http://", netguard.GuardAdapter())
        with self.assertRaises(requests.exceptions.ConnectionError):
            s.get(self.base + "/page", timeout=5)

    def test_allow_private_fetches_and_follows_redirects(self):
        r = netguard.fetch(self.base + "/redirect", {}, allow_private=True)
        self.assertEqual(r.content, PAGE)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.url.endswith("/page"))

    def test_redirect_loop(self):
        with self.assertRaises(requests.exceptions.TooManyRedirects):
            netguard.fetch(self.base + "/loop", {}, allow_private=True)

    def test_size_cap(self):
        with self.assertRaises(PeparError):
            netguard.fetch(self.base + "/big", {}, allow_private=True)


class LoadWebTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), Handler)
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/page"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        self._saved = dict(translator.settings)

    def tearDown(self):
        translator.settings.update(self._saved)

    def load(self, **kw):
        src = translator.Source("web", "w-000000000000", self.url)
        return translator.load_web(src, **kw)

    def test_private_url_rejected_unless_allowed(self):
        translator.settings["allow_private_urls"] = False
        with self.assertRaises(PeparError):
            self.load(keep_scripts=False)
        translator.settings["allow_private_urls"] = True
        self.assertIn("Hello there", str(self.load(keep_scripts=False).soup))

    def test_scripts_removed_or_kept(self):
        translator.settings["allow_private_urls"] = True
        self.assertNotIn("alert(1)", str(self.load(keep_scripts=False).soup))
        self.assertIn("alert(1)", str(self.load(keep_scripts=True).soup))


if __name__ == "__main__":
    unittest.main()
