import json
import time
import unittest

from _app import pepar  # noqa: E402

IP_A = {"CF-Connecting-IP": "203.0.113.1"}
IP_B = {"CF-Connecting-IP": "203.0.113.2"}


def cookie_value(resp):
    return resp.headers["Set-Cookie"].split("pepar_auth=")[1].split(";")[0]


class AuthBase(unittest.TestCase):
    def setUp(self):
        self.client = pepar.app.test_client()
        self._saved = (pepar.PASSWORD, pepar.TRUSTED_PROXIES, pepar.SESSION_LIFETIME, pepar.time.sleep)
        pepar.time.sleep = lambda s: None
        pepar.failed_logins.clear()
        pepar._sessions = None
        pepar.SESSIONS.unlink(missing_ok=True)

    def tearDown(self):
        pepar.PASSWORD, pepar.TRUSTED_PROXIES, pepar.SESSION_LIFETIME, pepar.time.sleep = self._saved

    def login(self, password, client=None, **kw):
        return (client or self.client).post("/login", data={"password": password, **kw.pop("data", {})}, **kw)

    def trust_loopback(self):
        pepar.TRUSTED_PROXIES = [pepar.ipaddress.ip_network("127.0.0.1/32")]


class AuthTest(AuthBase):
    def test_open_when_no_password(self):
        pepar.PASSWORD = ""
        self.assertEqual(self.client.get("/api/jobs").status_code, 200)
        self.assertEqual(self.client.get("/login").status_code, 302)

    def test_redirects_pages_and_rejects_api(self):
        pepar.PASSWORD = "비밀 pw"
        r = self.client.get("/view/m/2401.00001?retranslate=1")
        self.assertEqual(r.status_code, 302)
        self.assertIn("/login?next=", r.headers["Location"])
        self.assertEqual(self.client.get("/").status_code, 302)
        self.assertEqual(self.client.get("/image/i-000000000000").status_code, 302)
        r = self.client.get("/api/jobs")
        self.assertEqual(r.status_code, 401)
        self.assertIn("error", r.get_json())
        self.assertEqual(self.client.delete("/api/jobs/m/2401.00001").status_code, 401)
        self.assertEqual(self.client.get("/login").status_code, 200)

    def test_wrong_password(self):
        pepar.PASSWORD = "secret"
        r = self.login("nope")
        self.assertEqual(r.status_code, 401)
        self.assertIn("틀렸습니다", r.get_data(as_text=True))
        self.assertEqual(self.client.get("/api/jobs").status_code, 401)

    def test_login_then_access_then_logout(self):
        pepar.PASSWORD = "비밀 pw"
        r = self.login("비밀 pw", data={"next": "/view/m/2401.00001"})
        self.assertEqual(r.status_code, 302)
        self.assertEqual(r.headers["Location"], "/view/m/2401.00001")
        cookie = r.headers["Set-Cookie"]
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Strict", cookie)
        self.assertEqual(self.client.get("/api/jobs").status_code, 200)
        self.assertEqual(self.client.get("/logout").status_code, 302)
        self.assertEqual(self.client.get("/api/jobs").status_code, 401)

    def test_next_cannot_leave_the_site(self):
        pepar.PASSWORD = "pw"
        for nxt in ("//evil.com", "https://evil.com", "/\\evil.com", "evil"):
            self.assertEqual(self.login("pw", data={"next": nxt}).headers["Location"], "/", nxt)
            self.client.get("/logout")

    def test_next_is_escaped_in_login_page(self):
        pepar.PASSWORD = "pw"
        body = self.client.get('/login?next=/"><script>alert(1)</script>').get_data(as_text=True)
        self.assertNotIn("<script>alert(1)", body)

    def test_lockout_after_repeated_failures(self):
        pepar.PASSWORD = "pw"
        for _ in range(pepar.LOGIN_FAILS):
            self.assertEqual(self.login("bad").status_code, 401)
        r = self.login("pw")
        self.assertEqual(r.status_code, 429)
        self.assertIn("너무 많습니다", r.get_data(as_text=True))
        self.assertEqual(self.client.get("/api/jobs").status_code, 401)

    def test_lockout_expires_and_success_clears(self):
        pepar.PASSWORD = "pw"
        for _ in range(pepar.LOGIN_FAILS):
            self.login("bad")
        pepar.failed_logins["127.0.0.1"] = [time.time() - pepar.LOGIN_WINDOW - 1] * pepar.LOGIN_FAILS
        self.assertEqual(self.login("pw").status_code, 302)
        self.assertNotIn("127.0.0.1", pepar.failed_logins)


class SessionTest(AuthBase):
    def test_token_is_random_per_login_and_not_derived_from_password(self):
        pepar.PASSWORD = "pw"
        a = cookie_value(self.login("pw"))
        b = cookie_value(self.login("pw"))
        self.assertNotEqual(a, b)
        self.assertGreaterEqual(len(a), 40)

    def test_logout_revokes_the_session_itself(self):
        pepar.PASSWORD = "pw"
        token = cookie_value(self.login("pw"))
        self.client.get("/logout")
        replay = pepar.app.test_client()
        replay.set_cookie("pepar_auth", token)
        self.assertEqual(replay.get("/api/jobs").status_code, 401)

    def test_other_sessions_survive_a_logout(self):
        pepar.PASSWORD = "pw"
        other = pepar.app.test_client()
        self.login("pw", client=other)
        self.login("pw")
        self.client.get("/logout")
        self.assertEqual(other.get("/api/jobs").status_code, 200)

    def test_forged_cookies_rejected(self):
        pepar.PASSWORD = "pw"
        token = cookie_value(self.login("pw"))
        digest = pepar._digest(token)
        old_scheme = pepar.hmac.new(b"pw", b"pepar-session-v1", "sha256").hexdigest()
        for forged in ("", "x", "A" * 43, token[:-1], token + "A", "A" * 20000, "한글", digest, old_scheme, token.upper()):
            c = pepar.app.test_client()
            c.set_cookie("pepar_auth", forged)
            self.assertEqual(c.get("/api/jobs").status_code, 401, forged[:20])

    def test_session_expires_on_the_server(self):
        pepar.PASSWORD = "pw"
        pepar.SESSION_LIFETIME = 3600
        token = cookie_value(self.login("pw"))
        self.assertEqual(self.client.get("/api/jobs").status_code, 200)
        state = pepar._sessions["sessions"]
        state[pepar._digest(token)] = time.time() - 1
        self.assertEqual(self.client.get("/api/jobs").status_code, 401)

    def test_password_change_invalidates_sessions(self):
        pepar.PASSWORD = "one"
        self.login("one")
        self.assertEqual(self.client.get("/api/jobs").status_code, 200)
        pepar.PASSWORD = "two"
        self.assertEqual(self.client.get("/api/jobs").status_code, 401)
        pepar.PASSWORD = "one"
        self.assertEqual(self.client.get("/api/jobs").status_code, 401)

    def test_sessions_survive_restart_but_not_password_change(self):
        pepar.PASSWORD = "one"
        token = cookie_value(self.login("one"))
        pepar._sessions = None
        c = pepar.app.test_client()
        c.set_cookie("pepar_auth", token)
        self.assertEqual(c.get("/api/jobs").status_code, 200)
        pepar._sessions = None
        pepar.PASSWORD = "two"
        self.assertEqual(c.get("/api/jobs").status_code, 401)

    def test_store_holds_only_hashes_and_is_private(self):
        pepar.PASSWORD = "pw"
        token = cookie_value(self.login("pw"))
        raw = pepar.SESSIONS.read_text()
        self.assertNotIn(token, raw)
        self.assertNotIn("pw", json.loads(raw).get("fp", ""))
        self.assertEqual(pepar.SESSIONS.stat().st_mode & 0o777, 0o600)

    def test_session_count_is_capped(self):
        pepar.PASSWORD = "pw"
        for _ in range(230):
            pepar.start_session()
        self.assertLessEqual(len(pepar._sessions["sessions"]), 200)

    def test_responses_are_not_cacheable_by_shared_caches(self):
        pepar.PASSWORD = "pw"
        self.login("pw")
        for path in ("/", "/api/jobs", "/image/i-000000000000", "/login"):
            self.assertEqual(self.client.get(path).headers.get("Cache-Control"), "private, no-store", path)


class ClientIpTest(AuthBase):
    def test_without_trusted_proxy_header_is_ignored(self):
        pepar.PASSWORD = "pw"
        for i in range(pepar.LOGIN_FAILS):
            self.login("bad", headers={"CF-Connecting-IP": f"203.0.113.{i + 10}"})
        self.assertEqual(self.login("pw", headers={"CF-Connecting-IP": "203.0.113.99"}).status_code, 429)

    def test_trusted_proxy_separates_real_clients(self):
        pepar.PASSWORD = "pw"
        self.trust_loopback()
        for _ in range(pepar.LOGIN_FAILS):
            self.assertEqual(self.login("bad", headers=IP_A).status_code, 401)
        self.assertEqual(self.login("pw", headers=IP_A).status_code, 429)
        self.assertEqual(self.login("pw", headers=IP_B).status_code, 302)

    def test_trusted_proxy_cannot_be_bypassed_by_rotating_header_from_other_peers(self):
        pepar.PASSWORD = "pw"
        pepar.TRUSTED_PROXIES = [pepar.ipaddress.ip_network("10.9.9.9/32")]
        for i in range(pepar.LOGIN_FAILS):
            self.login("bad", headers={"CF-Connecting-IP": f"203.0.113.{i + 10}"})
        self.assertEqual(self.login("pw", headers={"CF-Connecting-IP": "203.0.113.99"}).status_code, 429)

    def test_ipv6_rotation_inside_a_64_shares_one_bucket(self):
        pepar.PASSWORD = "pw"
        self.trust_loopback()
        for i in range(pepar.LOGIN_FAILS):
            self.login("bad", headers={"CF-Connecting-IP": f"2001:db8:1:2::{i + 1:x}"})
        self.assertEqual(self.login("pw", headers={"CF-Connecting-IP": "2001:db8:1:2:ffff::9"}).status_code, 429)
        self.assertEqual(self.login("pw", headers={"CF-Connecting-IP": "2001:db8:1:3::1"}).status_code, 302)

    def test_garbage_header_falls_back_to_the_peer(self):
        pepar.PASSWORD = "pw"
        self.trust_loopback()
        for bad in ("not-an-ip", "1.2.3.4, 5.6.7.8", "", "999.1.1.1", "x" * 5000):
            self.assertEqual(self.login("bad", headers={"CF-Connecting-IP": bad}).status_code, 401)

    def test_x_forwarded_for_is_not_trusted(self):
        pepar.PASSWORD = "pw"
        self.trust_loopback()
        for i in range(pepar.LOGIN_FAILS):
            self.login("bad", headers={"X-Forwarded-For": f"203.0.113.{i + 10}"})
        self.assertEqual(self.login("pw", headers={"X-Forwarded-For": "203.0.113.99"}).status_code, 429)

    def test_secure_cookie_only_when_proxy_says_https_and_is_trusted(self):
        pepar.PASSWORD = "pw"
        https = {"X-Forwarded-Proto": "https"}
        self.assertNotIn("Secure", self.login("pw", headers=https).headers["Set-Cookie"])
        self.trust_loopback()
        self.assertIn("Secure", self.login("pw", headers=https).headers["Set-Cookie"])
        self.assertIn("Secure", self.login("pw", headers={"CF-Visitor": '{"scheme":"https"}'}).headers["Set-Cookie"])
        self.assertNotIn("Secure", self.login("pw").headers["Set-Cookie"])


class RouteCoverageTest(AuthBase):
    OPEN = {"login", "logout"}

    def sample_path(self, rule):
        path = rule.rule
        for arg in rule.arguments:
            path = path.replace(f"<{arg}>", {"model": "m", "key": "2401.00001"}.get(arg, "x"))
        return path.replace("<path:filename>", "x")

    def test_every_route_requires_login(self):
        pepar.PASSWORD = "pw"
        rules = [r for r in pepar.app.url_map.iter_rules() if r.endpoint not in self.OPEN]
        self.assertGreater(len(rules), 10)
        for rule in rules:
            for method in sorted(rule.methods - {"HEAD", "OPTIONS"}):
                r = self.client.open(self.sample_path(rule), method=method)
                self.assertIn(r.status_code, (302, 401), f"{method} {rule.rule} -> {r.status_code}")
                if r.status_code == 302:
                    self.assertIn("/login", r.headers["Location"], rule.rule)

    def test_options_head_and_odd_paths_are_blocked_too(self):
        pepar.PASSWORD = "pw"
        for method, path in (("OPTIONS", "/api/jobs"), ("HEAD", "/api/docs"), ("GET", "/%61pi/jobs"), ("GET", "/api/jobs/"),
                             ("GET", "//api/jobs"), ("GET", "/API/jobs"), ("GET", "/api/../api/jobs"), ("GET", "/static/x"),
                             ("GET", "/login/../api/jobs"), ("POST", "/login/x"), ("GET", "/nope")):
            r = self.client.open(path, method=method)
            self.assertIn(r.status_code, (301, 302, 308, 401, 404), f"{method} {path} -> {r.status_code}")
            self.assertNotIn(b"jobs\":", r.data[:200] if r.status_code == 200 else b"")
            if r.status_code == 200:
                self.fail(f"{method} {path} served without login")

    def test_login_logout_are_the_only_open_routes(self):
        names = {r.endpoint for r in pepar.app.url_map.iter_rules()}
        self.assertTrue(self.OPEN <= names)
        self.assertNotIn("static", names)


if __name__ == "__main__":
    unittest.main()
