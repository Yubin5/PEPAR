import importlib
import os
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load(config: str | None = None, **env):
    import config as mod

    saved = {k: v for k, v in os.environ.items() if k.startswith("PEPAR_")}
    for k in saved:
        del os.environ[k]
    tmp = tempfile.TemporaryDirectory()
    try:
        path = Path(tmp.name) / "config.toml"
        if config is not None:
            path.write_text(config, encoding="utf-8")
        os.environ["PEPAR_CONFIG"] = str(path)
        for k, v in env.items():
            os.environ[k] = v
        return importlib.reload(mod).settings
    finally:
        for k in [k for k in os.environ if k.startswith("PEPAR_")]:
            del os.environ[k]
        os.environ.update(saved)
        tmp.cleanup()


class ConfigTest(unittest.TestCase):
    def test_example_matches_defaults(self):
        import config

        example = tomllib.loads((ROOT / "config.example.toml").read_text(encoding="utf-8"))
        self.assertEqual(example, config.DEFAULTS)

    def test_defaults_without_file(self):
        s = load()
        self.assertEqual(s["host"], ["127.0.0.1"])
        self.assertEqual(s["port"], 8765)
        self.assertTrue(s["cache_dir"].is_absolute())

    def test_partial_file_fills_defaults(self):
        s = load('host = ["127.0.0.1", "100.1.2.3"]')
        self.assertEqual(s["host"], ["127.0.0.1", "100.1.2.3"])
        self.assertEqual(s["port"], 8765)

    def test_env_overrides_file(self):
        s = load("port = 1111", PEPAR_PORT="2222", PEPAR_HOST="0.0.0.0, 127.0.0.1")
        self.assertEqual(s["port"], 2222)
        self.assertEqual(s["host"], ["0.0.0.0", "127.0.0.1"])

    def test_unknown_key_rejected(self):
        with self.assertRaises(SystemExit):
            load("prot = 1")

    def test_bad_number_rejected(self):
        with self.assertRaises(SystemExit):
            load('port = "abc"')

    def test_invalid_toml_rejected(self):
        with self.assertRaises(SystemExit):
            load("host = [")

    def test_password_from_file_and_env(self):
        self.assertEqual(load()["password"], "")
        self.assertEqual(load('password = "a"')["password"], "a")
        self.assertEqual(load('password = "a"', PEPAR_PASSWORD="b")["password"], "b")

    def test_booleans(self):
        s = load()
        self.assertEqual((s["allow_private_urls"], s["keep_scripts"]), (False, False))
        s = load("keep_scripts = true")
        self.assertEqual((s["allow_private_urls"], s["keep_scripts"]), (False, True))
        self.assertTrue(load(PEPAR_ALLOW_PRIVATE_URLS="true")["allow_private_urls"])
        self.assertFalse(load("keep_scripts = true", PEPAR_KEEP_SCRIPTS="0")["keep_scripts"])
        with self.assertRaises(SystemExit):
            load('keep_scripts = "yes"')

    def test_empty_host_rejected(self):
        with self.assertRaises(SystemExit):
            load("host = []")

    def test_absolute_cache_dir_kept(self):
        self.assertEqual(str(load('cache_dir = "/tmp/x"')["cache_dir"]), "/tmp/x")


class DefaultModelTest(unittest.TestCase):
    def test_prefers_loaded_model(self):
        import translator

        models = [{"id": "a", "loaded": False}, {"id": "b", "loaded": True}]
        self.assertEqual(translator.pick_default_model(models), "b")

    def test_falls_back_to_first(self):
        import translator

        self.assertEqual(translator.pick_default_model([{"id": "a", "loaded": False}, {"id": "b", "loaded": False}]), "a")

    def test_empty_list(self):
        import translator

        self.assertEqual(translator.pick_default_model([]), "")


if __name__ == "__main__":
    unittest.main()
