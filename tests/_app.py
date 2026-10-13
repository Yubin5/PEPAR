import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tmp = tempfile.TemporaryDirectory()
os.environ["PEPAR_CACHE_DIR"] = _tmp.name
os.environ["PEPAR_CONFIG"] = str(Path(_tmp.name) / "no-config.toml")
os.environ["PEPAR_LLM_URL"] = "http://127.0.0.1:1/v1/chat/completions"

import app as pepar  # noqa: E402
