"""Settings live in config.toml (one file, also used by Docker). It is optional: without it, or for
missing keys, DEFAULTS apply.

PEPAR_* environment variables still override the file for advanced use; the Docker image sets only
container facts with them (listen on 0.0.0.0:8765, cache in /app/cache).
"""

import ipaddress
import os
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent

DEFAULTS = {
    "host": ["127.0.0.1"],
    "port": 8765,
    "llm_url": "http://127.0.0.1:9090",
    "cache_dir": "cache",
    "password": "",
    "trusted_proxies": [],
    "allow_private_urls": False,
    "keep_scripts": False,
}


def _load() -> dict:
    path = Path(os.environ.get("PEPAR_CONFIG", ROOT / "config.toml"))
    cfg = dict(DEFAULTS)
    if path.is_dir():
        # 예전 compose.yml은 config.toml 파일 하나를 연결했는데, 파일이 없으면 Docker가 그 이름으로 빈 폴더를 만든다
        print(f"Ignoring {path}: it is a directory, not a file (Docker creates one when a mounted file is missing). "
              "Remove it (rmdir config.toml) and copy config.example.toml to config.toml to change settings.",
              file=sys.stderr, flush=True)
    elif path.exists():
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise SystemExit(f"Cannot read config file ({path}): {e}") from e
        unknown = set(data) - set(DEFAULTS)
        if unknown:
            raise SystemExit(f"Unknown keys in config file ({path}): {', '.join(sorted(unknown))}")
        cfg.update(data)

    env = os.environ.get
    if env("PEPAR_HOST"):
        cfg["host"] = env("PEPAR_HOST")
    if env("PEPAR_PORT"):
        cfg["port"] = env("PEPAR_PORT")
    if env("PEPAR_LLM_URL"):
        cfg["llm_url"] = env("PEPAR_LLM_URL")
    if env("PEPAR_CACHE_DIR"):
        cfg["cache_dir"] = env("PEPAR_CACHE_DIR")
    if env("PEPAR_TRUSTED_PROXIES"):
        cfg["trusted_proxies"] = env("PEPAR_TRUSTED_PROXIES")
    if env("PEPAR_PASSWORD"):
        cfg["password"] = env("PEPAR_PASSWORD")
    for key in ("allow_private_urls", "keep_scripts"):
        if env(f"PEPAR_{key.upper()}"):
            cfg[key] = env(f"PEPAR_{key.upper()}").lower() in ("1", "true", "yes", "on")
        elif not isinstance(cfg[key], bool):
            raise SystemExit(f"{key} must be true or false.")

    if isinstance(cfg["host"], str):  # comma-separated string
        cfg["host"] = cfg["host"].split(",")
    cfg["host"] = [h.strip() for h in cfg["host"] if h.strip()]
    try:
        cfg["port"] = int(cfg["port"])
    except (TypeError, ValueError) as e:
        raise SystemExit(f"port must be a number: {e}") from e
    cfg["password"] = str(cfg["password"])
    if isinstance(cfg["trusted_proxies"], str):
        cfg["trusted_proxies"] = cfg["trusted_proxies"].split(",")
    try:
        cfg["trusted_proxies"] = [str(ipaddress.ip_network(p.strip(), strict=False)) for p in cfg["trusted_proxies"] if p.strip()]
    except (TypeError, ValueError) as e:
        raise SystemExit(f"trusted_proxies must be IP addresses or CIDR ranges: {e}") from e
    if not cfg["host"]:
        raise SystemExit("host must not be empty.")
    cache = Path(cfg["cache_dir"]).expanduser()
    cfg["cache_dir"] = cache if cache.is_absolute() else ROOT / cache
    cfg["llm_url"] = llm_endpoint(str(cfg["llm_url"]))
    return cfg


def llm_endpoint(url: str) -> str:
    """llm_url은 주소와 포트만 적어도 된다: http://127.0.0.1:8080 → …/v1/chat/completions. 끝까지 적었으면 그대로."""
    url = url.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if not url.endswith("/v1"):
        url += "/v1"
    return url + "/chat/completions"


settings = _load()
