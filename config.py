"""Settings: PEPAR_* environment variables > config.toml > defaults."""

import ipaddress
import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent

DEFAULTS = {
    "host": ["127.0.0.1"],
    "port": 8765,
    "llm_url": "http://127.0.0.1:9090/v1/chat/completions",
    "cache_dir": "cache",
    "password": "",
    "trusted_proxies": [],
    "allow_private_urls": False,
    "keep_scripts": False,
}


def _load() -> dict:
    path = Path(os.environ.get("PEPAR_CONFIG", ROOT / "config.toml"))
    cfg = dict(DEFAULTS)
    if path.exists():
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
    return cfg


settings = _load()
