#!/bin/bash
# 수동 실행. 설정은 config.toml 또는 PEPAR_* 환경변수.
cd "$(dirname "$0")"
exec .venv/bin/python app.py
