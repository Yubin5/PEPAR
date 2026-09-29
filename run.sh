#!/bin/bash
# pepar 수동 실행 (평소에는 systemd 사용자 서비스 pepar.service로 돈다). 환경 변수로 설정 변경 가능:
#   PEPAR_HOST (쉼표 구분, 기본 127.0.0.1,100.92.150.128), PEPAR_PORT (기본 8765)
#   PEPAR_LLM_URL (기본 http://127.0.0.1:9090/v1/chat/completions), PEPAR_MODEL (기본 gemma4-12b)
cd "$(dirname "$0")"
exec .venv/bin/python app.py
