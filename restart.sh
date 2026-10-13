#!/bin/bash
# 작업이 진행 중이면 재시작을 거부한다 (-f로 강제).
cd "$(dirname "$0")"
PORT=${PEPAR_PORT:-8765}
PASSWORD=$(.venv/bin/python -c "import config; print(config.settings['password'])")
JAR=$(mktemp)
trap 'rm -f "$JAR"' EXIT
[ -n "$PASSWORD" ] && curl -s -m 5 -c "$JAR" --data-urlencode "password=$PASSWORD" "http://127.0.0.1:$PORT/login" -o /dev/null
if [ "$1" != "-f" ]; then
  jobs=$(curl -s -m 5 -b "$JAR" "http://127.0.0.1:$PORT/api/jobs" | python3 -c "
import json, sys
try: print(sum(1 for j in json.load(sys.stdin) if j['stage'] != 'error'))
except Exception: print(0)")
  if [ "${jobs:-0}" -gt 0 ]; then
    echo "진행 중인 작업 ${jobs}개가 있어 재시작하지 않습니다. 강제로 하려면: $0 -f" >&2
    exit 1
  fi
fi
systemctl --user restart pepar
sleep 2
systemctl --user --no-pager status pepar | head -5
