#!/bin/bash
# pepar 재시작 (systemd 사용자 서비스). 번역 중이거나 대기 중인 작업이 있으면 거부한다 (-f로 강제).
# 로그: journalctl --user -u pepar -f
PORT=${PEPAR_PORT:-8765}
if [ "$1" != "-f" ]; then
  jobs=$(curl -s -m 5 "http://127.0.0.1:$PORT/api/jobs" | python3 -c "
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
