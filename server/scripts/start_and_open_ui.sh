#!/usr/bin/env bash
set -euo pipefail

# 실행 위치를 repository의 server 폴더로 이동
cd "$(dirname "$0")/.."

echo "Starting Docker services..."
# 빌드 및 백그라운드 실행
sudo docker compose up -d --build

# 헬스 체크가 준비될 때까지 최대 30초 대기
echo "Waiting for server health..."
for i in $(seq 1 30); do
  if curl -sSf http://localhost:8000/health >/dev/null 2>&1; then
    echo "Server is healthy"
    break
  fi
  sleep 1
done

# GUI 환경이면 브라우저 열기 (xdg-open 사용, 없으면 python webbrowser 사용)
URL="http://localhost:8000/admin"
if command -v xdg-open >/dev/null 2>&1 && [ -n "${DISPLAY-}" ]; then
  echo "Opening $URL in default browser..."
  xdg-open "$URL" || true
else
  echo "Attempting to open default browser via Python module..."
  python3 - <<PY
import webbrowser
try:
    webbrowser.open('''$URL''')
    print('Browser open command issued')
except Exception as e:
    print('Failed to open browser:', e)
PY
fi

echo "Done. If the browser didn't open, visit $URL manually."
