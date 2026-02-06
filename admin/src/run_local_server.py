"""
로컬 서버 실행 스크립트
Docker 없이 PostgreSQL + 메모리 캐시로 실행
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env.local')

# 데이터 디렉토리 생성
data_dir = PROJECT_ROOT / 'data'
data_dir.mkdir(exist_ok=True)

print("=== 출결관리 시스템 - 로컬 서버 시작 ===")
print()

# 1. 데이터베이스 초기화
print("[1/3] 데이터베이스 초기화...")
from lib.database_local import init_database
init_database()
print("✓ PostgreSQL 데이터베이스 준비 완료")

# 2. ZeroMQ 서버 시작 (백그라운드)
print()
print("[2/3] ZeroMQ 서버 시작...")
from lib.zmq_server_local import start_zmq_server
import threading
zmq_thread = threading.Thread(target=start_zmq_server, daemon=True)
zmq_thread.start()
print("✓ ZeroMQ 서버 시작됨 (tcp://localhost:5555)")

# 3. Flask 웹 서버 시작
print()
print("[3/3] Flask 웹 서버 시작...")
print()
print("=" * 50)
print("✓ 서버 준비 완료!")
print("=" * 50)
print()
print("관리자 UI: http://127.0.0.1:5000/admin")
print("사용자명: admin")
print("비밀번호: admin123")
print()
print("Ctrl+C로 종료")
print()

from app_local import app
app.run(host='127.0.0.1', port=5000, debug=False)
