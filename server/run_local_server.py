"""
간단 로컬 실행 스크립트: redis가 준비되어 있지 않아도 로그인/대시보드 확인 가능
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

print('=== Server (Admin UI) - Local Run ===')
print('Admin UI: http://127.0.0.1:5000/admin (user=admin / pass=admin123)')

from app import app
app.run(host='127.0.0.1', port=5000, debug=True)
