# Server - Admin UI (웹)

이 디렉터리는 관리자 웹 UI(Flask 기반) 스켈레톤을 포함합니다. mini project-2의 관리자 UI를 참고하여 간단한 대시보드와 로그인 페이지를 구성했습니다.

## 실행 (로컬 테스트)
1. 가상환경 생성: `python -m venv .venv`
2. 활성화: `.\.venv\Scripts\Activate.ps1` (Windows) 또는 `source .venv/bin/activate` (Linux)
3. 패키지 설치: `pip install -r requirements.txt`
4. 환경 변수: `.env` 파일에 `SECRET_KEY` 등을 설정 (선택)
5. 실행: `python app.py` (Flask 개발 서버 활성화; 기본 포트 5000)

## 엔드포인트
- `/admin` - 관리자 대시보드 (로그인 필요)
- `/admin/login` - 로그인
- `/admin/api/statistics` - 통계 (임시, Redis 연동 시 실제 값 반환)
- `/admin/api/attendance-events` - 출석 이벤트 (Redis에서 읽음)

## 다음 단계 제안
- Redis/DB 연동 테스트 및 데이터 소스 연결
- 인증(회원 관리) 및 권한(세션) 강화
- 실시간 모니터(웹소켓) 추가
