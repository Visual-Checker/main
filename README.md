# Multimodal Attendance System

얼굴, 제스처, 음성 정보를 활용해 사용자를 식별하고 출석 이벤트를 기록하는 프로젝트입니다.

## 핵심 구성

- **Admin (서버/관리자)**: Flask 기반 API 및 관리자 UI, ZMQ 수신/캐시
- **Client (클라이언트)**: PyQt5 기반 UI, 실시간 인식 및 이벤트 전송
- **DB**: PostgreSQL + SQLAlchemy

## 폴더 구조

- **admin/**: 서버 및 관리자 UI
- **client/**: 클라이언트 UI
- **data/**: 로컬 실행 시 생성되는 데이터(리포지토리에 포함하지 않음)
- **run_client_admin.ps1**: 관리자/클라이언트 실행 메뉴

## 요구사항

- Python 3.10 이상
- PostgreSQL (로컬 또는 서버)
- Windows 10/11 (클라이언트), Ubuntu 24.04 (서버 권장)

모델 파일(.ckpt)은 Git LFS로 관리됩니다. 최초 클론 후 아래 명령을 권장합니다.

```bash
git lfs install
git lfs pull
```

## 빠른 시작 (Windows)

### 1) 의존성 설치

```powershell
cd admin
pip install -r requirements.txt

cd ..\client
pip install -r requirements.txt
```

### 2) 로컬 환경 변수 설정

admin 폴더에 `.env.local`을 만들고 최소 아래 항목을 설정하세요.

```
DB_USER=admin
DB_PASSWORD=admin123
DB_HOST=localhost
DB_PORT=5432
DB_NAME=attendance_system
SECRET_KEY=local-dev-secret-key
```

### 3) 로컬 서버 실행 (PostgreSQL + ZMQ + Flask)

```powershell
cd admin
python src\run_local_server.py
```

관리자 UI: http://127.0.0.1:5000/admin

### 4) 관리자/클라이언트 UI 실행

```powershell
# 관리자 UI
cd admin
python src\admin_ui.py

# 클라이언트 UI
cd ..\client
python src\client_ui.py
```

또는 다음 스크립트로 실행할 수 있습니다.

```powershell
\run_client_admin.ps1
```

## 서버 실행 (API 모드)

```bash
cd admin
python src/app.py
```

API 기본 경로: `http://<host>:5000/`

## 데이터베이스

- 로컬/서버 모두 PostgreSQL 사용
- SQLAlchemy 모델은 [admin/src/lib/database.py](admin/src/lib/database.py)에 정의
- 로컬 서버용 경량 테이블은 [admin/src/lib/database_local.py](admin/src/lib/database_local.py)에서 관리

## 참고 문서

- 서버 설치: [admin/SERVER_SETUP.md](admin/SERVER_SETUP.md)
- 클라이언트 설치: [client/CLIENT_SETUP.md](client/CLIENT_SETUP.md)

## 문제 해결

- PostgreSQL 연결 오류: 호스트/포트/계정 정보 확인 후 재시도
- ZMQ 수신 확인: `tcp://localhost:5555` 포트가 열려 있는지 확인