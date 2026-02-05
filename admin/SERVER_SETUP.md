# 서버 설정 가이드 (Ubuntu 24.04)

## 시스템 요구사항

- Ubuntu 24.04 LTS
- Python 3.10 이상
- pip
- Docker (선택사항, 데이터베이스용)

## 설치 순서

### 1. Python 및 pip 설치

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
```

### 2. PostgreSQL 클라이언트 라이브러리 설치

```bash
sudo apt install libpq-dev -y
```

### 3. 프로젝트 설정

```bash
# 프로젝트 디렉토리로 이동
cd /path/to/mini-project-2/server

# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 4. 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집
nano .env
```

필수 설정 항목:
- `SECRET_KEY`: Flask 시크릿 키 (랜덤 문자열)
- `JWT_SECRET_KEY`: JWT 시크릿 키 (랜덤 문자열)
- `DB_HOST`: 데이터베이스 호스트 (Docker 사용시 localhost)
- `DB_PASSWORD`: 데이터베이스 비밀번호

### 5. 데이터베이스 연결 확인

```bash
# Python 쉘에서 테스트
python3 << EOF
from database import get_db
db = get_db()
if db.conn:
    print("데이터베이스 연결 성공!")
    db.disconnect()
else:
    print("데이터베이스 연결 실패!")
EOF
```

### 6. 서버 실행

#### 개발 모드
```bash
python app.py
```

#### 프로덕션 모드 (Gunicorn 사용)
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 7. 방화벽 설정 (필요시)

```bash
sudo ufw allow 5000/tcp
```

### 8. 시스템 서비스 등록 (선택사항)

systemd 서비스 파일 생성:

```bash
sudo nano /etc/systemd/system/attendance-server.service
```

내용:
```ini
[Unit]
Description=Attendance Management Server
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/mini-project-2/server
Environment="PATH=/path/to/mini-project-2/server/venv/bin"
ExecStart=/path/to/mini-project-2/server/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

서비스 활성화:
```bash
sudo systemctl daemon-reload
sudo systemctl enable attendance-server
sudo systemctl start attendance-server
sudo systemctl status attendance-server
```

## 로그 확인

```bash
# 개발 모드 - 콘솔 출력

# 서비스 모드
sudo journalctl -u attendance-server -f
```

## 문제 해결

### PostgreSQL 연결 오류
```bash
# PostgreSQL이 실행 중인지 확인
docker ps | grep postgres

# 또는 로컬 PostgreSQL 확인
sudo systemctl status postgresql
```

### 포트 충돌
```bash
# 5000 포트 사용 확인
sudo lsof -i :5000

# 프로세스 종료
sudo kill -9 <PID>
```

### 권한 오류
```bash
# 프로젝트 디렉토리 권한 확인
ls -la /path/to/mini-project-2/server

# 필요시 권한 수정
sudo chown -R $USER:$USER /path/to/mini-project-2/server
```
