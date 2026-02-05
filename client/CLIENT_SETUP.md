# 클라이언트 설정 가이드 (Windows 10)

## Python SQLAlchemy 기반 클라이언트

이 클라이언트는 Electron 대신 **Python + PyQt5 + SQLAlchemy**를 사용하여 데이터베이스에 직접 접근합니다.

## 시스템 요구사항

- Windows 10 (64-bit)
- Python 3.10 이상
- 웹캠 또는 USB 카메라
- NVIDIA GPU (선택사항, CUDA 지원)

## 설치 순서

### 1. Node.js 설치

[Node.js 공식 웹사이트](https://nodejs.org/)에서 LTS 버전을 다운로드하여 설치합니다.

설치 확인:
```powershell
node --version
npm --version
```

### 2. 프로젝트 설정

```powershell
# 프로젝트 디렉토리로 이동
cd D:\nextcloud\YOLO_program\mini-project-2\client

# 의존성 설치
npm install
```

### 3. 서버 연결 설정

[src/renderer/app.js](src/renderer/app.js) 파일에서 API 서버 주소를 확인/수정:

```javascript
const API_BASE_URL = 'http://localhost:5000/api';
```

서버가 다른 머신에 있는 경우:
```javascript
const API_BASE_URL = 'http://서버IP주소:5000/api';
```

### 4. 클라이언트 실행

#### 개발 모드
```powershell
npm run dev
```
DevTools가 자동으로 열립니다.

#### 일반 실행
```powershell
npm start
```

### 5. 실행 파일 빌드 (선택사항)

```powershell
# electron-builder 설치 (아직 설치 안했다면)
npm install electron-builder --save-dev

# 빌드
npm run build
```

빌드된 실행 파일은 `dist/` 폴더에 생성됩니다.

## 사용 방법

### 로그인

기본 테스트 계정:
- **관리자**: admin / admin123
- **교수**: prof_kim / prof123
- **학생**: student1 / student123

### 주요 기능

1. **대시보드**: 전체 통계 확인
2. **과목 관리**: 과목 목록 조회 및 관리
3. **출석 관리**: 출석 체크 및 기록
4. **학생 관리**: 학생 목록 조회
5. **통계/보고서**: 출석 통계 확인

## 문제 해결

### 서버 연결 오류

```
Error: Network Error
```

**해결 방법:**
1. 서버가 실행 중인지 확인
2. 방화벽 설정 확인
3. API_BASE_URL이 올바른지 확인

### Electron 실행 오류

```
Error: Electron failed to install correctly
```

**해결 방법:**
```powershell
# node_modules 삭제
Remove-Item -Recurse -Force node_modules

# package-lock.json 삭제
Remove-Item package-lock.json

# 재설치
npm install
```

### DevTools 열기

개발 중 문제 해결을 위해 DevTools를 열려면:
- `Ctrl + Shift + I` (개발 모드가 아닐 때)
- 또는 `npm run dev`로 실행

### 캐시 삭제

로그인 정보나 설정이 꼬였을 때:

```powershell
# AppData 폴더의 Electron 캐시 삭제
Remove-Item -Recurse "$env:APPDATA\attendance-client"
```

## 개발 팁

### 실시간 리로드

개발 중 HTML/CSS/JS 파일을 수정하면:
1. `Ctrl + R`: 페이지 새로고침
2. `Ctrl + Shift + R`: 캐시 삭제 후 새로고침

### 콘솔 로그 확인

- 메인 프로세스 로그: PowerShell 터미널에 출력
- 렌더러 프로세스 로그: DevTools 콘솔에 출력

### 네트워크 요청 모니터링

DevTools의 Network 탭에서 API 요청/응답을 확인할 수 있습니다.

## 배포

### Installer 생성

`package.json`에 빌드 설정 추가:

```json
{
  "build": {
    "appId": "com.attendance.client",
    "productName": "출결관리 시스템",
    "directories": {
      "output": "dist"
    },
    "win": {
      "target": "nsis",
      "icon": "assets/icon.ico"
    }
  }
}
```

빌드:
```powershell
npm run build
```

### 자동 업데이트 (선택사항)

electron-updater를 사용하여 자동 업데이트 기능 구현 가능
