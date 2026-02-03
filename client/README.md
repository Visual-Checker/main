# Client UI (출결관리 클라이언트)

이 디렉터리는 PyQt5 기반의 출석 체크용 클라이언트 UI를 포함합니다. 이전 프로젝트(`mini project-2`)에서 UI 코드를 가져와 재구성한 것입니다.

## 주요 파일
- `client_ui.py`: 메인 UI 코드 (카메라 뷰, 모드 버튼, 사용자 정보 패널 등)
- `ui_config_lib.py`: UI 설정(크기, 색상, 버튼 목록 등)
- `requirements.txt`: 실행에 필요한 Python 패키지 목록

## 요구 환경
- OS: Windows 10 (개발/테스트 환경)
- Python 3.10+ 권장

## 실행 방법 (Windows)
1. 가상환경 생성: `python -m venv .venv`
2. 가상환경 활성화: `.\.venv\Scripts\Activate.ps1` (PowerShell)
3. 패키지 설치: `pip install -r requirements.txt`
4. UI 실행: `python client_ui.py`

참고: MediaPipe, SpeechBrain 등 일부 모델/패키지는 추가 설정이나 모델 파일 다운로드가 필요합니다. 모델 파일 경로는 기본적으로 `../models/` 를 참조합니다. 서버 쪽에서 모델을 관리할 계획이면 해당 경로를 맞춰주세요.

원하시면 이 UI를 React/web 기반으로 포팅하거나, UI 컴포넌트를 더 모듈화(예: Qt Designer 사용)해 드릴 수 있습니다.