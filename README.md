# Multimodal User Recognition System 🚀

## 1. 시스템 목적 🎯

본 시스템은 **얼굴·제스처·음성 기반 멀티모달 인식**을 통해 사용자를 식별하고, 출입(Check-in / Check-out) 및 권한 승인을 수행하는 것을 목표로 한다.

- **Server / Admin**: 데이터 등록(Enrollment) 및 관리
- **Client**: 실시간 추론(Inference) 및 사용자 상호작용

> 일단 기억해둠 (초기 설계 명세)

---

## 2. 전체 아키텍처 개요 🏗️

### 2.1 구성 요소

**Server / Admin**
- Face Recognition: DeepInsight / InsightFace
- Gesture Recognition: MediaPipe Hands
- Voice Recognition: SpeechBrain (ECAPA-TDNN)
- Decision / Log Service

**Client**
- User Recognition (실시간 추론)
- Check-in / Check-out Checker (vector 기반)
- Logger

**DB 계층**
- PostgreSQL (메타데이터, 권한, 로그)
- Vector Store (얼굴/제스처/음성 임베딩)

---

## 3. 개발 / 배포 환경 ⚙️

- **서버 OS**: `Ubuntu 24.04`
- **클라이언트 OS**: `Windows 10`

---

## 4. 초기 작업 제안 🔧

- `server/`에 `Dockerfile` 또는 `devcontainer` 추가 (기본 이미지: `ubuntu:24.04`)
- `server/`에 Flask 기반 **관리자(Admin) 웹 UI** 스켈레톤(`templates/admin`, `routes/admin`) 추가 (로그인, 대시보드, 통계 API) ✅
- `client/`에 `README.md`로 Windows 10 설치 및 실행 안내 추가
- `client/`에 PyQt 기반 UI 샘플(`client_ui.py`)과 의존성 없이 실행 가능한 라이트 모드(`client_ui_light.py`) 추가
- CI: GitHub Actions에서 `ubuntu-24.04` / `windows-latest` 매트릭스 설정

---

추가로 반영할 요구사항이나 수정할 내용이 있으면 알려주세요.