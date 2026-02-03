# 클라이언트 UI 설정 파일
# 위치, 크기, 색상 등을 자유롭게 수정 가능

# ========== 윈도우 설정 ==========
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "출결관리 시스템 - 출석 체크"

# ========== 색상 설정 ==========
BG_COLOR = "#1E272E"  # 배경색 (다크 그레이)
SIDEBAR_COLOR = "#2C3A47"  # 사이드바 색상
BUTTON_COLOR = "#5F27CD"  # 버튼 색상 (보라색)
BUTTON_HOVER_COLOR = "#341F97"  # 버튼 호버 색상
TEXT_COLOR = "#F5F6FA"  # 텍스트 색상
ACCENT_COLOR = "#00D2D3"  # 강조 색상 (청록색)
SUCCESS_COLOR = "#10AC84"  # 성공 색상 (초록색)
WARNING_COLOR = "#FFA502"  # 경고 색상 (주황색)

# ========== 좌측 사이드바 설정 ==========
SIDEBAR_WIDTH = 250
SIDEBAR_PADDING = 20

# 클라이언트 모드 라벨
CLIENT_LABEL_HEIGHT = 60
CLIENT_LABEL_FONT_SIZE = 18
CLIENT_LABEL_FONT_WEIGHT = "bold"

# 좌측 버튼 설정
LEFT_BUTTON_WIDTH = 210
LEFT_BUTTON_HEIGHT = 50
LEFT_BUTTON_SPACING = 15
LEFT_BUTTON_START_Y = 100
LEFT_BUTTON_FONT_SIZE = 12

# 좌측 버튼 목록
LEFT_BUTTONS = [
    {"text": "😊 얼굴 인식 출석", "name": "face_attendance"},
    {"text": "✋ 제스처 출석", "name": "gesture_attendance"},
    {"text": "🎤 음성 인식 출석", "name": "voice_attendance"},
    {"text": "📊 출석 현황", "name": "attendance_status"},
    {"text": "🧠 자동 인식 모드", "name": "automation"},
    {"text": "⚙ 설정", "name": "settings"},
]

# ========== 중앙 카메라 영역 설정 ==========
CAM_X = SIDEBAR_WIDTH + 30
CAM_Y = 80
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_BG_COLOR = "#0F1419"  # 카메라 배경색

# ========== 우측 정보 영역 설정 ==========
RIGHT_PANEL_X = CAM_X + CAM_WIDTH + 20
RIGHT_PANEL_Y = CAM_Y
RIGHT_PANEL_WIDTH = 280
RIGHT_PANEL_HEIGHT = CAM_HEIGHT

# 사용자 정보 표시
USER_INFO_HEIGHT = 150
USER_INFO_BG_COLOR = "#2C3A47"

# 출석 상태 표시
ATTENDANCE_STATUS_HEIGHT = 200
ATTENDANCE_STATUS_BG_COLOR = "#2C3A47"

# ========== 제스처 가이드 ==========
GESTURE_GUIDE_HEIGHT = 120
GESTURE_GUIDE_BG_COLOR = "#2C3A47"

# ========== 상태 표시 영역 ==========
STATUS_BAR_HEIGHT = 40
STATUS_BAR_BG_COLOR = "#0B1016"
STATUS_FONT_SIZE = 10

# ========== 카메라 설정 ==========
CAMERA_INDEX = 0
CAMERA_FPS = 30

# ========== 제스처 설정 ==========
GESTURES = {
    "thumbs_up": {"emoji": "👍", "text": "출석 체크", "color": SUCCESS_COLOR},
    "peace": {"emoji": "✌️", "text": "확인", "color": ACCENT_COLOR},
    "fist": {"emoji": "✊", "text": "취소", "color": WARNING_COLOR},
    "open_palm": {"emoji": "✋", "text": "대기", "color": TEXT_COLOR},
    "pointing": {"emoji": "☝️", "text": "선택", "color": BUTTON_COLOR},
}
