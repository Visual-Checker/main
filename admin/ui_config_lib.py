# 관리자 UI 설정 파일
# 위치, 크기, 색상 등을 자유롭게 수정 가능

# ========== 윈도우 설정 ==========
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "출결관리 시스템 - 관리자 모드"

# ========== 색상 설정 ==========
BG_COLOR = "#2C3E50"  # 배경색 (다크 블루)
SIDEBAR_COLOR = "#34495E"  # 사이드바 색상
BUTTON_COLOR = "#3498DB"  # 버튼 색상 (파란색)
BUTTON_HOVER_COLOR = "#2980B9"  # 버튼 호버 색상
TEXT_COLOR = "#ECF0F1"  # 텍스트 색상 (밝은 회색)
ACCENT_COLOR = "#E74C3C"  # 강조 색상 (빨간색)

# ========== 좌측 사이드바 설정 ==========
SIDEBAR_WIDTH = 250
SIDEBAR_PADDING = 20

# 관리자 모드 라벨
ADMIN_LABEL_HEIGHT = 60
ADMIN_LABEL_FONT_SIZE = 18
ADMIN_LABEL_FONT_WEIGHT = "bold"

# 좌측 버튼 설정
LEFT_BUTTON_WIDTH = 210
LEFT_BUTTON_HEIGHT = 50
LEFT_BUTTON_SPACING = 15
LEFT_BUTTON_START_Y = 100  # 첫 번째 버튼 시작 Y 위치
LEFT_BUTTON_FONT_SIZE = 12

# 좌측 버튼 목록
LEFT_BUTTONS = [
    {"text": "📷 사진 등록", "name": "photo_register"},
    {"text": "🎤 목소리 등록", "name": "voice_register"},
    {"text": "👋 제스처 등록", "name": "gesture_register"},
]

# ========== 중앙 카메라 영역 설정 ==========
CAM_X = SIDEBAR_WIDTH + 30
CAM_Y = 80
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_BG_COLOR = "#1C2833"  # 카메라 배경색

# ========== 우측 버튼 영역 설정 ==========
RIGHT_BUTTON_X = CAM_X + CAM_WIDTH + 20
RIGHT_BUTTON_Y = CAM_Y
RIGHT_BUTTON_WIDTH = 180
RIGHT_BUTTON_HEIGHT = 45
RIGHT_BUTTON_SPACING = 20
RIGHT_BUTTON_FONT_SIZE = 11

# 우측 버튼 목록
RIGHT_BUTTONS = [
    {"text": "📸 사진찍기", "name": "capture"},
    {"text": "💾 사진 저장", "name": "save"},
    {"text": "✏️ 이름 입력", "name": "input_info"},
]

# ========== 입력 필드 설정 ==========
INPUT_FIELD_WIDTH = 180
INPUT_FIELD_HEIGHT = 35
INPUT_FIELD_Y = RIGHT_BUTTON_Y + (RIGHT_BUTTON_HEIGHT + RIGHT_BUTTON_SPACING) * len(RIGHT_BUTTONS) + 30

# ========== 상태 표시 영역 ==========
STATUS_BAR_HEIGHT = 40
STATUS_BAR_BG_COLOR = "#1A252F"
STATUS_FONT_SIZE = 10

# ========== 카메라 설정 ==========
CAMERA_INDEX = 0
CAMERA_FPS = 30
