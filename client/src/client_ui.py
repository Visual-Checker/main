"""
클라이언트 UI - 출결관리 시스템
제스처 및 얼굴 인식 출석 체크
"""

import sys
import cv2
import os
import pickle
import time
import numpy as np
import torch
import threading
from datetime import datetime
import matplotlib.ticker as ticker
from matplotlib import font_manager, rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from dotenv import load_dotenv
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QMessageBox, QFrame, QFileDialog, QStackedLayout, QInputDialog
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor

load_dotenv()

# MediaPipe import
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    # Task API import 시도
    try:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from mediapipe import Image as MPImage
        MEDIAPIPE_AVAILABLE = True
        USE_TASK_API = True
        print("✓ MediaPipe Task API 사용 가능")
    except:
        # Task API 없으면 OpenCV만 사용
        USE_TASK_API = False
        MEDIAPIPE_AVAILABLE = False
        print("ℹ️  MediaPipe Task API를 사용할 수 없습니다. OpenCV로 대체합니다.")
except ImportError:
    print("⚠️  MediaPipe가 설치되지 않았습니다.")

# SpeechBrain 음성 인식 모델 (옵션)
SPEECHBRAIN_AVAILABLE = False
try:
    import torchaudio

    # torchaudio backend 체크 무력화 (Windows/환경 이슈 대응)
    def _noop(*args, **kwargs):
        return None

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: []
    if hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = _noop
    if hasattr(torchaudio, "backend") and hasattr(torchaudio.backend, "utils"):
        if hasattr(torchaudio.backend.utils, "set_audio_backend"):
            torchaudio.backend.utils.set_audio_backend = _noop
    if hasattr(torchaudio, "utils") and hasattr(torchaudio.utils, "check_torchaudio_backend"):
        torchaudio.utils.check_torchaudio_backend = _noop

    from speechbrain.inference.speaker import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
    print("✓ SpeechBrain 사용 가능")
except Exception:
    SPEECHBRAIN_AVAILABLE = False
    print("ℹ️  SpeechBrain(음성 모델)을 사용할 수 없습니다.")

# UI 설정 임포트
from lib.ui_config_lib import *


class ClientUI(QMainWindow):
    """클라이언트 출석 체크 윈도우"""

    voice_attendance_result = pyqtSignal(str, float, str)
    
    def __init__(self):
        super().__init__()
        
        # 카메라 초기화
        self.camera = None
        self.current_frame = None
        self.current_mode = None  # 'gesture', 'face', 'voice', None
        self.current_user = None
        self.attendance_check_type = "in"  # 'in' or 'out'
        
        # 얼굴 감지기 초기화
        self.face_detector = None
        self.gesture_recognizer = None
        
        # 얼굴 인식 설정
        self.face_confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', 0.70))
        self.face_similarity_threshold = float(os.getenv('FACE_SIMILARITY_THRESHOLD', 0.70))
        
        # 제스처 인식 설정
        self.gesture_confidence_threshold = float(os.getenv('GESTURE_CONFIDENCE_THRESHOLD', 0.5))
        self.gesture_cooldown = float(os.getenv('GESTURE_COOLDOWN', 3.0))
        self.last_gesture_time = {}  # {gesture_type: timestamp}
        self.detected_gestures = []  # 감지된 제스처 히스토리
        self.last_automation_gesture = None  # (type, confidence, timestamp)
        
        # 음성 인식 설정
        self.voice_encoder = None
        self.known_voice_embeddings = []
        self.known_voice_names = []
        self.voice_similarity_threshold = float(os.getenv('VOICE_SIMILARITY_THRESHOLD', 0.7))
        self.voice_sensitivity_multiplier = float(os.getenv('VOICE_SENSITIVITY_MULTIPLIER', 3.0))
        self.voice_activity_threshold = float(os.getenv('VOICE_ACTIVITY_THRESHOLD', 0.01))
        self.voice_model_path = os.getenv('VOICE_MODEL_PATH', 'models/spkrec-ecapa-voxceleb')
        self.last_voice_result = None  # 마지막 음성 인식 결과 (name, confidence)
        self.voice_result_time = 0  # 마지막 음성 인식 시간
        self.voice_live_running = False
        self.voice_live_interval_ms = int(os.getenv("VOICE_LIVE_INTERVAL_MS", "1500"))
        self.voice_live_duration = float(os.getenv("VOICE_LIVE_DURATION", "1.5"))
        
        if MEDIAPIPE_AVAILABLE and USE_TASK_API:
            try:
                # 얼굴 감지기
                base_options_face = python.BaseOptions(model_asset_path='client/models/blaze_face_short_range.tflite')
                face_options = vision.FaceDetectorOptions(base_options=base_options_face)
                self.face_detector = vision.FaceDetector.create_from_options(face_options)
                
                # 제스처 인식기
                base_options_gesture = python.BaseOptions(model_asset_path='client/models/gesture_recognizer.task')
                gesture_options = vision.GestureRecognizerOptions(base_options=base_options_gesture)
                self.gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_options)
                
                print("✓ MediaPipe Task API 초기화 성공")
            except Exception as e:
                print(f"⚠️  MediaPipe 초기화 실패: {e}")
                print("ℹ️  모델 파일을 확인하세요: models/blaze_face_short_range.tflite, models/gesture_recognizer.task")
        
        # 음성 인식 모델 초기화
        if SPEECHBRAIN_AVAILABLE:
            try:
                # 모델은 최초 실행 시 Hugging Face에서 다운로드됩니다
                from pathlib import Path
                original_symlink_to = Path.symlink_to

                def _patched_symlink_to(self, target, target_is_directory=False):
                    import shutil
                    target = Path(target)
                    self.parent.mkdir(parents=True, exist_ok=True)
                    if target.is_file():
                        shutil.copy2(target, self)
                    elif target.is_dir():
                        if self.exists():
                            shutil.rmtree(self)
                        shutil.copytree(target, self)

                Path.symlink_to = _patched_symlink_to

                try:
                    self.voice_encoder = EncoderClassifier.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb",
                        savedir=self.voice_model_path
                    )
                finally:
                    Path.symlink_to = original_symlink_to
                self.load_voice_data()
                print("✓ 음성 인식 모델 초기화 성공")
            except Exception as e:
                print(f"⚠️  음성 인식 모델 초기화 실패: {e}")
        
        
        # 얼굴인식 데이터 로드
        self.known_face_features = []
        self.known_face_names = []
        self.load_face_data()
        
        # UI 초기화
        self.init_ui()

        # 음성 인식 결과 시그널
        self.voice_attendance_result.connect(self.handle_voice_attendance_result)

        # 자동 음성 인식 타이머 (자동 모드에서 주기적 실행)
        self.voice_auto_timer = QTimer()
        self.voice_auto_timer.timeout.connect(self._start_voice_auto_recognition)
        self.voice_auto_interval_ms = int(os.getenv("VOICE_AUTO_INTERVAL_MS", "8000"))
        self.voice_auto_running = False

        # 실시간 음성 인식 타이머 (음성 모드에서 주기적 실행)
        self.voice_live_timer = QTimer()
        self.voice_live_timer.timeout.connect(self._start_voice_live_recognition)
        
        # 카메라 시작
        self.start_camera()
        
    def init_ui(self):
        """UI 초기화"""
        # 윈도우 설정
        self.setWindowTitle(WINDOW_TITLE)
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(f"background-color: {BG_COLOR};")
        
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃 (수평)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 좌측 사이드바 생성
        sidebar = self.create_sidebar()
        main_layout.addWidget(sidebar)
        
        # 중앙 + 우측 영역
        center_right_layout = QVBoxLayout()
        center_right_layout.setContentsMargins(20, 20, 20, 20)
        
        # 중앙(카메라) + 우측(정보) 영역
        cam_info_layout = QHBoxLayout()
        
        # 카메라 영역
        self.camera_widget = self.create_camera_view()
        cam_info_layout.addWidget(self.camera_widget)
        
        # 우측 정보 패널
        right_panel = self.create_right_panel()
        cam_info_layout.addWidget(right_panel)
        
        center_right_layout.addLayout(cam_info_layout)
        
        # 하단 제스처 가이드
        gesture_guide = self.create_gesture_guide()
        center_right_layout.addWidget(gesture_guide)
        
        # 하단 상태바
        status_bar = self.create_status_bar()
        center_right_layout.addWidget(status_bar)
        
        main_layout.addLayout(center_right_layout)
        
    def create_sidebar(self):
        """좌측 사이드바 생성"""
        sidebar = QWidget()
        sidebar.setFixedWidth(SIDEBAR_WIDTH)
        sidebar.setStyleSheet(f"background-color: {SIDEBAR_COLOR};")
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(SIDEBAR_PADDING, SIDEBAR_PADDING, SIDEBAR_PADDING, SIDEBAR_PADDING)
        layout.setSpacing(0)
        
        # 클라이언트 모드 라벨
        client_label = QLabel("📱 출석 체크")
        client_label.setFixedHeight(CLIENT_LABEL_HEIGHT)
        client_label.setAlignment(Qt.AlignCenter)
        client_label.setStyleSheet(f"""
            color: {TEXT_COLOR};
            font-size: {CLIENT_LABEL_FONT_SIZE}px;
            font-weight: {CLIENT_LABEL_FONT_WEIGHT};
            background-color: {ACCENT_COLOR};
            border-radius: 5px;
            padding: 10px;
        """)
        layout.addWidget(client_label)
        
        layout.addSpacing(LEFT_BUTTON_START_Y - CLIENT_LABEL_HEIGHT - SIDEBAR_PADDING)
        
        # 좌측 버튼들 생성
        self.left_buttons = {}
        for btn_config in LEFT_BUTTONS:
            btn = QPushButton(btn_config["text"])
            btn.setFixedSize(LEFT_BUTTON_WIDTH, LEFT_BUTTON_HEIGHT)
            btn.setStyleSheet(self.get_button_style())
            btn.setCursor(Qt.PointingHandCursor)
            
            # 버튼 이벤트 연결
            btn_name = btn_config["name"]
            btn.clicked.connect(lambda checked, name=btn_name: self.on_mode_button_click(name))
            
            self.left_buttons[btn_name] = btn
            layout.addWidget(btn)
            layout.addSpacing(LEFT_BUTTON_SPACING)
        
        layout.addStretch()
        
        return sidebar
    
    def create_camera_view(self):
        """카메라 뷰 + 출석 현황 플롯 영역 생성"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # 출석 현황 툴바 (플롯 위)
        self.attendance_toolbar = QWidget()
        toolbar_layout = QHBoxLayout(self.attendance_toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(10)

        self.today_attendance_btn = QPushButton("금일 출석 조회")
        self.today_attendance_btn.setFixedHeight(30)
        self.today_attendance_btn.setStyleSheet(self.get_button_style())
        self.today_attendance_btn.setCursor(Qt.PointingHandCursor)
        self.today_attendance_btn.clicked.connect(self._on_today_attendance_clicked)

        self.date_attendance_btn = QPushButton("날짜별 조회")
        self.date_attendance_btn.setFixedHeight(30)
        self.date_attendance_btn.setStyleSheet(self.get_button_style())
        self.date_attendance_btn.setCursor(Qt.PointingHandCursor)
        self.date_attendance_btn.clicked.connect(self._on_date_attendance_clicked)

        self.range_attendance_btn = QPushButton("기간 조회")
        self.range_attendance_btn.setFixedHeight(30)
        self.range_attendance_btn.setStyleSheet(self.get_button_style())
        self.range_attendance_btn.setCursor(Qt.PointingHandCursor)
        self.range_attendance_btn.clicked.connect(self._on_range_attendance_clicked)

        toolbar_layout.addWidget(self.today_attendance_btn)
        toolbar_layout.addWidget(self.date_attendance_btn)
        toolbar_layout.addWidget(self.range_attendance_btn)
        toolbar_layout.addStretch()

        layout.addWidget(self.attendance_toolbar)

        # 스택 영역 (카메라/플롯)
        self.camera_stack = QStackedLayout()
        layout.addLayout(self.camera_stack)

        # 카메라 라벨
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(CAM_WIDTH, CAM_HEIGHT)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet(f"""
            background-color: {CAM_BG_COLOR};
            border: 3px solid {ACCENT_COLOR};
            border-radius: 10px;
        """)
        self.camera_label.setText("📹 카메라 로딩 중...")
        self.camera_label.setFont(QFont("Arial", 14))
        self.camera_label.setStyleSheet(self.camera_label.styleSheet() + f"color: {TEXT_COLOR};")

        # 플롯 캔버스
        self.plot_container = QWidget()
        plot_layout = QVBoxLayout(self.plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)
        self.plot_figure = Figure(figsize=(6, 4), dpi=100)
        self.plot_canvas = FigureCanvas(self.plot_figure)
        self.plot_canvas.setFixedSize(CAM_WIDTH, CAM_HEIGHT)
        plot_layout.addWidget(self.plot_canvas)

        self.camera_stack.addWidget(self.camera_label)
        self.camera_stack.addWidget(self.plot_container)
        self.camera_stack.setCurrentWidget(self.camera_label)
        self.attendance_toolbar.setVisible(False)

        return container
    
    def create_right_panel(self):
        """우측 정보 패널 생성"""
        panel = QWidget()
        panel.setFixedWidth(RIGHT_PANEL_WIDTH)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)
        
        # 사용자 정보 프레임
        user_info_frame = QFrame()
        user_info_frame.setFixedHeight(USER_INFO_HEIGHT)
        user_info_frame.setStyleSheet(f"""
            background-color: {USER_INFO_BG_COLOR};
            border-radius: 8px;
            padding: 10px;
        """)
        
        user_info_layout = QVBoxLayout(user_info_frame)
        
        user_title = QLabel("👤 사용자 정보")
        user_title.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 14px; font-weight: bold;")
        user_info_layout.addWidget(user_title)
        
        self.user_name_label = QLabel("이름: -")
        self.user_name_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 12px;")
        user_info_layout.addWidget(self.user_name_label)
        
        self.user_id_label = QLabel("학번: -")
        self.user_id_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 12px;")
        user_info_layout.addWidget(self.user_id_label)
        
        user_info_layout.addStretch()
        
        layout.addWidget(user_info_frame)
        
        # 출석 상태 프레임
        status_frame = QFrame()
        status_frame.setFixedHeight(ATTENDANCE_STATUS_HEIGHT)
        status_frame.setStyleSheet(f"""
            background-color: {ATTENDANCE_STATUS_BG_COLOR};
            border-radius: 8px;
            padding: 10px;
        """)
        
        status_layout = QVBoxLayout(status_frame)
        
        status_title = QLabel("📊 출석 상태")
        status_title.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 14px; font-weight: bold;")
        status_layout.addWidget(status_title)
        
        self.attendance_status_label = QLabel("대기 중...")
        self.attendance_status_label.setStyleSheet(f"color: {WARNING_COLOR}; font-size: 16px; font-weight: bold;")
        self.attendance_status_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.attendance_status_label)

        self.check_type_container = QWidget()
        check_type_layout = QHBoxLayout(self.check_type_container)
        check_type_layout.setContentsMargins(0, 0, 0, 0)
        check_type_layout.setSpacing(8)

        self.check_in_btn = QPushButton("입실")
        self.check_out_btn = QPushButton("퇴실")
        for btn in (self.check_in_btn, self.check_out_btn):
            btn.setFixedHeight(26)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(self.get_button_style(font_size=11))

        self.check_in_btn.clicked.connect(lambda: self._set_check_type("in"))
        self.check_out_btn.clicked.connect(lambda: self._set_check_type("out"))

        check_type_layout.addWidget(self.check_in_btn)
        check_type_layout.addWidget(self.check_out_btn)
        status_layout.addWidget(self.check_type_container)
        self.check_type_container.setVisible(False)
        
        self.detected_gesture_label = QLabel("제스처: -")
        self.detected_gesture_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 12px;")
        status_layout.addWidget(self.detected_gesture_label)
        
        status_layout.addStretch()
        
        layout.addWidget(status_frame)
        
        layout.addStretch()
        
        return panel
    
    def create_gesture_guide(self):
        """하단 제스처 가이드 생성"""
        guide_frame = QFrame()
        guide_frame.setFixedHeight(GESTURE_GUIDE_HEIGHT)
        guide_frame.setStyleSheet(f"""
            background-color: {GESTURE_GUIDE_BG_COLOR};
            border-radius: 8px;
            padding: 10px;
        """)
        
        layout = QVBoxLayout(guide_frame)
        
        title = QLabel("👋 제스처 가이드")
        title.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 13px; font-weight: bold;")
        layout.addWidget(title)
        
        # 제스처 목록
        gestures_layout = QHBoxLayout()
        
        for gesture_name, gesture_info in GESTURES.items():
            gesture_widget = QLabel(f"{gesture_info['emoji']}\n{gesture_info['text']}")
            gesture_widget.setAlignment(Qt.AlignCenter)
            gesture_widget.setStyleSheet(f"""
                color: {gesture_info['color']};
                font-size: 11px;
                padding: 5px;
            """)
            gestures_layout.addWidget(gesture_widget)
        
        layout.addLayout(gestures_layout)
        
        return guide_frame
    
    def create_status_bar(self):
        """하단 상태바 생성"""
        status_bar = QLabel("✅ 준비 완료 - 모드를 선택하세요")
        status_bar.setFixedHeight(STATUS_BAR_HEIGHT)
        status_bar.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        status_bar.setStyleSheet(f"""
            background-color: {STATUS_BAR_BG_COLOR};
            color: {TEXT_COLOR};
            font-size: {STATUS_FONT_SIZE}px;
            padding-left: 15px;
            border-radius: 5px;
        """)
        self.status_bar = status_bar
        return status_bar
    
    def get_button_style(self, font_size=LEFT_BUTTON_FONT_SIZE):
        """버튼 스타일 반환"""
        return f"""
            QPushButton {{
                background-color: {BUTTON_COLOR};
                color: {TEXT_COLOR};
                border: none;
                border-radius: 5px;
                font-size: {font_size}px;
                font-weight: bold;
                padding: 5px;
            }}
            QPushButton:hover {{
                background-color: {BUTTON_HOVER_COLOR};
            }}
            QPushButton:pressed {{
                background-color: #1B1464;
            }}
        """
    
    def start_camera(self):
        """카메라 시작"""
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        
        if not self.camera.isOpened():
            self.update_status("❌ 카메라를 열 수 없습니다")
            return
        
        # 카메라 해상도 설정
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        
        # 타이머로 프레임 업데이트
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / CAMERA_FPS))
        
        self.update_status("📹 카메라 활성화됨")
    
    def load_voice_data(self):
        """저장된 음성(임베딩) 데이터 로드"""
        # Resolve path relative to this file to the repository-level data/voice
        voice_data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'voice', 'voice_embeddings.pkl'))

        if os.path.exists(voice_data_file):
            try:
                with open(voice_data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_voice_embeddings = data.get('embeddings', [])
                    self.known_voice_names = data.get('names', [])
                print(f"✓ {len(self.known_voice_names)}명의 음성 데이터 로드됨 ({voice_data_file})")
                if self.known_voice_embeddings:
                    first_vec = np.array(self.known_voice_embeddings[0]).flatten()
                    first_norm = float(np.linalg.norm(first_vec)) if first_vec.size > 0 else 0.0
                    print(f"🔬 등록 임베딩[0] norm: {first_norm:.6f}, shape: {first_vec.shape}")
            except Exception as e:
                print(f"⚠️  음성 데이터 로드 실패: {e}")
        else:
            print(f"ℹ️  등록된 음성 데이터가 없습니다. ({voice_data_file})")
    
    def save_voice_data(self):
        """음성 임베딩 저장"""
        voice_data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'voice', 'voice_embeddings.pkl'))
        os.makedirs(os.path.dirname(voice_data_file), exist_ok=True)

        data = {
            'embeddings': self.known_voice_embeddings,
            'names': self.known_voice_names
        }

        try:
            with open(voice_data_file, 'wb') as f:
                pickle.dump(data, f)
            print("✓ 음성 데이터 저장됨")
        except Exception as e:
            print(f"⚠️  음성 데이터 저장 실패: {e}")
    
    def extract_voice_embedding(self, audio_file):
        """음성 파일에서 임베딩 추출"""
        if not SPEECHBRAIN_AVAILABLE or self.voice_encoder is None:
            print("ℹ️  SpeechBrain이 준비되지 않았습니다.")
            return None
        
        try:
            import soundfile as sf
            from scipy.signal import resample

            audio, sr = sf.read(audio_file, dtype='float32')
            if audio.ndim > 1:
                audio = audio[:, 0]

            if sr != 16000:
                num_samples = int(len(audio) * 16000 / sr)
                audio = resample(audio, num_samples)

            signal = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            emb = self.voice_encoder.encode_batch(signal)
            return emb.detach().cpu().numpy()
        except Exception as e:
            print(f"⚠️  음성 임베딩 추출 실패: {e}")
            return None
    
    def recognize_voice(self, audio_file):
        """음성 파일 인식"""
        embedding = self.extract_voice_embedding(audio_file)
        
        if embedding is None:
            return "Unknown", 0.0

        embedding = np.array(embedding).flatten()
        emb_norm = float(np.linalg.norm(embedding)) if embedding.size > 0 else 0.0
        print(f"🔬 추출 임베딩 norm: {emb_norm:.6f}, shape: {embedding.shape}")
        if not np.isfinite(emb_norm) or emb_norm == 0.0:
            print("⚠️  추출 임베딩이 비정상입니다 (NaN/0).")
            return "Unknown", 0.0
        
        best_match_name = "Unknown"
        best_similarity = 0.0
        
        # 저장된 음성과 비교
        for known_emb, known_name in zip(self.known_voice_embeddings, self.known_voice_names):
            known_vec = np.array(known_emb).flatten()
            known_norm = float(np.linalg.norm(known_vec)) if known_vec.size > 0 else 0.0
            if not np.isfinite(known_norm) or known_norm == 0.0:
                print(f"⚠️  등록 임베딩 비정상: {known_name} (norm={known_norm})")
                continue

            similarity = self.cosine_similarity(embedding, known_vec)
            similarity = min(float(similarity) * self.voice_sensitivity_multiplier, 1.0)
            if not np.isfinite(similarity):
                print(f"⚠️  유사도 NaN: {known_name}")
                continue
            print(f"🔎 후보: {known_name} | sim={similarity:.6f} | norm={known_norm:.6f}")
            if similarity > best_similarity:
                best_similarity = float(similarity)
                best_match_name = known_name
        
        # 임계값 미달이면 이름만 Unknown 처리 (유사도는 유지)
        if best_similarity < self.voice_similarity_threshold:
            best_match_name = "Unknown"
        
        return best_match_name, float(best_similarity)

    def select_input_device(self, sd):
        """입력 마이크 선택 (VOICE_INPUT_DEVICE 환경변수 우선)"""
        try:
            preferred = os.getenv("VOICE_INPUT_DEVICE", "WO Mic").strip()
            devices = sd.query_devices()

            # 숫자 지정 (장치 인덱스)
            if preferred.isdigit():
                idx = int(preferred)
                if 0 <= idx < len(devices) and devices[idx]["max_input_channels"] > 0:
                    return idx, devices[idx]["name"]

            # 이름 부분 일치
            preferred_lower = preferred.lower()
            for i, device in enumerate(devices):
                if device.get("max_input_channels", 0) > 0:
                    if preferred_lower in device["name"].lower():
                        return i, device["name"]

            # 기본 입력 장치 fallback
            for i, device in enumerate(devices):
                if device.get("max_input_channels", 0) > 0:
                    return i, device["name"]
        except Exception as e:
            print(f"⚠️  입력 장치 선택 실패: {e}")

        return None, None

    def record_voice_and_recognize(self, duration=3, sample_rate=16000):
        """마이크에서 음성 녹음 후 인식"""
        return self.record_voice_and_recognize_internal(duration, sample_rate, ui_updates=True)

    def record_voice_and_recognize_internal(self, duration=3, sample_rate=16000, ui_updates=True):
        """마이크에서 음성 녹음 후 인식 (UI 업데이트 옵션)"""
        if not SPEECHBRAIN_AVAILABLE or self.voice_encoder is None:
            print("ℹ️  SpeechBrain이 준비되지 않았습니다.")
            return "Unknown", 0.0, "SpeechBrain이 준비되지 않았습니다."

        if not self.known_voice_embeddings:
            if ui_updates:
                self.update_status("⚠️  등록된 음성 데이터가 없습니다.")
            return "Unknown", 0.0, "등록된 음성 데이터가 없습니다."

        try:
            import sounddevice as sd
            import soundfile as sf

            temp_audio_file = "./tmp_voice_attendance.wav"

            # 입력 장치 선택 (환경변수 VOICE_INPUT_DEVICE 우선)
            device_id, device_name = self.select_input_device(sd)
            if device_id is None:
                self.update_status("❌ 마이크 입력 장치를 찾을 수 없습니다")
                return "Unknown", 0.0
            print(f"🎙️  사용 마이크: [{device_id}] {device_name}")

            # 녹음
            if ui_updates:
                self.update_status(f"🎤 음성 녹음 중... ({duration}초)")
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='float32',
                device=device_id
            )
            sd.wait()

            # 음성 에너지 확인
            rms = float(np.sqrt(np.mean(np.square(audio_data)))) if audio_data is not None else 0.0
            print(f"🔊 녹음 RMS: {rms:.6f}")
            if rms < self.voice_activity_threshold:
                if ui_updates:
                    self.update_status("🔇 음성 미감지")
                return "Unknown", 0.0, "no_voice"

            # 파일로 저장
            sf.write(temp_audio_file, audio_data, sample_rate)

            # 음성 인식 실행
            name, confidence = self.recognize_voice(temp_audio_file)

            # 임시 파일 삭제
            if os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)

            return name, confidence, ""
        except ImportError:
            if ui_updates:
                self.update_status("❌ 음성 녹음 라이브러리 없음")
                QMessageBox.warning(
                    self,
                    "라이브러리 오류",
                    "sounddevice 및 soundfile이 설치되어 있지 않습니다.\n설치 후 다시 시도하세요."
                )
            return "Unknown", 0.0, "sounddevice/soundfile 미설치"
        except Exception as e:
            if ui_updates:
                self.update_status(f"❌ 음성 녹음 오류: {str(e)}")
                QMessageBox.warning(
                    self,
                    "녹음 오류",
                    f"음성 녹음 중 오류가 발생했습니다:\n{str(e)}"
                )
            return "Unknown", 0.0, str(e)
    
    def record_voice_and_extract(self, filename="./tmp_voice.wav", duration=3, fs=16000):
        """음성 녹음 및 임베딩 추출 (placeholder)"""
        if not SPEECHBRAIN_AVAILABLE or self.voice_encoder is None:
            print("ℹ️  SpeechBrain이 준비되지 않았습니다.")
            return None
        try:
            import soundfile as sf
            from scipy.signal import resample

            audio, sr = sf.read(filename, dtype='float32')
            if audio.ndim > 1:
                audio = audio[:, 0]

            if sr != 16000:
                num_samples = int(len(audio) * 16000 / sr)
                audio = resample(audio, num_samples)

            signal = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            emb = self.voice_encoder.encode_batch(signal)
            return emb.detach().cpu().numpy()
        except Exception as e:
            print(f"⚠️  음성 임베딩 추출 실패: {e}")
            return None
    
    def process_voice_event(self, name, confidence):
        """감지된 음성 처리"""
        self.user_name_label.setText(f"이름: {name}")
        self.attendance_status_label.setText(f"음성 인식됨 ({confidence:.1%})")
        self.attendance_status_label.setStyleSheet(
            f"color: {SUCCESS_COLOR}; font-size: 16px; font-weight: bold;"
        )
        print(f"📤 음성 이벤트: {name} ({confidence:.1%})")
    
    def load_face_data(self):
        """저장된 얼굴 데이터 로드"""
        # Resolve path relative to this file to reach the repository-level data directory
        face_data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'face_data.pkl'))

        if os.path.exists(face_data_file):
            try:
                with open(face_data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_features = data.get('features', [])
                    self.known_face_names = data.get('names', [])
                print(f"✓ {len(self.known_face_names)}명의 얼굴 데이터 로드됨")
            except Exception as e:
                print(f"⚠️  얼굴 데이터 로드 실패: {e}")
        else:
            print(f"ℹ️  등록된 얼굴 데이터가 없습니다. (checked {face_data_file})")
    
    def extract_face_features(self, detection, image_width, image_height):
        """얼굴 감지 결과에서 특징 벡터 추출"""
        # 바운딩 박스 좌표를 특징으로 사용
        bbox = detection.bounding_box
        features = [
            bbox.origin_x / image_width,
            bbox.origin_y / image_height,
            bbox.width / image_width,
            bbox.height / image_height
        ]
        return np.array(features)
    
    def cosine_similarity(self, vec1, vec2):
        """코사인 유사도 계산"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def recognize_faces(self, frame):
        """프레임에서 얼굴 인식 (MediaPipe Task API 기반)"""
        if not self.face_detector:
            # MediaPipe가 없으면 OpenCV Haar Cascade 사용
            return self.recognize_faces_opencv(frame)
        
        # MediaPipe Image 객체 생성
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 얼굴 감지
        detection_result = self.face_detector.detect(mp_image)
        
        recognized_names = []
        h, w, _ = frame.shape
        
        if detection_result.detections:
            for detection in detection_result.detections:
                # 바운딩 박스 추출
                bbox = detection.bounding_box
                x_min = int(bbox.origin_x)
                y_min = int(bbox.origin_y)
                x_max = int(bbox.origin_x + bbox.width)
                y_max = int(bbox.origin_y + bbox.height)
                
                # 얼굴 특징 추출
                current_features = self.extract_face_features(detection, w, h)
                
                # 등록된 얼굴과 비교
                best_match_name = "Unknown"
                best_similarity = 0
                
                for known_features, known_name in zip(self.known_face_features, self.known_face_names):
                    similarity = self.cosine_similarity(current_features, known_features)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_name = known_name
                
                # 유사도 임계값 (설정값 이상이면 같은 사람)
                confidence = best_similarity * 100
                if (best_similarity * 100) < (self.face_similarity_threshold * 100):
                    best_match_name = "Unknown"
                    confidence = 0
                
                if best_match_name != "Unknown":
                    recognized_names.append((best_match_name, confidence))
                
                # 얼굴 박스 그리기
                color = (0, 255, 0) if best_match_name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                
                # 키포인트 그리기 (눈, 코, 입 등)
                for keypoint in detection.keypoints:
                    kp_x = int(keypoint.x * w)
                    kp_y = int(keypoint.y * h)
                    cv2.circle(frame, (kp_x, kp_y), 2, (0, 255, 255), -1)
                
                # 이름과 신뢰도 표시
                label_height = 40
                cv2.rectangle(frame, (x_min, y_max), (x_max, y_max + label_height), color, cv2.FILLED)
                
                if best_match_name != "Unknown":
                    cv2.putText(frame, best_match_name, (x_min + 6, y_max + 15), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, f"{confidence:.1f}%", (x_min + 6, y_max + 35), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                else:
                    cv2.putText(frame, "Unknown", (x_min + 6, y_max + 25), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame, recognized_names
    
    def recognize_faces_opencv(self, frame):
        """폴백: OpenCV Haar Cascade로 얼굴 감지"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Haar Cascade 불러오기
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        recognized_names = []
        
        for (x, y, w, h) in faces:
            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame, recognized_names
    
    def recognize_gesture(self, frame, skip_cooldown=False):
        """프레임에서 제스처 인식 (Quality Gate 포함)
        
        Args:
            frame: 입력 프레임
            skip_cooldown: True일 경우 쿨다운 무시 (자동 인식 모드용)
        """
        if not MEDIAPIPE_AVAILABLE or not self.gesture_recognizer:
            return [], frame
        
        try:
            # MediaPipe Image 객체 생성
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # 제스처 인식
            result = self.gesture_recognizer.recognize(mp_image)
            
            detected_gestures = []
            current_time = time.time()
            
            if hasattr(result, 'gestures') and result.gestures:
                for gesture_list in result.gestures:
                    if gesture_list:  # 제스처 리스트가 비어있지 않으면
                        gesture = gesture_list[0]  # 가장 높은 신뢰도의 제스처
                        gesture_name = gesture.category_name
                        confidence = gesture.score
                        
                        # Quality Gate: 신뢰도 임계값 확인
                        if confidence < self.gesture_confidence_threshold:
                            continue
                        
                        # Cooldown 확인: 같은 제스처가 최근에 감지되었는가?
                        # (자동 인식 모드에서는 cooldown 무시)
                        if not skip_cooldown:
                            if gesture_name in self.last_gesture_time:
                                if current_time - self.last_gesture_time[gesture_name] < self.gesture_cooldown:
                                    continue  # 쿨다운 중이면 무시
                        
                        # 유효한 제스처: 업데이트
                        self.last_gesture_time[gesture_name] = current_time
                        detected_gestures.append({
                            'type': gesture_name,
                            'confidence': confidence,
                            'timestamp': current_time
                        })
                        
                        print(f"✓ 제스처 인식: {gesture_name} ({confidence:.2f})")
            
            return detected_gestures, frame
            
        except Exception as e:
            print(f"⚠️  제스처 인식 오류: {e}")
            return [], frame
    
    def process_gesture_event(self, gesture_data):
        """감지된 제스처 처리"""
        gesture_type = gesture_data['type']
        confidence = gesture_data['confidence']
        
        self.detected_gesture_label.setText(f"제스처: {gesture_type} ({confidence:.1%})")
        
        # ZeroMQ로 서버에 전송 (필요시)
        print(f"📤 제스처 이벤트: {gesture_type} ({confidence:.1%})")
    
    def update_frame(self):
        """카메라 프레임 업데이트"""
        ret, frame = self.camera.read()
        
        if ret:
            self.current_frame = frame
            display_frame = frame.copy()
            
            # 얼굴 인식 모드
            if self.current_mode == "face_attendance":
                display_frame, recognized_names = self.recognize_faces(display_frame)
                
                if recognized_names:
                    name, confidence = recognized_names[0]
                    self.user_name_label.setText(f"이름: {name}")
                    self.attendance_status_label.setText(f"인식됨 ({confidence:.1f}%)")
                    self.attendance_status_label.setStyleSheet(
                        f"color: {SUCCESS_COLOR}; font-size: 16px; font-weight: bold;"
                    )
                    
                    if confidence > 80:
                        self.detected_gesture_label.setText(f"✓ {name} 출석 확인")
                else:
                    self.detected_gesture_label.setText("얼굴: 감지 안됨")
            
            # 제스처 인식 모드
            elif self.current_mode == "gesture_attendance":
                gestures, display_frame = self.recognize_gesture(display_frame)
                
                # 제스처 오버레이 표시
                overlay_y = 30
                if gestures:
                    for gesture in gestures:
                        self.process_gesture_event(gesture)
                        gesture_type = gesture['type']
                        confidence = gesture['confidence']
                        
                        gesture_text = f"Gesture: {gesture_type} ({confidence*100:.1f}%)"
                        cv2.putText(display_frame, gesture_text, (10, overlay_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        overlay_y += 30
                        
                        self.attendance_status_label.setText(f"제스처 감지됨!")
                        self.attendance_status_label.setStyleSheet(
                            f"color: {SUCCESS_COLOR}; font-size: 16px; font-weight: bold;"
                        )
                else:
                    cv2.putText(display_frame, "Gesture: Waiting...", (10, overlay_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
                    self.attendance_status_label.setText("제스처 대기 중...")
                    self.attendance_status_label.setStyleSheet(f"color: {WARNING_COLOR}; font-size: 16px; font-weight: bold;")
            
            # 자동 인식 모드 (얼굴 + 제스처 + 음성 동시 인식)
            elif self.current_mode == "automation":
                display_frame, face_results = self.recognize_faces(display_frame)
                # 자동 인식 모드에서는 제스처 쿨다운 무시
                gestures, display_frame = self.recognize_gesture(display_frame, skip_cooldown=True)
                
                # Decision Fusion Logic: 얼굴 + 제스처 종합 판단
                face_score = 0
                face_name = "Unknown"
                
                if face_results:
                    face_name, face_confidence = face_results[0]
                    face_score = face_confidence / 100.0
                
                gesture_score = 0
                gesture_detected = None

                if gestures:
                    gesture_detected = gestures[0]['type']
                    gesture_score = gestures[0]['confidence']
                    self.last_automation_gesture = (gesture_detected, gesture_score, time.time())
                elif self.last_automation_gesture:
                    gesture_detected, gesture_score, _ = self.last_automation_gesture
                
                # 통합 점수 계산 (얼굴 70%, 제스처 30%)
                fusion_score = (face_score * 0.7) + (gesture_score * 0.3)
                
                # 동시 오버레이: 카메라 프레임에 실시간 표시
                #h, w, _ = display_frame.shape
                
                # 1. 얼굴 정보 오버레이
                overlay_y = 30
                if face_name != "Unknown":
                    face_text = f"Face: {face_name} ({face_score*100:.1f}%)"
                    cv2.putText(display_frame, face_text, (10, overlay_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    overlay_y += 30
                else:
                    cv2.putText(display_frame, "Face: Detecting...", (10, overlay_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    overlay_y += 30
                
                # 2. 제스처 정보 오버레이
                if gesture_detected:
                    gesture_text = f"Gesture: {gesture_detected} ({gesture_score*100:.1f}%)"
                    cv2.putText(display_frame, gesture_text, (10, overlay_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    overlay_y += 30
                else:
                    cv2.putText(display_frame, "Gesture: Waiting...", (10, overlay_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
                    overlay_y += 30
                
                # 3. 음성 정보 오버레이 (최근 5초 이내 결과 표시)
                current_time = time.time()
                voice_name = "Unknown"
                voice_score = 0.0
                voice_score_for_fusion = 0.0
                
                if self.last_voice_result and (current_time - self.voice_result_time) < 5.0:
                    voice_name, voice_score = self.last_voice_result
                    voice_score_for_fusion = voice_score
                    voice_text = f"Voice: {voice_name} ({voice_score:.2f})"
                    cv2.putText(display_frame, voice_text, (10, overlay_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "Voice: Listening...", (10, overlay_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
                overlay_y += 30
                
                # 4. 종합 점수 오버레이 (얼굴 50%, 제스처 25%, 음성 25%)
                fusion_score = (face_score * 0.5) + (gesture_score * 0.25) + (voice_score_for_fusion * 0.25)
                fusion_text = f"Fusion Score: {fusion_score*100:.1f}%"
                cv2.putText(display_frame, fusion_text, (10, overlay_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # 자동 인식 기준: 얼굴 인식률 > 70% 또는 (얼굴 > 50% AND 제스처 감지)
                if fusion_score >= 0.90:
                    display_name = face_name if face_name != "Unknown" else voice_name
                    if display_name == "Unknown":
                        display_name = "사용자"
                    self.user_name_label.setText(f"이름: {display_name}")
                    self.attendance_status_label.setText(f"✓ {display_name} 출석 완료")
                    self.attendance_status_label.setStyleSheet(
                        f"color: {SUCCESS_COLOR}; font-size: 16px; font-weight: bold;"
                    )
                    status_msg = f"Fusion Score: {fusion_score*100:.1f}%"
                    if gesture_detected:
                        status_msg += f" + 제스처: {gesture_detected}"
                    if voice_score_for_fusion > 0.0:
                        status_msg += f" + 음성: {voice_score_for_fusion*100:.1f}%"
                    self.detected_gesture_label.setText(status_msg)
                    self._mark_attendance_if_needed(display_name, fusion_score)
                elif face_name != "Unknown" and face_score > 0.7:
                    # 얼굴 인식 성공
                    self.user_name_label.setText(f"이름: {face_name}")
                    self.attendance_status_label.setText(f"✓ Fusion Score ({fusion_score*100:.1f}%)")
                    self.attendance_status_label.setStyleSheet(
                        f"color: {SUCCESS_COLOR}; font-size: 16px; font-weight: bold;"
                    )
                    
                    status_msg = f"✓ {face_name} - 얼굴: {face_score*100:.1f}%"
                    if gesture_detected:
                        status_msg += f" + 제스처: {gesture_detected}"
                    self.detected_gesture_label.setText(status_msg)
                    
                elif face_name != "Unknown" and face_score > 0.5 and gesture_detected:
                    # 얼굴 + 제스처 조합으로 인식
                    self.user_name_label.setText(f"이름: {face_name}")
                    self.attendance_status_label.setText(f"✓ 다중 모달 인식 ({fusion_score*100:.1f}%)")
                    self.attendance_status_label.setStyleSheet(
                        f"color: {SUCCESS_COLOR}; font-size: 16px; font-weight: bold;"
                    )
                    self.detected_gesture_label.setText(f"얼굴: {face_score*100:.1f}% + 제스처: {gesture_detected}")
                    if fusion_score >= 0.80:
                        self._mark_attendance_if_needed(face_name, fusion_score)
                else:
                    # 대기
                    self.attendance_status_label.setText("자동 인식 중...")
                    self.attendance_status_label.setStyleSheet(f"color: {ACCENT_COLOR}; font-size: 16px; font-weight: bold;")
                    
                    status_msg = ""
                    if face_name != "Unknown":
                        status_msg = f"얼굴: {face_score*100:.1f}%"
                    if gesture_detected:
                        if status_msg:
                            status_msg += f" + 제스처: {gesture_detected}"
                        else:
                            status_msg = f"제스처: {gesture_detected}"
                    
                    if voice_score_for_fusion > 0.0:
                        status_msg = f"{status_msg} + 음성: {voice_score_for_fusion*100:.1f}%" if status_msg else f"음성: {voice_score_for_fusion*100:.1f}%"

                    if status_msg:
                        self.detected_gesture_label.setText(status_msg)
                    else:
                        self.detected_gesture_label.setText("대기 중...")
            
            # 모드 표시
            if self.current_mode:
                mode_text = {
                    "gesture_attendance": "제스처 출석 모드",
                    "face_attendance": "얼굴 인식 모드",
                    "automation": "🧠 자동 인식 모드 (얼굴+제스처+음성)",
                }
                cv2.putText(display_frame, mode_text.get(self.current_mode, ""), 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # BGR을 RGB로 변환
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # QImage로 변환
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # QLabel에 표시
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(CAM_WIDTH, CAM_HEIGHT, Qt.KeepAspectRatio)
            self.camera_label.setPixmap(scaled_pixmap)
    
    def on_mode_button_click(self, mode_name):
        """모드 버튼 클릭 이벤트"""
        self.current_mode = mode_name
        if self.voice_auto_timer.isActive() and mode_name != "automation":
            self.voice_auto_timer.stop()
        if self.voice_live_timer.isActive() and mode_name != "voice_attendance":
            self.voice_live_timer.stop()
        
        if mode_name == "gesture_attendance":
            self._show_camera_view()
            self.update_status("✋ 제스처 출석 모드 활성화")
            self.attendance_status_label.setText("제스처 대기 중...")
            self.attendance_status_label.setStyleSheet(f"color: {WARNING_COLOR}; font-size: 16px; font-weight: bold;")
            
        elif mode_name == "face_attendance":
            self._show_camera_view()
            self.update_status("😊 얼굴 인식 출석 모드 활성화")
            self.attendance_status_label.setText("얼굴 인식 중...")
            self.attendance_status_label.setStyleSheet(f"color: {ACCENT_COLOR}; font-size: 16px; font-weight: bold;")
            
        elif mode_name == "voice_attendance":
            self._show_camera_view()
            self.update_status("🎤 음성 인식 출석 모드 활성화")
            self.attendance_status_label.setText("음성 녹음 대기 중...")
            self.attendance_status_label.setStyleSheet(f"color: {WARNING_COLOR}; font-size: 16px; font-weight: bold;")
            # 실시간 음성 인식 시작 (짧은 구간 반복)
            self.voice_live_timer.start(self.voice_live_interval_ms)
            
        elif mode_name == "automation":
            self._show_camera_view()
            self.update_status("🧠 자동 인식 모드 활성화 (얼굴+제스처+음성)")
            self.attendance_status_label.setText("자동 인식 중...")
            self.attendance_status_label.setStyleSheet(f"color: {ACCENT_COLOR}; font-size: 16px; font-weight: bold;")
            self._set_check_type("in")
            if hasattr(self, "check_type_container"):
                self.check_type_container.setVisible(True)
            # 자동 음성 인식 시작
            self.voice_auto_timer.start(self.voice_auto_interval_ms)
            
        elif mode_name == "attendance_status":
            self.update_status("📊 출석 현황 조회")
            self._show_plot_view()
            self.show_attendance_status_plot(date=datetime.now().date())
        else:
            if hasattr(self, "check_type_container"):
                self.check_type_container.setVisible(False)
    
    def update_status(self, message):
        """상태바 업데이트"""
        self.status_bar.setText(message)

    def _show_camera_view(self):
        if hasattr(self, "camera_stack"):
            self.camera_stack.setCurrentWidget(self.camera_label)
        if hasattr(self, "attendance_toolbar"):
            self.attendance_toolbar.setVisible(False)
        if hasattr(self, "check_type_container"):
            self.check_type_container.setVisible(False)

    def _show_plot_view(self):
        if hasattr(self, "camera_stack"):
            self.camera_stack.setCurrentWidget(self.plot_container)
        if hasattr(self, "attendance_toolbar"):
            self.attendance_toolbar.setVisible(True)
        if hasattr(self, "check_type_container"):
            self.check_type_container.setVisible(False)

    def _set_check_type(self, mode):
        self.attendance_check_type = mode
        if mode == "in":
            self.check_in_btn.setStyleSheet(self.get_button_style(font_size=11) + f"border: 2px solid {SUCCESS_COLOR};")
            self.check_out_btn.setStyleSheet(self.get_button_style(font_size=11))
        else:
            self.check_out_btn.setStyleSheet(self.get_button_style(font_size=11) + f"border: 2px solid {WARNING_COLOR};")
            self.check_in_btn.setStyleSheet(self.get_button_style(font_size=11))

    def _on_today_attendance_clicked(self):
        self.show_attendance_status_plot(date=datetime.now().date())

    def _on_date_attendance_clicked(self):
        date_str, ok = QInputDialog.getText(self, "날짜별 조회", "날짜를 입력하세요 (YYYY-MM-DD):")
        if not ok or not date_str.strip():
            return
        try:
            selected_date = datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
        except ValueError:
            QMessageBox.warning(self, "출석 현황", "날짜 형식이 올바르지 않습니다.")
            return

        student_name, ok = QInputDialog.getText(self, "학생 지정", "학생 이름을 입력하세요:")
        if not ok or not student_name.strip():
            QMessageBox.warning(self, "출석 현황", "학생 이름을 입력해야 합니다.")
            return

        self.show_attendance_status_plot(date=selected_date, student_name=student_name.strip())

    def _on_range_attendance_clicked(self):
        start_str, ok = QInputDialog.getText(self, "기간 조회", "시작 날짜를 입력하세요 (YYYY-MM-DD):")
        if not ok or not start_str.strip():
            return
        end_str, ok = QInputDialog.getText(self, "기간 조회", "종료 날짜를 입력하세요 (YYYY-MM-DD):")
        if not ok or not end_str.strip():
            return

        try:
            start_date = datetime.strptime(start_str.strip(), "%Y-%m-%d").date()
            end_date = datetime.strptime(end_str.strip(), "%Y-%m-%d").date()
        except ValueError:
            QMessageBox.warning(self, "출석 현황", "날짜 형식이 올바르지 않습니다.")
            return

        if end_date < start_date:
            QMessageBox.warning(self, "출석 현황", "종료 날짜는 시작 날짜보다 빠를 수 없습니다.")
            return

        student_name, ok = QInputDialog.getText(self, "학생 지정", "학생 이름을 입력하세요 (선택):")
        if not ok:
            return

        student_name = student_name.strip() if student_name else None
        self.show_attendance_status_plot(start_date=start_date, end_date=end_date, student_name=student_name)

    def _get_attendance_db_config(self):
        return {
            "host": os.getenv("DB_HOST", "192.168.0.41"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "dbname": os.getenv("DB_NAME", "devserver"),
            "user": os.getenv("DB_USER", "orugu"),
            "password": os.getenv("DB_PASSWORD", "orugu#0916"),
        }

    def _fetch_attendance_records(self, limit=200, date_filter=None, start_date=None, end_date=None, employee_no=None, student_name=None):
        cfg = self._get_attendance_db_config()
        records = []
        sql = (
            'SELECT id, employee_no, name, email, created_at, check_in_time, check_out_time '
            'FROM "UserData"."userdata"'
        )
        conditions = []
        params = []

        if date_filter:
            conditions.append('DATE(check_in_time) = %s')
            params.append(date_filter)
        if start_date and end_date:
            conditions.append('DATE(check_in_time) BETWEEN %s AND %s')
            params.extend([start_date, end_date])
        if employee_no:
            conditions.append('employee_no = %s')
            params.append(employee_no)
        if student_name:
            conditions.append('name = %s')
            params.append(student_name)

        if conditions:
            sql += ' WHERE ' + ' AND '.join(conditions)

        sql += ' ORDER BY id DESC LIMIT %s'
        params.append(limit)

        encoding_primary = os.getenv("ATTENDANCE_DB_ENCODING", "UTF8")
        encoding_fallback = os.getenv("ATTENDANCE_DB_ENCODING_FALLBACK", "EUC_KR")
        encoding_last_resort = os.getenv("ATTENDANCE_DB_ENCODING_LAST", "LATIN1")

        def _run_query(client_encoding: str):
            os.environ["PGCLIENTENCODING"] = client_encoding
            import psycopg2
            with psycopg2.connect(
                host=cfg["host"],
                port=cfg["port"],
                dbname=cfg["dbname"],
                user=cfg["user"],
                password=cfg["password"],
                connect_timeout=5,
                options=f"-c client_encoding={client_encoding} -c lc_messages=C"
            ) as conn:
                conn.set_client_encoding(client_encoding)
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    return cur.fetchall()

        try:
            rows = _run_query(encoding_primary)
        except UnicodeDecodeError:
            try:
                rows = _run_query(encoding_fallback)
            except UnicodeDecodeError:
                rows = _run_query(encoding_last_resort)

        for row in rows:
            records.append({
                "id": row[0],
                "employee_no": row[1],
                "name": row[2],
                "email": row[3],
                "created_at": row[4],
                "check_in_time": row[5],
                "check_out_time": row[6],
            })
        return records

    def _ensure_userdata_table_schema(self, conn):
        """userdata 테이블 스키마 확인 및 자동 생성/수정"""
        try:
            with conn.cursor() as cur:
                # 1. 스키마 존재 확인
                cur.execute("""
                    SELECT schema_name FROM information_schema.schemata 
                    WHERE schema_name = 'UserData'
                """)
                if not cur.fetchone():
                    print("✓ UserData 스키마 생성 중...")
                    cur.execute('CREATE SCHEMA "UserData"')
                    conn.commit()

                # 2. 테이블 존재 확인
                cur.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'UserData' AND table_name = 'userdata'
                """)
                table_exists = cur.fetchone() is not None

                if not table_exists:
                    print("✓ userdata 테이블 생성 중...")
                    cur.execute("""
                        CREATE TABLE "UserData"."userdata" (
                            id SERIAL PRIMARY KEY,
                            employee_no VARCHAR NOT NULL,
                            name VARCHAR NOT NULL,
                            email VARCHAR NOT NULL,
                            created_at TIMESTAMP NOT NULL,
                            check_in_time TIMESTAMP,
                            check_out_time TIMESTAMP
                        )
                    """)
                    conn.commit()
                    print("✓ userdata 테이블 생성 완료")
                    return

                # 3. 필수 컬럼 확인 및 추가
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'UserData' AND table_name = 'userdata'
                """)
                existing_columns = {row[0] for row in cur.fetchall()}

                required_columns = {
                    'id': 'SERIAL PRIMARY KEY',
                    'employee_no': 'VARCHAR NOT NULL',
                    'name': 'VARCHAR NOT NULL',
                    'email': 'VARCHAR NOT NULL',
                    'created_at': 'TIMESTAMP NOT NULL',
                    'check_in_time': 'TIMESTAMP',
                    'check_out_time': 'TIMESTAMP'
                }

                for col_name, col_type in required_columns.items():
                    if col_name not in existing_columns:
                        print(f"✓ 컬럼 추가 중: {col_name}")
                        if col_name == 'id':
                            # id 컬럼의 경우 특별 처리
                            cur.execute(f'''
                                ALTER TABLE "UserData"."userdata" 
                                ADD COLUMN {col_name} SERIAL PRIMARY KEY
                            ''')
                        else:
                            # 다른 컬럼들
                            if 'NOT NULL' in col_type:
                                # NOT NULL 컬럼은 기본값 제공
                                default_val = "''" if 'VARCHAR' in col_type else 'NOW()'
                                cur.execute(f'''
                                    ALTER TABLE "UserData"."userdata" 
                                    ADD COLUMN {col_name} {col_type.replace('NOT NULL', '')} DEFAULT {default_val}
                                ''')
                                cur.execute(f'''
                                    ALTER TABLE "UserData"."userdata" 
                                    ALTER COLUMN {col_name} SET NOT NULL
                                ''')
                            else:
                                cur.execute(f'''
                                    ALTER TABLE "UserData"."userdata" 
                                    ADD COLUMN {col_name} {col_type}
                                ''')
                        conn.commit()
                        print(f"✓ 컬럼 추가 완료: {col_name}")

        except Exception as e:
            print(f"⚠️  테이블 스키마 확인 중 오류: {e}")
            conn.rollback()

    def _mark_attendance_if_needed(self, name, fusion_score):
        if not name or name == "Unknown":
            return
        now = datetime.now()
        cfg = self._get_attendance_db_config()

        encoding_primary = os.getenv("ATTENDANCE_DB_ENCODING", "UTF8")
        encoding_fallback = os.getenv("ATTENDANCE_DB_ENCODING_FALLBACK", "EUC_KR")
        encoding_last_resort = os.getenv("ATTENDANCE_DB_ENCODING_LAST", "LATIN1")

        def _run(client_encoding: str):
            os.environ["PGCLIENTENCODING"] = client_encoding
            import psycopg2
            with psycopg2.connect(
                host=cfg["host"],
                port=cfg["port"],
                dbname=cfg["dbname"],
                user=cfg["user"],
                password=cfg["password"],
                connect_timeout=5,
                options=f"-c client_encoding={client_encoding} -c lc_messages=C"
            ) as conn:
                conn.set_client_encoding(client_encoding)
                
                # 테이블 스키마 확인 및 자동 생성
                self._ensure_userdata_table_schema(conn)
                
                with conn.cursor() as cur:
                    cur.execute(
                        'SELECT user_id, email FROM "UserData".userbasicdata WHERE name = %s ORDER BY user_id LIMIT 1',
                        (name,)
                    )
                    row = cur.fetchone()
                    employee_no = row[0] if row else "UNKNOWN"
                    email = row[1] if row else "unknown@example.com"

                    cur.execute(
                        'SELECT id FROM "UserData"."userdata" WHERE employee_no = %s AND DATE(check_in_time) = %s LIMIT 1',
                        (employee_no, now.date())
                    )
                    existing = cur.fetchone()

                    if self.attendance_check_type == "out":
                        if existing:
                            cur.execute(
                                'UPDATE "UserData"."userdata" SET check_out_time = %s WHERE id = %s',
                                (now, existing[0])
                            )
                        else:
                            cur.execute(
                                'INSERT INTO "UserData"."userdata" (employee_no, name, email, created_at, check_in_time, check_out_time) '
                                'VALUES (%s, %s, %s, %s, %s, %s)',
                                (employee_no, name, email, now, now, now)
                            )
                        return

                    if existing:
                        return

                    cur.execute(
                        'INSERT INTO "UserData"."userdata" (employee_no, name, email, created_at, check_in_time) '
                        'VALUES (%s, %s, %s, %s, %s)',
                        (employee_no, name, email, now, now)
                    )
                conn.commit()

        try:
            _run(encoding_primary)
        except UnicodeDecodeError:
            try:
                _run(encoding_fallback)
            except UnicodeDecodeError:
                _run(encoding_last_resort)

    def show_attendance_status_plot(self, date=None, start_date=None, end_date=None, employee_no=None, student_name=None):
        # Matplotlib 한글 폰트 설정 (Windows 기본: 맑은 고딕)
        try:
            malgun = "C:\\Windows\\Fonts\\malgun.ttf"
            if os.path.exists(malgun):
                font_manager.fontManager.addfont(malgun)
                rcParams["font.family"] = "Malgun Gothic"
            else:
                for name in ["NanumGothic", "AppleGothic", "Malgun Gothic"]:
                    if any(f.name == name for f in font_manager.fontManager.ttflist):
                        rcParams["font.family"] = name
                        break
            rcParams["axes.unicode_minus"] = False
        except Exception:
            pass

        try:
            records = self._fetch_attendance_records(
                limit=200,
                date_filter=date,
                start_date=start_date,
                end_date=end_date,
                employee_no=employee_no,
                student_name=student_name
            )
        except UnicodeDecodeError:
            cfg = self._get_attendance_db_config()
            QMessageBox.warning(
                self,
                "출석 현황",
                "DB 연결 중 인코딩 오류가 발생했습니다.\n"
                "DB_USER/DB_PASSWORD 또는 DB_HOST/DB_NAME이 다른 서버와 맞지 않을 수 있습니다.\n\n"
                f"현재 설정: {cfg['host']}:{cfg['port']} / {cfg['dbname']} / {cfg['user']}"
            )
            return
        except Exception as e:
            QMessageBox.warning(self, "출석 현황", f"DB 조회 실패: {e}")
            return

        if not records:
            self.plot_figure.clear()
            ax = self.plot_figure.add_subplot(111)
            ax.axis('off')
            ax.text(0.5, 0.5, "자료가 없어요 ㅠ.ㅠ", ha='center', va='center', fontsize=14)
            self.plot_canvas.draw()
            return

        records = list(reversed(records))
        labels = [f"{r['employee_no']} {r['name']}" for r in records]

        def dt_to_minutes(dt_value):
            if not isinstance(dt_value, datetime):
                return None
            return dt_value.hour * 60 + dt_value.minute + dt_value.second / 60.0

        check_in = [dt_to_minutes(r["check_in_time"]) for r in records]
        check_out = [dt_to_minutes(r["check_out_time"]) for r in records]

        x = list(range(len(labels)))

        self.plot_figure.clear()
        ax = self.plot_figure.add_subplot(111)
        ax.plot(x, check_in, marker='o', label='Check-in')
        ax.plot(x, check_out, marker='o', label='Check-out')

        title = "출석 현황 (Check-in/Check-out)"
        if start_date and end_date:
            if employee_no or student_name:
                target = employee_no or student_name
                title = f"출석 현황 - {target} ({start_date}~{end_date})"
            else:
                title = f"기간 출석 현황 ({start_date}~{end_date})"
        elif date and (employee_no or student_name):
            target = employee_no or student_name
            title = f"출석 현황 - {target} ({date})"
        elif date:
            title = f"금일 출석 현황 ({date})"
        ax.set_title(title)
        ax.set_xlabel("사용자")
        ax.set_ylabel("시간")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.legend()

        def minutes_to_hhmm(value, _):
            if value is None:
                return ""
            hours = int(value // 60)
            minutes = int(value % 60)
            return f"{hours:02d}:{minutes:02d}"

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(minutes_to_hhmm))
        self.plot_figure.tight_layout()
        self.plot_canvas.draw()
    
    def keyPressEvent(self, event):
        """키보드 이벤트 핸들러"""
        super().keyPressEvent(event)

    def _start_voice_auto_recognition(self):
        """자동 모드에서 주기적 음성 인식 실행"""
        if self.current_mode != "automation":
            if self.voice_auto_timer.isActive():
                self.voice_auto_timer.stop()
            return

        if self.voice_auto_running:
            return

        if not SPEECHBRAIN_AVAILABLE or self.voice_encoder is None:
            self.update_status("⚠️  음성 인식 모델이 준비되지 않았습니다")
            return

        self.voice_auto_running = True
        threading.Thread(target=self._voice_auto_worker, daemon=True).start()

    def _start_voice_live_recognition(self):
        """음성 모드에서 실시간 음성 인식 실행"""
        if self.current_mode != "voice_attendance":
            if self.voice_live_timer.isActive():
                self.voice_live_timer.stop()
            return

        if self.voice_live_running:
            return

        if not SPEECHBRAIN_AVAILABLE or self.voice_encoder is None:
            self.update_status("⚠️  음성 인식 모델이 준비되지 않았습니다")
            return

        self.voice_live_running = True
        threading.Thread(target=self._voice_live_worker, daemon=True).start()

    def _voice_auto_worker(self):
        try:
            name, confidence, error = self.record_voice_and_recognize_internal(
                duration=3,
                sample_rate=16000,
                ui_updates=False
            )
            if error:
                return

            if name != "Unknown":
                self.last_voice_result = (name, confidence)
                self.voice_result_time = time.time()
        finally:
            self.voice_auto_running = False

    def _voice_live_worker(self):
        try:
            name, confidence, error = self.record_voice_and_recognize_internal(
                duration=self.voice_live_duration,
                sample_rate=16000,
                ui_updates=False
            )
            if error == "no_voice":
                return
            self.voice_attendance_result.emit(name, confidence, error)
        finally:
            self.voice_live_running = False

    def _voice_attendance_worker(self):
        name, confidence, error = self.record_voice_and_recognize_internal(
            duration=3,
            sample_rate=16000,
            ui_updates=False
        )
        self.voice_attendance_result.emit(name, confidence, error)

    def handle_voice_attendance_result(self, name, confidence, error):
        threshold = self.voice_similarity_threshold

        if error == "no_voice":
            return

        if error:
            self.attendance_status_label.setText("음성 인식 실패")
            self.attendance_status_label.setStyleSheet(
                f"color: {WARNING_COLOR}; font-size: 16px; font-weight: bold;"
            )
            self.update_status(f"❌ 음성 인식 실패: {error}")
            return

        print(f"🔎 음성 인식 결과: {name} (유사도: {confidence:.3f}, 임계값: {threshold})")
        print(f"🔎 등록된 음성 수: {len(self.known_voice_names)}")

        if name != "Unknown" and confidence >= threshold:
            self.user_name_label.setText(f"이름: {name}")
            self.attendance_status_label.setText(f"음성 인식됨 ({confidence:.2f})")
            self.attendance_status_label.setStyleSheet(
                f"color: {SUCCESS_COLOR}; font-size: 16px; font-weight: bold;"
            )
            self.update_status(f"✅ {name} 음성 인식 성공")
            self.process_voice_event(name, confidence)

            QMessageBox.information(
                self,
                "음성출석 완료",
                "음성출석 완료!"
            )
        else:
            self.attendance_status_label.setText("음성 인식 실패")
            self.attendance_status_label.setStyleSheet(
                f"color: {WARNING_COLOR}; font-size: 16px; font-weight: bold;"
            )
            self.update_status(f"❌ 음성 인식 실패 (유사도: {confidence:.2f}, 임계값: {threshold})")
    
    def closeEvent(self, event):
        """윈도우 종료 이벤트"""
        if self.camera is not None:
            self.camera.release()
        if MEDIAPIPE_AVAILABLE:
            if hasattr(self, 'face_detector') and self.face_detector:
                self.face_detector.close()
            if hasattr(self, 'gesture_recognizer') and self.gesture_recognizer:
                self.gesture_recognizer.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    # 다크 테마 적용
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 39, 46))
    palette.setColor(QPalette.WindowText, Qt.white)
    app.setPalette(palette)
    
    window = ClientUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
