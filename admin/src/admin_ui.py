"""
관리자 UI - 출결관리 시스템
사진, 목소리, 제스처 등록 기능
"""

import sys
import cv2
import os
import pickle
import numpy as np
import psycopg2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QLineEdit, QMessageBox, QInputDialog, QFileDialog,
    QDialog, QComboBox, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor

# 음성 인식 서비스 import
from lib.voice_service import VoiceService

# 제스처 인식 서비스 import
from lib.gesture_service import GestureService

# MediaPipe import
MEDIAPIPE_AVAILABLE = False
USE_TASK_API = False
try:
    import mediapipe as mp
    try:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from mediapipe import Image as MPImage
        MEDIAPIPE_AVAILABLE = True
        USE_TASK_API = True
        print("✓ MediaPipe Task API 사용 가능")
    except:
        USE_TASK_API = False
        MEDIAPIPE_AVAILABLE = False
        print("ℹ️  MediaPipe Task API를 사용할 수 없습니다. OpenCV로 대체합니다.")
except ImportError:
    print("⚠️  MediaPipe가 설치되지 않았습니다.")

# UI 설정 임포트
from lib.ui_config_lib import *


class AdminUI(QMainWindow):
    """관리자 모드 메인 윈도우"""
    
    def __init__(self):
        super().__init__()
        
        # 카메라 초기화
        self.camera = None
        self.current_frame = None
        self.captured_frame = None
        self.current_name = ""
        
        # 얼굴 감지기 초기화
        self.face_detector = None
        
        # 음성 서비스 초기화
        self.voice_service = VoiceService()
        self.voice_service.load_voice_data()
        
        # 제스처 서비스 초기화
        self.gesture_service = GestureService()
        
        if MEDIAPIPE_AVAILABLE and USE_TASK_API:
            try:
                base_options_face = python.BaseOptions(model_asset_path='admin/models/blaze_face_short_range.tflite')
                face_options = vision.FaceDetectorOptions(base_options=base_options_face)
                self.face_detector = vision.FaceDetector.create_from_options(face_options)
                print("✓ MediaPipe 얼굴 감지기 초기화 성공")
            except Exception as e:
                print(f"⚠️  MediaPipe 초기화 실패: {e}")
                print("ℹ️  모델 파일을 확인하세요: models/blaze_face_short_range.tflite")
        
        # 얼굴 데이터 로드
        self.known_face_features = []
        self.known_face_names = []
        self.load_face_data()
        
        # UI 초기화
        self.init_ui()
        
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
        
        # 중앙(카메라) + 우측(버튼) 영역
        cam_control_layout = QHBoxLayout()
        
        # 카메라 영역
        self.camera_label = self.create_camera_view()
        cam_control_layout.addWidget(self.camera_label)
        
        # 우측 컨트롤 영역
        right_panel = self.create_right_panel()
        cam_control_layout.addWidget(right_panel)
        
        center_right_layout.addLayout(cam_control_layout)
        
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
        
        # 관리자 모드 라벨
        admin_label = QLabel("🔐 관리자 모드")
        admin_label.setFixedHeight(ADMIN_LABEL_HEIGHT)
        admin_label.setAlignment(Qt.AlignCenter)
        admin_label.setStyleSheet(f"""
            color: {TEXT_COLOR};
            font-size: {ADMIN_LABEL_FONT_SIZE}px;
            font-weight: {ADMIN_LABEL_FONT_WEIGHT};
            background-color: {ACCENT_COLOR};
            border-radius: 5px;
            padding: 10px;
        """)
        layout.addWidget(admin_label)
        
        layout.addSpacing(LEFT_BUTTON_START_Y - ADMIN_LABEL_HEIGHT - SIDEBAR_PADDING)
        
        # 좌측 버튼들 생성
        self.left_buttons = {}
        for btn_config in LEFT_BUTTONS:
            btn = QPushButton(btn_config["text"])
            btn.setFixedSize(LEFT_BUTTON_WIDTH, LEFT_BUTTON_HEIGHT)
            btn.setStyleSheet(self.get_button_style())
            btn.setCursor(Qt.PointingHandCursor)
            
            # 버튼 이벤트 연결
            btn_name = btn_config["name"]
            btn.clicked.connect(lambda checked, name=btn_name: self.on_left_button_click(name))
            
            self.left_buttons[btn_name] = btn
            layout.addWidget(btn)
            layout.addSpacing(LEFT_BUTTON_SPACING)
        
        layout.addStretch()
        
        return sidebar
    
    def create_camera_view(self):
        """카메라 뷰 생성"""
        camera_label = QLabel()
        camera_label.setFixedSize(CAM_WIDTH, CAM_HEIGHT)
        camera_label.setAlignment(Qt.AlignCenter)
        camera_label.setStyleSheet(f"""
            background-color: {CAM_BG_COLOR};
            border: 2px solid {BUTTON_COLOR};
            border-radius: 5px;
        """)
        camera_label.setText("📹 카메라 로딩 중...")
        camera_label.setFont(QFont("Arial", 14))
        camera_label.setStyleSheet(camera_label.styleSheet() + f"color: {TEXT_COLOR};")
        
        return camera_label
    
    def create_right_panel(self):
        """우측 컨트롤 패널 생성"""
        panel = QWidget()
        panel.setFixedWidth(RIGHT_BUTTON_WIDTH + 20)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(RIGHT_BUTTON_SPACING)
        
        # 우측 버튼들 생성
        self.right_buttons = {}
        for btn_config in RIGHT_BUTTONS:
            btn = QPushButton(btn_config["text"])
            btn.setFixedSize(RIGHT_BUTTON_WIDTH, RIGHT_BUTTON_HEIGHT)
            btn.setStyleSheet(self.get_button_style(font_size=RIGHT_BUTTON_FONT_SIZE))
            btn.setCursor(Qt.PointingHandCursor)
            
            # 버튼 이벤트 연결
            btn_name = btn_config["name"]
            btn.clicked.connect(lambda checked, name=btn_name: self.on_right_button_click(name))
            
            self.right_buttons[btn_name] = btn
            layout.addWidget(btn)
        
        # 이름 입력 필드
        layout.addSpacing(30)
        
        name_label = QLabel("이름:")
        name_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 11px;")
        layout.addWidget(name_label)
        
        self.name_input = QLineEdit()
        self.name_input.setFixedSize(INPUT_FIELD_WIDTH, INPUT_FIELD_HEIGHT)
        self.name_input.setPlaceholderText("이름 입력")
        self.name_input.setStyleSheet(self.get_input_style())
        layout.addWidget(self.name_input)

        id_label = QLabel("ID:")
        id_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 11px;")
        layout.addWidget(id_label)

        self.user_id_input = QLineEdit()
        self.user_id_input.setFixedSize(INPUT_FIELD_WIDTH, INPUT_FIELD_HEIGHT)
        self.user_id_input.setPlaceholderText("ID 입력")
        self.user_id_input.setStyleSheet(self.get_input_style())
        layout.addWidget(self.user_id_input)

        email_label = QLabel("이메일:")
        email_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 11px;")
        layout.addWidget(email_label)

        self.email_input = QLineEdit()
        self.email_input.setFixedSize(INPUT_FIELD_WIDTH, INPUT_FIELD_HEIGHT)
        self.email_input.setPlaceholderText("email@example.com")
        self.email_input.setStyleSheet(self.get_input_style())
        layout.addWidget(self.email_input)

        phone_label = QLabel("전화번호:")
        phone_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 11px;")
        layout.addWidget(phone_label)

        self.phone_input = QLineEdit()
        self.phone_input.setFixedSize(INPUT_FIELD_WIDTH, INPUT_FIELD_HEIGHT)
        self.phone_input.setPlaceholderText("010-0000-0000")
        self.phone_input.setStyleSheet(self.get_input_style())
        layout.addWidget(self.phone_input)
        
        layout.addStretch()
        
        return panel
    
    def create_status_bar(self):
        """하단 상태바 생성"""
        status_bar = QLabel("✅ 준비 완료")
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
    
    def load_face_data(self):
        """저장된 얼굴 데이터 로드"""
        face_data_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'face_data.pkl')
        )
        
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
            print("ℹ️  등록된 얼굴 데이터가 없습니다.")
    
    def save_face_data(self):
        """얼굴 데이터 저장"""
        face_data_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'face_data.pkl')
        )
        os.makedirs(os.path.dirname(face_data_file), exist_ok=True)
        
        data = {
            'features': self.known_face_features,
            'names': self.known_face_names
        }
        
        try:
            with open(face_data_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ 얼굴 데이터 저장됨: {len(self.known_face_names)}명")
        except Exception as e:
            print(f"⚠️  얼굴 데이터 저장 실패: {e}")
    
    def extract_face_features(self, detection, image_width, image_height):
        """얼굴 감지 결과에서 특징 벡터 추출"""
        bbox = detection.bounding_box
        features = [
            bbox.origin_x / image_width,
            bbox.origin_y / image_height,
            bbox.width / image_width,
            bbox.height / image_height
        ]
        return np.array(features)
    
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
                background-color: #1F618D;
            }}
        """
    
    def get_input_style(self):
        """입력 필드 스타일 반환"""
        return f"""
            QLineEdit {{
                background-color: {CAM_BG_COLOR};
                color: {TEXT_COLOR};
                border: 2px solid {BUTTON_COLOR};
                border-radius: 5px;
                padding: 5px;
                font-size: 11px;
            }}
            QLineEdit:focus {{
                border: 2px solid {BUTTON_HOVER_COLOR};
            }}
        """
    
    def start_camera(self):
        """카메라 시작"""
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        
        if not self.camera.isOpened():
            self.update_status("❌ 카메라를 열 수 없습니다", error=True)
            return
        
        # 카메라 해상도 설정
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        
        # 타이머로 프레임 업데이트
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / CAMERA_FPS))
        
        self.update_status("📹 카메라 활성화됨")
    
    def update_frame(self):
        """카메라 프레임 업데이트"""
        ret, frame = self.camera.read()
        
        if ret:
            self.current_frame = frame
            display_frame = frame.copy()
            
            # 얼굴 감지 오버레이
            if self.face_detector:
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.face_detector.detect(mp_image)
                
                if detection_result.detections:
                    for detection in detection_result.detections:
                        bbox = detection.bounding_box
                        x_min = int(bbox.origin_x)
                        y_min = int(bbox.origin_y)
                        x_max = int(bbox.origin_x + bbox.width)
                        y_max = int(bbox.origin_y + bbox.height)
                        
                        cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        
                        for keypoint in detection.keypoints:
                            h, w, _ = display_frame.shape
                            kp_x = int(keypoint.x * w)
                            kp_y = int(keypoint.y * h)
                            cv2.circle(display_frame, (kp_x, kp_y), 3, (0, 255, 255), -1)
                    
                    cv2.putText(display_frame, "Face Detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "No Face Detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # OpenCV Haar Cascade fallback
                gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                if len(faces) > 0:
                    cv2.putText(display_frame, "Face Detected (OpenCV)", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "No Face Detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
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
    
    def on_left_button_click(self, button_name):
        """좌측 버튼 클릭 이벤트"""
        if button_name == "photo_register":
            self.update_status("📷 사진 등록 모드")
            QMessageBox.information(self, "사진 등록", "사진 등록 기능이 선택되었습니다.")
        elif button_name == "voice_register":
            # 음성 등록/인식 옵션 선택
            options = ["음성 등록", "음성 인식"]
            choice, ok = QInputDialog.getItem(
                self,
                "음성 모드 선택",
                "수행할 작업을 선택하세요:",
                options,
                0,
                False
            )
            
            if ok:
                if choice == "음성 등록":
                    self.voice_register_mode()
                elif choice == "음성 인식":
                    self.voice_recognize_mode()
        elif button_name == "gesture_register":
            self.gesture_register_mode()
        elif button_name == "data_manage":
            self.open_data_management()

    def open_data_management(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("등록 데이터 관리")
        dialog.setMinimumSize(700, 500)

        main_layout = QVBoxLayout(dialog)

        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("데이터 유형:"))

        type_combo = QComboBox()
        type_combo.addItems(["얼굴 인식", "손동작 인식", "음성 인식"])
        control_layout.addWidget(type_combo)

        refresh_btn = QPushButton("새로고침")
        delete_btn = QPushButton("삭제")

        for btn in [refresh_btn, delete_btn]:
            btn.setStyleSheet(self.get_button_style(font_size=10))

        control_layout.addWidget(refresh_btn)
        control_layout.addWidget(delete_btn)
        control_layout.addStretch()

        main_layout.addLayout(control_layout)

        table = QTableWidget(0, 5)
        table.setHorizontalHeaderLabels(["선택", "Source", "ID", "Label", "Created At"])
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.MultiSelection)
        table.horizontalHeader().setStretchLastSection(True)
        main_layout.addWidget(table)

        def _get_local_labels():
            index = type_combo.currentIndex()
            if index == 0:
                return sorted(set(self.known_face_names))
            if index == 1:
                labels = []
                for entries in self.gesture_service.known_gestures.values():
                    labels.extend([e.get("name") for e in entries if e.get("name")])
                return sorted(set(labels))
            return sorted(set(self.voice_service.known_voice_names))

        def _load_table():
            table.setRowCount(0)
            for label in _get_local_labels():
                row = table.rowCount()
                table.insertRow(row)
                check_item = QTableWidgetItem()
                check_item.setCheckState(Qt.Unchecked)
                table.setItem(row, 0, check_item)
                table.setItem(row, 1, QTableWidgetItem("LOCAL"))
                table.setItem(row, 2, QTableWidgetItem("-"))
                table.setItem(row, 3, QTableWidgetItem(label))
                table.setItem(row, 4, QTableWidgetItem("-"))

        def _get_checked_rows():
            checked = []
            for row in range(table.rowCount()):
                item = table.item(row, 0)
                if item and item.checkState() == Qt.Checked:
                    source_item = table.item(row, 1)
                    id_item = table.item(row, 2)
                    label_item = table.item(row, 3)
                    source = source_item.text() if source_item else None
                    record_id = int(id_item.text()) if id_item and id_item.text().isdigit() else None
                    label = label_item.text() if label_item else None
                    checked.append((source, record_id, label))
            return checked

        def _handle_refresh():
            _load_table()

        def _handle_delete():
            checked = _get_checked_rows()
            if not checked:
                QMessageBox.warning(dialog, "선택 필요", "삭제할 항목을 체크하세요.")
                return
            local_rows = [r for r in checked if r[0] == "LOCAL" and r[2]]

            total_count = len(local_rows)
            if total_count == 0:
                QMessageBox.warning(dialog, "선택 필요", "삭제할 항목을 체크하세요.")
                return

            if QMessageBox.question(dialog, "삭제 확인", f"선택한 {total_count}건을 삭제하시겠습니까?") != QMessageBox.Yes:
                return

            if local_rows:
                index = type_combo.currentIndex()
                labels = [r[2] for r in local_rows]
                if index == 0:
                    filtered = [(f, n) for f, n in zip(self.known_face_features, self.known_face_names) if n not in labels]
                    self.known_face_features = [f for f, _ in filtered]
                    self.known_face_names = [n for _, n in filtered]
                    self.save_face_data()
                elif index == 1:
                    for gesture_type in list(self.gesture_service.known_gestures.keys()):
                        entries = self.gesture_service.known_gestures.get(gesture_type, [])
                        entries = [e for e in entries if e.get("name") not in labels]
                        if entries:
                            self.gesture_service.known_gestures[gesture_type] = entries
                        else:
                            self.gesture_service.known_gestures.pop(gesture_type, None)
                    self.gesture_service.save_gesture_data()
                else:
                    filtered = [(e, n) for e, n in zip(self.voice_service.known_voice_embeddings, self.voice_service.known_voice_names) if n not in labels]
                    self.voice_service.known_voice_embeddings = [e for e, _ in filtered]
                    self.voice_service.known_voice_names = [n for _, n in filtered]
                    self.voice_service.save_voice_data()

            _load_table()

        refresh_btn.clicked.connect(_handle_refresh)
        delete_btn.clicked.connect(_handle_delete)
        type_combo.currentIndexChanged.connect(_load_table)

        _load_table()
        dialog.exec_()
    
    def on_right_button_click(self, button_name):
        """우측 버튼 클릭 이벤트"""
        if button_name == "capture":
            self.capture_photo()
        elif button_name == "save":
            self.save_photo()
        elif button_name == "input_info":
            self.input_user_info()
    
    def capture_photo(self):
        """사진 찍기"""
        if self.current_frame is not None:
            self.captured_frame = self.current_frame.copy()
            self.update_status("📸 사진이 캡처되었습니다")
            QMessageBox.information(self, "캡처 완료", "사진이 캡처되었습니다.\n'사진 저장' 버튼을 눌러 저장하세요.")
        else:
            self.update_status("❌ 카메라가 활성화되지 않았습니다", error=True)
    
    def save_photo(self):
        """사진 저장"""
        if self.captured_frame is None:
            self.update_status("❌ 저장할 사진이 없습니다. 먼저 사진을 찍어주세요.", error=True)
            QMessageBox.warning(self, "저장 실패", "먼저 '사진찍기' 버튼을 눌러 사진을 찍어주세요.")
            return
        
        name = self.name_input.text().strip()
        user_id = self.user_id_input.text().strip() if hasattr(self, "user_id_input") else ""
        email = self.email_input.text().strip() if hasattr(self, "email_input") else ""
        phone = self.phone_input.text().strip() if hasattr(self, "phone_input") else ""
        
        if not name:
            self.update_status("❌ 이름을 입력해주세요", error=True)
            QMessageBox.warning(self, "입력 필요", "이름을 입력해주세요.")
            return

        if not user_id or not email or not phone:
            self.update_status("❌ ID/이메일/전화번호를 입력해주세요", error=True)
            QMessageBox.warning(self, "입력 필요", "ID, 이메일, 전화번호를 모두 입력해주세요.")
            return

        if not self._upsert_user_basic_data(user_id, name, email, phone):
            return
        
        # 얼굴 감지 및 특징 추출
        if MEDIAPIPE_AVAILABLE and self.face_detector:
            rgb_frame = cv2.cvtColor(self.captured_frame, cv2.COLOR_BGR2RGB)
            mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.face_detector.detect(mp_image)
            
            if not detection_result.detections:
                reply = QMessageBox.question(
                    self, 
                    "얼굴 감지 안됨", 
                    "사진에서 얼굴을 감지하지 못했습니다.\n그래도 저장하시겠습니까?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
            elif len(detection_result.detections) > 1:
                QMessageBox.warning(
                    self, 
                    "여러 얼굴 감지", 
                    f"{len(detection_result.detections)}개의 얼굴이 감지되었습니다.\n한 명만 촬영해주세요."
                )
                return
            
            # 얼굴 특징 추출
            if detection_result.detections and len(detection_result.detections) == 1:
                h, w, _ = self.captured_frame.shape
                face_features = self.extract_face_features(detection_result.detections[0], w, h)
                
                # 기존 데이터에 추가
                self.known_face_features.append(face_features)
                self.known_face_names.append(name)
                
                # 파일로 저장
                self.save_face_data()
                
                self.update_status(f"✓ 얼굴 데이터 저장됨: {name}")
                face_registered = "O"
            else:
                face_registered = "X"
        else:
            face_registered = "X"
        
        # 파일명 생성
        import datetime
        
        save_dir = "../data/photos"
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/{name}_{timestamp}.jpg"
        
        # 이미지 저장
        cv2.imwrite(filename, self.captured_frame)
        
        
        self.update_status(f"💾 저장 완료: {filename}")
        QMessageBox.information(self, "저장 완료", f"사진이 저장되었습니다.\n\n파일: {filename}\n얼굴 등록: {face_registered}")
        
        # 입력 필드 초기화
        self.name_input.clear()
        if hasattr(self, "user_id_input"):
            self.user_id_input.clear()
        if hasattr(self, "email_input"):
            self.email_input.clear()
        if hasattr(self, "phone_input"):
            self.phone_input.clear()
        self.captured_frame = None
    
    def input_user_info(self):
        """사용자 정보 입력 다이얼로그"""
        name, ok = QInputDialog.getText(self, "이름 입력", "이름:")
        if ok and name:
            self.name_input.setText(name)
            self.update_status(f"✏️ 입력 완료: {name}")

    def _get_userbasic_db_config(self):
        return {
            "host": os.getenv("DB_HOST", "192.168.0.41"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "dbname": os.getenv("DB_NAME", "devserver"),
            "user": os.getenv("DB_USER", "orugu"),
            "password": os.getenv("DB_PASSWORD", "orugu#0916"),
        }

    def _ensure_userbasicdata_schema(self, conn):
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS "UserData".userbasicdata (
                    user_id VARCHAR PRIMARY KEY,
                    name VARCHAR,
                    email VARCHAR,
                    phone VARCHAR,
                    created_at TIMESTAMP DEFAULT NOW()
                )
                """
            )
            cur.execute("ALTER TABLE \"UserData\".userbasicdata ADD COLUMN IF NOT EXISTS user_id VARCHAR")
            cur.execute("ALTER TABLE \"UserData\".userbasicdata ADD COLUMN IF NOT EXISTS name VARCHAR")
            cur.execute("ALTER TABLE \"UserData\".userbasicdata ADD COLUMN IF NOT EXISTS email VARCHAR")
            cur.execute("ALTER TABLE \"UserData\".userbasicdata ADD COLUMN IF NOT EXISTS phone VARCHAR")
            cur.execute("ALTER TABLE \"UserData\".userbasicdata ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW()")
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS userbasicdata_user_id_uidx ON \"UserData\".userbasicdata (user_id)"
            )

    def _upsert_user_basic_data(self, user_id, name, email, phone):
        cfg = self._get_userbasic_db_config()
        try:
            with psycopg2.connect(
                host=cfg["host"],
                port=cfg["port"],
                dbname=cfg["dbname"],
                user=cfg["user"],
                password=cfg["password"],
                connect_timeout=5,
                options="-c lc_messages=C"
            ) as conn:
                self._ensure_userbasicdata_schema(conn)
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO "UserData".userbasicdata (user_id, name, email, phone, created_at)
                        VALUES (%s, %s, %s, %s, NOW())
                        ON CONFLICT (user_id)
                        DO UPDATE SET name = EXCLUDED.name, email = EXCLUDED.email, phone = EXCLUDED.phone
                        """,
                        (user_id, name, email, phone)
                    )
                conn.commit()
            return True
        except Exception as e:
            self.update_status("❌ 사용자 정보 저장 실패", error=True)
            QMessageBox.warning(self, "저장 실패", f"사용자 정보 저장 실패: {e}")
            return False
    
    def update_status(self, message, error=False):
        """상태바 업데이트"""
        if error:
            self.status_bar.setStyleSheet(
                self.status_bar.styleSheet().replace(STATUS_BAR_BG_COLOR, ACCENT_COLOR)
            )
        else:
            self.status_bar.setStyleSheet(
                self.status_bar.styleSheet().replace(ACCENT_COLOR, STATUS_BAR_BG_COLOR)
            )
        self.status_bar.setText(message)
    
    def voice_register_mode(self):
        """음성 등록 모드 (음성 녹음)"""
        # 모델이 로드되지 않았다면 먼저 로드
        if not self.voice_service.is_ready():
            self.update_status("🔄 음성 인식 모델 로딩 중... (처음 실행 시 시간이 걸릴 수 있습니다)")
            QApplication.processEvents()  # UI 업데이트
            
            if not self.voice_service.ensure_model_loaded():
                error_msg = self.voice_service.get_error_message()
                self.update_status(f"❌ 모델 로드 실패: {error_msg}")
                QMessageBox.critical(
                    self,
                    "모델 로드 실패",
                    f"음성 인식 모델을 로드할 수 없습니다.\n\n{error_msg}\n\n설치 방법:\npip install speechbrain"
                )
                return
            
            self.update_status("✅ 음성 인식 모델 로드 완료")
        
        # 사용자 이름 입력
        name, ok = QInputDialog.getText(
            self,
            "사용자 이름 입력",
            "등록할 사용자의 이름을 입력하세요:"
        )
        
        if not ok or not name.strip():
            self.update_status("❌ 사용자 이름 입력 취소됨")
            return
        
        name = name.strip()
        
        # 녹음 시간 입력 (기본값: 3초)
        duration, ok = QInputDialog.getInt(
            self,
            "녹음 시간 설정",
            "녹음 시간(초)을 입력하세요:",
            3,
            1,
            10
        )
        
        if not ok:
            self.update_status("❌ 녹음 시간 설정 취소됨")
            return
        
        # 녹음 확인
        confirm = QMessageBox.question(
            self,
            "녹음 시작",
            f"{name}의 음성을 {duration}초간 녹음합니다.\n마이크를 준비하세요.\n계속할까요?"
        )
        
        if confirm != QMessageBox.Yes:
            self.update_status("❌ 음성 녹음 취소됨")
            return
        
        # 음성 녹음
        self.update_status(f"🎤 음성 녹음 중... ({duration}초)")
        
        try:
            import sounddevice as sd
            import soundfile as sf
            import numpy as np
            
            # 오디오 장치 확인
            try:
                devices = sd.query_devices()
                print("사용 가능한 오디오 장치:")
                print(devices)
                
                # 입력 장치 찾기
                input_devices = []
                for i, device in enumerate(devices):
                    if device['max_input_channels'] > 0:
                        input_devices.append((i, device['name']))
                
                if not input_devices:
                    raise Exception("마이크 입력 장치를 찾을 수 없습니다.")
                
                print(f"발견된 입력 장치: {len(input_devices)}개")
                for idx, device_label in input_devices:
                    print(f"  [{idx}] {device_label}")
                
                # 첫 번째 입력 장치를 기본으로 사용
                device_id = input_devices[0][0]
                device_name = input_devices[0][1]
                print(f"사용할 장치: [{device_id}] {device_name}")
                
            except Exception as device_error:
                self.update_status(f"❌ 오디오 장치 확인 실패: {str(device_error)}")
                QMessageBox.warning(
                    self,
                    "오디오 장치 오류",
                    f"오디오 장치를 찾을 수 없습니다.\n\n마이크가 연결되어 있는지 확인하세요.\n\n에러: {str(device_error)}"
                )
                return
            
            # 임시 음성 파일 저장
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            voice_dir = os.path.join(base_dir, "data", "voice")
            os.makedirs(voice_dir, exist_ok=True)
            temp_audio_file = os.path.join(voice_dir, "tmp_voice_recording.wav")
            sample_rate = 16000
            
            # 녹음
            print(f"🎤 녹음 시작... ({duration}초)")
            QMessageBox.information(
                self,
                "녹음 시작",
                f"{duration}초 후 자동으로 종료됩니다.\n지금부터 말씀하세요!"
            )
            
            audio_data = sd.rec(
                int(duration * sample_rate), 
                samplerate=sample_rate, 
                channels=1, 
                dtype='float32',
                device=device_id  # 명시적으로 장치 지정
            )
            sd.wait()  # 녹음 완료 대기
            
            # 파일로 저장
            sf.write(temp_audio_file, audio_data, sample_rate)
            print(f"✓ 녹음 완료: {temp_audio_file}")
            
            # 음성 데이터 등록
            self.update_status(f"🎤 음성 데이터 처리 중... ({name})")
            
            success = self.voice_service.register_voice(temp_audio_file, name)
            
            if success:
                self.voice_service.save_voice_data()
                self.update_status(f"✅ {name}의 음성 데이터 등록 완료")
                QMessageBox.information(
                    self,
                    "등록 완료",
                    f"{name}의 음성 데이터가 성공적으로 등록되었습니다."
                )
            else:
                self.update_status("❌ 음성 데이터 등록 실패")
                QMessageBox.warning(
                    self,
                    "등록 실패",
                    "음성 데이터 등록 중 오류가 발생했습니다.\n다시 시도하세요."
                )
            
            # 임시 파일 삭제
            if os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)
                
        except ImportError:
            self.update_status("❌ 음성 녹음 라이브러리 없음")
            QMessageBox.warning(
                self,
                "라이브러리 오류",
                "sounddevice 및 soundfile이 설치되어 있지 않습니다.\n설치 후 다시 시도하세요."
            )
        except Exception as e:
            self.update_status(f"❌ 음성 녹음 오류: {str(e)}")
            QMessageBox.warning(
                self,
                "녹음 오류",
                f"음성 녹음 중 오류가 발생했습니다:\n{str(e)}"
            )
    
    def gesture_register_mode(self):
        """제스처 등록 모드"""
        self.update_status("👋 제스처 등록 모드 활성화")
        
        # 제스처 타입 선택
        gesture_types = ["OK", "Pointing_Up", "Thumbs_Down", "Thumbs_Up", "Victory", "Open_Palm", "Closed_Fist"]
        
        gesture_dialog = QInputDialog()
        gesture_type, ok = gesture_dialog.getItem(
            self,
            "제스처 타입 선택",
            "등록할 제스처를 선택하세요:",
            gesture_types,
            0,
            False
        )
        
        if not ok or not gesture_type:
            self.update_status("❌ 제스처 타입 선택 취소됨")
            return
        
        # 사용자 이름 입력
        user_name, ok = QInputDialog.getText(
            self,
            "사용자 정보 입력",
            "등록할 사용자의 이름을 입력하세요:"
        )
        
        if not ok or not user_name.strip():
            self.update_status("❌ 사용자 이름 입력 취소됨")
            return
        
        user_name = user_name.strip()
        
        # 카메라에서 제스처 캡처
        self.update_status(f"👋 {gesture_type} 제스처를 보여주세요... (카메라 확인)")
        
        # 3초 동안 프레임 캡처
        capture_count = 0
        max_captures = 5  # 5개 프레임 캡처
        success_count = 0
        
        for i in range(90):  # 3초 (30fps * 3)
            ret, frame = self.camera.read()
            
            if ret:
                self.current_frame = frame
                
                # 매 18프레임마다 캡처 시도 (대략 0.6초 간격)
                if i % 18 == 0 and capture_count < max_captures:
                    if self.gesture_service.register_gesture(frame, gesture_type, user_name):
                        success_count += 1
                    capture_count += 1
                
                # UI 업데이트 (디스플레이만)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(CAM_WIDTH, CAM_HEIGHT, Qt.KeepAspectRatio)
                self.camera_label.setPixmap(scaled_pixmap)
                
                QApplication.processEvents()
        
        # 결과 처리
        if success_count > 0:
            self.gesture_service.save_gesture_data()
            self.update_status(f"✅ {user_name}의 '{gesture_type}' 제스처 {success_count}개 등록 완료")
            QMessageBox.information(
                self,
                "등록 완료",
                f"{user_name}의 '{gesture_type}' 제스처 {success_count}개가 등록되었습니다."
            )
        else:
            self.update_status("❌ 제스처 등록 실패 - 다시 시도하세요")
            QMessageBox.warning(
                self,
                "등록 실패",
                "제스처 등록 중 오류가 발생했습니다.\n제스처를 명확하게 보여주세요."
            )
    
    def voice_recognize_mode(self):
        """음성 인식 모드"""
        self.update_status("🎤 음성 파일 선택 대기 중...")
        
        # 음성 파일 선택
        audio_file, _ = QFileDialog.getOpenFileName(
            self,
            "음성 파일 선택",
            "",
            "음성 파일 (*.wav *.mp3 *.flac);;모든 파일 (*)"
        )
        
        if not audio_file:
            self.update_status("❌ 음성 파일 선택 취소됨")
            return
        
        # 음성 인식 실행
        self.update_status(f"🎤 음성 인식 중... ({os.path.basename(audio_file)})")
        
        name, similarity = self.voice_service.recognize_voice(audio_file)
        
        if name != "Unknown" and similarity > self.voice_service.voice_similarity_threshold:
            self.update_status(f"✅ {name} 인식됨 (유사도: {similarity:.3f})")
            QMessageBox.information(
                self,
                "인식 성공",
                f"음성 인식 완료:\n이름: {name}\n유사도: {similarity:.3f}"
            )
        else:
            self.update_status("❌ 음성 인식 실패 - 등록된 사용자를 찾을 수 없습니다")
            QMessageBox.warning(
                self,
                "인식 실패",
                "등록된 사용자의 음성과 일치하지 않습니다."
            )
    
    def closeEvent(self, event):
        """윈도우 종료 이벤트"""
        if self.camera is not None:
            self.camera.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    # 다크 테마 적용
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(44, 62, 80))
    palette.setColor(QPalette.WindowText, Qt.white)
    app.setPalette(palette)
    
    window = AdminUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
