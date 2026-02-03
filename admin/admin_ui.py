"""
관리자 UI - 출결관리 시스템
사진, 목소리, 제스처 등록 기능
"""

import sys
import cv2
import os
import pickle
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QLineEdit, QMessageBox, QInputDialog
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor

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
from ui_config_lib import *


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
        
        if MEDIAPIPE_AVAILABLE and USE_TASK_API:
            try:
                base_options_face = python.BaseOptions(model_asset_path='models/blaze_face_short_range.tflite')
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
        face_data_file = "../data/face_data.pkl"
        
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
        face_data_file = "../data/face_data.pkl"
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
            self.update_status("🎤 목소리 등록 모드")
            QMessageBox.information(self, "목소리 등록", "목소리 등록 기능이 선택되었습니다.")
        elif button_name == "gesture_register":
            self.update_status("👋 제스처 등록 모드")
            QMessageBox.information(self, "제스처 등록", "제스처 등록 기능이 선택되었습니다.")
    
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
        
        if not name:
            self.update_status("❌ 이름을 입력해주세요", error=True)
            QMessageBox.warning(self, "입력 필요", "이름을 입력해주세요.")
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
        self.captured_frame = None
    
    def input_user_info(self):
        """사용자 정보 입력 다이얼로그"""
        name, ok = QInputDialog.getText(self, "이름 입력", "이름:")
        if ok and name:
            self.name_input.setText(name)
            self.update_status(f"✏️ 입력 완료: {name}")
    
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
