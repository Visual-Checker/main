"""
클라이언트 UI - 출결관리 시스템
제스처 및 얼굴 인식 출석 체크
"""

import sys
import cv2
import os
import pickle
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QMessageBox, QFrame
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor

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

# UI 설정 임포트
from ui_config_lib import *


class ClientUI(QMainWindow):
    """클라이언트 출석 체크 윈도우"""
    
    def __init__(self):
        super().__init__()
        
        # 카메라 초기화
        self.camera = None
        self.current_frame = None
        self.current_mode = None  # 'gesture', 'face', None
        self.current_user = None
        
        # 얼굴 감지기 초기화
        self.face_detector = None
        self.gesture_recognizer = None
        
        if MEDIAPIPE_AVAILABLE and USE_TASK_API:
            try:
                # 얼굴 감지기
                base_options_face = python.BaseOptions(model_asset_path='models/blaze_face_short_range.tflite')
                face_options = vision.FaceDetectorOptions(base_options=base_options_face)
                self.face_detector = vision.FaceDetector.create_from_options(face_options)
                
                # 제스처 인식기
                base_options_gesture = python.BaseOptions(model_asset_path='models/gesture_recognizer.task')
                gesture_options = vision.GestureRecognizerOptions(base_options=base_options_gesture)
                self.gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_options)
                
                print("✓ MediaPipe Task API 초기화 성공")
            except Exception as e:
                print(f"⚠️  MediaPipe 초기화 실패: {e}")
                print("ℹ️  모델 파일을 확인하세요: models/blaze_face_short_range.tflite, models/gesture_recognizer.task")
        
        # 얼굴인식 데이터 로드
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
        
        # 중앙(카메라) + 우측(정보) 영역
        cam_info_layout = QHBoxLayout()
        
        # 카메라 영역
        self.camera_label = self.create_camera_view()
        cam_info_layout.addWidget(self.camera_label)
        
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
        """카메라 뷰 생성"""
        camera_label = QLabel()
        camera_label.setFixedSize(CAM_WIDTH, CAM_HEIGHT)
        camera_label.setAlignment(Qt.AlignCenter)
        camera_label.setStyleSheet(f"""
            background-color: {CAM_BG_COLOR};
            border: 3px solid {ACCENT_COLOR};
            border-radius: 10px;
        """)
        camera_label.setText("📹 카메라 로딩 중...")
        camera_label.setFont(QFont("Arial", 14))
        camera_label.setStyleSheet(camera_label.styleSheet() + f"color: {TEXT_COLOR};")
        
        return camera_label
    
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
                
                # 유사도 임계값 (0.98 이상이면 같은 사람)
                confidence = best_similarity * 100
                if confidence < 98:
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
    
    def update_frame(self):
        """카메라 프레임 업데이트"""
        ret, frame = self.camera.read()
        
        if ret:
            self.current_frame = frame
            display_frame = frame.copy()
            
            # 얼굴 인식 모드일 때 처리
            if self.current_mode == "face_attendance":
                display_frame, recognized_names = self.recognize_faces(display_frame)
                
                # 인식된 사람 정보 업데이트
                if recognized_names:
                    name, confidence = recognized_names[0]  # 첫 번째 인식된 사람
                    self.user_name_label.setText(f"이름: {name}")
                    self.attendance_status_label.setText(f"인식됨 ({confidence:.1f}%)")
                    self.attendance_status_label.setStyleSheet(
                        f"color: {SUCCESS_COLOR}; font-size: 16px; font-weight: bold;"
                    )
                    
                    # 자동 출석 처리 (confidence > 80%)
                    if confidence > 80:
                        self.detected_gesture_label.setText(f"✓ {name} 출석 확인")
                else:
                    self.detected_gesture_label.setText("얼굴: 감지 안됨")
            
            # 모드 표시
            if self.current_mode:
                mode_text = {
                    "gesture_attendance": "제스처 출석 모드",
                    "face_attendance": "얼굴 인식 모드",
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
        
        if mode_name == "gesture_attendance":
            self.update_status("✋ 제스처 출석 모드 활성화")
            self.attendance_status_label.setText("제스처 대기 중...")
            self.attendance_status_label.setStyleSheet(f"color: {WARNING_COLOR}; font-size: 16px; font-weight: bold;")
            
        elif mode_name == "face_attendance":
            self.update_status("😊 얼굴 인식 출석 모드 활성화")
            self.attendance_status_label.setText("얼굴 인식 중...")
            self.attendance_status_label.setStyleSheet(f"color: {ACCENT_COLOR}; font-size: 16px; font-weight: bold;")
            
        elif mode_name == "attendance_status":
            self.update_status("📊 출석 현황 조회")
            QMessageBox.information(self, "출석 현황", "출석 현황 조회 기능입니다.")
    
    def update_status(self, message):
        """상태바 업데이트"""
        self.status_bar.setText(message)
    
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
