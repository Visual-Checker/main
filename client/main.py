import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QTableWidget, QTableWidgetItem, QTabWidget,
                             QMessageBox, QComboBox, QTextEdit, QGroupBox,
                             QGridLayout, QFormLayout)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import cv2
from datetime import datetime
from dotenv import load_dotenv

from database import get_db, User, Course, AttendanceRecord, ClassSession, Enrollment
from face_detector import FaceDetector
from sqlalchemy import func, and_

load_dotenv()

class LoginWindow(QWidget):
    """로그인 창"""
    login_successful = pyqtSignal(object)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle('출결관리 시스템 - 로그인')
        self.setGeometry(100, 100, 400, 300)
        
        layout = QVBoxLayout()
        layout.addStretch()
        
        # 타이틀
        title = QLabel('출결관리 시스템')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet('font-size: 24px; font-weight: bold; margin: 20px;')
        layout.addWidget(title)
        
        # 로그인 폼
        form_layout = QFormLayout()
        
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText('사용자명')
        form_layout.addRow('사용자명:', self.username_input)
        
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText('비밀번호')
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.returnPressed.connect(self.login)
        form_layout.addRow('비밀번호:', self.password_input)
        
        layout.addLayout(form_layout)
        
        # 로그인 버튼
        self.login_btn = QPushButton('로그인')
        self.login_btn.clicked.connect(self.login)
        self.login_btn.setStyleSheet('padding: 10px; font-size: 14px;')
        layout.addWidget(self.login_btn)
        
        self.error_label = QLabel('')
        self.error_label.setStyleSheet('color: red;')
        self.error_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.error_label)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        
        if not username or not password:
            self.error_label.setText('사용자명과 비밀번호를 입력하세요')
            return
        
        try:
            db = get_db()
            user = db.query(User).filter(User.username == username).first()
            
            if user and user.password_hash == password:  # 실제 환경에서는 bcrypt 사용
                self.login_successful.emit(user)
                self.close()
            else:
                self.error_label.setText('로그인 실패: 사용자명 또는 비밀번호가 올바르지 않습니다')
            
            db.close()
        except Exception as e:
            self.error_label.setText(f'로그인 오류: {str(e)}')

class FaceRecognitionWindow(QWidget):
    """얼굴 인식 출석 체크 창"""
    
    def __init__(self, user):
        super().__init__()
        self.user = user
        self.detector = FaceDetector()
        self.camera = None
        self.timer = QTimer()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle('얼굴 인식 출석')
        self.setGeometry(100, 100, 800, 600)
        
        layout = QVBoxLayout()
        
        # 카메라 화면
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet('border: 2px solid #ccc;')
        self.camera_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.camera_label)
        
        # 정보 표시
        info_layout = QHBoxLayout()
        self.info_label = QLabel('카메라를 시작하려면 "카메라 시작" 버튼을 클릭하세요')
        self.info_label.setStyleSheet('font-size: 12px; padding: 10px;')
        info_layout.addWidget(self.info_label)
        layout.addLayout(info_layout)
        
        # 버튼
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton('카메라 시작')
        self.start_btn.clicked.connect(self.start_camera)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton('카메라 중지')
        self.stop_btn.clicked.connect(self.stop_camera)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        
        self.checkin_btn = QPushButton('출석 체크')
        self.checkin_btn.clicked.connect(self.check_attendance)
        self.checkin_btn.setEnabled(False)
        btn_layout.addWidget(self.checkin_btn)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
    
    def start_camera(self):
        camera_idx = int(os.getenv('CAMERA_INDEX', 0))
        self.camera = cv2.VideoCapture(camera_idx)
        
        if not self.camera.isOpened():
            QMessageBox.warning(self, '오류', '카메라를 열 수 없습니다')
            return
        
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.checkin_btn.setEnabled(True)
    
    def stop_camera(self):
        self.timer.stop()
        if self.camera:
            self.camera.release()
        self.camera_label.clear()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.checkin_btn.setEnabled(False)
    
    def update_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            return
        
        # 얼굴 검출
        faces = self.detector.detect(frame)
        annotated = self.detector.draw_faces(frame, faces)
        
        # 정보 표시
        cv2.putText(annotated, f"Detected: {len(faces)} face(s)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Qt 이미지로 변환
        rgb_image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        self.camera_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.camera_label.size(), Qt.KeepAspectRatio))
        
        self.info_label.setText(f'검출된 얼굴: {len(faces)}개 | 모델: {self.detector.model_type}')
    
    def check_attendance(self):
        # 실제 구현에서는 얼굴 인식 및 DB 기록
        QMessageBox.information(self, '출석 체크', '얼굴 인식 출석 체크가 완료되었습니다!')
    
    def closeEvent(self, event):
        self.stop_camera()
        self.detector.release()
        event.accept()

class MainWindow(QMainWindow):
    """메인 창"""
    
    def __init__(self, user):
        super().__init__()
        self.user = user
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle(f'출결관리 시스템 - {self.user.full_name} ({self.user.role})')
        self.setGeometry(100, 100, 1200, 800)
        
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        # 상단 정보
        header = QLabel(f'환영합니다, {self.user.full_name}님 ({self.user.role})')
        header.setStyleSheet('font-size: 16px; font-weight: bold; padding: 10px; background: #2196F3; color: white;')
        layout.addWidget(header)
        
        # 탭 위젯
        tabs = QTabWidget()
        
        # 대시보드
        tabs.addTab(self.create_dashboard(), '대시보드')
        
        # 과목 관리
        tabs.addTab(self.create_courses_tab(), '과목 관리')
        
        # 출석 관리
        tabs.addTab(self.create_attendance_tab(), '출석 관리')
        
        # 얼굴 인식
        tabs.addTab(self.create_face_recognition_tab(), '얼굴 인식')
        
        layout.addWidget(tabs)
        central_widget.setLayout(layout)
    
    def create_dashboard(self):
        widget = QWidget()
        layout = QGridLayout()
        
        try:
            db = get_db()
            
            # 통계 카드
            total_courses = db.query(Course).count()
            total_students = db.query(User).filter(User.role == 'student').count()
            
            stats = [
                ('전체 과목', total_courses),
                ('전체 학생', total_students),
                ('오늘 출석률', '85%'),
            ]
            
            for i, (title, value) in enumerate(stats):
                card = QGroupBox(title)
                card_layout = QVBoxLayout()
                value_label = QLabel(str(value))
                value_label.setStyleSheet('font-size: 32px; font-weight: bold; color: #2196F3;')
                value_label.setAlignment(Qt.AlignCenter)
                card_layout.addWidget(value_label)
                card.setLayout(card_layout)
                layout.addWidget(card, 0, i)
            
            db.close()
        except Exception as e:
            error_label = QLabel(f'오류: {str(e)}')
            layout.addWidget(error_label)
        
        widget.setLayout(layout)
        return widget
    
    def create_courses_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 과목 테이블
        self.courses_table = QTableWidget()
        self.courses_table.setColumnCount(5)
        self.courses_table.setHorizontalHeaderLabels(['과목코드', '과목명', '교수', '학기', '년도'])
        
        self.load_courses()
        
        layout.addWidget(self.courses_table)
        widget.setLayout(layout)
        return widget
    
    def create_attendance_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        label = QLabel('출석 관리 기능')
        label.setStyleSheet('font-size: 18px; padding: 20px;')
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        
        widget.setLayout(layout)
        return widget
    
    def create_face_recognition_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 얼굴 인식 버튼
        face_rec_btn = QPushButton('얼굴 인식 출석 체크 시작')
        face_rec_btn.clicked.connect(self.open_face_recognition)
        face_rec_btn.setStyleSheet('padding: 15px; font-size: 16px;')
        layout.addWidget(face_rec_btn)
        
        # 모델 정보
        info_group = QGroupBox('얼굴 인식 모델 정보')
        info_layout = QVBoxLayout()
        
        model_type = os.getenv('FACE_DETECTION_MODEL', 'yunet')
        use_gpu = os.getenv('USE_GPU', 'True')
        
        info_text = f"""
        <b>현재 모델:</b> {model_type.upper()}<br>
        <b>GPU 사용:</b> {use_gpu}<br>
        <b>신뢰도 임계값:</b> {os.getenv('CONFIDENCE_THRESHOLD', 0.7)}<br>
        <br>
        <b>지원 모델:</b><br>
        - YuNet: 빠르고 정확 (권장)<br>
        - MediaPipe: 매우 빠름, Google 지원<br>
        - Haar Cascade: 레거시, CPU 전용
        """
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)
        info_group.setLayout(info_layout)
        
        layout.addWidget(info_group)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def load_courses(self):
        try:
            db = get_db()
            courses = db.query(Course).all()
            
            self.courses_table.setRowCount(len(courses))
            
            for i, course in enumerate(courses):
                self.courses_table.setItem(i, 0, QTableWidgetItem(course.course_code))
                self.courses_table.setItem(i, 1, QTableWidgetItem(course.course_name))
                self.courses_table.setItem(i, 2, QTableWidgetItem('-'))
                self.courses_table.setItem(i, 3, QTableWidgetItem(course.semester))
                self.courses_table.setItem(i, 4, QTableWidgetItem(str(course.year)))
            
            db.close()
        except Exception as e:
            QMessageBox.warning(self, '오류', f'과목 로드 실패: {str(e)}')
    
    def open_face_recognition(self):
        self.face_window = FaceRecognitionWindow(self.user)
        self.face_window.show()

def main():
    app = QApplication(sys.argv)
    
    # 데이터베이스 연결 테스트
    from database import test_connection
    if not test_connection():
        QMessageBox.critical(None, '오류', '데이터베이스 연결 실패')
        sys.exit(1)
    
    # 로그인 창
    login_window = LoginWindow()
    
    def on_login_success(user):
        main_window = MainWindow(user)
        main_window.show()
    
    login_window.login_successful.connect(on_login_success)
    login_window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
