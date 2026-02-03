"""
Lightweight UI for development/testing without heavy ML dependencies.
"""
import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QMainWindow
from client_ui import ClientUI

class LightClientUI(ClientUI):
    def __init__(self):
        # Skip heavy model initialization by not calling parent __init__ fully
        QMainWindow.__init__(self)
        # Minimal attributes used by parent methods
        self.current_mode = None
        # Apply UI config
        self.init_ui()
        # Start a simple timer to draw a fake camera feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_fake_frame)
        self.timer.start(1000 // 10)  # 10 FPS
        self.frame_counter = 0

    def start_camera(self):
        # Overridden to avoid cv2.VideoCapture
        self.update_status("📹 라이트 모드: 카메라 시뮬레이션")

    def update_fake_frame(self):
        w, h = 640, 480
        t = self.frame_counter / 10.0
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # moving circle to simulate motion
        cx = int((np.sin(t) * 0.4 + 0.5) * w)
        cy = int((np.cos(t) * 0.4 + 0.5) * h)
        cv2.circle(img, (cx, cy), 60, (0, 180, 255), -1)
        # mode text
        if self.current_mode == 'face_attendance':
            cv2.putText(img, '얼굴 모드 (샘플)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        elif self.current_mode == 'gesture_attendance':
            cv2.putText(img, '제스처 모드 (샘플)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        rgb_for_qt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h2, w2, ch = rgb_for_qt.shape
        bytes_per_line = ch * w2
        qt_image = QImage(rgb_for_qt.data, w2, h2, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.camera_label.width(), self.camera_label.height(), Qt.KeepAspectRatio)
        self.camera_label.setPixmap(scaled_pixmap)
        self.frame_counter += 1

def main():
    app = QApplication(sys.argv)
    window = LightClientUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
