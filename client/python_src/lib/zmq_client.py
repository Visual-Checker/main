import zmq
import json
import time
import cv2
import numpy as np
from datetime import datetime
from mediapipe_detector import MediaPipeDetector
import os
from dotenv import load_dotenv

load_dotenv()

class ZMQPublisher:
    """
    ZeroMQ Publisher - 클라이언트에서 서버로 데이터 전송
    얼굴 인식 + 제스처 인식 데이터를 실시간 스트리밍
    """
    
    def __init__(self, host="*", port=5555):
        """
        Args:
            host: "*" for binding (서버 역할), "localhost" for connecting
            port: ZeroMQ 포트
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        
        # Publisher로 바인딩
        self.address = f"tcp://{host}:{port}"
        self.socket.bind(self.address)
        
        print(f"✓ ZeroMQ Publisher started at {self.address}")
        print("  Waiting for subscribers...")
        
        # 구독자 연결 대기
        time.sleep(1)
    
    def send_detection_data(self, faces, gestures, student_id=None):
        """
        얼굴/제스처 검출 데이터 전송
        
        Args:
            faces: 얼굴 검출 결과
            gestures: 제스처 인식 결과
            student_id: 학생 ID (옵션)
        """
        message = {
            'type': 'detection',
            'timestamp': datetime.now().timestamp(),
            'student_id': student_id,
            'faces': faces,
            'gestures': gestures
        }
        
        self.socket.send_json(message)
    
    def send_attendance_event(self, student_id, gesture_type, confidence):
        """
        출석 이벤트 전송
        
        Args:
            student_id: 학생 ID
            gesture_type: 제스처 타입 (thumbs_up 등)
            confidence: 신뢰도
        """
        message = {
            'type': 'attendance_event',
            'timestamp': datetime.now().timestamp(),
            'student_id': student_id,
            'gesture': gesture_type,
            'confidence': confidence,
            'status': 'present'  # present, late, absent
        }
        
        self.socket.send_json(message)
        print(f"📤 Attendance event sent: Student {student_id} - {gesture_type}")
    
    def send_heartbeat(self):
        """하트비트 전송 (연결 유지)"""
        message = {
            'type': 'heartbeat',
            'timestamp': datetime.now().timestamp()
        }
        self.socket.send_json(message)
    
    def close(self):
        """연결 종료"""
        self.socket.close()
        self.context.term()
        print("✓ ZeroMQ Publisher closed")


class AttendanceClient:
    """
    출결관리 클라이언트
    MediaPipe + ZeroMQ 통합
    """
    
    def __init__(self, zmq_host="*", zmq_port=5555):
        self.detector = MediaPipeDetector()
        self.publisher = ZMQPublisher(host=zmq_host, port=zmq_port)
        self.camera = None
        
        # 출석 체크 설정
        self.gesture_cooldown = 3.0  # 제스처 재인식 쿨다운 (초)
        self.last_gesture_time = {}
        
        # 현재 로그인 학생 (실제로는 로그인 시스템에서)
        self.current_student_id = None
    
    def start_camera(self, camera_index=0):
        """카메라 시작"""
        self.camera = cv2.VideoCapture(camera_index)
        
        if not self.camera.isOpened():
            raise RuntimeError("카메라를 열 수 없습니다")
        
        # 카메라 설정
        width = int(os.getenv('CAMERA_WIDTH', 640))
        height = int(os.getenv('CAMERA_HEIGHT', 480))
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        print(f"✓ Camera started ({width}x{height})")
    
    def process_gesture_event(self, gesture):
        """
        제스처 이벤트 처리
        
        Thumbs Up → 출석 체크
        Peace → 확인
        Fist → 취소
        """
        gesture_type = gesture['type']
        confidence = gesture['confidence']
        
        # 쿨다운 체크
        current_time = time.time()
        if gesture_type in self.last_gesture_time:
            if current_time - self.last_gesture_time[gesture_type] < self.gesture_cooldown:
                return  # 쿨다운 중
        
        # 제스처별 처리
        if gesture_type == 'thumbs_up' and confidence > 0.8:
            if self.current_student_id:
                self.publisher.send_attendance_event(
                    self.current_student_id,
                    'thumbs_up',
                    confidence
                )
                self.last_gesture_time[gesture_type] = current_time
                print(f"✅ 출석 체크: 학생 {self.current_student_id}")
        
        elif gesture_type == 'peace' and confidence > 0.8:
            print(f"✌️  확인 제스처")
            self.last_gesture_time[gesture_type] = current_time
        
        elif gesture_type == 'fist' and confidence > 0.8:
            print(f"✊ 취소 제스처")
            self.last_gesture_time[gesture_type] = current_time
    
    def run(self, send_interval=0.1):
        """
        메인 루프
        
        Args:
            send_interval: ZeroMQ 전송 간격 (초)
        """
        if self.camera is None:
            self.start_camera()
        
        print("\n=== 출결관리 클라이언트 시작 ===")
        print("제스처 가이드:")
        print("  👍 Thumbs Up: 출석 체크")
        print("  ✌️  Peace: 확인")
        print("  ✊ Fist: 취소")
        print("\nPress 'q' to quit, 'l' to login")
        
        frame_count = 0
        last_send_time = 0
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                # 프레임 처리
                result = self.detector.process_frame(frame)
                
                # ZeroMQ로 데이터 전송 (간격 제어)
                current_time = time.time()
                if current_time - last_send_time >= send_interval:
                    self.publisher.send_detection_data(
                        result['faces'],
                        result['gestures'],
                        self.current_student_id
                    )
                    last_send_time = current_time
                
                # 제스처 이벤트 처리
                for gesture in result['gestures']:
                    self.process_gesture_event(gesture)
                
                # 화면 표시
                annotated = result['frame_annotated']
                
                # 정보 오버레이
                info_lines = [
                    f"Student ID: {self.current_student_id or 'Not logged in'}",
                    f"Faces: {len(result['faces'])} | Gestures: {len(result['gestures'])}",
                    f"Frame: {frame_count}"
                ]
                
                for i, line in enumerate(info_lines):
                    cv2.putText(annotated, line, (10, annotated.shape[0] - 60 + i * 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Attendance Client', annotated)
                
                # 키보드 입력
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('l'):
                    # 간단한 로그인 (실제로는 GUI 또는 인증 시스템)
                    student_id = input("\nEnter Student ID: ")
                    self.current_student_id = student_id
                    print(f"✓ Logged in as Student {student_id}")
                
                frame_count += 1
                
                # 하트비트 전송 (10초마다)
                if frame_count % 300 == 0:
                    self.publisher.send_heartbeat()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        self.detector.release()
        self.publisher.close()
        print("\n✓ Client stopped")


if __name__ == "__main__":
    # ZeroMQ 설정 (환경변수 또는 기본값)
    zmq_host = os.getenv('ZMQ_HOST', '*')
    zmq_port = int(os.getenv('ZMQ_PORT', 5555))
    
    # 클라이언트 실행
    client = AttendanceClient(zmq_host=zmq_host, zmq_port=zmq_port)
    
    # 테스트용 학생 ID 설정
    client.current_student_id = "2024001"  # 실제로는 로그인 시스템에서
    
    try:
        client.run(send_interval=0.1)  # 100ms마다 전송
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
