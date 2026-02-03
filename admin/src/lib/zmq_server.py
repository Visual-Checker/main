import zmq
import json
import time
from datetime import datetime
from threading import Thread
import redis
import os
from dotenv import load_dotenv

load_dotenv()

class ZMQSubscriber:
    """
    ZeroMQ Subscriber - 클라이언트로부터 데이터 수신
    """
    
    def __init__(self, client_addresses, port=5555):
        """
        Args:
            client_addresses: 클라이언트 주소 리스트 ["192.168.1.100", "192.168.1.101", ...]
            port: ZeroMQ 포트
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        
        # 모든 클라이언트에 연결
        self.client_addresses = client_addresses
        for address in client_addresses:
            conn_str = f"tcp://{address}:{port}"
            self.socket.connect(conn_str)
            print(f"✓ Connected to client: {conn_str}")
        
        # 모든 메시지 구독
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # Redis 연결 (데이터 캐싱)
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            decode_responses=True
        )
        
        # 콜백 함수들
        self.callbacks = {
            'detection': [],
            'attendance_event': [],
            'heartbeat': []
        }
        
        self.running = False
        self.thread = None
    
    def register_callback(self, message_type, callback_func):
        """
        콜백 함수 등록
        
        Args:
            message_type: 'detection', 'attendance_event', 'heartbeat'
            callback_func: def callback(data): ...
        """
        if message_type in self.callbacks:
            self.callbacks[message_type].append(callback_func)
    
    def start(self):
        """백그라운드 수신 시작"""
        self.running = True
        self.thread = Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        print("✓ ZeroMQ Subscriber started")
    
    def _receive_loop(self):
        """메시지 수신 루프"""
        while self.running:
            try:
                # 메시지 수신 (타임아웃 1초)
                if self.socket.poll(1000):
                    message = self.socket.recv_json()
                    self._process_message(message)
            except zmq.ZMQError as e:
                print(f"ZMQ Error: {e}")
            except Exception as e:
                print(f"Error processing message: {e}")
    
    def _process_message(self, message):
        """메시지 처리"""
        msg_type = message.get('type')
        
        # Redis 캐싱
        cache_key = f"zmq:{msg_type}:{message.get('student_id', 'unknown')}:{message['timestamp']}"
        self.redis_client.setex(cache_key, 300, json.dumps(message))  # 5분 TTL
        
        # 타입별 처리
        if msg_type == 'detection':
            self._handle_detection(message)
        elif msg_type == 'attendance_event':
            self._handle_attendance_event(message)
        elif msg_type == 'heartbeat':
            self._handle_heartbeat(message)
        
        # 콜백 실행
        for callback in self.callbacks.get(msg_type, []):
            try:
                callback(message)
            except Exception as e:
                print(f"Callback error: {e}")
    
    def _handle_detection(self, message):
        """얼굴/제스처 검출 데이터 처리"""
        student_id = message.get('student_id')
        faces = message.get('faces', [])
        gestures = message.get('gestures', [])
        
        # 통계 업데이트
        if student_id:
            self.redis_client.hincrby(f"stats:student:{student_id}", "detection_count", 1)
        
        # 로그
        if faces or gestures:
            print(f"📊 Detection - Student: {student_id}, Faces: {len(faces)}, Gestures: {len(gestures)}")
    
    def _handle_attendance_event(self, message):
        """출석 이벤트 처리"""
        student_id = message['student_id']
        gesture = message['gesture']
        confidence = message['confidence']
        timestamp = message['timestamp']
        
        print(f"✅ Attendance Event - Student: {student_id}, Gesture: {gesture}, Confidence: {confidence:.2f}")
        
        # Redis에 출석 이벤트 저장
        event_data = {
            'student_id': student_id,
            'gesture': gesture,
            'confidence': confidence,
            'timestamp': timestamp,
            'status': message.get('status', 'present')
        }
        
        self.redis_client.lpush('attendance_events', json.dumps(event_data))
        self.redis_client.ltrim('attendance_events', 0, 999)  # 최대 1000개 유지
    
    def _handle_heartbeat(self, message):
        """하트비트 처리"""
        # 클라이언트 연결 상태 업데이트
        # (실제로는 클라이언트 식별자 필요)
        pass
    
    def stop(self):
        """수신 중지"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.socket.close()
        self.context.term()
        print("✓ ZeroMQ Subscriber stopped")


class AttendanceServer:
    """
    출결관리 서버
    ZeroMQ + Redis + PostgreSQL
    """
    
    def __init__(self, client_addresses):
        self.subscriber = ZMQSubscriber(client_addresses)
        
        # 콜백 등록
        self.subscriber.register_callback('attendance_event', self.save_attendance)
        
        # PostgreSQL 연결 (추후 추가)
        # self.db = ...
    
    def save_attendance(self, event_data):
        """
        출석 데이터를 PostgreSQL에 저장
        """
        student_id = event_data['student_id']
        timestamp = datetime.fromtimestamp(event_data['timestamp'])
        gesture = event_data['gesture']
        confidence = event_data['confidence']
        
        # TODO: PostgreSQL에 저장
        # db.execute("INSERT INTO attendance_records ...")
        
        print(f"💾 Saved attendance: Student {student_id} at {timestamp}")
    
    def start(self):
        """서버 시작"""
        self.subscriber.start()
        print("\n=== Attendance Server Started ===")
        print("Listening for client data...")
    
    def run_forever(self):
        """서버 실행 유지"""
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⚠️  Server stopping...")
            self.subscriber.stop()


if __name__ == "__main__":
    # 클라이언트 주소 설정
    # 실제 환경에서는 설정 파일이나 환경변수에서 읽기
    client_addresses = [
        "localhost",  # 테스트용
        # "192.168.1.100",  # 실제 클라이언트 IP
        # "192.168.1.101",
    ]
    
    server = AttendanceServer(client_addresses)
    server.start()
    server.run_forever()
