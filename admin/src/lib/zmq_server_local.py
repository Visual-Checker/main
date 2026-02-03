"""
로컬 ZeroMQ 서버
메모리 캐시 사용 (Redis 대신)
"""

import zmq
import json
import threading
from datetime import datetime
from collections import deque

# 메모리 캐시
cache = {
    'detections': deque(maxlen=100),  # 최근 100개 검출
    'attendance_events': deque(maxlen=100),  # 최근 100개 출석
    'statistics': {
        'total_detections': 0,
        'active_clients': set()
    }
}

def start_zmq_server():
    """ZeroMQ 서버 시작"""
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.bind("tcp://*:5555")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    
    print("✓ ZeroMQ 서버 리스닝 중: tcp://*:5555")
    
    while True:
        try:
            message = socket.recv_string()
            data = json.loads(message)
            
            # 메시지 타입별 처리
            msg_type = data.get('type')
            
            if msg_type == 'detection':
                handle_detection(data)
            elif msg_type == 'attendance_event':
                handle_attendance(data)
            elif msg_type == 'heartbeat':
                handle_heartbeat(data)
                
        except Exception as e:
            print(f"ZeroMQ 에러: {e}")

def handle_detection(data):
    """검출 데이터 처리"""
    cache['detections'].append({
        'timestamp': datetime.now().isoformat(),
        'faces': data.get('faces', []),
        'gestures': data.get('gestures', [])
    })
    cache['statistics']['total_detections'] += 1

def handle_attendance(data):
    """출석 이벤트 처리"""
    from server.database_local import add_user, get_user_by_name, add_attendance
    
    student_name = data.get('student_name')
    gesture = data.get('gesture')
    confidence = data.get('confidence', 0.0)
    
    # 사용자 확인 또는 생성
    user = get_user_by_name(student_name)
    if not user:
        user_id = add_user(student_name)
    else:
        user_id = user['id']
    
    # 출석 기록
    add_attendance(user_id, 'gesture', gesture, confidence)
    
    # 캐시에 저장
    cache['attendance_events'].append({
        'timestamp': datetime.now().isoformat(),
        'student_name': student_name,
        'gesture': gesture,
        'confidence': confidence,
        'status': 'success'
    })
    
    print(f"✓ 출석 기록: {student_name} ({gesture})")

def handle_heartbeat(data):
    """클라이언트 연결 상태 처리"""
    client_id = data.get('client_id')
    cache['statistics']['active_clients'].add(client_id)

def get_cached_data(key):
    """캐시 데이터 조회"""
    return cache.get(key)

if __name__ == '__main__':
    start_zmq_server()
