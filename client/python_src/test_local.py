# Windows 로컬 전용 간단 테스트 스크립트
import os
import sys

def check_environment():
    """환경 확인"""
    print("=== 환경 확인 ===")
    
    # Python 버전
    print(f"Python: {sys.version}")
    
    # 필수 패키지 확인
    required = ['cv2', 'mediapipe', 'zmq', 'PyQt5', 'sqlalchemy', 'psycopg2']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
            print(f"✓ {pkg}")
        except ImportError:
            print(f"✗ {pkg} - 설치 필요")
            missing.append(pkg)
    
    if missing:
        print(f"\n누락 패키지: {', '.join(missing)}")
        print("설치: pip install -r requirements.txt")
        return False
    
    print("\n✓ 모든 패키지 설치됨")
    return True

def test_mediapipe():
    """MediaPipe 간단 테스트"""
    print("\n=== MediaPipe 테스트 ===")
    
    try:
        import cv2
        import mediapipe as mp
        
        # 카메라 확인
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ 카메라를 열 수 없습니다")
            return False
        
        # MediaPipe 초기화
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands()
        
        print("✓ MediaPipe 초기화 성공")
        print("✓ 카메라 연결 성공")
        
        ret, frame = cap.read()
        if ret:
            print(f"✓ 카메라 해상도: {frame.shape[1]}x{frame.shape[0]}")
        
        cap.release()
        hands.close()
        
        return True
        
    except Exception as e:
        print(f"✗ 오류: {e}")
        return False

def test_database():
    """데이터베이스 연결 테스트"""
    print("\n=== 데이터베이스 테스트 ===")
    
    try:
        import psycopg2
        
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='attendance_system',
            user='admin',
            password='admin123'
        )
        
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()
        
        print(f"✓ PostgreSQL 연결 성공")
        print(f"  버전: {version[0][:50]}...")
        
        cur.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"✗ 데이터베이스 연결 실패: {e}")
        print("  Docker가 실행 중인지 확인하세요:")
        print("  docker ps | findstr postgres")
        return False

def test_zmq():
    """ZeroMQ 테스트"""
    print("\n=== ZeroMQ 테스트 ===")
    
    try:
        import zmq
        
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.bind("tcp://*:5556")  # 테스트용 포트
        
        print("✓ ZeroMQ Publisher 생성 성공")
        
        socket.close()
        context.term()
        
        return True
        
    except Exception as e:
        print(f"✗ ZeroMQ 오류: {e}")
        return False

def main():
    print("╔═══════════════════════════════════════╗")
    print("║  출결관리 시스템 - 로컬 환경 테스트  ║")
    print("╚═══════════════════════════════════════╝")
    print()
    
    # 1. 환경 확인
    if not check_environment():
        return
    
    # 2. MediaPipe 테스트
    mediapipe_ok = test_mediapipe()
    
    # 3. 데이터베이스 테스트
    db_ok = test_database()
    
    # 4. ZeroMQ 테스트
    zmq_ok = test_zmq()
    
    # 결과
    print("\n" + "="*40)
    print("테스트 결과:")
    print(f"  MediaPipe:   {'✓' if mediapipe_ok else '✗'}")
    print(f"  Database:    {'✓' if db_ok else '✗'}")
    print(f"  ZeroMQ:      {'✓' if zmq_ok else '✗'}")
    print("="*40)
    
    if mediapipe_ok and zmq_ok:
        print("\n✓ 클라이언트 실행 준비 완료!")
        print("  실행: python mediapipe_detector.py")
        print("  또는: python zmq_client.py")
    
    if db_ok:
        print("\n✓ 데이터베이스 연결 가능")
    else:
        print("\n⚠️  Docker 서비스를 시작하세요:")
        print("  cd docker")
        print("  docker-compose up -d")

if __name__ == "__main__":
    main()
