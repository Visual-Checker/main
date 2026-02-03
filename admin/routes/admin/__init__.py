from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
import redis
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Admin UI Blueprint
admin_bp = Blueprint('admin', __name__, url_prefix='/admin', template_folder='templates/admin')

# Redis 연결
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    decode_responses=True
)

@admin_bp.route('/')
def index():
    """관리자 대시보드"""
    # 로그인 확인
    if 'user' not in session or session.get('role') != 'admin':
        return redirect(url_for('admin.login'))
    
    return render_template('dashboard.html')

@admin_bp.route('/login', methods=['GET', 'POST'])
def login():
    """관리자 로그인"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # 간단한 인증 (실제로는 DB 확인)
        if username == 'admin' and password == 'admin123':
            session['user'] = username
            session['role'] = 'admin'
            return redirect(url_for('admin.index'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@admin_bp.route('/logout')
def logout():
    """로그아웃"""
    session.clear()
    return redirect(url_for('admin.login'))

@admin_bp.route('/api/realtime-detections')
def get_realtime_detections():
    """실시간 검출 데이터 조회"""
    # Redis에서 최근 검출 데이터 가져오기
    detections = []
    
    # ZMQ 메시지 캐시에서 읽기
    keys = redis_client.keys('zmq:detection:*')
    for key in keys[-20:]:  # 최근 20개
        data = redis_client.get(key)
        if data:
            detections.append(json.loads(data))
    
    return jsonify(detections)

@admin_bp.route('/api/attendance-events')
def get_attendance_events():
    """출석 이벤트 조회"""
    # Redis 리스트에서 출석 이벤트 가져오기
    events_raw = redis_client.lrange('attendance_events', 0, 99)  # 최근 100개
    events = [json.loads(e) for e in events_raw]
    
    return jsonify(events)

@admin_bp.route('/api/statistics')
def get_statistics():
    """통계 데이터"""
    stats = {
        'total_detections': 0,
        'total_students': 0,
        'today_attendance': 0,
        'active_clients': 0
    }
    
    # Redis에서 통계 계산
    # 학생별 검출 횟수
    student_keys = redis_client.keys('stats:student:*')
    stats['total_students'] = len(student_keys)
    
    for key in student_keys:
        count = redis_client.hget(key, 'detection_count')
        if count:
            stats['total_detections'] += int(count)
    
    # 오늘 출석 수
    today = datetime.now().date()
    events_raw = redis_client.lrange('attendance_events', 0, -1)
    for event_str in events_raw:
        event = json.loads(event_str)
        event_date = datetime.fromtimestamp(event['timestamp']).date()
        if event_date == today:
            stats['today_attendance'] += 1
    
    return jsonify(stats)

@admin_bp.route('/api/students')
def get_students():
    """학생 목록"""
    # TODO: PostgreSQL에서 조회
    # 임시 데이터
    students = [
        {'id': '2024001', 'name': '홍길동', 'status': 'present'},
        {'id': '2024002', 'name': '이영희', 'status': 'absent'},
    ]
    return jsonify(students)

@admin_bp.route('/realtime')
def realtime_monitor():
    """실시간 모니터링 페이지"""
    if 'user' not in session:
        return redirect(url_for('admin.login'))
    return render_template('realtime.html')

@admin_bp.route('/students')
def students():
    """학생 관리 페이지"""
    if 'user' not in session:
        return redirect(url_for('admin.login'))
    return render_template('students.html')

@admin_bp.route('/reports')
def reports():
    """보고서 페이지"""
    if 'user' not in session:
        return redirect(url_for('admin.login'))
    return render_template('reports.html')
