from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
import redis
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Admin UI Blueprint
admin_bp = Blueprint('admin', __name__, url_prefix='/admin', template_folder='templates/admin')

# Redis 연결 (선택적)
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        decode_responses=True
    )
except Exception:
    redis_client = None

@admin_bp.route('/')
def index():
    """관리자 대시보드"""
    if 'user' not in session or session.get('role') != 'admin':
        return redirect(url_for('admin.login'))
    return render_template('dashboard.html')

@admin_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        # 간단 인증 (로컬 테스트용)
        if username == 'admin' and password == 'admin123':
            session['user'] = username
            session['role'] = 'admin'
            return redirect(url_for('admin.index'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@admin_bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('admin.login'))

@admin_bp.route('/api/statistics')
def get_statistics():
    stats = {
        'total_detections': 0,
        'total_students': 0,
        'today_attendance': 0,
        'active_clients': 0
    }

    if redis_client:
        student_keys = redis_client.keys('stats:student:*')
        stats['total_students'] = len(student_keys)
        for key in student_keys:
            count = redis_client.hget(key, 'detection_count')
            if count:
                stats['total_detections'] += int(count)

        # 오늘 출석 수 계산
        events_raw = redis_client.lrange('attendance_events', 0, -1)
        today = datetime.now().date()
        for event_str in events_raw:
            try:
                event = json.loads(event_str)
                event_date = datetime.fromtimestamp(event['timestamp']).date()
                if event_date == today:
                    stats['today_attendance'] += 1
            except Exception:
                continue

    return jsonify(stats)

@admin_bp.route('/api/attendance-events')
def get_attendance_events():
    events = []
    if redis_client:
        events_raw = redis_client.lrange('attendance_events', 0, 99)
        for e in events_raw:
            try:
                events.append(json.loads(e))
            except Exception:
                continue
    return jsonify(events)

@admin_bp.route('/api/students')
def get_students():
    students = [
        {'id': '2024001', 'name': '홍길동', 'status': 'present'},
        {'id': '2024002', 'name': '이영희', 'status': 'absent'},
    ]
    return jsonify(students)

@admin_bp.route('/realtime')
def realtime_monitor():
    if 'user' not in session:
        return redirect(url_for('admin.login'))
    return render_template('realtime.html')

@admin_bp.route('/students')
def students():
    if 'user' not in session:
        return redirect(url_for('admin.login'))
    return render_template('students.html')

@admin_bp.route('/reports')
def reports():
    if 'user' not in session:
        return redirect(url_for('admin.login'))
    return render_template('reports.html')
