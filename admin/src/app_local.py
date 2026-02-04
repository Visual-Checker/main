"""
로컬 Flask 앱
SQLite + 메모리 캐시 사용
"""

from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from datetime import datetime
import os

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'local-dev-secret-key')

# 데이터베이스 및 캐시 임포트
from lib.database_local import (
    get_all_users, get_today_attendance, get_statistics
)
from lib.zmq_server_local import get_cached_data

# 관리자 로그인
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # 간단한 인증
        if username == 'admin' and password == 'admin123':
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template('admin/login.html', error='잘못된 계정 정보입니다')
    
    return render_template('admin/login.html')

@app.route('/admin/logout')
def admin_logout():
    session.clear()
    return redirect(url_for('admin_login'))

@app.route('/admin')
@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('admin_login'))
    
    return render_template('admin/dashboard.html')

# API 엔드포인트
@app.route('/admin/api/statistics')
def get_api_statistics():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    stats = get_statistics()
    cached_stats = get_cached_data('statistics')
    
    return jsonify({
        'total_users': stats['total_users'],
        'today_attendance': stats['today_attendance'],
        'total_detections': cached_stats['total_detections'] if cached_stats else 0,
        'active_clients': len(cached_stats['active_clients']) if cached_stats else 0
    })

@app.route('/admin/api/attendance-events')
def get_api_attendance_events():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    events = list(get_cached_data('attendance_events') or [])
    return jsonify(events[-50:])  # 최근 50개

@app.route('/admin/api/users')
def get_api_users():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    users = get_all_users()
    return jsonify(users)

@app.route('/')
def index():
    return redirect(url_for('admin_login'))

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
