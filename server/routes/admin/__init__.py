from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
import redis
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import logfire
import psycopg2

logfire.configure(send_to_logfire=False)

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

# PostgreSQL log store
POSTGRES_LOGS = []


class PostgresLogEntry(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    level: str
    message: str
    host: str | None = None
    port: int | None = None
    dbname: str | None = None
    user: str | None = None
    detail: str | None = None


def _get_db_config():
    return {
        'host': os.getenv('DB_HOST'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'dbname': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD')
    }


def _add_postgres_log(entry: PostgresLogEntry):
    POSTGRES_LOGS.append(entry.model_dump())
    if len(POSTGRES_LOGS) > 200:
        POSTGRES_LOGS.pop(0)
    payload = entry.model_dump()
    if entry.level.upper() == 'ERROR':
        logfire.error(entry.message, **payload)
    elif entry.level.upper() == 'WARNING':
        logfire.warning(entry.message, **payload)
    else:
        logfire.info(entry.message, **payload)

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


@admin_bp.route('/api/postgres-logs')
def get_postgres_logs():
    """최근 PostgreSQL 로그를 반환"""
    return jsonify(POSTGRES_LOGS[-100:])


@admin_bp.route('/api/postgres-ping')
def postgres_ping():
    """PostgreSQL 연결을 확인하고 로그를 남김"""
    cfg = _get_db_config()
    try:
        with psycopg2.connect(
            host=cfg['host'],
            port=cfg['port'],
            dbname=cfg['dbname'],
            user=cfg['user'],
            password=cfg['password'],
            connect_timeout=3
        ) as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT version()')
                version = cur.fetchone()[0]
        _add_postgres_log(
            PostgresLogEntry(
                level='INFO',
                message='PostgreSQL connection OK',
                host=cfg['host'],
                port=cfg['port'],
                dbname=cfg['dbname'],
                user=cfg['user'],
                detail=version
            )
        )
        return jsonify({'status': 'ok', 'version': version})
    except Exception as e:
        _add_postgres_log(
            PostgresLogEntry(
                level='ERROR',
                message='PostgreSQL connection failed',
                host=cfg['host'],
                port=cfg['port'],
                dbname=cfg['dbname'],
                user=cfg['user'],
                detail=str(e)
            )
        )
        return jsonify({'status': 'error', 'message': str(e)}), 500

@admin_bp.route('/api/students')
def get_students():
    students = [
        {'id': '2024001', 'name': '홍길동', 'status': 'present'},
        {'id': '2024002', 'name': '이영희', 'status': 'absent'},
    ]
    return jsonify(students)

@admin_bp.route('/realtime')
@admin_bp.route('/realtime.html')
def realtime_monitor():
    if 'user' not in session:
        return redirect(url_for('admin.login'))
    return render_template('realtime.html')

@admin_bp.route('/students')
@admin_bp.route('/students.html')
def students():
    if 'user' not in session:
        return redirect(url_for('admin.login'))
    return render_template('students.html')

@admin_bp.route('/reports')
@admin_bp.route('/reports.html')
def reports():
    if 'user' not in session:
        return redirect(url_for('admin.login'))
    return render_template('reports.html')


@admin_bp.route('/pydantic')
@admin_bp.route('/pydantic.html')
def pydantic_logs():
    if 'user' not in session:
        return redirect(url_for('admin.login'))
    return render_template('pydantic_logs.html')
