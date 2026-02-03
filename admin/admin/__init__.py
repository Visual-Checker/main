from flask import Blueprint, render_template, request, redirect, url_for, session
import os
import psycopg2
import redis
from dotenv import load_dotenv

load_dotenv()

admin_bp = Blueprint('admin', __name__, url_prefix='/admin', template_folder='templates')

# Helpers to connect to Redis and Postgres

def get_redis_client():
    try:
        url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        r = redis.from_url(url, decode_responses=True)
        r.ping()
        return r
    except Exception:
        return None


def get_db_conn():
    try:
        dsn = os.getenv('DATABASE_URL', None)
        if dsn is None:
            user = os.getenv('POSTGRES_USER', 'postgres')
            password = os.getenv('POSTGRES_PASSWORD', 'example')
            host = os.getenv('POSTGRES_HOST', 'localhost')
            port = os.getenv('POSTGRES_PORT', '5432')
            dbname = os.getenv('POSTGRES_DB', 'appdb')
            dsn = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        conn = psycopg2.connect(dsn)
        return conn
    except Exception:
        return None


@admin_bp.route('/')
def index():
    if 'user' not in session or session.get('role') != 'admin':
        return redirect(url_for('admin.login'))

    stats = {
        'tables': 0,
        'students': None,
        'attendance_events': 0,
        'total_students': 0,
    }

    r = get_redis_client()
    if r:
        try:
            stats['attendance_events'] = r.llen('attendance_events')
            keys = r.keys('stats:student:*')
            stats['total_students'] = len(keys)
        except Exception:
            pass

    conn = get_db_conn()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema='public';")
            stats['tables'] = cur.fetchone()[0]

            # check for students table
            cur.execute("SELECT to_regclass('public.students') IS NOT NULL;")
            exists = cur.fetchone()[0]
            if exists:
                cur.execute('SELECT count(*) FROM students;')
                stats['students'] = cur.fetchone()[0]
            cur.close()
        except Exception:
            pass
        finally:
            conn.close()

    return render_template('dashboard.html', stats=stats, user=session.get('user'))


@admin_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'admin123':
            session['user'] = username
            session['role'] = 'admin'
            return redirect(url_for('admin.index'))
        return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')


@admin_bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('admin.login'))
