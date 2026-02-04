"""
로컬 데이터베이스 모듈 (SQLite)
Docker 없이 로컬에서 실행
"""

import sqlite3
from pathlib import Path
from datetime import datetime

# 데이터베이스 경로
DB_PATH = Path(__file__).parent.parent / 'data' / 'attendance.db'

def get_connection():
    """SQLite 연결 반환"""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row  # 딕셔너리처럼 접근 가능
    return conn

def init_database():
    """데이터베이스 초기화"""
    DB_PATH.parent.mkdir(exist_ok=True)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # 사용자 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            face_encoding BLOB,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 출석 기록 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            check_in_method TEXT,
            gesture_type TEXT,
            confidence REAL,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # 제스처 등록 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gesture_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            gesture_type TEXT,
            template_data BLOB,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # 목소리 등록 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS voice_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            audio_data BLOB,
            features BLOB,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # 기본 관리자 추가
    cursor.execute('''
        INSERT OR IGNORE INTO users (id, name) VALUES (1, 'admin')
    ''')
    
    conn.commit()
    conn.close()
    
    print(f"✓ 데이터베이스 생성: {DB_PATH}")

def add_user(name, face_encoding=None):
    """사용자 추가"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        'INSERT INTO users (name, face_encoding) VALUES (?, ?)',
        (name, face_encoding)
    )
    
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return user_id

def get_user_by_name(name):
    """이름으로 사용자 조회"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM users WHERE name = ?', (name,))
    user = cursor.fetchone()
    
    conn.close()
    return dict(user) if user else None

def get_all_users():
    """모든 사용자 조회"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, name, created_at FROM users WHERE id > 1 ORDER BY created_at DESC')
    users = cursor.fetchall()
    
    conn.close()
    return [dict(user) for user in users]

def add_attendance(user_id, check_in_method, gesture_type=None, confidence=None):
    """출석 기록 추가"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        '''INSERT INTO attendance_records 
           (user_id, check_in_method, gesture_type, confidence) 
           VALUES (?, ?, ?, ?)''',
        (user_id, check_in_method, gesture_type, confidence)
    )
    
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return record_id

def get_today_attendance():
    """오늘 출석 기록 조회"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT a.*, u.name
        FROM attendance_records a
        JOIN users u ON a.user_id = u.id
        WHERE DATE(a.timestamp) = DATE('now', 'localtime')
        ORDER BY a.timestamp DESC
    ''')
    
    records = cursor.fetchall()
    conn.close()
    
    return [dict(record) for record in records]

def get_statistics():
    """통계 조회"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 전체 사용자 수
    cursor.execute('SELECT COUNT(*) as total FROM users WHERE id > 1')
    total_users = cursor.fetchone()['total']
    
    # 오늘 출석 수
    cursor.execute('''
        SELECT COUNT(*) as total 
        FROM attendance_records 
        WHERE DATE(timestamp) = DATE('now', 'localtime')
    ''')
    today_attendance = cursor.fetchone()['total']
    
    # 전체 출석 기록 수
    cursor.execute('SELECT COUNT(*) as total FROM attendance_records')
    total_records = cursor.fetchone()['total']
    
    conn.close()
    
    return {
        'total_users': total_users,
        'today_attendance': today_attendance,
        'total_records': total_records
    }

if __name__ == '__main__':
    init_database()
    print("데이터베이스 초기화 완료!")
