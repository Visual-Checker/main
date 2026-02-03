import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

class Database:
    def __init__(self):
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """데이터베이스 연결"""
        try:
            self.conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                port=os.getenv('DB_PORT', 5432),
                database=os.getenv('DB_NAME', 'attendance_system'),
                user=os.getenv('DB_USER', 'admin'),
                password=os.getenv('DB_PASSWORD', 'admin123')
            )
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            return True
        except Exception as e:
            print(f"데이터베이스 연결 오류: {e}")
            return False
    
    def disconnect(self):
        """데이터베이스 연결 해제"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
    
    def execute(self, query, params=None):
        """쿼리 실행"""
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"쿼리 실행 오류: {e}")
            return False
    
    def fetch_one(self, query, params=None):
        """단일 결과 조회"""
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            return self.cursor.fetchone()
        except Exception as e:
            print(f"조회 오류: {e}")
            return None
    
    def fetch_all(self, query, params=None):
        """다중 결과 조회"""
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            return self.cursor.fetchall()
        except Exception as e:
            print(f"조회 오류: {e}")
            return []

def get_db():
    """데이터베이스 연결 가져오기"""
    db = Database()
    db.connect()
    return db
