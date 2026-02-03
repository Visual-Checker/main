from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, Date, Time, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# ===== 로컬 데이터베이스 연결 =====
LOCAL_DATABASE_URL = f"postgresql://{os.getenv('DB_USER', 'admin')}:{os.getenv('DB_PASSWORD', 'admin123')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', 5432)}/{os.getenv('DB_NAME', 'attendance_system')}"

# ===== 서버 데이터베이스 연결 =====
SERVER_DATABASE_URL = f"postgresql://{os.getenv('SERVER_DB_USER', 'admin')}:{os.getenv('SERVER_DB_PASSWORD', 'admin123')}@{os.getenv('SERVER_DB_HOST', 'localhost')}:{os.getenv('SERVER_DB_PORT', 5432)}/{os.getenv('SERVER_DB_NAME', 'attendance_system')}"

# SQLAlchemy 엔진 생성
local_engine = create_engine(LOCAL_DATABASE_URL, echo=False, pool_pre_ping=True)
server_engine = create_engine(SERVER_DATABASE_URL, echo=False, pool_pre_ping=True)

# 세션 팩토리
LocalSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=local_engine)
ServerSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=server_engine)

# Base 클래스
Base = declarative_base()

# ===== 모델 정의 =====
class User(Base):
    __tablename__ = "users"
    
    user_id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    role = Column(String(20), nullable=False)  # 'student', 'professor', 'admin'
    student_number = Column(String(20), unique=True)
    face_encoding = Column(Text)  # 얼굴 인코딩 저장
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    enrollments = relationship("Enrollment", back_populates="student")
    attendance_records = relationship("AttendanceRecord", back_populates="student")
    courses_taught = relationship("Course", back_populates="professor")

class Course(Base):
    __tablename__ = "courses"
    
    course_id = Column(Integer, primary_key=True)
    course_code = Column(String(20), unique=True, nullable=False)
    course_name = Column(String(100), nullable=False)
    professor_id = Column(Integer, ForeignKey('users.user_id'))
    semester = Column(String(20), nullable=False)
    year = Column(Integer, nullable=False)
    description = Column(Text)
    max_students = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    professor = relationship("User", back_populates="courses_taught")
    enrollments = relationship("Enrollment", back_populates="course")
    sessions = relationship("ClassSession", back_populates="course")

class Enrollment(Base):
    __tablename__ = "enrollments"
    
    enrollment_id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey('users.user_id'), nullable=False)
    course_id = Column(Integer, ForeignKey('courses.course_id'), nullable=False)
    enrolled_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default='active')  # 'active', 'dropped', 'completed'
    
    # Relationships
    student = relationship("User", back_populates="enrollments")
    course = relationship("Course", back_populates="enrollments")

class ClassSession(Base):
    __tablename__ = "class_sessions"
    
    session_id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey('courses.course_id'), nullable=False)
    session_date = Column(Date, nullable=False)
    session_time = Column(Time, nullable=False)
    duration = Column(Integer, nullable=False)  # 분 단위
    session_type = Column(String(20), default='lecture')  # 'lecture', 'lab', 'seminar'
    location = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    course = relationship("Course", back_populates="sessions")
    attendance_records = relationship("AttendanceRecord", back_populates="session")

class AttendanceRecord(Base):
    __tablename__ = "attendance_records"
    
    record_id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('class_sessions.session_id'), nullable=False)
    student_id = Column(Integer, ForeignKey('users.user_id'), nullable=False)
    status = Column(String(20), nullable=False)  # 'present', 'absent', 'late', 'excused'
    check_in_time = Column(DateTime)
    check_in_method = Column(String(20))  # 'manual', 'face', 'qr'
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    session = relationship("ClassSession", back_populates="attendance_records")
    student = relationship("User", back_populates="attendance_records")

# ===== 유틸리티 함수 =====
def get_local_session():
    """로컬 DB 세션 반환"""
    return LocalSessionLocal()

def get_server_session():
    """서버 DB 세션 반환"""
    return ServerSessionLocal()

def close_session(session):
    """세션 종료"""
    if session:
        session.close()
