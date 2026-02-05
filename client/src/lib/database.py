from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, Date, Time
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# ===== 로컬 데이터베이스 연결 (필요시) =====
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

# 모델 정의
class User(Base):
    __tablename__ = "users"
    
    user_id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    role = Column(String(20), nullable=False)
    student_number = Column(String(20), unique=True)
    face_encoding = Column(Text)  # 얼굴 인코딩 저장
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    enrollments = relationship("Enrollment", back_populates="student")
    attendance_records = relationship("AttendanceRecord", back_populates="student")

class Course(Base):
    __tablename__ = "courses"
    
    course_id = Column(Integer, primary_key=True)
    course_code = Column(String(20), unique=True, nullable=False)
    course_name = Column(String(100), nullable=False)
    professor_id = Column(Integer, ForeignKey('users.user_id'))
    semester = Column(String(20), nullable=False)
    year = Column(Integer, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    enrollments = relationship("Enrollment", back_populates="course")
    sessions = relationship("ClassSession", back_populates="course")

class Enrollment(Base):
    __tablename__ = "enrollments"
    
    enrollment_id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey('users.user_id'), nullable=False)
    course_id = Column(Integer, ForeignKey('courses.course_id'), nullable=False)
    enrolled_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    student = relationship("User", back_populates="enrollments")
    course = relationship("Course", back_populates="enrollments")

class ClassSession(Base):
    __tablename__ = "class_sessions"
    
    session_id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey('courses.course_id'), nullable=False)
    session_date = Column(Date, nullable=False)
    session_time = Column(Time, nullable=False)
    duration = Column(Integer, nullable=False)
    session_type = Column(String(20), default='lecture')
    location = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    course = relationship("Course", back_populates="sessions")
    attendance_records = relationship("AttendanceRecord", back_populates="session")

class AttendanceRecord(Base):
    __tablename__ = "attendance_records"
    
    record_id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('class_sessions.session_id'), nullable=False)
    student_id = Column(Integer, ForeignKey('users.user_id'), nullable=False)
    status = Column(String(20), nullable=False)
    check_in_time = Column(DateTime)
    check_in_method = Column(String(20))  # 'manual', 'face', 'qr'
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("ClassSession", back_populates="attendance_records")
    student = relationship("User", back_populates="attendance_records")

# Base 클래스
Base = declarative_base()

# 모델 정의
class User(Base):
    __tablename__ = "users"
    
    user_id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    role = Column(String(20), nullable=False)
    student_number = Column(String(20), unique=True)
    face_encoding = Column(Text)  # 얼굴 인코딩 저장
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    enrollments = relationship("Enrollment", back_populates="student")
    attendance_records = relationship("AttendanceRecord", back_populates="student")

class Course(Base):
    __tablename__ = "courses"
    
    course_id = Column(Integer, primary_key=True)
    course_code = Column(String(20), unique=True, nullable=False)
    course_name = Column(String(100), nullable=False)
    professor_id = Column(Integer, ForeignKey('users.user_id'))
    semester = Column(String(20), nullable=False)
    year = Column(Integer, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    enrollments = relationship("Enrollment", back_populates="course")
    sessions = relationship("ClassSession", back_populates="course")

class Enrollment(Base):
    __tablename__ = "enrollments"
    
    enrollment_id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey('users.user_id'), nullable=False)
    course_id = Column(Integer, ForeignKey('courses.course_id'), nullable=False)
    enrolled_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    student = relationship("User", back_populates="enrollments")
    course = relationship("Course", back_populates="enrollments")

class ClassSession(Base):
    __tablename__ = "class_sessions"
    
    session_id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey('courses.course_id'), nullable=False)
    session_date = Column(Date, nullable=False)
    session_time = Column(Time, nullable=False)
    duration = Column(Integer, nullable=False)
    session_type = Column(String(20), default='lecture')
    location = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    course = relationship("Course", back_populates="sessions")
    attendance_records = relationship("AttendanceRecord", back_populates="session")

class AttendanceRecord(Base):
    __tablename__ = "attendance_records"
    
    record_id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('class_sessions.session_id'), nullable=False)
    student_id = Column(Integer, ForeignKey('users.user_id'), nullable=False)
    status = Column(String(20), nullable=False)
    check_in_time = Column(DateTime)
    check_in_method = Column(String(20))  # 'manual', 'face', 'qr'
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("ClassSession", back_populates="attendance_records")
    student = relationship("User", back_populates="attendance_records")

# 데이터베이스 세션 가져오기
def get_db():
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        print(f"Database connection error: {e}")
        db.close()
        raise

# 연결 테스트
def test_connection():
    try:
        db = get_db()
        db.execute("SELECT 1")
        print("✓ Database connection successful!")
        db.close()
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False
