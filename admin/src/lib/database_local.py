"""
로컬 데이터베이스 모듈 (PostgreSQL + SQLAlchemy)
Docker 없이 로컬에서 실행
"""

from datetime import datetime
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, LargeBinary, Float, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

load_dotenv()

# 로컬 PostgreSQL 연결 문자열
LOCAL_DATABASE_URL = (
    f"postgresql://{os.getenv('DB_USER', 'admin')}:"
    f"{os.getenv('DB_PASSWORD', 'admin123')}@"
    f"{os.getenv('DB_HOST', 'localhost')}:"
    f"{os.getenv('DB_PORT', 5432)}/"
    f"{os.getenv('DB_NAME', 'attendance_system')}"
)

engine = create_engine(LOCAL_DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class LocalUser(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    face_encoding = Column(LargeBinary)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    attendance_records = relationship('AttendanceRecord', back_populates='user')
    gesture_templates = relationship('GestureTemplate', back_populates='user')
    voice_templates = relationship('VoiceTemplate', back_populates='user')

class AttendanceRecord(Base):
    __tablename__ = 'attendance_records'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    check_in_method = Column(String(50))
    gesture_type = Column(String(50))
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship('LocalUser', back_populates='attendance_records')

class GestureTemplate(Base):
    __tablename__ = 'gesture_templates'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    gesture_type = Column(String(50))
    template_data = Column(LargeBinary)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship('LocalUser', back_populates='gesture_templates')

class VoiceTemplate(Base):
    __tablename__ = 'voice_templates'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    audio_data = Column(LargeBinary)
    features = Column(LargeBinary)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship('LocalUser', back_populates='voice_templates')

def get_session():
    """SQLAlchemy 세션 반환"""
    return SessionLocal()

def init_database():
    """데이터베이스 초기화"""
    Base.metadata.create_all(bind=engine)

    session = get_session()
    try:
        admin_user = session.query(LocalUser).filter(LocalUser.id == 1).first()
        if not admin_user:
            session.add(LocalUser(id=1, name='admin'))
            session.commit()
    finally:
        session.close()

    print("✓ PostgreSQL 데이터베이스 준비 완료")

def _user_to_dict(user):
    if not user:
        return None
    return {
        'id': user.id,
        'name': user.name,
        'face_encoding': user.face_encoding,
        'created_at': user.created_at.isoformat() if user.created_at else None,
        'updated_at': user.updated_at.isoformat() if user.updated_at else None
    }

def add_user(name, face_encoding=None):
    """사용자 추가"""
    session = get_session()
    try:
        user = LocalUser(name=name, face_encoding=face_encoding)
        session.add(user)
        session.commit()
        session.refresh(user)
        return user.id
    finally:
        session.close()

def get_user_by_name(name):
    """이름으로 사용자 조회"""
    session = get_session()
    try:
        user = session.query(LocalUser).filter(LocalUser.name == name).first()
        return _user_to_dict(user)
    finally:
        session.close()

def get_all_users():
    """모든 사용자 조회"""
    session = get_session()
    try:
        users = (
            session.query(LocalUser)
            .filter(LocalUser.id > 1)
            .order_by(LocalUser.created_at.desc())
            .all()
        )
        return [_user_to_dict(user) for user in users]
    finally:
        session.close()

def add_attendance(user_id, check_in_method, gesture_type=None, confidence=None):
    """출석 기록 추가"""
    session = get_session()
    try:
        record = AttendanceRecord(
            user_id=user_id,
            check_in_method=check_in_method,
            gesture_type=gesture_type,
            confidence=confidence
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record.id
    finally:
        session.close()

def get_today_attendance():
    """오늘 출석 기록 조회"""
    session = get_session()
    try:
        records = (
            session.query(AttendanceRecord, LocalUser)
            .join(LocalUser, AttendanceRecord.user_id == LocalUser.id)
            .filter(func.date(AttendanceRecord.timestamp) == func.current_date())
            .order_by(AttendanceRecord.timestamp.desc())
            .all()
        )

        result = []
        for record, user in records:
            result.append({
                'id': record.id,
                'user_id': record.user_id,
                'name': user.name,
                'check_in_method': record.check_in_method,
                'gesture_type': record.gesture_type,
                'confidence': record.confidence,
                'timestamp': record.timestamp.isoformat() if record.timestamp else None
            })

        return result
    finally:
        session.close()

def get_statistics():
    """통계 조회"""
    session = get_session()
    try:
        total_users = session.query(func.count(LocalUser.id)).filter(LocalUser.id > 1).scalar() or 0
        today_attendance = (
            session.query(func.count(AttendanceRecord.id))
            .filter(func.date(AttendanceRecord.timestamp) == func.current_date())
            .scalar()
            or 0
        )
        total_records = session.query(func.count(AttendanceRecord.id)).scalar() or 0

        return {
            'total_users': total_users,
            'today_attendance': today_attendance,
            'total_records': total_records
        }
    finally:
        session.close()

if __name__ == '__main__':
    init_database()
    print("데이터베이스 초기화 완료!")
