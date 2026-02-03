from sqlalchemy import Column, Integer, String, DateTime, func, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Student(Base):
    __tablename__ = 'students'
    id = Column(Integer, primary_key=True)
    student_id = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

class AttendanceEvent(Base):
    __tablename__ = 'attendance_events'
    id = Column(Integer, primary_key=True)
    student_id = Column(String, nullable=True)
    timestamp = Column(DateTime, server_default=func.now())
    source = Column(String)

class FaceData(Base):
    __tablename__ = 'face_data'
    id = Column(Integer, primary_key=True)
    student_id = Column(String, nullable=False)
    image_path = Column(String, nullable=False)
    embedding = Column(Text)
    created_at = Column(DateTime, server_default=func.now())

class GestureData(Base):
    __tablename__ = 'gesture_data'
    id = Column(Integer, primary_key=True)
    student_id = Column(String, nullable=False)
    gesture_type = Column(String)
    gesture_json = Column(Text)
    created_at = Column(DateTime, server_default=func.now())

class VoiceData(Base):
    __tablename__ = 'voice_data'
    id = Column(Integer, primary_key=True)
    student_id = Column(String, nullable=False)
    audio_path = Column(String, nullable=False)
    embedding = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
