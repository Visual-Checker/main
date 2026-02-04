"""
VectorDB ORM (pgvector) 연결 유틸
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, DateTime, text
from sqlalchemy.orm import sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector

load_dotenv()

VECTOR_DB_HOST = os.getenv("VECTOR_DB_HOST", "192.168.0.41")
VECTOR_DB_PORT = int(os.getenv("VECTOR_DB_PORT", "5432"))
VECTOR_DB_NAME = os.getenv("VECTOR_DB_NAME", "VectorDB")
VECTOR_DB_USER = os.getenv("VECTOR_DB_USER", "orugu")
VECTOR_DB_PASSWORD = os.getenv("VECTOR_DB_PASSWORD", "orugu#0916")

VECTOR_DB_URL = (
    f"postgresql://{VECTOR_DB_USER}:{VECTOR_DB_PASSWORD}"
    f"@{VECTOR_DB_HOST}:{VECTOR_DB_PORT}/{VECTOR_DB_NAME}"
)

engine = create_engine(VECTOR_DB_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class FaceRecog(Base):
    __tablename__ = "face_recog_db"

    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, nullable=False, index=True)
    embedding = Column(Vector, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class HandGesture(Base):
    __tablename__ = "hand_gesture_db"

    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, nullable=False, index=True)
    embedding = Column(Vector, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class VoiceRecog(Base):
    __tablename__ = "voice_recog_db"

    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, nullable=False, index=True)
    embedding = Column(Vector, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def _upsert_by_label(session, model_cls, label, embedding):
    session.query(model_cls).filter(model_cls.label == label).delete()
    session.add(model_cls(label=label, embedding=embedding))
    session.commit()


def create_entry(model_cls, label, embedding):
    with SessionLocal() as session:
        session.add(model_cls(label=label, embedding=embedding))
        session.commit()


def list_entries(model_cls):
    with SessionLocal() as session:
        rows = session.query(model_cls).order_by(model_cls.id.desc()).all()
        return [(r.id, r.label, r.created_at) for r in rows]


def update_label(model_cls, record_id, new_label):
    with SessionLocal() as session:
        row = session.query(model_cls).filter(model_cls.id == record_id).first()
        if not row:
            return False
        row.label = new_label
        session.commit()
        return True


def delete_entry(model_cls, record_id):
    with SessionLocal() as session:
        row = session.query(model_cls).filter(model_cls.id == record_id).first()
        if not row:
            return False
        session.delete(row)
        session.commit()
        return True


def upsert_face(label, embedding):
    with SessionLocal() as session:
        _upsert_by_label(session, FaceRecog, label, embedding)


def upsert_gesture(label, embedding):
    with SessionLocal() as session:
        _upsert_by_label(session, HandGesture, label, embedding)


def upsert_voice(label, embedding):
    with SessionLocal() as session:
        _upsert_by_label(session, VoiceRecog, label, embedding)


def search_face(embedding, limit=1):
    """코사인 유사도 기반 얼굴 검색"""
    sql = text(
        "SELECT label, (1 - (embedding <=> :emb)) AS similarity "
        "FROM face_recog_db ORDER BY embedding <=> :emb LIMIT :limit"
    )
    with SessionLocal() as session:
        rows = session.execute(sql, {"emb": embedding, "limit": limit}).all()
        return [(r[0], float(r[1])) for r in rows]


def search_voice(embedding, limit=1):
    """코사인 유사도 기반 음성 검색"""
    sql = text(
        "SELECT label, (1 - (embedding <=> :emb)) AS similarity "
        "FROM voice_recog_db ORDER BY embedding <=> :emb LIMIT :limit"
    )
    with SessionLocal() as session:
        rows = session.execute(sql, {"emb": embedding, "limit": limit}).all()
        return [(r[0], float(r[1])) for r in rows]


def search_gesture(embedding, limit=1):
    """코사인 유사도 기반 제스처 검색"""
    sql = text(
        "SELECT label, (1 - (embedding <=> :emb)) AS similarity "
        "FROM hand_gesture_db ORDER BY embedding <=> :emb LIMIT :limit"
    )
    with SessionLocal() as session:
        rows = session.execute(sql, {"emb": embedding, "limit": limit}).all()
        return [(r[0], float(r[1])) for r in rows]
