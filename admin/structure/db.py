import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv('DATABASE_URL') or (
    f"postgresql://{os.getenv('POSTGRES_USER','postgres')}:{os.getenv('POSTGRES_PASSWORD','example')}@"
    f"{os.getenv('POSTGRES_HOST','localhost')}:{os.getenv('POSTGRES_PORT','5432')}/{os.getenv('POSTGRES_DB','appdb')}"
)

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    from .models import Base
    Base.metadata.create_all(bind=engine)


def raw_table_count():
    # information_schema query
    with engine.connect() as conn:
        res = conn.execute(text("SELECT count(*) FROM information_schema.tables WHERE table_schema='public';"))
        return int(res.scalar() or 0)


def list_face_data(student_id=None, limit=100):
    from .models import FaceData
    with SessionLocal() as session:
        q = session.query(FaceData)
        if student_id:
            q = q.filter(FaceData.student_id == student_id)
        return q.order_by(FaceData.id.desc()).limit(limit).all()


def list_gesture_data(student_id=None, limit=100):
    from .models import GestureData
    with SessionLocal() as session:
        q = session.query(GestureData)
        if student_id:
            q = q.filter(GestureData.student_id == student_id)
        return q.order_by(GestureData.id.desc()).limit(limit).all()


def list_voice_data(student_id=None, limit=100):
    from .models import VoiceData
    with SessionLocal() as session:
        q = session.query(VoiceData)
        if student_id:
            q = q.filter(VoiceData.student_id == student_id)
        return q.order_by(VoiceData.id.desc()).limit(limit).all()
