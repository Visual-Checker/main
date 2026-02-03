import os
import redis
from .db import SessionLocal, raw_table_count, init_db
from .models import Student, AttendanceEvent
from sqlalchemy.exc import IntegrityError

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')


def get_table_count():
    try:
        return raw_table_count()
    except Exception:
        return None


def get_students_count():
    try:
        with SessionLocal() as session:
            return session.query(Student).count()
    except Exception:
        return None


def list_students(limit=100):
    try:
        with SessionLocal() as session:
            students = session.query(Student).order_by(Student.id.desc()).limit(limit).all()
            return [{'id': s.id, 'student_id': s.student_id, 'name': s.name} for s in students]
    except Exception:
        return []


def add_student(student_id, name):
    try:
        with SessionLocal() as session:
            s = Student(student_id=student_id, name=name)
            session.add(s)
            session.commit()
            return True
    except IntegrityError:
        return False
    except Exception:
        return False


def add_face_data(student_id, image_path, embedding):
    try:
        from .db import SessionLocal
        from .models import FaceData
        with SessionLocal() as session:
            f = FaceData(student_id=student_id, image_path=image_path, embedding=embedding)
            session.add(f)
            session.commit()
            return True
    except Exception:
        return False


def list_face_data(student_id=None, limit=100):
    try:
        from .db import list_face_data
        return list_face_data(student_id, limit)
    except Exception:
        return []


def add_gesture_data(student_id, gesture_type, gesture_json):
    try:
        from .db import SessionLocal
        from .models import GestureData
        with SessionLocal() as session:
            g = GestureData(student_id=student_id, gesture_type=gesture_type, gesture_json=gesture_json)
            session.add(g)
            session.commit()
            return True
    except Exception:
        return False


def add_voice_data(student_id, audio_path, embedding):
    try:
        from .db import SessionLocal
        from .models import VoiceData
        with SessionLocal() as session:
            v = VoiceData(student_id=student_id, audio_path=audio_path, embedding=embedding)
            session.add(v)
            session.commit()
            return True
    except Exception:
        return False


def list_voice_data(student_id=None, limit=100):
    try:
        from .db import list_voice_data
        return list_voice_data(student_id, limit)
    except Exception:
        return []


# Enrollment helpers: save files under admin/data and create DB records
def _ensure_data_dirs():
    base = os.path.join(os.path.dirname(__file__), '..', 'data')
    base = os.path.abspath(base)
    os.makedirs(os.path.join(base, 'faces'), exist_ok=True)
    os.makedirs(os.path.join(base, 'gestures'), exist_ok=True)
    os.makedirs(os.path.join(base, 'voices'), exist_ok=True)
    return base


def enroll_face_from_file(student_id, src_image_path):
    """Compute embedding for image and store the file + embedding in DB"""
    try:
        base = _ensure_data_dirs()
        # dst dir
        dst_dir = os.path.join(base, 'faces', student_id)
        os.makedirs(dst_dir, exist_ok=True)
        import shutil, time, json
        filename = f"{student_id}_{int(time.time())}.jpg"
        dst = os.path.join(dst_dir, filename)
        shutil.copyfile(src_image_path, dst)

        # compute embedding
        from .recognition import FaceEmbedder
        fe = FaceEmbedder()
        emb = fe.embed(dst)
        if emb is None:
            return False, 'Face not detected or embedding failed'
        emb_json = json.dumps(emb.tolist())
        ok = add_face_data(student_id, dst, emb_json)
        return ok, None if ok else 'DB insert failed'
    except Exception as e:
        return False, str(e)


def enroll_gesture_from_file(student_id, src_image_path, gesture_type='unknown'):
    try:
        base = _ensure_data_dirs()
        dst_dir = os.path.join(base, 'gestures', student_id)
        os.makedirs(dst_dir, exist_ok=True)
        import shutil, time, json
        filename = f"{student_id}_{gesture_type}_{int(time.time())}.jpg"
        dst = os.path.join(dst_dir, filename)
        shutil.copyfile(src_image_path, dst)

        from .recognition import GestureExtractor
        ge = GestureExtractor()
        feats = ge.extract(dst)
        if feats is None:
            return False, 'Gesture not detected'
        gesture_json = json.dumps(feats.tolist())
        ok = add_gesture_data(student_id, gesture_type, gesture_json)
        return ok, None if ok else 'DB insert failed'
    except Exception as e:
        return False, str(e)


def enroll_voice_from_file(student_id, src_wav_path):
    try:
        base = _ensure_data_dirs()
        dst_dir = os.path.join(base, 'voices', student_id)
        os.makedirs(dst_dir, exist_ok=True)
        import shutil, time, json
        filename = f"{student_id}_{int(time.time())}.wav"
        dst = os.path.join(dst_dir, filename)
        shutil.copyfile(src_wav_path, dst)

        from .recognition import VoiceEmbedder
        ve = VoiceEmbedder()
        emb = ve.embed_from_file(dst)
        if emb is None:
            return False, 'Embedding failed (model missing?)'
        emb_json = json.dumps(emb.tolist())
        ok = add_voice_data(student_id, dst, emb_json)
        return ok, None if ok else 'DB insert failed'
    except Exception as e:
        return False, str(e)


def delete_student_by_student_id(student_id):
    try:
        with SessionLocal() as session:
            s = session.query(Student).filter_by(student_id=student_id).first()
            if s:
                session.delete(s)
                session.commit()
                return True
            return False
    except Exception:
        return False


def get_attendance_events_count():
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True)
        return r.llen('attendance_events')
    except Exception:
        return None


def seed_sample_data():
    """Create tables and insert sample students and attendance events"""
    init_db()
    sample_students = [
        ('2024001', '홍길동'),
        ('2024002', '이영희'),
        ('2024003', '김철수')
    ]
    for sid, name in sample_students:
        add_student(sid, name)

    # add some attendance events to redis
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True)
        r.lpush('attendance_events', *[{'student_id': sid, 'timestamp': 0} for sid, _ in sample_students])
    except Exception:
        pass
