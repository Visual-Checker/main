from flask import request, jsonify
from datetime import datetime
from routes.attendance import bp
from database import get_db

@bp.route('/sessions', methods=['POST'])
def create_session():
    """강의 세션 생성"""
    data = request.get_json()
    
    required_fields = ['course_id', 'session_date', 'session_time', 'duration']
    for field in required_fields:
        if not data.get(field):
            return jsonify({'error': f'{field}를 입력하세요'}), 400
    
    db = get_db()
    success = db.execute(
        """INSERT INTO class_sessions (course_id, session_date, session_time, duration, session_type, location)
           VALUES (%s, %s, %s, %s, %s, %s)""",
        (data['course_id'], data['session_date'], data['session_time'],
         data['duration'], data.get('session_type', 'lecture'), data.get('location'))
    )
    db.disconnect()
    
    if success:
        return jsonify({'message': '강의 세션이 생성되었습니다'}), 201
    else:
        return jsonify({'error': '세션 생성 중 오류가 발생했습니다'}), 500

@bp.route('/check-in', methods=['POST'])
def check_in():
    """출석 체크"""
    data = request.get_json()
    
    required_fields = ['session_id', 'student_id', 'status']
    for field in required_fields:
        if not data.get(field):
            return jsonify({'error': f'{field}를 입력하세요'}), 400
    
    db = get_db()
    
    # 기존 출석 기록 확인
    existing = db.fetch_one(
        "SELECT record_id FROM attendance_records WHERE session_id = %s AND student_id = %s",
        (data['session_id'], data['student_id'])
    )
    
    if existing:
        # 업데이트
        success = db.execute(
            """UPDATE attendance_records 
               SET status = %s, check_in_time = %s, notes = %s
               WHERE session_id = %s AND student_id = %s""",
            (data['status'], datetime.now(), data.get('notes'),
             data['session_id'], data['student_id'])
        )
    else:
        # 신규 생성
        success = db.execute(
            """INSERT INTO attendance_records (session_id, student_id, status, check_in_time, notes)
               VALUES (%s, %s, %s, %s, %s)""",
            (data['session_id'], data['student_id'], data['status'],
             datetime.now(), data.get('notes'))
        )
    
    db.disconnect()
    
    if success:
        return jsonify({'message': '출석이 기록되었습니다'}), 201
    else:
        return jsonify({'error': '출석 기록 중 오류가 발생했습니다'}), 500

@bp.route('/student/<int:student_id>', methods=['GET'])
def get_student_attendance(student_id):
    """학생 출석 현황 조회"""
    course_id = request.args.get('course_id')
    
    db = get_db()
    if course_id:
        records = db.fetch_all(
            """SELECT ar.*, cs.session_date, cs.session_time, c.course_name
               FROM attendance_records ar
               JOIN class_sessions cs ON ar.session_id = cs.session_id
               JOIN courses c ON cs.course_id = c.course_id
               WHERE ar.student_id = %s AND cs.course_id = %s
               ORDER BY cs.session_date DESC, cs.session_time DESC""",
            (student_id, course_id)
        )
    else:
        records = db.fetch_all(
            """SELECT ar.*, cs.session_date, cs.session_time, c.course_name
               FROM attendance_records ar
               JOIN class_sessions cs ON ar.session_id = cs.session_id
               JOIN courses c ON cs.course_id = c.course_id
               WHERE ar.student_id = %s
               ORDER BY cs.session_date DESC, cs.session_time DESC""",
            (student_id,)
        )
    db.disconnect()
    
    return jsonify(records)

@bp.route('/session/<int:session_id>', methods=['GET'])
def get_session_attendance(session_id):
    """세션별 출석 현황 조회"""
    db = get_db()
    records = db.fetch_all(
        """SELECT ar.*, u.full_name, u.student_number
           FROM attendance_records ar
           JOIN users u ON ar.student_id = u.user_id
           WHERE ar.session_id = %s""",
        (session_id,)
    )
    db.disconnect()
    
    return jsonify(records)
