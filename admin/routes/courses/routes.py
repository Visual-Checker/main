from flask import request, jsonify
from routes.courses import bp
from database import get_db

@bp.route('/', methods=['GET'])
def get_courses():
    """전체 과목 조회"""
    db = get_db()
    courses = db.fetch_all(
        """SELECT c.*, u.full_name as professor_name 
           FROM courses c 
           LEFT JOIN users u ON c.professor_id = u.user_id"""
    )
    db.disconnect()
    
    return jsonify(courses)

@bp.route('/', methods=['POST'])
def create_course():
    """과목 생성"""
    data = request.get_json()
    
    required_fields = ['course_code', 'course_name', 'semester', 'year']
    for field in required_fields:
        if not data.get(field):
            return jsonify({'error': f'{field}를 입력하세요'}), 400
    
    db = get_db()
    success = db.execute(
        """INSERT INTO courses (course_code, course_name, professor_id, semester, year, description)
           VALUES (%s, %s, %s, %s, %s, %s)""",
        (data['course_code'], data['course_name'], data.get('professor_id'),
         data['semester'], data['year'], data.get('description'))
    )
    db.disconnect()
    
    if success:
        return jsonify({'message': '과목이 생성되었습니다'}), 201
    else:
        return jsonify({'error': '과목 생성 중 오류가 발생했습니다'}), 500

@bp.route('/<int:course_id>/students', methods=['GET'])
def get_course_students(course_id):
    """과목 수강생 조회"""
    db = get_db()
    students = db.fetch_all(
        """SELECT u.user_id, u.username, u.full_name, u.student_number, e.enrolled_at
           FROM enrollments e
           JOIN users u ON e.student_id = u.user_id
           WHERE e.course_id = %s""",
        (course_id,)
    )
    db.disconnect()
    
    return jsonify(students)
