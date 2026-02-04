from flask import request, jsonify
from routes.users import bp
from database import get_db

@bp.route('/', methods=['GET'])
def get_users():
    """전체 사용자 조회"""
    role = request.args.get('role')
    
    db = get_db()
    if role:
        users = db.fetch_all(
            "SELECT user_id, username, full_name, email, role, student_number FROM users WHERE role = %s",
            (role,)
        )
    else:
        users = db.fetch_all(
            "SELECT user_id, username, full_name, email, role, student_number FROM users"
        )
    db.disconnect()
    
    return jsonify(users)

@bp.route('/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """특정 사용자 조회"""
    db = get_db()
    user = db.fetch_one(
        "SELECT user_id, username, full_name, email, role, student_number FROM users WHERE user_id = %s",
        (user_id,)
    )
    db.disconnect()
    
    if user:
        return jsonify(user)
    else:
        return jsonify({'error': '사용자를 찾을 수 없습니다'}), 404
