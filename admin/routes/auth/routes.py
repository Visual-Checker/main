from flask import request, jsonify
import bcrypt
import jwt
import os
from datetime import datetime, timedelta
from routes.auth import bp
from database import get_db

@bp.route('/login', methods=['POST'])
def login():
    """로그인"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': '사용자명과 비밀번호를 입력하세요'}), 400
    
    db = get_db()
    user = db.fetch_one(
        "SELECT * FROM users WHERE username = %s",
        (username,)
    )
    db.disconnect()
    
    if not user:
        return jsonify({'error': '사용자를 찾을 수 없습니다'}), 401
    
    # 비밀번호 확인 (현재는 단순 비교, 실제로는 bcrypt 사용)
    if user['password_hash'] != password:  # 실제 환경에서는 bcrypt.checkpw() 사용
        return jsonify({'error': '비밀번호가 일치하지 않습니다'}), 401
    
    # JWT 토큰 생성
    token_payload = {
        'user_id': user['user_id'],
        'username': user['username'],
        'role': user['role'],
        'exp': datetime.utcnow() + timedelta(hours=24)
    }
    
    token = jwt.encode(
        token_payload,
        os.getenv('JWT_SECRET_KEY', 'dev-jwt-secret'),
        algorithm='HS256'
    )
    
    return jsonify({
        'token': token,
        'user': {
            'user_id': user['user_id'],
            'username': user['username'],
            'full_name': user['full_name'],
            'email': user['email'],
            'role': user['role']
        }
    })

@bp.route('/register', methods=['POST'])
def register():
    """회원가입"""
    data = request.get_json()
    
    required_fields = ['username', 'password', 'full_name', 'email', 'role']
    for field in required_fields:
        if not data.get(field):
            return jsonify({'error': f'{field}를 입력하세요'}), 400
    
    db = get_db()
    
    # 중복 확인
    existing_user = db.fetch_one(
        "SELECT user_id FROM users WHERE username = %s OR email = %s",
        (data['username'], data['email'])
    )
    
    if existing_user:
        db.disconnect()
        return jsonify({'error': '이미 존재하는 사용자명 또는 이메일입니다'}), 409
    
    # 사용자 생성 (실제로는 bcrypt로 비밀번호 해시화)
    success = db.execute(
        """INSERT INTO users (username, password_hash, full_name, email, role, student_number)
           VALUES (%s, %s, %s, %s, %s, %s)""",
        (data['username'], data['password'], data['full_name'], 
         data['email'], data['role'], data.get('student_number'))
    )
    
    db.disconnect()
    
    if success:
        return jsonify({'message': '회원가입이 완료되었습니다'}), 201
    else:
        return jsonify({'error': '회원가입 중 오류가 발생했습니다'}), 500
