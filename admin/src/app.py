from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()

# Flask 앱 생성
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
app.config['SESSION_TYPE'] = 'filesystem'

# CORS 설정
CORS(app, resources={r"/api/*": {"origins": "*"}})

# 라우트 import
from routes import auth, users, courses, attendance
from routes.face import bp as face_bp
from routes.admin import admin_bp

# 블루프린트 등록
app.register_blueprint(auth.bp)
app.register_blueprint(users.bp)
app.register_blueprint(courses.bp)
app.register_blueprint(attendance.bp)
app.register_blueprint(face_bp)
app.register_blueprint(admin_bp)

@app.route('/')
def index():
    return jsonify({
        'message': '출결관리 시스템 API 서버',
        'version': '1.0.0',
        'status': 'running'
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True') == 'True'
    
    app.run(host=host, port=port, debug=debug)
