from flask import Blueprint, request, jsonify
import redis
import os
from dotenv import load_dotenv
from face_service import get_face_service
import cv2
import numpy as np
import base64

load_dotenv()

bp = Blueprint('face', __name__, url_prefix='/api/face')

# Redis 연결
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    password=os.getenv('REDIS_PASSWORD', None),
    db=int(os.getenv('REDIS_DB', 0)),
    decode_responses=False
)

@bp.route('/detect', methods=['POST'])
def detect_faces():
    """
    얼굴 검출 API
    
    Request:
        - image: base64 encoded image or multipart file
        
    Response:
        {
            'success': bool,
            'faces': [{'bbox': [x,y,w,h], 'confidence': float}, ...],
            'count': int
        }
    """
    try:
        # 이미지 가져오기
        if 'image' in request.files:
            file = request.files['image']
            image_data = file.read()
        elif 'image' in request.json:
            # base64 디코딩
            image_b64 = request.json['image']
            image_data = base64.b64decode(image_b64)
        else:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        # 얼굴 검출
        face_service = get_face_service()
        result = face_service.detect_faces(image_data)
        
        return jsonify({
            'success': True,
            **result
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/encode', methods=['POST'])
def encode_face():
    """
    얼굴 인코딩 생성 (얼굴 임베딩)
    
    Request:
        - image: 이미지 데이터
        - bbox: [x, y, w, h] (옵션)
        
    Response:
        {
            'success': bool,
            'embedding': [float, ...],
            'length': int
        }
    """
    try:
        # 이미지 가져오기
        if 'image' in request.files:
            file = request.files['image']
            image_data = file.read()
        elif 'image' in request.json:
            image_b64 = request.json['image']
            image_data = base64.b64decode(image_b64)
        else:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        bbox = request.json.get('bbox') if request.json else None
        
        # 얼굴 인코딩
        face_service = get_face_service()
        embedding = face_service.encode_face(image_data, bbox)
        
        if embedding is None:
            return jsonify({
                'success': False,
                'error': 'No face detected or encoding not supported'
            }), 400
        
        return jsonify({
            'success': True,
            'embedding': embedding,
            'length': len(embedding)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/compare', methods=['POST'])
def compare_faces():
    """
    두 얼굴 비교
    
    Request:
        - embedding1: [float, ...]
        - embedding2: [float, ...]
        
    Response:
        {
            'success': bool,
            'similarity': float,  # 0-1
            'match': bool
        }
    """
    try:
        data = request.get_json()
        emb1 = np.array(data['embedding1'])
        emb2 = np.array(data['embedding2'])
        
        # 코사인 유사도
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        similarity = float(similarity)
        
        threshold = 0.6  # 임계값
        match = similarity > threshold
        
        return jsonify({
            'success': True,
            'similarity': similarity,
            'match': match,
            'threshold': threshold
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/cache/<key>', methods=['GET', 'POST', 'DELETE'])
def cache_operations(key):
    """
    Redis 캐시 조작
    
    GET: 캐시 조회
    POST: 캐시 저장 (body: {'value': ..., 'ttl': seconds})
    DELETE: 캐시 삭제
    """
    try:
        if request.method == 'GET':
            value = redis_client.get(key)
            if value is None:
                return jsonify({'success': False, 'error': 'Key not found'}), 404
            return jsonify({'success': True, 'value': value.decode()})
        
        elif request.method == 'POST':
            data = request.get_json()
            value = data['value']
            ttl = data.get('ttl', 3600)  # 기본 1시간
            
            redis_client.setex(key, ttl, value)
            return jsonify({'success': True, 'message': 'Cached successfully'})
        
        elif request.method == 'DELETE':
            redis_client.delete(key)
            return jsonify({'success': True, 'message': 'Deleted successfully'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/health', methods=['GET'])
def health_check():
    """서비스 상태 확인"""
    try:
        # Redis 체크
        redis_client.ping()
        redis_ok = True
    except:
        redis_ok = False
    
    # Face service 체크
    try:
        face_service = get_face_service()
        face_ok = face_service.detector is not None
        model_info = {
            'type': face_service.model_type,
            'backend': face_service.backend
        }
    except:
        face_ok = False
        model_info = {}
    
    return jsonify({
        'success': True,
        'redis': redis_ok,
        'face_detection': face_ok,
        'model': model_info
    })
