"""
Admin 음성 인식 라우터
"""

from flask import Blueprint, request, jsonify
import os
from dotenv import load_dotenv
from voice_service import VoiceService

load_dotenv()

bp = Blueprint('voice', __name__, url_prefix='/api/voice')
voice_service = VoiceService()


@bp.route('/recognize', methods=['POST'])
def recognize_voice():
    """음성 파일 인식"""
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': '음성 파일이 없습니다'}), 400
        
        audio_file = request.files['audio']
        temp_path = f"/tmp/{audio_file.filename}"
        audio_file.save(temp_path)
        
        # 음성 인식
        name, similarity = voice_service.recognize_voice(temp_path)
        
        # 임시 파일 삭제
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'name': name,
            'similarity': float(similarity),
            'threshold': voice_service.voice_similarity_threshold
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/register', methods=['POST'])
def register_voice():
    """음성 데이터 등록"""
    try:
        if 'audio' not in request.files or 'name' not in request.form:
            return jsonify({'success': False, 'error': '음성 파일 또는 이름이 없습니다'}), 400
        
        audio_file = request.files['audio']
        name = request.form['name']
        
        temp_path = f"/tmp/{audio_file.filename}"
        audio_file.save(temp_path)
        
        # 음성 등록
        success = voice_service.register_voice(temp_path, name)
        
        # 임시 파일 삭제
        os.remove(temp_path)
        
        # 데이터 저장
        voice_service.save_voice_data()
        
        return jsonify({
            'success': success,
            'message': f'{name}의 음성 데이터가 등록되었습니다.' if success else '등록 실패'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/health', methods=['GET'])
def health_check():
    """서비스 상태 확인"""
    return jsonify({
        'success': True,
        'status': 'active',
        'model': 'speechbrain/spkrec-ecapa-voxceleb',
        'registered_speakers': len(voice_service.known_voice_names)
    })
