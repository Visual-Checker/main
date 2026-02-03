"""
음성 인식 모듈 - SpeechBrain ECAPA-TDNN 기반
"""

import os
import pickle
import numpy as np
from dotenv import load_dotenv

load_dotenv()

SPEECHBRAIN_AVAILABLE = False
try:
    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except Exception:
    SPEECHBRAIN_AVAILABLE = False


class VoiceRecognizer:
    """음성 인식 및 스피커 식별"""
    
    def __init__(self, model_path="../models/spkrec-ecapa-voxceleb"):
        """
        Args:
            model_path: 모델 저장 경로
        """
        self.voice_encoder = None
        self.known_voice_embeddings = []
        self.known_voice_names = []
        self.voice_similarity_threshold = float(os.getenv('VOICE_SIMILARITY_THRESHOLD', 0.7))
        self.model_path = model_path
        
        if SPEECHBRAIN_AVAILABLE:
            try:
                self.voice_encoder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=model_path
                )
                print("✓ 음성 인식 모델 초기화 성공")
            except Exception as e:
                print(f"⚠️  음성 인식 모델 초기화 실패: {e}")
    
    def load_voice_data(self, voice_data_file="../data/voice_data.pkl"):
        """저장된 음성 데이터 로드"""
        if os.path.exists(voice_data_file):
            try:
                with open(voice_data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_voice_embeddings = data.get('embeddings', [])
                    self.known_voice_names = data.get('names', [])
                print(f"✓ {len(self.known_voice_names)}명의 음성 데이터 로드됨")
            except Exception as e:
                print(f"⚠️  음성 데이터 로드 실패: {e}")
        else:
            print("ℹ️  등록된 음성 데이터가 없습니다.")
    
    def save_voice_data(self, voice_data_file="../data/voice_data.pkl"):
        """음성 임베딩 저장"""
        voice_data_dir = os.path.dirname(voice_data_file)
        os.makedirs(voice_data_dir, exist_ok=True)
        
        data = {
            'embeddings': self.known_voice_embeddings,
            'names': self.known_voice_names
        }
        
        try:
            with open(voice_data_file, 'wb') as f:
                pickle.dump(data, f)
            print("✓ 음성 데이터 저장됨")
        except Exception as e:
            print(f"⚠️  음성 데이터 저장 실패: {e}")
    
    def extract_voice_embedding(self, audio_file):
        """음성 파일에서 임베딩 추출"""
        if not SPEECHBRAIN_AVAILABLE or self.voice_encoder is None:
            print("ℹ️  SpeechBrain이 준비되지 않았습니다.")
            return None
        
        try:
            signal, sr = torchaudio.load(audio_file)
            
            # 샘플레이트 변환 (16kHz)
            if sr != 16000:
                resample = torchaudio.transforms.Resample(sr, 16000)
                signal = resample(signal)
            
            # 임베딩 추출
            emb = self.voice_encoder.encode_batch(signal)
            embedding = emb.detach().cpu().numpy().flatten()
            
            return embedding
            
        except Exception as e:
            print(f"⚠️  음성 임베딩 추출 실패: {e}")
            return None
    
    def recognize_voice(self, audio_file):
        """음성 파일 인식 (등록된 음성과 비교)"""
        embedding = self.extract_voice_embedding(audio_file)
        
        if embedding is None:
            return None, 0.0
        
        best_match_name = "Unknown"
        best_similarity = 0.0
        
        # 저장된 음성과 비교
        for known_emb, known_name in zip(self.known_voice_embeddings, self.known_voice_names):
            similarity = self.cosine_similarity(embedding, np.array(known_emb))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_name = known_name
        
        # 임계값 확인
        if best_similarity < self.voice_similarity_threshold:
            best_match_name = "Unknown"
            best_similarity = 0.0
        
        return best_match_name, best_similarity
    
    def register_voice(self, audio_file, name):
        """새로운 음성 데이터 등록"""
        embedding = self.extract_voice_embedding(audio_file)
        
        if embedding is None:
            return False
        
        # 중복 확인
        if name in self.known_voice_names:
            idx = self.known_voice_names.index(name)
            self.known_voice_embeddings[idx] = embedding.tolist()
            print(f"✓ {name}의 음성 데이터 업데이트됨")
        else:
            self.known_voice_embeddings.append(embedding.tolist())
            self.known_voice_names.append(name)
            print(f"✓ {name}의 음성 데이터 등록됨")
        
        return True
    
    @staticmethod
    def cosine_similarity(vec1, vec2):
        """코사인 유사도 계산"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
