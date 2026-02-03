"""
Admin 음성 인식 서비스
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


class VoiceService:
    """음성 인식 및 스피커 식별 서비스"""
    
    def __init__(self, model_path=None, voice_data_file=None):
        """
        Args:
            model_path: 모델 저장 경로 (.env의 VOICE_MODEL_PATH 사용)
            voice_data_file: 음성 데이터 파일 경로 (.env의 VOICE_DATA_FILE 사용)
        """
        self.voice_encoder = None
        self.known_voice_embeddings = []
        self.known_voice_names = []
        self.voice_similarity_threshold = float(os.getenv('VOICE_SIMILARITY_THRESHOLD', 0.7))
        self.model_path = model_path or os.getenv('VOICE_MODEL_PATH', './models/spkrec-ecapa-voxceleb')
        # 음성 데이터를 data/voice 폴더에 저장
        self.voice_data_file = voice_data_file or os.getenv('VOICE_DATA_FILE', './data/voice/voice_embeddings.pkl')
        
        if SPEECHBRAIN_AVAILABLE:
            try:
                self.voice_encoder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=self.model_path
                )
                self.load_voice_data()
                print("✓ 음성 인식 서비스 초기화 성공")
            except Exception as e:
                print(f"⚠️  음성 인식 서비스 초기화 실패: {e}")
    
    def load_voice_data(self):
        """저장된 음성 데이터 로드"""
        if os.path.exists(self.voice_data_file):
            try:
                with open(self.voice_data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_voice_embeddings = data.get('embeddings', [])
                    self.known_voice_names = data.get('names', [])
                print(f"✓ {len(self.known_voice_names)}명의 음성 데이터 로드됨")
            except Exception as e:
                print(f"⚠️  음성 데이터 로드 실패: {e}")
        else:
            print("ℹ️  등록된 음성 데이터가 없습니다.")
    
    def save_voice_data(self):
        """음성 임베딩 저장"""
        os.makedirs(os.path.dirname(self.voice_data_file), exist_ok=True)
        
        data = {
            'embeddings': self.known_voice_embeddings,
            'names': self.known_voice_names,
            'threshold': self.voice_similarity_threshold
        }
        
        try:
            with open(self.voice_data_file, 'wb') as f:
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
            
            print(f"✓ 음성 임베딩 추출 성공 (차원: {len(embedding)})")
            return embedding
            
        except Exception as e:
            print(f"⚠️  음성 임베딩 추출 실패: {e}")
            return None
    
    def recognize_voice(self, audio_file):
        """음성 파일 인식 (등록된 음성과 비교)"""
        embedding = self.extract_voice_embedding(audio_file)
        
        if embedding is None:
            return "Unknown", 0.0
        
        best_match_name = "Unknown"
        best_similarity = 0.0
        
        # 저장된 음성과 비교
        if not self.known_voice_embeddings:
            print("⚠️  등록된 음성 데이터가 없습니다.")
            return "Unknown", 0.0
        
        for known_emb, known_name in zip(self.known_voice_embeddings, self.known_voice_names):
            similarity = self.cosine_similarity(embedding, np.array(known_emb))
            
            if similarity > best_similarity:
                best_similarity = float(similarity)
                best_match_name = known_name
        
        # 임계값 확인
        if best_similarity < self.voice_similarity_threshold:
            print(f"⚠️  유사도 {best_similarity:.3f}가 임계값 {self.voice_similarity_threshold}보다 낮습니다.")
            best_match_name = "Unknown"
            best_similarity = 0.0
        else:
            print(f"✓ 음성 인식: {best_match_name} (유사도: {best_similarity:.3f})")
        
        return best_match_name, best_similarity
    
    def register_voice(self, audio_file, name):
        """새로운 음성 데이터 등록 (Quality Gate 적용)"""
        embedding = self.extract_voice_embedding(audio_file)
        
        if embedding is None:
            return False
        
        # Quality Gate: 임베딩 벡터의 norm 확인 (이상값 감지)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            print(f"⚠️  음성 임베딩의 norm이 0입니다. 등록 실패.")
            return False
        
        # 중복 확인
        if name in self.known_voice_names:
            idx = self.known_voice_names.index(name)
            self.known_voice_embeddings[idx] = embedding.tolist()
            print(f"✓ {name}의 음성 데이터 업데이트됨")
        else:
            self.known_voice_embeddings.append(embedding.tolist())
            self.known_voice_names.append(name)
            print(f"✓ {name}의 음성 데이터 등록됨 (임베딩 차원: {len(embedding)})")
        
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
    
    def get_stats(self):
        """통계 반환"""
        return {
            'registered_speakers': len(self.known_voice_names),
            'similarity_threshold': self.voice_similarity_threshold,
            'speakers': self.known_voice_names
        }
