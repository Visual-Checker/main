"""
Admin 제스처 인식 서비스
"""

import os
import pickle
import numpy as np
from dotenv import load_dotenv

load_dotenv()

MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe import Image as MPImage
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False


class GestureService:
    """제스처 인식 및 등록 서비스"""
    
    def __init__(self, model_path="./models/gesture_recognizer.task"):
        """
        Args:
            model_path: 제스처 인식 모델 경로
        """
        self.gesture_recognizer = None
        self.gesture_confidence_threshold = float(os.getenv('GESTURE_CONFIDENCE_THRESHOLD', 0.7))
        self.gesture_cooldown = float(os.getenv('GESTURE_COOLDOWN', 3.0))
        self.model_path = model_path
        self.gesture_data_file = os.getenv('GESTURE_DATA_FILE', './data/admin_gesture_data.pkl')
        self.known_gestures = {}  # {gesture_type: [{landmarks: [...], name: ...}, ...]}
        
        if MEDIAPIPE_AVAILABLE:
            try:
                base_options = python.BaseOptions(model_asset_path=model_path)
                gesture_options = vision.GestureRecognizerOptions(base_options=base_options)
                self.gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_options)
                self.load_gesture_data()
                print("✓ 제스처 인식 서비스 초기화 성공")
            except Exception as e:
                print(f"⚠️  제스처 인식 서비스 초기화 실패: {e}")
    
    def load_gesture_data(self):
        """저장된 제스처 데이터 로드"""
        if os.path.exists(self.gesture_data_file):
            try:
                with open(self.gesture_data_file, 'rb') as f:
                    self.known_gestures = pickle.load(f)
                total_gestures = sum(len(v) for v in self.known_gestures.values())
                print(f"✓ 제스처 데이터 로드됨 (타입: {len(self.known_gestures)}, 총: {total_gestures}개)")
            except Exception as e:
                print(f"⚠️  제스처 데이터 로드 실패: {e}")
        else:
            print("ℹ️  등록된 제스처 데이터가 없습니다.")
    
    def save_gesture_data(self):
        """제스처 데이터 저장"""
        os.makedirs(os.path.dirname(self.gesture_data_file), exist_ok=True)
        
        try:
            with open(self.gesture_data_file, 'wb') as f:
                pickle.dump(self.known_gestures, f)
            total_gestures = sum(len(v) for v in self.known_gestures.values())
            print(f"✓ 제스처 데이터 저장됨 (총: {total_gestures}개)")
        except Exception as e:
            print(f"⚠️  제스처 데이터 저장 실패: {e}")
    
    def extract_hand_landmarks(self, frame):
        """프레임에서 손 랜드마크 추출"""
        if not MEDIAPIPE_AVAILABLE or self.gesture_recognizer is None:
            print("ℹ️  MediaPipe가 준비되지 않았습니다.")
            return None, None
        
        try:
            import cv2
            # RGB 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # 제스처 인식
            result = self.gesture_recognizer.recognize(mp_image)
            
            if hasattr(result, 'gestures') and result.gestures and result.gestures[0]:
                gesture = result.gestures[0][0]
                gesture_name = gesture.category_name
                confidence = gesture.score
                
                # Hand landmarks 추출
                if hasattr(result, 'hand_landmarks') and result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    landmark_list = []
                    for landmark in landmarks:
                        landmark_list.append([landmark.x, landmark.y, landmark.z])
                    
                    return gesture_name, {
                        'landmarks': landmark_list,
                        'confidence': confidence
                    }
            
            return None, None
            
        except Exception as e:
            print(f"⚠️  손 랜드마크 추출 실패: {e}")
            return None, None
    
    def register_gesture(self, frame, gesture_type, user_name):
        """새로운 제스처 데이터 등록 (Quality Gate 적용)"""
        detected_gesture, landmark_data = self.extract_hand_landmarks(frame)
        
        if detected_gesture is None or landmark_data is None:
            print(f"⚠️  제스처 감지 실패")
            return False
        
        # Quality Gate: confidence 확인
        if landmark_data['confidence'] < self.gesture_confidence_threshold:
            print(f"⚠️  제스처 신뢰도 {landmark_data['confidence']:.3f}가 임계값 {self.gesture_confidence_threshold}보다 낮습니다.")
            return False
        
        # 제스처 타입이 기록되지 않았으면 초기화
        if gesture_type not in self.known_gestures:
            self.known_gestures[gesture_type] = []
        
        # 랜드마크 데이터 추가
        gesture_entry = {
            'landmarks': landmark_data['landmarks'],
            'confidence': landmark_data['confidence'],
            'name': user_name
        }
        
        self.known_gestures[gesture_type].append(gesture_entry)
        print(f"✓ {user_name}의 '{gesture_type}' 제스처 등록됨 (신뢰도: {landmark_data['confidence']:.3f})")
        
        return True
    
    def recognize_gesture(self, frame):
        """프레임에서 제스처 인식"""
        detected_gesture, landmark_data = self.extract_hand_landmarks(frame)
        
        if detected_gesture is None or landmark_data is None:
            return "Unknown", 0.0
        
        # Quality Gate: confidence 확인
        if landmark_data['confidence'] < self.gesture_confidence_threshold:
            return "Unknown", 0.0
        
        # 저장된 제스처와 비교
        if detected_gesture not in self.known_gestures:
            return "Unknown", 0.0
        
        # 가장 유사한 제스처 찾기 (간단히 신뢰도 기반)
        best_match_name = "Unknown"
        best_confidence = 0.0
        
        for gesture_entry in self.known_gestures[detected_gesture]:
            similarity = self.calculate_landmark_similarity(
                landmark_data['landmarks'],
                gesture_entry['landmarks']
            )
            
            if similarity > best_confidence:
                best_confidence = similarity
                best_match_name = gesture_entry['name']
        
        return best_match_name, best_confidence
    
    @staticmethod
    def calculate_landmark_similarity(landmarks1, landmarks2):
        """두 랜드마크 간 유사도 계산 (유클리드 거리)"""
        if len(landmarks1) != len(landmarks2):
            return 0.0
        
        distances = []
        for lm1, lm2 in zip(landmarks1, landmarks2):
            dist = np.sqrt((lm1[0] - lm2[0])**2 + (lm1[1] - lm2[1])**2 + (lm1[2] - lm2[2])**2)
            distances.append(dist)
        
        # 평균 거리를 유사도로 변환 (작을수록 유사)
        avg_distance = np.mean(distances)
        similarity = max(0, 1 - avg_distance)  # 0~1 범위
        
        return similarity
    
    def get_stats(self):
        """통계 반환"""
        gesture_types = list(self.known_gestures.keys())
        total_samples = sum(len(v) for v in self.known_gestures.values())
        
        return {
            'gesture_types': gesture_types,
            'total_samples': total_samples,
            'threshold': self.gesture_confidence_threshold,
            'details': {gt: len(samples) for gt, samples in self.known_gestures.items()}
        }
