import cv2
import numpy as np
import os
from dotenv import load_dotenv
import mediapipe as mp

load_dotenv()

class FaceDetectionService:
    """서버용 얼굴 검출 서비스"""
    
    def __init__(self):
        self.model_type = os.getenv('FACE_DETECTION_MODEL', 'yunet')
        self.backend = os.getenv('FACE_DETECTION_BACKEND', 'gpu')
        self.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', 0.7))
        self.detector = None
        self._initialize_detector()
    
    def _initialize_detector(self):
        """모델 초기화"""
        if self.model_type == 'yunet':
            self._init_yunet()
        elif self.model_type == 'mediapipe':
            self._init_mediapipe()
        elif self.model_type == 'retinaface':
            self._init_retinaface()
        else:
            raise ValueError(f"Unsupported model: {self.model_type}")
    
    def _init_yunet(self):
        """YuNet 초기화"""
        model_path = "models/face_detection_yunet_2023mar.onnx"
        
        os.makedirs("models", exist_ok=True)
        
        if not os.path.exists(model_path):
            print("Downloading YuNet model...")
            import urllib.request
            url = "https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            urllib.request.urlretrieve(url, model_path)
        
        self.detector = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320),
            self.confidence_threshold, 0.3, 5000
        )
        print(f"✓ YuNet loaded ({self.backend})")
    
    def _init_mediapipe(self):
        """MediaPipe 초기화"""
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=self.confidence_threshold
        )
        print(f"✓ MediaPipe loaded")
    
    def _init_retinaface(self):
        """RetinaFace 초기화 (InsightFace 사용)"""
        try:
            from insightface.app import FaceAnalysis
            
            self.detector = FaceAnalysis(
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] 
                if self.backend == 'gpu' else ['CPUExecutionProvider']
            )
            self.detector.prepare(ctx_id=0 if self.backend == 'gpu' else -1)
            print(f"✓ RetinaFace loaded ({self.backend})")
        except ImportError:
            raise ImportError("insightface not installed. Run: pip install insightface")
    
    def detect_faces(self, image_data):
        """
        얼굴 검출
        
        Args:
            image_data: numpy array (BGR) or bytes
            
        Returns:
            {
                'faces': [{'bbox': [x, y, w, h], 'confidence': float}, ...],
                'count': int
            }
        """
        # bytes to numpy
        if isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            frame = image_data
        
        if self.model_type == 'yunet':
            return self._detect_yunet(frame)
        elif self.model_type == 'mediapipe':
            return self._detect_mediapipe(frame)
        elif self.model_type == 'retinaface':
            return self._detect_retinaface(frame)
    
    def _detect_yunet(self, frame):
        """YuNet 검출"""
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        
        _, faces = self.detector.detect(frame)
        
        result = {'faces': [], 'count': 0}
        
        if faces is not None:
            for face in faces:
                x, y, w, h = face[:4].astype(int)
                confidence = float(face[-1])
                result['faces'].append({
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'confidence': confidence
                })
            result['count'] = len(faces)
        
        return result
    
    def _detect_mediapipe(self, frame):
        """MediaPipe 검출"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)
        
        result = {'faces': [], 'count': 0}
        
        if results.detections:
            height, width = frame.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                confidence = float(detection.score[0])
                
                result['faces'].append({
                    'bbox': [x, y, w, h],
                    'confidence': confidence
                })
            result['count'] = len(results.detections)
        
        return result
    
    def _detect_retinaface(self, frame):
        """RetinaFace 검출"""
        faces = self.detector.get(frame)
        
        result = {'faces': [], 'count': len(faces)}
        
        for face in faces:
            bbox = face.bbox.astype(int)
            x, y, x2, y2 = bbox
            w, h = x2 - x, y2 - y
            
            result['faces'].append({
                'bbox': [int(x), int(y), int(w), int(h)],
                'confidence': float(face.det_score),
                'landmarks': face.kps.tolist() if hasattr(face, 'kps') else None,
                'embedding': face.embedding.tolist() if hasattr(face, 'embedding') else None
            })
        
        return result
    
    def encode_face(self, image_data, bbox=None):
        """
        얼굴 인코딩 (임베딩) 생성
        
        Args:
            image_data: 이미지 데이터
            bbox: [x, y, w, h] 또는 None (자동 검출)
            
        Returns:
            embedding vector (list) or None
        """
        if self.model_type != 'retinaface':
            return None  # 현재 RetinaFace만 지원
        
        if isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            frame = image_data
        
        faces = self.detector.get(frame)
        
        if len(faces) > 0:
            return faces[0].embedding.tolist()
        
        return None

# 전역 인스턴스
_face_service = None

def get_face_service():
    """싱글톤 패턴으로 서비스 반환"""
    global _face_service
    if _face_service is None:
        _face_service = FaceDetectionService()
    return _face_service
