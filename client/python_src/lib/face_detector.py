import cv2
import numpy as np
import mediapipe as mp
import os
from dotenv import load_dotenv

load_dotenv()

class FaceDetector:
    """얼굴 인식 클래스 - 다양한 모델 지원"""
    
    def __init__(self, model_type=None, use_gpu=True):
        """
        Args:
            model_type: 'yunet', 'mediapipe', 'haar' (기본값은 환경변수에서)
            use_gpu: GPU 사용 여부
        """
        self.model_type = model_type or os.getenv('FACE_DETECTION_MODEL', 'yunet')
        self.use_gpu = use_gpu and os.getenv('USE_GPU', 'True') == 'True'
        self.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', 0.7))
        self.detector = None
        
        self._initialize_detector()
    
    def _initialize_detector(self):
        """선택된 모델 초기화"""
        if self.model_type == 'yunet':
            self._init_yunet()
        elif self.model_type == 'mediapipe':
            self._init_mediapipe()
        elif self.model_type == 'haar':
            self._init_haar()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _init_yunet(self):
        """YuNet 모델 초기화 (OpenCV)"""
        # YuNet 모델 다운로드 URL
        model_path = "face_detection_yunet_2023mar.onnx"
        
        if not os.path.exists(model_path):
            print("Downloading YuNet model...")
            url = "https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            import urllib.request
            urllib.request.urlretrieve(url, model_path)
        
        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            self.confidence_threshold,
            0.3,  # NMS threshold
            5000
        )
        print(f"✓ YuNet model loaded (GPU: {self.use_gpu})")
    
    def _init_mediapipe(self):
        """MediaPipe 모델 초기화"""
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0: short range, 1: full range
            min_detection_confidence=self.confidence_threshold
        )
        print(f"✓ MediaPipe Face Detection loaded")
    
    def _init_haar(self):
        """Haar Cascade 초기화 (레거시, CPU 전용)"""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        print(f"✓ Haar Cascade loaded")
    
    def detect(self, frame):
        """
        얼굴 검출
        
        Args:
            frame: OpenCV 이미지 (BGR)
            
        Returns:
            faces: [(x, y, w, h, confidence), ...]
        """
        if self.model_type == 'yunet':
            return self._detect_yunet(frame)
        elif self.model_type == 'mediapipe':
            return self._detect_mediapipe(frame)
        elif self.model_type == 'haar':
            return self._detect_haar(frame)
        return []
    
    def _detect_yunet(self, frame):
        """YuNet으로 얼굴 검출"""
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        
        _, faces = self.detector.detect(frame)
        
        if faces is None:
            return []
        
        results = []
        for face in faces:
            x, y, w, h = face[:4].astype(int)
            confidence = face[-1]
            results.append((x, y, w, h, confidence))
        
        return results
    
    def _detect_mediapipe(self, frame):
        """MediaPipe로 얼굴 검출"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)
        
        if not results.detections:
            return []
        
        faces = []
        height, width = frame.shape[:2]
        
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * width)
            y = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)
            confidence = detection.score[0]
            
            faces.append((x, y, w, h, confidence))
        
        return faces
    
    def _detect_haar(self, frame):
        """Haar Cascade로 얼굴 검출"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Haar는 confidence를 제공하지 않으므로 1.0으로 설정
        return [(x, y, w, h, 1.0) for (x, y, w, h) in faces]
    
    def draw_faces(self, frame, faces, show_confidence=True):
        """
        검출된 얼굴에 박스 그리기
        
        Args:
            frame: 원본 이미지
            faces: detect() 결과
            show_confidence: 신뢰도 표시 여부
            
        Returns:
            annotated_frame: 박스가 그려진 이미지
        """
        annotated = frame.copy()
        
        for face in faces:
            x, y, w, h, conf = face
            
            # 박스 그리기
            color = (0, 255, 0) if conf > 0.8 else (0, 165, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # 신뢰도 표시
            if show_confidence:
                label = f"{conf:.2f}"
                cv2.putText(annotated, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated
    
    def get_face_region(self, frame, face, padding=0.2):
        """
        얼굴 영역 추출 (패딩 포함)
        
        Args:
            frame: 원본 이미지
            face: (x, y, w, h, conf)
            padding: 패딩 비율
            
        Returns:
            face_img: 얼굴 영역 이미지
        """
        x, y, w, h, _ = face
        
        # 패딩 추가
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(frame.shape[1], x + w + pad_w)
        y2 = min(frame.shape[0], y + h + pad_h)
        
        return frame[y1:y2, x1:x2]
    
    def release(self):
        """리소스 해제"""
        if self.model_type == 'mediapipe' and self.detector:
            self.detector.close()

# 테스트 코드
if __name__ == "__main__":
    # 웹캠으로 테스트
    detector = FaceDetector(model_type='yunet')
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit, 'm' to switch model")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 얼굴 검출
        faces = detector.detect(frame)
        
        # 결과 그리기
        annotated = detector.draw_faces(frame, faces)
        
        # 정보 표시
        cv2.putText(annotated, f"Model: {detector.model_type}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"Faces: {len(faces)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Face Detection', annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            # 모델 전환
            detector.release()
            models = ['yunet', 'mediapipe', 'haar']
            current_idx = models.index(detector.model_type)
            next_model = models[(current_idx + 1) % len(models)]
            detector = FaceDetector(model_type=next_model)
    
    cap.release()
    cv2.destroyAllWindows()
    detector.release()
