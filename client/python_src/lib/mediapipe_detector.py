import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import json

class MediaPipeDetector:
    """
    MediaPipe 기반 얼굴 검출 + 제스처 인식
    - Face Detection
    - Hand Gesture Recognition (thumbs up, peace sign, etc.)
    """
    
    def __init__(self):
        # Face Detection 초기화
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0: short range (2m), 1: full range (5m)
            min_detection_confidence=0.7
        )
        
        # Hand Detection 초기화
        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        print("✓ MediaPipe Face + Gesture Detector initialized")
    
    def detect_faces(self, frame):
        """
        얼굴 검출
        
        Returns:
            faces: [{'bbox': [x, y, w, h], 'confidence': float, 'landmarks': {...}}, ...]
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)
        
        faces = []
        if results.detections:
            height, width = frame.shape[:2]
            
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                confidence = detection.score[0]
                
                # Landmarks (눈, 코, 입 등)
                landmarks = {}
                if detection.location_data.relative_keypoints:
                    for idx, keypoint in enumerate(detection.location_data.relative_keypoints):
                        landmarks[f'point_{idx}'] = {
                            'x': int(keypoint.x * width),
                            'y': int(keypoint.y * height)
                        }
                
                faces.append({
                    'bbox': [x, y, w, h],
                    'confidence': float(confidence),
                    'landmarks': landmarks
                })
        
        return faces
    
    def detect_gestures(self, frame):
        """
        손 제스처 인식
        
        Returns:
            gestures: [{'type': str, 'hand': 'left'|'right', 'confidence': float, 'landmarks': [...]}, ...]
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands_detector.process(rgb_frame)
        
        gestures = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # 손의 왼쪽/오른쪽 구분
                hand_label = handedness.classification[0].label  # 'Left' or 'Right'
                hand_confidence = handedness.classification[0].score
                
                # Landmarks 추출
                landmarks_list = []
                for landmark in hand_landmarks.landmark:
                    landmarks_list.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                
                # 제스처 인식
                gesture_type = self._recognize_gesture(hand_landmarks)
                
                gestures.append({
                    'type': gesture_type,
                    'hand': hand_label.lower(),
                    'confidence': float(hand_confidence),
                    'landmarks': landmarks_list
                })
        
        return gestures
    
    def _recognize_gesture(self, hand_landmarks):
        """
        손 랜드마크로부터 제스처 인식
        
        Gestures:
        - thumbs_up: 엄지 올림 (출석 체크)
        - peace: 브이 사인 (확인)
        - fist: 주먹 (취소)
        - open_palm: 손바닥 펼침 (대기)
        - pointing: 검지 가리킴
        """
        landmarks = hand_landmarks.landmark
        
        # 손가락 끝점과 관절점
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
        
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        
        middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        
        ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        
        pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = landmarks[self.mp_hands.HandLandmark.PINKY_PIP]
        
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        
        # 각 손가락이 펴져있는지 확인
        thumb_extended = thumb_tip.y < thumb_ip.y
        index_extended = index_tip.y < index_pip.y
        middle_extended = middle_tip.y < middle_pip.y
        ring_extended = ring_tip.y < ring_pip.y
        pinky_extended = pinky_tip.y < pinky_pip.y
        
        # 제스처 판별
        
        # Thumbs Up (엄지만 펴짐, 나머지 접힘)
        if thumb_extended and not (index_extended or middle_extended or ring_extended or pinky_extended):
            if thumb_tip.y < wrist.y:  # 엄지가 위로
                return "thumbs_up"
        
        # Peace Sign (검지, 중지만 펴짐)
        if index_extended and middle_extended and not (ring_extended or pinky_extended):
            return "peace"
        
        # Fist (모두 접힘)
        if not (thumb_extended or index_extended or middle_extended or ring_extended or pinky_extended):
            return "fist"
        
        # Open Palm (모두 펴짐)
        if thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
            return "open_palm"
        
        # Pointing (검지만 펴짐)
        if index_extended and not (middle_extended or ring_extended or pinky_extended):
            return "pointing"
        
        return "unknown"
    
    def draw_detections(self, frame, faces, gestures):
        """
        검출 결과를 프레임에 그리기
        """
        annotated = frame.copy()
        
        # 얼굴 박스 그리기
        for face in faces:
            x, y, w, h = face['bbox']
            confidence = face['confidence']
            
            # 박스
            color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # 신뢰도
            label = f"Face: {confidence:.2f}"
            cv2.putText(annotated, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 랜드마크
            for point_name, point in face['landmarks'].items():
                cv2.circle(annotated, (point['x'], point['y']), 3, (255, 0, 0), -1)
        
        # 제스처 정보 표시
        y_offset = 30
        for i, gesture in enumerate(gestures):
            gesture_text = f"{gesture['hand'].upper()} Hand: {gesture['type']} ({gesture['confidence']:.2f})"
            
            # 제스처별 색상
            color_map = {
                'thumbs_up': (0, 255, 0),
                'peace': (255, 255, 0),
                'fist': (0, 0, 255),
                'open_palm': (255, 255, 255),
                'pointing': (255, 128, 0)
            }
            color = color_map.get(gesture['type'], (128, 128, 128))
            
            cv2.putText(annotated, gesture_text, (10, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return annotated
    
    def process_frame(self, frame):
        """
        프레임 처리 - 얼굴 + 제스처 통합
        
        Returns:
            {
                'faces': [...],
                'gestures': [...],
                'timestamp': float,
                'frame_annotated': numpy.ndarray
            }
        """
        faces = self.detect_faces(frame)
        gestures = self.detect_gestures(frame)
        annotated = self.draw_detections(frame, faces, gestures)
        
        return {
            'faces': faces,
            'gestures': gestures,
            'timestamp': datetime.now().timestamp(),
            'frame_annotated': annotated
        }
    
    def to_json(self, result):
        """결과를 JSON으로 변환 (ZeroMQ 전송용)"""
        return json.dumps({
            'faces': result['faces'],
            'gestures': [{
                'type': g['type'],
                'hand': g['hand'],
                'confidence': g['confidence']
            } for g in result['gestures']],
            'timestamp': result['timestamp']
        })
    
    def release(self):
        """리소스 해제"""
        self.face_detector.close()
        self.hands_detector.close()


# 테스트 코드
if __name__ == "__main__":
    detector = MediaPipeDetector()
    cap = cv2.VideoCapture(0)
    
    print("\n=== 제스처 가이드 ===")
    print("👍 Thumbs Up: 출석 체크")
    print("✌️  Peace Sign: 확인")
    print("✊ Fist: 취소")
    print("✋ Open Palm: 대기")
    print("☝️  Pointing: 선택")
    print("\nPress 'q' to quit, 's' to save screenshot")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 매 프레임 처리 (성능을 위해 매 3프레임마다 처리 가능)
        if frame_count % 1 == 0:
            result = detector.process_frame(frame)
            
            # 정보 표시
            info_text = f"Faces: {len(result['faces'])} | Gestures: {len(result['gestures'])}"
            cv2.putText(result['frame_annotated'], info_text, (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('MediaPipe Face + Gesture Detection', result['frame_annotated'])
            
            # 제스처 이벤트 처리
            for gesture in result['gestures']:
                if gesture['type'] == 'thumbs_up' and gesture['confidence'] > 0.8:
                    print(f"✅ 출석 체크 인식! (손: {gesture['hand']})")
                elif gesture['type'] == 'peace':
                    print(f"✌️  확인 제스처 (손: {gesture['hand']})")
        
        frame_count += 1
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"capture_{timestamp}.jpg", frame)
            print(f"📸 Screenshot saved: capture_{timestamp}.jpg")
    
    cap.release()
    cv2.destroyAllWindows()
    detector.release()
