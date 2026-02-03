"""Recognition utilities for face, gesture, and voice.
- Face: ArcFace/InsightFace if available, fallback to color histogram (not ideal but usable)
- Gesture: MediaPipe Hands landmarks extractor
- Voice: SpeechBrain ECAPA-TDNN embedder
"""
import os
import numpy as np

# Face
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except Exception:
    INSIGHTFACE_AVAILABLE = False

# MediaPipe for gesture
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False

# SpeechBrain for voice
try:
    from speechbrain.pretrained import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except Exception:
    SPEECHBRAIN_AVAILABLE = False


class FaceEmbedder:
    def __init__(self, model_name='arcface_r50', device='cpu'):
        self.device = device
        self.model_name = model_name
        if INSIGHTFACE_AVAILABLE:
            # initialize insightface FaceAnalysis
            self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'] if device=='cpu' else None)
            self.app.prepare(ctx_id=0 if device!='cpu' else -1)
        else:
            self.app = None

    def embed(self, image):
        """Return embedding (np.array) given an image (BGR numpy) or image path"""
        if isinstance(image, str):
            import cv2
            image = cv2.imread(image)
        if image is None:
            return None
        if self.app:
            try:
                faces = self.app.get(image)
                if not faces:
                    return None
                # take first face
                embedding = faces[0].embedding
                return np.array(embedding, dtype=np.float32)
            except Exception:
                return None
        else:
            # fallback: simple color histogram reduced to 512 dims
            import cv2
            face = cv2.resize(image, (160, 160))
            hist = cv2.calcHist([face], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
            hist = cv2.normalize(hist, hist).flatten()
            # pad/trim to 512
            vec = hist
            if vec.size < 512:
                vec = np.pad(vec, (0, 512-vec.size))
            else:
                vec = vec[:512]
            return vec.astype(np.float32)

    @staticmethod
    def cosine_similarity(a, b):
        if a is None or b is None:
            return 0.0
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        num = np.dot(a, b)
        den = np.linalg.norm(a) * np.linalg.norm(b)
        if den == 0:
            return 0.0
        return float(num/den)


class GestureExtractor:
    def __init__(self):
        if MEDIAPIPE_AVAILABLE:
            self.hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)
        else:
            self.hands = None

    def extract(self, image):
        """Return normalized landmark vector or None"""
        if isinstance(image, str):
            import cv2
            image = cv2.imread(image)
        if image is None:
            return None
        if not self.hands:
            return None
        import cv2
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = self.hands.process(img_rgb)
        if not res.multi_hand_landmarks:
            return None
        lm = res.multi_hand_landmarks[0]
        pts = np.array([[p.x, p.y, p.z] for p in lm.landmark])
        # normalize by wrist and scale
        base = pts[0]
        pts = pts - base
        scale = np.linalg.norm(pts)
        if scale == 0:
            return None
        pts = pts.flatten() / scale
        return pts.astype(np.float32)

    @staticmethod
    def compare(a, b):
        if a is None or b is None:
            return 0.0
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        # cosine
        num = np.dot(a, b)
        den = np.linalg.norm(a) * np.linalg.norm(b)
        if den == 0:
            return 0.0
        return float(num/den)


class VoiceEmbedder:
    def __init__(self, source='speechbrain/spkrec-ecapa-voxceleb', savedir='./models/spkrec-ecapa-voxceleb'):
        self.source = source
        self.savedir = savedir
        self.model = None
        if SPEECHBRAIN_AVAILABLE:
            try:
                self.model = EncoderClassifier.from_hparams(source=self.source, savedir=self.savedir)
            except Exception:
                self.model = None

    def embed_from_file(self, wav_path):
        if self.model is None:
            return None
        try:
            emb = self.model.encode_file(wav_path)
            # emb is tensor, convert to numpy 1D
            return np.array(emb.detach().cpu()).reshape(-1)
        except Exception:
            return None

    @staticmethod
    def cosine_similarity(a, b):
        if a is None or b is None:
            return 0.0
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        num = np.dot(a, b)
        den = np.linalg.norm(a) * np.linalg.norm(b)
        if den == 0:
            return 0.0
        return float(num/den)
