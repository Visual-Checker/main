import torch
import sounddevice as sd
import numpy as np
import os
import warnings
from scipy.io import wavfile
from scipy.signal import resample
from pathlib import Path
import sys
from dotenv import load_dotenv
from speechbrain.pretrained import EncoderClassifier
# 경고 무시
warnings.filterwarnings("ignore")
os.environ['TORCHAUDIO_USE_SOX_EFFECTS'] = '0'
load_dotenv()

# ====== CRITICAL: speechbrain import 전에 백엔드 체크 완전 차단 ======
import torchaudio
import torchaudio.backend.utils
import torchaudio.utils

# 백엔드 체크 함수 무시
def noop(*args, **kwargs):
    pass

amplifier = int(os.getenv("voice_amplifier"))

torchaudio.set_audio_backend = noop
torchaudio.backend.utils.set_audio_backend = noop
if hasattr(torchaudio, 'utils'):
    if hasattr(torchaudio.utils, 'check_torchaudio_backend'):
        torchaudio.utils.check_torchaudio_backend = noop

# ====== 이제 speechbrain import ======

# Windows에서 symlink 에러 해결을 위해 pathlib 패치
original_symlink_to = Path.symlink_to
def patched_symlink_to(self, target, target_is_directory=False):
    """symlink 대신 파일 복사 사용"""
    import shutil
    target = Path(target)
    if target.is_file():
        self.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, self)
    elif target.is_dir():
        self.parent.mkdir(parents=True, exist_ok=True)
        if self.exists():
            shutil.rmtree(self)
        shutil.copytree(target, self)

Path.symlink_to = patched_symlink_to

SAMPLE_RATE = 16000
DURATION = 4  # seconds per recognition chunk

print("모델 로딩 중...")
try:
    
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/ecapa"
    )
    print("✓ 모델 로드 완료")
except Exception as e:
    print(f"✗ 모델 로드 실패: {e}")
    import traceback
    traceback.print_exc()
    raise

# 원본 함수 복원
Path.symlink_to = original_symlink_to

import queue
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """오디오 콜백 함수"""
    audio_queue.put(indata.copy())

def load_audio_file(file_path):
    """scipy를 사용하여 wav 파일 로드"""
    sr, audio = wavfile.read(file_path)
    
    # 스테레오를 모노로 변환
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    
    # float32로 변환
    audio = audio.astype(np.float32)
    
    # 정규화 (int16 범위 고려)
    if audio.dtype == np.int16:
        audio = audio / 32768.0
    
    # 샘플레이트 리샘플링
    if sr != SAMPLE_RATE:
        num_samples = int(len(audio) * SAMPLE_RATE / sr)
        audio = resample(audio, num_samples)
    
    return torch.tensor(audio, dtype=torch.float32)

def record_audio_chunk():
    """마이크에서 오디오 청크 수집"""
    frames = []
    needed_frames = int(SAMPLE_RATE * DURATION)

    while len(frames) < needed_frames:
        data = audio_queue.get()
        frames.extend(data[:, 0])

    return np.array(frames[:needed_frames], dtype=np.float32)

def extract_embedding_from_array(audio_array):
    """오디오 임베딩 추출"""
    tensor_audio = torch.tensor(audio_array).unsqueeze(0)
    with torch.no_grad():
        embedding = classifier.encode_batch(tensor_audio)
    return embedding.squeeze(0)

def cosine_similarity(a, b):
    """코사인 유사도 계산 (스칼라 값 반환)"""
    # embedding이 1D 벡터인 경우
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    
    # 코사인 유사도 계산
    sim = torch.nn.functional.cosine_similarity(a, b)
    # 스칼라 값 반환
    return sim.mean() if sim.numel() > 1 else sim

print("등록된 화자 음성 로딩 중...")

try:
    registered_users = {
        "chulsu": extract_embedding_from_array(
            load_audio_file("register/chulsu.wav")
        ),
        "younghee": extract_embedding_from_array(
            load_audio_file("register/younghee.wav")
        ),
        "dongin": extract_embedding_from_array(
            load_audio_file("register/dongin.wav")
        )
    }
    print("✓ 화자 등록 완료")
except FileNotFoundError as e:
    print(f"✗ 오류: 등록된 wav 파일을 찾을 수 없습니다 - {e}")
    exit(1)
except Exception as e:
    print(f"✗ 오류 발생: {e}")
    exit(1)

print("\n실시간 화자 인식 시작 (Ctrl+C 종료)")
print("-" * 50)

try:
    with sd.InputStream(callback=audio_callback,samplerate=SAMPLE_RATE,channels=1):

        while True:
            print("음성 수신중... (4초 대기)")
            audio_chunk = record_audio_chunk()
            test_embedding = extract_embedding_from_array(audio_chunk)

            best_score = -1
            best_user = "Unknown"

            for name, emb in registered_users.items():
                score = cosine_similarity(test_embedding, emb)
                score_val = score.item() if torch.is_tensor(score) else float(score)
                if score_val > best_score:
                    best_score = score_val
                    best_user = name

            print(f"👤 인식된 화자: {best_user:10s} | 유사도: {best_score * amplifier:.4f}")
            print("-" * 50)

except KeyboardInterrupt:
    print("\n프로그램 종료")
except Exception as e:
    print(f"✗ 오류 발생: {e}")
    import traceback
    traceback.print_exc()
