"""SpeechBrain 설치 및 import 테스트"""
try:
    import speechbrain
    from speechbrain.inference.speaker import EncoderClassifier
    print('✓ SpeechBrain 정상 작동')
    print(f'버전: {speechbrain.__version__}')
    print('✓ EncoderClassifier import 성공')
except Exception as e:
    print(f'✗ 오류: {e}')
    import traceback
    traceback.print_exc()
