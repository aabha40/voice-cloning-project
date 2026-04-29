import librosa
import numpy as np
import soundfile as sf


def preprocess_audio(input_path, output_path, sr=16000):
    audio, _ = librosa.load(input_path, sr=sr, mono=True)

    # normalize
    peak = max(abs(audio)) if len(audio) > 0 else 1
    audio = audio / peak * 0.95

    sf.write(output_path, audio, sr)
    return output_path