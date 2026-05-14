import librosa
import soundfile as sf
import numpy as np

def preprocess_audio(input_path, output_path, target_sr=22050):
    # Load audio
    audio, sr = librosa.load(input_path, sr=None, mono=True)

    # Resample only if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Trim only leading/trailing silence — don't over-process
    audio, _ = librosa.effects.trim(audio, top_db=20)

    # Normalize volume
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.95

    # Must be at least 3 seconds for good cloning
    min_samples = target_sr * 3
    if len(audio) < min_samples:
        audio = np.tile(audio, int(np.ceil(min_samples / len(audio))))
        audio = audio[:min_samples]

    sf.write(output_path, audio, target_sr)