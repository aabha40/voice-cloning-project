import librosa
import numpy as np
from jiwer import wer


def compute_duration(path):
    audio, sr = librosa.load(path)
    return len(audio) / sr


def compute_similarity(a, b):
    # simple cosine similarity of MFCCs (approx)
    mfcc1 = librosa.feature.mfcc(y=librosa.load(a)[0])
    mfcc2 = librosa.feature.mfcc(y=librosa.load(b)[0])

    min_len = min(mfcc1.shape[1], mfcc2.shape[1])
    mfcc1 = mfcc1[:, :min_len]
    mfcc2 = mfcc2[:, :min_len]

    return np.dot(mfcc1.flatten(), mfcc2.flatten()) / (
        np.linalg.norm(mfcc1.flatten()) * np.linalg.norm(mfcc2.flatten())
    )


def evaluate(original_text, generated_text, ref_audio, gen_audio):
    print("\n--- Evaluation ---")

    print("WER:", wer(original_text, generated_text))

    sim = compute_similarity(ref_audio, gen_audio)
    print("Speaker Similarity (approx):", round(sim, 3))

    print("Generated Duration:", compute_duration(gen_audio), "sec")