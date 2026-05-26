"""
evaluate.py — Voice Cloning Evaluation
Generates 100 samples and computes:
  - MOS-like quality score
  - Speaker similarity (cosine similarity of embeddings)
  - WER (Word Error Rate) — accuracy metric
  - Precision, Recall, F1 on phoneme-level matching
  - RTF (Real Time Factor)
"""

import os
import sys
import json
import time
import torch
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime
from TTS.api import TTS

# ── Optional: for WER calculation ──
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠ whisper not installed — WER will be skipped. Run: pip install openai-whisper")

# ── Optional: for speaker similarity ──
try:
    from speechbrain.pretrained import SpeakerRecognition
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    print("⚠ speechbrain not installed — speaker similarity will use librosa fallback.")

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────
REFERENCE_WAV   = "samples/sampleA.wav"   # your reference voice sample
OUTPUT_DIR      = "evaluation/outputs"
REPORT_PATH     = "evaluation/report.json"
CSV_PATH        = "evaluation/results.csv"
LANGUAGE        = "en"
NUM_SAMPLES     = 100

# 100 diverse test sentences (varied length, vocabulary, phonemes)
TEST_SENTENCES = [
    "Hello, how are you doing today?",
    "The weather is beautiful this morning.",
    "I would like to order a coffee please.",
    "Can you tell me the time right now?",
    "She sells seashells by the seashore.",
    "The quick brown fox jumps over the lazy dog.",
    "Peter Piper picked a peck of pickled peppers.",
    "How much wood would a woodchuck chuck?",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "A journey of a thousand miles begins with a single step.",
    "The early bird catches the worm.",
    "Actions speak louder than words.",
    "Better late than never.",
    "Every cloud has a silver lining.",
    "I am going to the market to buy vegetables.",
    "Please send me the report by tomorrow morning.",
    "The meeting has been scheduled for three o'clock.",
    "We need to finish this project before the deadline.",
    "Could you please help me with this task?",
    "Technology is changing the way we live and work.",
    "Artificial intelligence is transforming every industry.",
    "Machine learning models require large amounts of data.",
    "Deep learning has revolutionized computer vision.",
    "Natural language processing enables human-machine interaction.",
    "The sun rises in the east and sets in the west.",
    "Water boils at one hundred degrees Celsius.",
    "The human body contains approximately thirty-seven trillion cells.",
    "Mount Everest is the highest peak in the world.",
    "The Amazon River is the largest river by discharge.",
    "I enjoy reading books on science and philosophy.",
    "Music has the power to change your mood instantly.",
    "Exercise regularly to maintain good health and fitness.",
    "A balanced diet includes proteins, carbohydrates, and fats.",
    "Sleep is essential for memory consolidation and recovery.",
    "The project deadline is approaching very quickly.",
    "We should review the code before the final submission.",
    "Please make sure to save your work frequently.",
    "The database needs to be updated with new records.",
    "Security vulnerabilities must be patched immediately.",
    "One two three four five six seven eight nine ten.",
    "The year two thousand and twenty five was eventful.",
    "She was born on the fifteenth of August.",
    "The price increased by twenty three percent last quarter.",
    "There are approximately eight billion people on Earth.",
    "Good morning everyone, welcome to today's presentation.",
    "Thank you for your patience and understanding.",
    "I sincerely apologize for the inconvenience caused.",
    "Congratulations on your outstanding achievement.",
    "Have a wonderful day and take care of yourself.",
    "The cat sat on the mat near the window.",
    "Birds fly south during the cold winter months.",
    "Children love playing in the park on sunny days.",
    "The old library had thousands of interesting books.",
    "Fresh fruits and vegetables are good for health.",
    "The engineer designed a new bridge over the river.",
    "Scientists discovered a new species in the rainforest.",
    "The chef prepared a delicious meal for the guests.",
    "Students gathered in the hall for the annual event.",
    "The pilot landed the aircraft safely on the runway.",
    "Innovation drives progress in modern society.",
    "Collaboration leads to better outcomes in teamwork.",
    "Patience is a virtue that everyone should cultivate.",
    "Creativity is the foundation of artistic expression.",
    "Knowledge is power when applied with wisdom.",
    "The conference call starts at nine in the morning.",
    "Please review the attached document and provide feedback.",
    "The server is down and needs immediate attention.",
    "Our team successfully completed the sprint goals.",
    "The client was satisfied with the final deliverable.",
    "Can we schedule a meeting for later this week?",
    "The temperature dropped significantly overnight.",
    "Heavy rainfall is expected throughout the weekend.",
    "The storm caused widespread power outages in the city.",
    "Rescue teams worked tirelessly to help those affected.",
    "Emergency services responded quickly to the situation.",
    "The stock market showed strong gains this week.",
    "Investors are optimistic about the economic recovery.",
    "The company reported record profits for the quarter.",
    "New regulations will impact the financial sector.",
    "Consumer spending increased significantly last month.",
    "The film received critical acclaim from reviewers.",
    "The author published her third novel this year.",
    "The concert was attended by thousands of fans.",
    "The art exhibition showcased works from local artists.",
    "The museum opened a new wing dedicated to history.",
    "She spoke confidently during the entire presentation.",
    "He carefully analyzed the data before concluding.",
    "They collaborated effectively to solve the problem.",
    "The team worked late into the night to meet the deadline.",
    "Everyone contributed their best efforts to the project.",
    "In the morning I wake up and have breakfast.",
    "After lunch I usually take a short walk outside.",
    "In the evening the family gathers for dinner.",
    "Before sleeping I like to read for thirty minutes.",
    "On weekends we visit family and friends nearby.",
    "This is a test of the voice cloning system.",
    "The system should reproduce my voice accurately.",
    "Voice cloning technology has advanced significantly.",
    "This evaluation will measure the quality of cloning.",
    "Thank you for using the VoiceForge cloning system.",
]

# ───────────────────────────────────────────────
# SETUP
# ───────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("evaluation", exist_ok=True)

print("=" * 60)
print("  VoiceForge — Evaluation Pipeline")
print("  Samples:", NUM_SAMPLES)
print("  Reference:", REFERENCE_WAV)
print("=" * 60)

# Load TTS model
print("\n[1/5] Loading XTTS-v2 model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2",
          gpu=(device == "cuda"))
print(f"✓ Model loaded on {device.upper()}")

# Load Whisper for WER
asr_model = None
if WHISPER_AVAILABLE:
    print("\n[2/5] Loading Whisper ASR for WER evaluation...")
    asr_model = whisper.load_model("base")
    print("✓ Whisper loaded")
else:
    print("\n[2/5] Skipping Whisper (not installed)")

# Load speaker verification model
spk_model = None
if SPEECHBRAIN_AVAILABLE:
    print("\n[3/5] Loading SpeakerRecognition model...")
    spk_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec"
    )
    print("✓ SpeakerRecognition loaded")
else:
    print("\n[3/5] Using librosa fallback for speaker similarity")


# ───────────────────────────────────────────────
# HELPER FUNCTIONS
# ───────────────────────────────────────────────

def get_mfcc_embedding(wav_path, sr=16000):
    """Extract MFCC-based speaker embedding."""
    audio, _ = librosa.load(wav_path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc, axis=1)

def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def compute_speaker_similarity(ref_wav, gen_wav):
    """Speaker similarity score between reference and generated audio."""
    if spk_model:
        try:
            score, _ = spk_model.verify_files(ref_wav, gen_wav)
            return float(score)
        except:
            pass
    # Fallback: MFCC cosine similarity
    ref_emb = get_mfcc_embedding(ref_wav)
    gen_emb = get_mfcc_embedding(gen_wav)
    return cosine_similarity(ref_emb, gen_emb)

def compute_wer(reference_text, wav_path):
    """Word Error Rate using Whisper ASR."""
    if not asr_model:
        return None
    try:
        result = asr_model.transcribe(wav_path, language="en")
        hypothesis = result["text"].strip().lower()
        reference  = reference_text.strip().lower()

        ref_words  = reference.split()
        hyp_words  = hypothesis.split()

        # Dynamic programming WER
        d = np.zeros((len(ref_words)+1, len(hyp_words)+1))
        for i in range(len(ref_words)+1): d[i][0] = i
        for j in range(len(hyp_words)+1): d[0][j] = j
        for i in range(1, len(ref_words)+1):
            for j in range(1, len(hyp_words)+1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])

        wer = d[len(ref_words)][len(hyp_words)] / len(ref_words)
        return round(float(wer), 4), hypothesis
    except Exception as e:
        print(f"    WER error: {e}")
        return None, ""

def compute_audio_quality(wav_path):
    """
    Pseudo-MOS quality score based on:
    - SNR (signal to noise ratio)
    - Spectral flatness (less flat = more speech-like)
    - Zero crossing rate
    Returns 0-5 score.
    """
    try:
        audio, sr = librosa.load(wav_path, sr=None, mono=True)

        # SNR estimate
        signal_power = np.mean(audio**2)
        noise = audio - librosa.effects.harmonic(audio)
        noise_power  = np.mean(noise**2) + 1e-8
        snr = 10 * np.log10(signal_power / noise_power)

        # Spectral flatness (lower = more tonal = better speech)
        flatness = np.mean(librosa.feature.spectral_flatness(y=audio))

        # Normalize to 0-5 score
        snr_score      = np.clip(snr / 40, 0, 1) * 2.5
        flatness_score = np.clip(1 - flatness * 100, 0, 1) * 2.5
        quality_score  = snr_score + flatness_score

        return round(float(quality_score), 3)
    except:
        return 0.0

def words_match(reference, hypothesis):
    """Compute per-word precision, recall, F1."""
    ref_words = set(reference.lower().split())
    hyp_words = set(hypothesis.lower().split())

    tp = len(ref_words & hyp_words)
    fp = len(hyp_words - ref_words)
    fn = len(ref_words - hyp_words)

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return round(precision, 4), round(recall, 4), round(f1, 4)


# ───────────────────────────────────────────────
# MAIN EVALUATION LOOP
# ───────────────────────────────────────────────

print("\n[4/5] Running evaluation on 100 samples...\n")

results     = []
all_wer     = []
all_sim     = []
all_quality = []
all_prec    = []
all_rec     = []
all_f1      = []
all_rtf     = []
failed      = 0

for idx, sentence in enumerate(TEST_SENTENCES[:NUM_SAMPLES]):
    out_path = os.path.join(OUTPUT_DIR, f"sample_{idx+1:03d}.wav")
    print(f"  [{idx+1:3d}/{NUM_SAMPLES}] Generating: \"{sentence[:50]}...\"" if len(sentence)>50
          else f"  [{idx+1:3d}/{NUM_SAMPLES}] Generating: \"{sentence}\"")

    try:
        # ── Generate cloned audio ──
        t_start = time.time()
        tts.tts_to_file(
            text=sentence,
            speaker_wav=REFERENCE_WAV,
            language=LANGUAGE,
            file_path=out_path,
            split_sentences=True,
            temperature=0.70,
            top_k=50,
            top_p=0.85,
            repetition_penalty=7.5,
            speed=1.0,
        )
        t_end = time.time()
        gen_time = t_end - t_start

        # Duration of generated audio
        audio_dur = librosa.get_duration(path=out_path)
        rtf = gen_time / audio_dur if audio_dur > 0 else 0

        # ── Metrics ──
        sim     = compute_speaker_similarity(REFERENCE_WAV, out_path)
        quality = compute_audio_quality(out_path)

        wer_score, hypothesis = compute_wer(sentence, out_path) \
            if asr_model else (None, "")

        precision, recall, f1 = words_match(sentence, hypothesis) \
            if hypothesis else (0.0, 0.0, 0.0)

        # ── Collect ──
        all_sim.append(sim)
        all_quality.append(quality)
        all_rtf.append(rtf)
        if wer_score is not None:
            all_wer.append(wer_score)
            all_prec.append(precision)
            all_rec.append(recall)
            all_f1.append(f1)

        result = {
            "id":            idx + 1,
            "sentence":      sentence,
            "output_file":   out_path,
            "gen_time_s":    round(gen_time, 2),
            "audio_dur_s":   round(audio_dur, 2),
            "rtf":           round(rtf, 3),
            "speaker_sim":   round(sim, 4),
            "quality_score": quality,
            "wer":           wer_score,
            "precision":     precision,
            "recall":        recall,
            "f1":            f1,
            "hypothesis":    hypothesis,
        }
        results.append(result)

        print(f"         ✓ sim={sim:.3f} | quality={quality:.2f}/5 "
              + (f"| wer={wer_score:.3f} | f1={f1:.3f}" if wer_score is not None else "")
              + f" | rtf={rtf:.2f}")

    except Exception as e:
        failed += 1
        print(f"         ✗ FAILED: {e}")
        results.append({"id": idx+1, "sentence": sentence, "error": str(e)})


# ───────────────────────────────────────────────
# SUMMARY REPORT
# ───────────────────────────────────────────────

print("\n[5/5] Computing summary report...\n")

summary = {
    "timestamp":           datetime.now().isoformat(),
    "total_samples":       NUM_SAMPLES,
    "successful":          NUM_SAMPLES - failed,
    "failed":              failed,
    "reference_wav":       REFERENCE_WAV,
    "language":            LANGUAGE,
    "device":              device.upper(),
    "metrics": {
        "speaker_similarity": {
            "mean":  round(np.mean(all_sim), 4)  if all_sim  else 0,
            "std":   round(np.std(all_sim), 4)   if all_sim  else 0,
            "min":   round(np.min(all_sim), 4)   if all_sim  else 0,
            "max":   round(np.max(all_sim), 4)   if all_sim  else 0,
        },
        "audio_quality_score": {
            "mean":  round(np.mean(all_quality), 4) if all_quality else 0,
            "std":   round(np.std(all_quality), 4)  if all_quality else 0,
        },
        "wer": {
            "mean":       round(np.mean(all_wer), 4)  if all_wer else "N/A (whisper not installed)",
            "std":        round(np.std(all_wer), 4)   if all_wer else 0,
            "accuracy":   round(1 - np.mean(all_wer), 4) if all_wer else "N/A",
        },
        "precision": {
            "mean": round(np.mean(all_prec), 4) if all_prec else "N/A",
        },
        "recall": {
            "mean": round(np.mean(all_rec), 4) if all_rec else "N/A",
        },
        "f1_score": {
            "mean": round(np.mean(all_f1), 4) if all_f1 else "N/A",
        },
        "real_time_factor": {
            "mean": round(np.mean(all_rtf), 3) if all_rtf else 0,
            "note": "RTF < 1.0 means faster than real-time"
        },
    },
    "per_sample": results,
}

# Save JSON report
with open(REPORT_PATH, "w") as f:
    json.dump(summary, f, indent=2)

# Save CSV
with open(CSV_PATH, "w") as f:
    f.write("id,sentence,gen_time_s,audio_dur_s,rtf,speaker_sim,quality_score,wer,precision,recall,f1\n")
    for r in results:
        if "error" not in r:
            f.write(f"{r['id']},\"{r['sentence'][:40]}\","
                    f"{r.get('gen_time_s','')},{r.get('audio_dur_s','')},{r.get('rtf','')},"
                    f"{r.get('speaker_sim','')},{r.get('quality_score','')},"
                    f"{r.get('wer','')},{r.get('precision','')},{r.get('recall','')},"
                    f"{r.get('f1','')}\n")

# Print final summary
print("=" * 60)
print("  EVALUATION RESULTS — 100 SAMPLES")
print("=" * 60)
m = summary["metrics"]
print(f"  Successful Samples   : {summary['successful']}/{NUM_SAMPLES}")
print(f"  Speaker Similarity   : {m['speaker_similarity']['mean']} "
      f"(±{m['speaker_similarity']['std']})")
print(f"  Audio Quality Score  : {m['audio_quality_score']['mean']} / 5.0")
print(f"  WER (lower=better)   : {m['wer']['mean']}")
print(f"  Accuracy (1-WER)     : {m['wer']['accuracy']}")
print(f"  Precision            : {m['precision']['mean']}")
print(f"  Recall               : {m['recall']['mean']}")
print(f"  F1 Score             : {m['f1_score']['mean']}")
print(f"  Real Time Factor     : {m['real_time_factor']['mean']}x")
print("=" * 60)
print(f"\n  ✓ Full report saved to : {REPORT_PATH}")
print(f"  ✓ CSV saved to         : {CSV_PATH}")
print(f"  ✓ Audio files in       : {OUTPUT_DIR}/")