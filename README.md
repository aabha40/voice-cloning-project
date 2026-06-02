# 🎙️ VoiceForge — AI Voice Cloning System

> **Zero-shot voice cloning from just 5–30 seconds of audio.**
> No GPU required · 17 languages · Browser-based · CPU-only inference

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://python.org/downloads/release/python-3100/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-black?logo=flask)](https://flask.palletsprojects.com)
[![XTTS-v2](https://img.shields.io/badge/Model-XTTS--v2%20750M-cyan)](https://github.com/coqui-ai/TTS)
[![CDAC Pune](https://img.shields.io/badge/Internship-CDAC%20Pune%202026-orange)](https://cdac.in)
[![CPU Only](https://img.shields.io/badge/Hardware-CPU%20Only-red)](https://github.com/aabha40/voice-cloning-project)

---

## 👥 Authors

| Name | Email | Primary Contribution |
|------|-------|---------------------|
| **Aabha Shukla** | aabhasiddhishukla@gmail.com | Model integration, Flask backend, audio preprocessing |
| **Prachi Jha** | prachijhaa.2901@gmail.com | Evaluation pipeline, WER analysis, frontend & documentation |

*6-Month Deep Learning Internship — CDAC Pune, 2025–26*

---

## 📌 Table of Contents

1. [What is VoiceForge?](#-what-is-voiceforge)
2. [Demo](#-demo)
3. [Evaluation Results](#-evaluation-results)
4. [How It Works](#-how-it-works)
5. [Model Architecture](#-model-architecture)
6. [Tech Stack](#-tech-stack)
7. [System Requirements](#-system-requirements)
8. [Installation](#-installation)
9. [Running the App](#-running-the-app)
10. [Project Structure](#-project-structure)
11. [Supported Languages](#-supported-languages)
12. [Ethical Use](#-ethical-use)
13. [References](#-references)

---

## 🧠 What is VoiceForge?

**VoiceForge** is a zero-shot AI voice cloning system built on [Coqui XTTS-v2](https://github.com/coqui-ai/TTS). It synthesises natural-sounding speech that mimics a target speaker from as little as **5 seconds** of reference audio — with no GPU and no speaker-specific training required.

The system is deployed as a **Flask REST API** with a browser-based frontend, making it accessible from any device without local installation beyond Python.

**Key capabilities:**
- 🎤 Clone any voice from a 5–30 second WAV, MP3, or FLAC clip
- 🌍 Generate cloned speech in 17 different languages
- 💻 Runs entirely on CPU — no NVIDIA GPU needed
- 🔓 100% open-source — Coqui TTS, Whisper, librosa, Flask
- 📊 Rigorous quantitative evaluation (WER, Speaker Similarity, RTF, F1)

---

## 🎬 Demo

| Step | Action |
|------|--------|
| **1** | Upload a WAV / MP3 reference audio (5–30 seconds) |
| **2** | Type the text you want the cloned voice to say |
| **3** | Select output language from 17 options |
| **4** | Click **Generate** — output ready in 1–3 minutes on CPU |

The web interface provides:
- Real-time waveform visualisation of uploaded audio
- Step-by-step progress indicator
- Inline audio playback + WAV download
- Duration, sample rate, and words-per-minute metrics

---

## 📊 Evaluation Results

Evaluated on **150 synthesised utterances across 30 speakers** (native English, Indian English, Hindi) on CPU-only hardware. All metrics automatically computed — no human annotators.

### Primary Metrics (150 utterances · 30 speakers)

| Metric | Score | Industry Threshold | Status |
|--------|-------|--------------------|--------|
| Word Error Rate (WER) | **7.4%** | < 10% | ✅ Meets |
| Transcription Accuracy | **92.6%** | > 90% | ✅ Meets |
| Speaker Similarity (MFCC) | **0.837** | > 0.80 | ✅ Exceeds |
| Precision | **92.31%** | > 85% | ✅ Exceeds |
| Recall | **91.67%** | > 85% | ✅ Exceeds |
| F1 Score | **92.05%** | > 85% | ✅ Exceeds |
| Real-Time Factor (RTF) | **2.66×** | < 10× | ✅ Meets |
| Generation Success Rate | **100%** | — | ✅ Perfect |

> **5 out of 6 metrics exceed industry benchmarks by more than 6 percentage points.**

### Single-Speaker Benchmark (100 samples)

| Metric | Value |
|--------|-------|
| Word Error Rate | **9.51%** — meets < 10% industry standard |
| Accuracy | **90.49%** |
| Precision | **92.31%** |
| Recall | **91.67%** |
| F1 Score | **92.05%** |
| Speaker Similarity | **98.68%** |

### WER Distribution (150 utterances)

| WER Range | Samples | Percentage | Interpretation |
|-----------|---------|------------|----------------|
| WER = 0.00 | 42 | **42%** | Perfect — every word correct |
| 0.00 < WER ≤ 0.10 | 28 | 28% | Low error — highly intelligible |
| 0.10 < WER ≤ 0.20 | 16 | 16% | Moderate — minor word errors |
| 0.20 < WER ≤ 0.33 | 9 | 9% | High — noticeable errors |
| WER > 0.33 | 5 | 5% | Poor — rare, complex phonemes |

> **70% of samples achieved WER ≤ 0.10 (low to perfect intelligibility)**

---

## ⚙️ How It Works

```
Reference Audio (5–30 s)
         │
    ┌────▼────────────────────────────┐
    │     PREPROCESSING               │
    │  Resample → 22,050 Hz           │
    │  Trim silence (top_db = 20)     │
    │  Normalise: â = 0.95·a/max|a|   │
    │  Pad to ≥ 3 s minimum           │
    └────┬────────────────────────────┘
         │
    ┌────▼────────────────────────────┐
    │   SPEAKER CONDITIONING ENCODER  │
    │  Conv + GRU · GE2E-trained      │
    │  q = SpeakerEncoder_θ(M)        │
    │  Frozen weights → zero-shot     │
    └────┬────────────────────────────┘
         │  speaker embedding q
    ┌────▼────────────────────────────┐
    │   GPT-2 AUTOREGRESSIVE DECODER  │
    │  443M params · EnCodec tokens   │
    │  Conditioned on (text, q)       │
    │  τ=0.65 · top-k=50 · p=0.85    │
    └────┬────────────────────────────┘
         │  codec token sequence
    ┌────▼────────────────────────────┐
    │   HiFi-GAN NEURAL VOCODER       │
    │  W = HiFiGAN_θ(C′, q)          │
    │  22,050 Hz WAV output           │
    └────┬────────────────────────────┘
         │
    Cloned Audio ✓
```

---

## 🏗️ Model Architecture

**XTTS-v2** by Coqui AI — 750M parameters total, pre-trained on 16,000+ hours across 17 languages.

| Component | Parameters | Function |
|-----------|-----------|----------|
| VQ-VAE Speaker Encoder | 13M | Voice fingerprint extraction via GE2E loss |
| GPT-2 Autoregressive Decoder | 443M | Predicts discrete EnCodec audio tokens |
| HiFi-GAN Neural Vocoder | 26M | Converts codec tokens → 22 kHz waveform |

**Key design choices:**
- Speaker encoder weights are **frozen** after pre-training → enables zero-shot generalisation
- Perceiver Resampler produces exactly **32 fixed embeddings** regardless of reference clip length
- HiFi-GAN is **shared across all 17 languages** — cross-lingual cloning without language-specific modules
- Indian English: `en-in` maps to `hi` phoneme set for authentic accent rendering

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Voice Cloning Model** | Coqui XTTS-v2 (750M params, zero-shot multilingual) |
| **Backend** | Python 3.10, Flask 3.0+ |
| **Audio Processing** | librosa, soundfile, scipy, numpy |
| **ASR Evaluation** | OpenAI Whisper (WER computation) |
| **Speaker Evaluation** | MFCC cosine similarity via librosa |
| **Deep Learning** | PyTorch (CPU inference) |
| **Frontend** | HTML5 / CSS3 / JavaScript (single file, no framework) |

---

## 💻 System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Windows 10 64-bit | Windows 11 64-bit |
| Python | **3.10 exactly** ⚠️ | 3.10.x |
| RAM | 8 GB | 16 GB |
| Disk space | 6 GB free | 10 GB free |
| Internet | Required (first run, ~2 GB model download) | — |
| GPU | ❌ Not required | CUDA GPU (speeds up ~10×) |

> ⚠️ **Python 3.10 is required.** 3.11 and 3.12 are not compatible with the Coqui TTS library at this time.

---

## 🚀 Installation

### Step 1 — Install Python 3.10

Download from https://www.python.org/downloads/release/python-3100/

> ✅ Check **"Add Python to PATH"** during installation

Verify:
```bash
python --version   # should show Python 3.10.x
```

### Step 2 — Install Visual Studio Build Tools

Download from https://visualstudio.microsoft.com/visual-cpp-build-tools/

During installation, select:
- ✅ Desktop development with C++
- ✅ Windows 10 SDK (or Windows 11 SDK)

### Step 3 — Install FFmpeg

```powershell
winget install ffmpeg
```

> Close and reopen PowerShell after this step.

### Step 4 — Clone the Repository

```powershell
git clone https://github.com/aabha40/voice-cloning-project.git
cd voice-cloning-project
```

### Step 5 — Create Virtual Environment

```powershell
python -m venv env
env\Scripts\activate
```

Your prompt should now show `(env)`.

### Step 6 — Set Windows SDK Paths

Paste all three lines into PowerShell (required for native audio library compilation):

```powershell
$env:INCLUDE = "C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\shared;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\um;" + $env:INCLUDE

$env:LIB = "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64;" + $env:LIB

$env:PATH = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64;" + $env:PATH
```

### Step 7 — Install Dependencies

```powershell
pip install flask torch torchaudio librosa soundfile numpy
pip install TTS
```

> ☕ This step takes 5–15 minutes. The TTS package downloads model weights (~2 GB) on first run.

---

## ▶️ Running the App

### Web Interface (Recommended)

```powershell
python app/voice_clone.py
```

**First run:** Downloads XTTS-v2 model (~2 GB) — takes 5–10 minutes once.  
**Subsequent runs:** Model loads in ~30 seconds.

```
Loading XTTS-v2 model (this takes ~30 seconds)...
✓ Model loaded on CPU!
Starting VoiceForge at http://localhost:5000
```

Open **http://localhost:5000** in your browser.

---

### Command Line Interface

```powershell
python app/voice_clone.py `
  --text "Hello, this is a cloned voice." `
  --speaker_wav samples/sampleA.wav `
  --out outputs/result.wav `
  --language en
```

---

### Evaluation Scripts

```powershell
# Full evaluation — generates 100 samples, computes all metrics
python evaluate.py

# WER / Precision / Recall / F1 on existing audio files
python wer_only.py

# Detailed analysis report and benchmark comparison
python wer_analysis.py
```

Results are saved to:
- `evaluation/report.json` — Full JSON report with per-sample data
- `evaluation/results.csv` — Per-sample metrics CSV
- `evaluation/analysis_report.txt` — Human-readable benchmark report

---

## 📁 Project Structure

```
voice-cloning-project/
│
├── app/
│   ├── voice_clone.py        ← Flask REST API + XTTS-v2 inference engine
│   └── static/
│       └── index.html        ← Web frontend (drag-drop UI, waveform preview)
│
├── evaluation/
│   ├── outputs/              ← 100 generated WAV files
│   ├── report.json           ← Full evaluation report (JSON)
│   ├── results.csv           ← Per-sample metrics
│   └── RESULTS_SUMMARY.md    ← Human-readable results summary
│
├── docs/
│   └── Report_Voice_Cloning_Final.pdf  ← Research paper
│
├── samples/
│   └── sampleA.wav           ← Reference audio sample
│
├── preprocess.py             ← Audio preprocessing pipeline
├── evaluate.py               ← Full evaluation pipeline (100 samples, all metrics)
├── wer_only.py               ← Standalone WER / Precision / Recall / F1
├── wer_analysis.py           ← Detailed analysis & benchmark comparison report
├── requirements.txt          ← Python dependencies (versioned)
├── run.sh                    ← Quick start script (Linux/Mac)
├── setup.py                  ← pip-installable package config
├── .gitignore                ← Excludes model weights, outputs, venv
└── README.md                 ← This file
```

---

## 🌍 Supported Languages

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| `en` | English | `hi` | Hindi | `es` | Spanish |
| `fr` | French | `de` | German | `it` | Italian |
| `pt` | Portuguese | `pl` | Polish | `tr` | Turkish |
| `ru` | Russian | `nl` | Dutch | `cs` | Czech |
| `ar` | Arabic | `zh-cn` | Chinese | `ja` | Japanese |
| `ko` | Korean | `hu` | Hungarian | `en-in` | English (India) 🇮🇳 |

> 🇮🇳 **Indian English note:** `en-in` maps internally to Hindi phoneme patterns, producing a more authentic Indian accent than standard `en`.

---

## 📄 Research Paper

Our full research paper is available in [`https://docs.google.com/document/d/15LNYdU4Ro2tndhO60j-p1HOm5O59NmTjl_iYatOm1o4/edit?usp=sharing`](docs/Report_Voice_Cloning_Final.pdf)

**Citation:**
```bibtex
@misc{voiceforge2026,
  title     = {Voice-Cloning Using Deep Learning},
  author    = {Shukla, Aabha and Jha, Prachi},
  year      = {2026},
  note      = {6-Month Internship Project, CDAC Pune},
  url       = {https://github.com/aabha40/voice-cloning-project}
}
```

---

## ⚠️ Ethical Use

VoiceForge is intended for legitimate, consensual use only.

| ✅ Acceptable | ❌ Not Acceptable |
|--------------|-----------------|
| Accessibility tools for speech-impaired individuals | Impersonating individuals without consent |
| Academic research and evaluation | Creating deepfakes for fraud or deception |
| Dubbing with full speaker consent | Identity theft or social engineering |
| Creative content production (with consent) | Any illegal or harmful purpose |

All outputs should be treated as AI-generated audio. We recommend watermarking outputs in production deployments.

---

## 🙏 Acknowledgements

- [**Coqui AI**](https://github.com/coqui-ai/TTS) — XTTS-v2 model and TTS library
- [**OpenAI Whisper**](https://github.com/openai/whisper) — ASR for WER evaluation
- [**librosa**](https://librosa.org) — Audio analysis and processing
- [**SpeechBrain**](https://speechbrain.github.io) — Speaker verification (optional)
- **CDAC Pune** — Internship infrastructure and mentorship

---

## 📚 References

1. Casanova, E. et al. (2024). *XTTS: A Massively Multilingual Zero-Shot TTS*. Coqui AI.
2. Jia, Y. et al. (2018). *Transfer Learning from Speaker Verification to Multispeaker TTS*. NeurIPS.
3. Wang, Y. et al. (2017). *Tacotron: Towards End-to-End Speech Synthesis*. INTERSPEECH.
4. Kim, J. et al. (2021). *VITS: Conditional VAE for End-to-End TTS*. ICML.
5. Radford, A. et al. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision*. ICML.
6. Kong, J. et al. (2020). *HiFi-GAN: High Fidelity Speech Synthesis*. NeurIPS.
7. Wan, L. et al. (2018). *Generalized End-to-End Loss for Speaker Verification*. ICASSP.

---

<div align="center">

**VoiceForge v1.0** · Built at **CDAC Pune** · 2025–26

*Aabha Shukla · Prachi Jha*

[GitHub](https://github.com/aabha40/voice-cloning-project) · [Research Paper](docs/Report_Voice_Cloning_Final.pdf) · [Coqui XTTS-v2](https://github.com/coqui-ai/TTS)

</div>
