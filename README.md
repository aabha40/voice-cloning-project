# VoiceForge — AI Voice Cloning System

> Clone any voice from 5–30 seconds of audio. No GPU required. 17 languages supported.

Built on **Coqui XTTS-v2** · Flask backend · Browser-based frontend · CPU-only inference

---

## Demo

| Step | Action |
|---|---|
| 1 | Upload a WAV / MP3 reference audio (5–30 sec) |
| 2 | Type the text you want the cloned voice to say |
| 3 | Select output language |
| 4 | Click **Generate** — output ready in 1–3 minutes |

---

## Evaluation Results — 100 Samples

| Metric | Value |
|---|---|
| Word Error Rate | **9.51%** — meets < 10% industry standard |
| Accuracy | **90.49%** |
| Precision | **92.31%** |
| Recall | **91.67%** |
| F1 Score | **92.05%** |
| Speaker Similarity | **98.68%** |

---

## Tech Stack

- **Model** — Coqui XTTS-v2 (750M parameters, zero-shot)
- **Backend** — Python 3.10, Flask
- **Audio** — Librosa, SoundFile
- **Evaluation** — OpenAI Whisper (WER), MFCC cosine similarity
- **Frontend** — HTML / CSS / JavaScript (single file)

---

## Requirements

- Windows 10 / 11 (64-bit)
- Python **3.10 exactly** — not 3.11 or 3.12
- 8 GB RAM minimum
- 6 GB free disk space
- Internet on first run (downloads ~2 GB model)
- No GPU needed

---

## Installation

### 1. Install Python 3.10
Download from https://www.python.org/downloads/release/python-3100/

> ⚠ Check **"Add Python to PATH"** during installation

```bash
python --version   # should show Python 3.10.x
```

### 2. Install Visual Studio Build Tools
Download from https://visualstudio.microsoft.com/visual-cpp-build-tools/

Select during install:
- Desktop development with C++
- Windows 10 SDK (or Windows 11 SDK)

### 3. Install FFmpeg
```powershell
winget install ffmpeg
```
Close and reopen PowerShell after this step.

### 4. Clone the repo
```powershell
git clone https://github.com/aabha40/voice-cloning-project.git
cd voice-cloning-project
```

### 5. Create virtual environment
```powershell
python -m venv env
env\Scripts\activate
```

### 6. Set Windows SDK paths
Paste all three lines into PowerShell:
```powershell
$env:INCLUDE = "C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\shared;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\um;" + $env:INCLUDE

$env:LIB = "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64;" + $env:LIB

$env:PATH = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64;" + $env:PATH
```

### 7. Install dependencies
```powershell
pip install flask torch torchaudio librosa soundfile numpy
pip install TTS
```

---

## Running the App

```powershell
python app/voice_clone.py
```

First run downloads the XTTS-v2 model (~2 GB). Takes 5–10 minutes once, then ~30 seconds every time after.

```
✓ Model loaded on CPU!
Starting VoiceForge at http://localhost:5000
```

Open your browser at **http://localhost:5000**

---

## Project Structure

```
voice-cloning-project/
├── app/
│   ├── static/
│   │   └── index.html        # Frontend UI
│   ├── voice_clone.py        # Flask server + cloning logic
│   └── preprocess.py         # Audio preprocessing
├── evaluation/
│   ├── outputs/              # 100 generated WAV files
│   ├── report.json           # Full evaluation report
│   └── results.csv           # Per-sample metrics
├── samples/                  # Reference audio files
├── evaluate.py               # Evaluation pipeline (100 samples)
├── wer_only.py               # WER / F1 / Precision / Recall
├── requirements.txt
└── README.md
```

---



## Supported Languages

English · Hindi · Spanish · French · German · Italian · Portuguese · Polish · Turkish · Russian · Dutch · Czech · Arabic · Chinese · Japanese · Korean · Hungarian

---

## References

- [Coqui XTTS-v2](https://github.com/coqui-ai/TTS)
- [OpenAI Whisper](https://github.com/openai/whisper)
- Casanova et al. (2024) — XTTS: A Massively Multilingual Zero-Shot TTS

---
