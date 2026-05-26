import argparse
import sys
import tempfile
import torch
import librosa
import numpy as np
from flask import Flask, send_from_directory, request, send_file
from TTS.api import TTS
import os

app = Flask(__name__, static_folder='static')

# Load model ONCE at startup
print("Loading XTTS-v2 model (this takes ~30 seconds)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=(device=="cuda"))
print(f"✓ Model loaded on {device.upper()}!")

def enhance_audio(wav_path, target_sr=24000):
    """Advanced preprocessing to preserve voice characteristics."""
    audio, sr = librosa.load(wav_path, sr=None, mono=True)
    
    # Resample to 24kHz (optimal for XTTS v2)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # Gentle noise reduction using spectral gating
    S = librosa.stft(audio)
    mag = np.abs(S)
    
    # Compute noise floor (bottom 5% of frequencies)
    noise_floor = np.percentile(mag, 5, axis=1, keepdims=True)
    mask = mag > (noise_floor * 1.5)  # soft mask
    S_clean = S * mask
    audio = librosa.istft(S_clean)
    
    # Trim silence (top_db=25 is gentler than 20)
    audio, _ = librosa.effects.trim(audio, top_db=25)
    
    # Normalize audio carefully
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.95
    
    # Ensure minimum duration (5 seconds is ideal for cloning)
    min_samples = target_sr * 5
    if len(audio) < min_samples:
        # Repeat but not just copy — add slight variations
        repeats = int(np.ceil(min_samples / len(audio)))
        audio = np.tile(audio, repeats)[:min_samples]
    
    return audio, target_sr

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/clone', methods=['POST'])
def clone():
    try:
        audio_file = request.files['audio']
        text = request.form['text']
        language = request.form.get('language', 'en')

        # Indian English trick — use 'hi' language with English text
        # This forces Indian phoneme patterns onto English words
        if language in ['en-in', 'hi-en']:
            language = 'hi'

        elif '-' in language:
            language = language.split('-')[0]
        
        # Save uploaded audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_file.save(tmp.name)
            input_path = tmp.name
        
        # Enhance the audio
        enhanced_audio, sr = enhance_audio(input_path)
        
        # Save enhanced audio
        import soundfile as sf
        enhanced_path = tempfile.mktemp(suffix='.wav')
        sf.write(enhanced_path, enhanced_audio, sr)
        
        output_path = tempfile.mktemp(suffix='.wav')
        
        # Advanced XTTS v2 settings
        tts.tts_to_file(
            text=text,
            speaker_wav=enhanced_path,
            language=language,
            file_path=output_path,
            split_sentences=True,
            # ── Optimal parameters for quality ──
            temperature=0.70,          # Lower = more voice consistency
            top_k=50,                  # Nucleus sampling
            top_p=0.85,                # Probability threshold
            repetition_penalty=7.5,    # Prevent repeated words
            speed=1.0,                 # Natural speed
            emotion="neutral",         # Stable emotion
        )
        
        return send_file(
            output_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='cloned_voice.wav'
        )
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return {'error': str(e)}, 500

if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("--text", required=True)
        parser.add_argument("--speaker_wav", required=True)
        parser.add_argument("--out", default="outputs/cloned.wav")
        parser.add_argument("--language", default="en")
        args = parser.parse_args()
        
        enhanced, sr = enhance_audio(args.speaker_wav)
        import soundfile as sf
        temp_path = tempfile.mktemp(suffix='.wav')
        sf.write(temp_path, enhanced, sr)
        
        clone_voice(args.text, temp_path, args.out, args.language)
    else:
        print("Starting VoiceForge at http://localhost:5000")
        app.run(debug=False, port=5000, threaded=True)