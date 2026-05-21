import argparse
import torch
import os
import sys
import tempfile
from TTS.api import TTS
from preprocess import preprocess_audio
from flask import Flask, send_from_directory, request, send_file

app = Flask(__name__, static_folder='static')

# Load model ONCE at startup (not on every request)
print("Loading XTTS-v2 model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
print("Model loaded!")

def clone_voice(text, speaker_wav, output_path, language="en"):
    device = "cpu"

    # Support multiple files
    if "," in speaker_wav:
        wavs = speaker_wav.split(",")
        clean_wavs = []
        for i, w in enumerate(wavs):
            clean = f"clean_{i}.wav"
            preprocess_audio(w.strip(), clean)
            clean_wavs.append(clean)
    else:
        clean = "clean_input.wav"
        preprocess_audio(speaker_wav, clean)
        clean_wavs = [clean]

    tts.tts_to_file(
        text=text,
        speaker_wav=clean_wavs,
        language=language,
        file_path=output_path,
        split_sentences=True,
        # ── Better cloning settings ──
        temperature=0.65,        # lower = more stable/accurate voice match
        speed=1.0,               # keep natural speed
        repetition_penalty=5.0, # reduce repetition artifacts
        top_k=50,
        top_p=0.85,
    )

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/clone', methods=['POST'])
def clone():
    try:
        audio = request.files['audio']
        text = request.form['text']
        language = request.form.get('language', 'en')

        # Fix language code — XTTS doesn't support 'en-in', use 'en'
        if '-' in language:
            language = language.split('-')[0]

        # Save uploaded audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_in:
            audio.save(tmp_in.name)
            input_path = tmp_in.name

        output_path = tempfile.mktemp(suffix='.wav')

        clone_voice(text, input_path, output_path, language)

        return send_file(
            output_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='cloned_voice.wav'
        )

    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("--text", required=True)
        parser.add_argument("--speaker_wav", required=True)
        parser.add_argument("--out", default="outputs/cloned.wav")
        parser.add_argument("--language", default="en")
        args = parser.parse_args()
        clone_voice(args.text, args.speaker_wav, args.out, args.language)
    else:
        print("Starting VoiceForge at http://localhost:5000")
        app.run(debug=False, port=5000)