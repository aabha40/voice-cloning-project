import argparse
import torch
from TTS.api import TTS
from preprocess import preprocess_audio

def clone_voice(text, speaker_wav, output_path, language="en"):
    device = "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    # support multiple files
    if "," in speaker_wav:
        wavs = speaker_wav.split(",")
        clean_wavs = []
        for i, w in enumerate(wavs):
            clean = f"clean_{i}.wav"
            preprocess_audio(w.strip(), clean)
            clean_wavs.append(clean)
    else:
        clean = "clean.wav"
        preprocess_audio(speaker_wav, clean)
        clean_wavs = [clean]
        

    tts.tts_to_file(
        text=text,
        speaker_wav=clean_wavs,
        language=language,
        file_path=output_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker_wav", required=True)
    parser.add_argument("--out", default="outputs/cloned.wav")
    parser.add_argument("--language", default="en")

    args = parser.parse_args()

    clone_voice(args.text, args.speaker_wav, args.out, args.language)