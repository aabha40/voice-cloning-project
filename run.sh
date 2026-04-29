#!/bin/bash

python app/voice_clone.py \
  --text "Hello, this is my cloned voice running on CPU." \
  --speaker_wav samples/speaker.wav \
  --out outputs/cloned.wav