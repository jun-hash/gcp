#!/bin/bash
set -e

echo "=== Starting VAD processing ==="
python pipe_vad_cut.py

echo "=== Starting Whisper transcription ==="
python asr.py

echo "=== Processing complete ===" 