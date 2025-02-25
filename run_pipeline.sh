#!/bin/bash
set -e

# CUDA 환경 변수 설정 (명시적으로 설정)
export CUDA_VISIBLE_DEVICES=0

# CUDA 상태 확인
echo "=== CUDA 상태 확인 ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi 명령을 찾을 수 없습니다. NVIDIA 드라이버가 설치되어 있지 않을 수 있습니다."
fi

# PyTorch CUDA 지원 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

echo "=== Starting VAD processing ==="
python pipe_vad_cut.py

echo "=== Starting Whisper transcription ==="
python asr.py

echo "=== Processing complete ===" 