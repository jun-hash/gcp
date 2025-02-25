#!/bin/bash
set -e

# 1) 기본 패키지 설치 (python3, pip, ffmpeg, unzip, git, curl, tmux)
sudo apt-get update
sudo apt-get install -y python3 python3-pip ffmpeg unzip git curl tmux

# 2) NVIDIA 드라이버 및 CUDA 상태 확인
echo "=== NVIDIA 드라이버 및 CUDA 상태 확인 ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi 명령을 찾을 수 없습니다. NVIDIA 드라이버가 설치되어 있지 않을 수 있습니다."
fi

# 3) 가상환경 설정
echo "=== 가상환경 설정 ==="
python3 -m venv venv
source venv/bin/activate

# 4) 저장소 클론 및 패키지 설치
git clone https://github.com/jun-hash/gcp.git || echo "저장소가 이미 존재합니다"
cd gcp

# 5) CUDA 환경 변수 설정 (명시적으로 설정)
export CUDA_VISIBLE_DEVICES=0

# 6) 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt

# 7) PyTorch CUDA 지원 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

# 8) 파이프라인 실행
echo "=== 파이프라인 실행 ==="
bash run_pipeline.sh