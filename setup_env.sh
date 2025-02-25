#!/bin/bash
set -e

echo "=== 환경 설정 시작 ==="

# CUDA 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0

# 기존 whisper 패키지 제거
pip uninstall -y whisper openai-whisper

# 필요한 패키지 설치
pip install faster-whisper librosa tqdm

# 환경 확인
python check_env.py

echo "=== 환경 설정 완료 ===" 