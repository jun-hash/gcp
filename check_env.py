#!/usr/bin/env python3
import os
import sys
import torch
import subprocess

print("=== Python 환경 정보 ===")
print(f"Python 버전: {sys.version}")
print(f"Python 경로: {sys.executable}")

print("\n=== CUDA 환경 정보 ===")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 버전: {torch.version.cuda}")
    print(f"CUDA 장치 수: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"CUDA 장치 {i}: {torch.cuda.get_device_name(i)}")

print("\n=== 설치된 패키지 ===")
subprocess.run([sys.executable, "-m", "pip", "list"])

print("\n=== NVIDIA 드라이버 정보 ===")
try:
    subprocess.run(["nvidia-smi"], check=False)
except:
    print("nvidia-smi 명령을 실행할 수 없습니다.") 