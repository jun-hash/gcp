#!/bin/bash
# startup.sh
# VM 부팅 시 실행되어 필요한 패키지 설치, 코드 다운로드, 그리고 파이프라인 실행을 수행합니다.

set -e

# 1) 시스템 업데이트 및 필수 패키지 설치
sudo apt-get update
sudo apt-get install -y python3 python3-pip ffmpeg unzip git curl tmux

# 2) pip 패키지 설치 (필요한 라이브러리)
sudo pip3 install requests tqdm termcolor

# gsutil은 기본적으로 Cloud SDK에 포함되어 있으므로 별도 설치 불필요

# 4) GitHub에서 파이프라인 코드 클론 (아래 URL은 예시, 실제로는 자신의 저장소 URL로 교체)
git clone https://github.com/jun-hash/gcp.git
cd gcp
pip3 install -r requirements.txt
