#!/bin/bash
# startup.sh
# VM 부팅 시 실행되어 필요한 패키지 설치, 코드 다운로드, 그리고 파이프라인 실행을 수행합니다.

set -e

# 1) 시스템 업데이트 및 필수 패키지 설치
sudo apt-get update
sudo apt-get install -y python3 python3-pip ffmpeg unzip git curl

# 2) pip 패키지 설치 (필요한 라이브러리)
sudo pip3 install requests tqdm termcolor

# gsutil은 기본적으로 Cloud SDK에 포함되어 있으므로 별도 설치 불필요

# 3) 작업 디렉토리 생성 및 이동
mkdir -p /home/pipeline
cd /home/pipeline

# 4) GitHub에서 파이프라인 코드 클론 (아래 URL은 예시, 실제로는 자신의 저장소 URL로 교체)
git clone https://github.com/jun-hash/gcp.git
cd libri-light-pipeline

# 5) 소규모 테스트용 Master URL 파일 생성
# 예: 10시간 분량을 위해 --max_urls=50 정도로 제한
python3 generate_urls.py --output_file librivox_urls.txt --language English --max_urls 50

# 6) 파이프라인 실행 (모든 단계)
python3 pipeline.py --step all \
  --master_file librivox_urls.txt \
  --partition_index 0 --total_partitions 1 \
  --download_dir raw_audio \
  --mp3_dir mp3 \
  --flac_dir flac \
  --snr_file calculated_snr.tsv \
  --split_dir split_audio \
  --librivox_meta_dir librivox_meta \
  --gcs_bucket your-test-bucket

# 7) 로그 출력
echo "Pipeline execution completed."
