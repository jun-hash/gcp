# 1) 기본 패키지 설치 (python3, pip, ffmpeg, unzip, git, curl, tmux)
sudo apt-get update
sudo apt-get install -y python3 python3-pip ffmpeg unzip git curl tmux

# 2) pip 패키지 설치 (requests, tqdm, termcolor)
sudo pip3 install requests tqdm termcolor

# 3) 파이썬 가상환경 관련 패키지 설치 (venv)
sudo apt-get install -y python3-venv


# 가상환경 생성 (myenv 라는 이름으로 생성)
python3 -m venv venv
# 가상환경 활성화
source venv/bin/activate
git clone https://github.com/jun-hash/gcp.git
cd gcp
pip install -r requirements.txt
tmux new -s my
cd ../
source venv/bin/activate
cd gcp
python pipe_vad_cut.py