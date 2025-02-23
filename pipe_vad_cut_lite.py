#!/usr/bin/env python3
import os
import sys
import argparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import subprocess
import multiprocessing
import json
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import tqdm
import numpy as np
import soundfile as sf
import torch
import torchaudio

# -------------------------------------------------
# 파이프라인 단계 정의
# -------------------------------------------------
PIPELINE_STAGES = {
    "download": "Download audio files",
    "vad":      "Voice Activity Detection",
    "cut":      "Cut segments",
    "intro":    "Remove intro segments"
}

# -------------------------------------------------
# 기본 설정값 (맥북 최적화)
# -------------------------------------------------
DEFAULT_BASE_DIR            = "data"
DEFAULT_SAMPLE_RATE        = 44100
DEFAULT_N_PROCESSES        = max(1, min(4, multiprocessing.cpu_count() - 1))  # 맥북 발열 방지
DEFAULT_MIN_SPEECH_DURATION = 0.5
DEFAULT_TARGET_LEN_SEC     = 30
DEFAULT_FORMAT             = ".mp3"
DEFAULT_THREADS_DOWNLOAD   = 4      # 다운로드 스레드 수 감소
DEFAULT_POOL_SIZE         = 10     # 커넥션 풀 크기 감소

# -------------------------------------------------
# 타임 트래커 유틸
# -------------------------------------------------
class StageTimer:
    def __init__(self):
        self.times = {}

    def start(self, stage_name):
        self.times[stage_name] = {"start": time.time(), "end": None, "duration": None}

    def end(self, stage_name):
        if stage_name not in self.times or self.times[stage_name]["start"] is None:
            print(f"[WARN] Stage '{stage_name}' was never started.")
            return
        self.times[stage_name]["end"] = time.time()
        self.times[stage_name]["duration"] = (
            self.times[stage_name]["end"] - self.times[stage_name]["start"]
        )

    def report(self):
        print("\n[Pipeline Stage Times]")
        for stage_name, tinfo in self.times.items():
            if all(v is not None for v in tinfo.values()):
                print(f"  - {stage_name:<20}: {tinfo['duration']:.2f} seconds")

# -------------------------------------------------
# Download (MP3)
# -------------------------------------------------
def create_session(max_retries=3, backoff_factor=0.5, pool_connections=10, pool_maxsize=10):
    session = requests.Session()
    retries = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(
        max_retries=retries,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def _download_file(url, output_dir, session=None):
    file_name = os.path.basename(url)
    save_path = os.path.join(output_dir, file_name)

    if os.path.exists(save_path):
        return file_name, True

    if session is None:
        session = create_session()

    try:
        with session.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            with open(save_path, "wb") as out_file:
                for chunk in response.iter_content(chunk_size=8192):  # 작은 청크 사이즈
                    if chunk:
                        out_file.write(chunk)
        return file_name, True
    except Exception as e:
        print(f"[ERROR] Download failed for {url}: {e}")
        return file_name, False

def download_urls(urls_file, output_dir, test_sample=None):
    os.makedirs(output_dir, exist_ok=True)

    with open(urls_file, "r", encoding="utf-8") as f:
        all_urls = [line.strip() for line in f if line.strip()]

    if test_sample:
        all_urls = all_urls[:test_sample]

    print(f"[Download] {len(all_urls)} URLs to download.")
    session = create_session(pool_connections=DEFAULT_POOL_SIZE, pool_maxsize=DEFAULT_POOL_SIZE)

    results = []
    with ThreadPoolExecutor(max_workers=DEFAULT_THREADS_DOWNLOAD) as executor:
        future_to_url = {
            executor.submit(_download_file, url, output_dir, session): url
            for url in all_urls
        }

        for future in tqdm.tqdm(as_completed(future_to_url), total=len(all_urls), desc="Downloading"):
            url = future_to_url[future]
            try:
                file_name, status = future.result()
                results.append((file_name, status))
            except Exception as e:
                print(f"[ERROR] Download failed for {url}: {e}")
                results.append(("Unknown", False))

    success_count = sum(1 for _, s in results if s)
    fail_count = len(results) - success_count
    print(f"[Download Summary] Success={success_count}, Failed={fail_count}")

# -------------------------------------------------
# Silero VAD
# -------------------------------------------------
def load_vad_model():
    print("[VAD] Loading model...")
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        trust_repo=True,
        force_reload=False
    )
    return model, utils[0]

def apply_vad(audio_path, vad_model, get_timestamps, sample_rate=16000):
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sr != sample_rate:
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = transform(waveform)

    timestamps = get_timestamps(
        waveform,
        vad_model,
        threshold=0.3,
        sampling_rate=sample_rate,
        min_speech_duration_ms=500,
        min_silence_duration_ms=500
    )
    
    return [{'start': ts['start'] / sample_rate, 'end': ts['end'] / sample_rate} 
            for ts in timestamps]

def save_vad_results(audio_path, vad_segments, output_file):
    data = {
        "audio_file": str(audio_path),
        "voice_activity": vad_segments
    }
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

def process_vad(input_dir, output_dir, test_sample=None):
    vad_model, get_timestamps = load_vad_model()
    
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.mp3'):
                audio_files.append(os.path.join(root, file))

    if test_sample:
        audio_files = audio_files[:test_sample]

    # 이미 처리된 파일 스킵하기
    tasks = []
    for audio_path in audio_files:
        print(f"audio_path: {audio_path}")
        rel_path = os.path.relpath(audio_path, input_dir)
        print(f"rel_path: {rel_path}")

        json_out = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.json')
        print(f"json_out: {json_out}")

        # 이미 결과 파일이 존재하면 스킵
        if os.path.exists(json_out):
            print(f"[VAD] Skipped already processed file: {rel_path}")
            continue
    print(f"[VAD] Processing {len(audio_files)} audio files.")
    os.makedirs(output_dir, exist_ok=True)

    for audio_path in tqdm.tqdm(audio_files, desc="VAD Processing"):
        try:
            segments = apply_vad(audio_path, vad_model, get_timestamps)
            rel_path = os.path.relpath(audio_path, input_dir)
            json_out = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.json')
            os.makedirs(os.path.dirname(json_out), exist_ok=True)
            save_vad_results(audio_path, segments, json_out)
        except Exception as e:
            print(f"[ERROR] VAD failed for {audio_path}: {e}")

# -------------------------------------------------
# Cutting
# -------------------------------------------------
def save_cut(seq, fname, index, extension, output_dir):
    output = np.hstack(seq)
    file_name = pathlib.Path(output_dir) / fname.name / (fname.stem + f"_{index:04}{extension}")
    file_name.parent.mkdir(exist_ok=True, parents=True)
    sf.write(file_name, output, samplerate=44100)

def cut_sequence(audio_path, vad, output_dir, min_len_sec=15, max_len_sec=30):
    data, samplerate = sf.read(audio_path)
    if len(data.shape) == 2:
        data = data.mean(axis=1)

    if samplerate != 44100:
        print(f"[WARN] {audio_path} samplerate {samplerate} != 44100")
        return

    to_stitch = []
    length_accumulated = 0.0
    segment_index = 0

    for segment in vad:
        start, end = segment['start'], segment['end']
        start_idx, end_idx = int(start * samplerate), int(end * samplerate)
        slice_audio = data[start_idx:end_idx]

        if length_accumulated + (end - start) > max_len_sec:
            if len(to_stitch) > 0 and length_accumulated >= min_len_sec:
                save_cut(to_stitch, pathlib.Path(audio_path), segment_index, ".mp3", output_dir)
                segment_index += 1
            to_stitch = []
            length_accumulated = 0.0

        to_stitch.append(slice_audio)
        length_accumulated += (end - start)

    if to_stitch and length_accumulated >= min_len_sec:
        save_cut(to_stitch, pathlib.Path(audio_path), segment_index, ".mp3", output_dir)

def process_cut(vad_dir, audio_dir, output_dir, test_sample=None):
    json_files = []
    for root, _, files in os.walk(vad_dir):
        for file in files:
            if file.lower().endswith('.json'):
                json_files.append(os.path.join(root, file))

    if test_sample:
        json_files = json_files[:test_sample]

    print(f"[Cut] Processing {len(json_files)} files.")
    os.makedirs(output_dir, exist_ok=True)

    for json_file in tqdm.tqdm(json_files, desc="Cutting Audio"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            vad = data.get("voice_activity", [])
            rel_path = os.path.relpath(json_file, vad_dir)
            audio_path = os.path.join(audio_dir, os.path.splitext(rel_path)[0] + '.mp3')

            if not os.path.exists(audio_path):
                print(f"[Cut] Audio file not found: {audio_path}")
                continue

            cut_sequence(audio_path, vad, output_dir)
        except Exception as e:
            print(f"[Cut] Error processing {json_file}: {e}")

# -------------------------------------------------
# Remove Intro Segments
# -------------------------------------------------
def remove_intro_segments(input_dir):
    removed_count = 0
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith("_0000.mp3"):
                try:
                    os.remove(os.path.join(root, file))
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing {file}: {e}")
    print(f"[RemoveIntro] Removed {removed_count} intro segments.")

# -------------------------------------------------
# 오디오 총 길이 계산 유틸
# -------------------------------------------------
def calculate_total_audio_hours(dir_path, ext=".mp3", max_workers=4):  # 맥북용으로 worker 수 감소
    """dir_path 내부의 모든 MP3 파일의 총 재생시간(시간 단위)을 리턴."""
    audio_files = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith(ext):
                audio_files.append(os.path.join(root, f))

    print(f"[Audio Length] Found {len(audio_files)} audio files")
    total_secs = 0

    def get_length(fpath):
        try:
            info = sf.info(fpath)
            return info.frames / info.samplerate
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            return 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(get_length, f) for f in audio_files]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Calc Audio Length", ncols=80):
            total_secs += future.result()

    return total_secs / 3600.0  # 초 단위 -> 시간 단위

# -------------------------------------------------
# 파이프라인 실행
# -------------------------------------------------
def run_pipeline(args):
    stage_timer = StageTimer()
    base_dir = pathlib.Path(args.base_dir)

    download_dir = base_dir / "downloads"
    vad_dir = base_dir / "vad"
    cut_dir = base_dir / "cut"

    start_stage_idx = list(PIPELINE_STAGES.keys()).index(args.start_stage)
    end_stage_idx = list(PIPELINE_STAGES.keys()).index(args.end_stage)
    stages_to_run = list(PIPELINE_STAGES.keys())[start_stage_idx:end_stage_idx + 1]

    for d in [download_dir, vad_dir, cut_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if "download" in stages_to_run:
        stage_timer.start("download")
        download_urls("urls.txt", download_dir, args.test_sample)
        stage_timer.end("download")
        
        # 다운로드 직후 오디오 총 길이 출력
        total_hours = calculate_total_audio_hours(download_dir)
        print(f"\n[Audio Length] After download = {total_hours:.2f} hours")

    if "vad" in stages_to_run:
        stage_timer.start("vad")
        process_vad(download_dir, vad_dir, args.test_sample)
        stage_timer.end("vad")

    if "cut" in stages_to_run:
        stage_timer.start("cut")
        process_cut(vad_dir, download_dir, cut_dir, args.test_sample)
        stage_timer.end("cut")

    if "intro" in stages_to_run:
        stage_timer.start("intro")
        remove_intro_segments(cut_dir)
        stage_timer.end("intro")

    # 최종 컷 후 오디오 길이 (cut 단계가 포함된 경우에만)
    if end_stage_idx >= list(PIPELINE_STAGES.keys()).index("cut"):
        final_hours = calculate_total_audio_hours(cut_dir)
        print(f"\n[Audio Length] After cutting = {final_hours:.2f} hours")

    stage_timer.report()

def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight Audio Processing Pipeline")
    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--test_sample", type=int, default=None, 
                       help="Process only N files (for testing)")
    parser.add_argument("--start_stage", type=str, choices=list(PIPELINE_STAGES.keys()),
                       default="download")
    parser.add_argument("--end_stage", type=str, choices=list(PIPELINE_STAGES.keys()),
                       default="intro")

    args = parser.parse_args()

    if list(PIPELINE_STAGES.keys()).index(args.start_stage) > list(PIPELINE_STAGES.keys()).index(args.end_stage):
        parser.error("start_stage cannot come after end_stage")

    return args

def main():
    args = parse_args()
    run_pipeline(args)

if __name__ == "__main__":
    main() 