#!/usr/bin/env python3
import os
import sys
import argparse
import requests
import urllib.request
import zipfile
import subprocess
import multiprocessing
import json
import pathlib
import shutil
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import tqdm
import numpy as np
import soundfile as sf
import torch
import torchaudio
from google.cloud import storage
import shlex

# -------------------------------------------------
# 기본 설정값
# -------------------------------------------------
DEFAULT_BASE_DIR = "data"
DEFAULT_LANGUAGE = "English"
DEFAULT_API_LIMIT = 200
DEFAULT_MAX_URLS = 5
DEFAULT_VM_INDEX = 0
DEFAULT_NUM_VMS = 1
DEFAULT_FORMAT = ".mp3"
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_N_PROCESSES = multiprocessing.cpu_count()
DEFAULT_MIN_SPEECH_DURATION = 0.5
DEFAULT_TARGET_LEN_SEC = 30
DEFAULT_GCS_BUCKET = "nari-librivox-test"
DEFAULT_GCS_PREFIX = "test"

_VAD_MODEL = None
_VAD_UTILS = None

# -------------------------------------------------
# 타임 트래커 유틸 (각 파이프라인 단계 시간 측정용)
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
        self.times[stage_name]["duration"] = self.times[stage_name]["end"] - self.times[stage_name]["start"]

    def report(self):
        print("\n[Pipeline Stage Times]")
        for stage_name, tinfo in self.times.items():
            start_t = tinfo["start"]
            end_t = tinfo["end"]
            duration = tinfo["duration"]
            if start_t is None or end_t is None or duration is None:
                continue
            print(f"  - {stage_name:<20}: {duration:.2f} seconds ({duration/60:.2f} minutes)")


def get_vad_model():
    global _VAD_MODEL, _VAD_UTILS
    if _VAD_MODEL is None:
        _VAD_MODEL, _VAD_UTILS = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True, force_reload=False)
    return _VAD_MODEL, _VAD_UTILS

# -------------------------------------------------
# 다운로드 함수들
# -------------------------------------------------
def _download_file(url, output_dir):
    """파일 1개를 다운로드"""
    file_name = os.path.basename(url)
    save_path = os.path.join(output_dir, file_name)

    if os.path.exists(save_path):
        # 이미 존재하면 스킵
        return file_name, True

    try:
        with urllib.request.urlopen(url) as response, open(save_path, 'wb') as out_file:
            total = int(response.info().get('Content-Length', -1))
            downloaded = 0
            block_size = 8192
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
        return file_name, True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return file_name, False

def download_urls(urls_file, output_dir, vm_index=0, num_vms=1, test_sample=None, n_threads=8):
    """VM 분산 + 멀티스레드로 다운로드."""
    os.makedirs(output_dir, exist_ok=True)
    with open(urls_file, "r", encoding="utf-8") as f:
        all_urls = [line.strip() for line in f if line.strip()]

    # VM 분산 (예: 4대 중 0번째 VM이면 i%4==0만 다운로드)
    urls = [url for i, url in enumerate(all_urls) if i % num_vms == vm_index]

    if test_sample:
        urls = urls[:test_sample]

    print(f"[Download] VM {vm_index}/{num_vms} => {len(urls)} URLs to download.")

    # ThreadPoolExecutor로 병렬 다운로드
    results = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        future_to_url = {executor.submit(_download_file, url, output_dir): url for url in urls}
        for future in tqdm.tqdm(as_completed(future_to_url), total=len(urls), desc="Downloading", ncols=80):
            url = future_to_url[future]
            try:
                file_name, status = future.result()
                results.append((file_name, status))
            except Exception as e:
                print(f"[ERROR] Download failed for {url}: {e}")

    # 다운로드 결과 요약
    success_count = sum(1 for _, s in results if s)
    fail_count = len(results) - success_count
    print(f"[Download Summary] Success={success_count}, Failed={fail_count} / Total={len(urls)}")


# -------------------------------------------------
# Unzip & Convert (ffmpeg)
# -------------------------------------------------
def _unzip_and_convert_single(task):
    zip_file, params = task
    in_dir = params["in_dir"]
    out_dir = params["out_dir"]
    sample_rate = params["sample_rate"]

    zip_path = os.path.join(in_dir, zip_file)
    out_base = os.path.join(out_dir, os.path.splitext(zip_file)[0])
    os.makedirs(out_base, exist_ok=True)

    temp_dir = os.path.join(out_base, "_temp_extracted")
    os.makedirs(temp_dir, exist_ok=True)

    # ZIP 풀기
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_dir)
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return

    # ffmpeg 변환 (mp3)
    for root, dirs, files in os.walk(temp_dir):
        for filename in files:
            if filename.lower().endswith(('.mp3', '.wav', '.ogg', '.flac')):
                in_file = os.path.join(root, filename)
                out_file_name = os.path.splitext(filename)[0] + ".mp3"
                out_file = os.path.join(out_base, out_file_name)

                if os.path.exists(out_file):
                    continue

                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-i", in_file,
                    "-ac", "1",
                    "-ar", str(sample_rate),
                    "-af", "aresample=resampler=soxr:precision=20",
                    "-f", "mp3",
                    out_file
                ]
                try:
                    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception as e:
                    print(f"Error converting {in_file} from {zip_path}: {e}")

    # 임시폴더 정리
    shutil.rmtree(temp_dir, ignore_errors=True)

def unzip_and_convert(in_dir, out_dir, sample_rate=44100, n_processes=1):
    os.makedirs(out_dir, exist_ok=True)
    zip_files = [f for f in os.listdir(in_dir) if f.lower().endswith('.zip')]
    print(f"Found {len(zip_files)} zip files to unzip and convert.")

    params = {
        "in_dir": in_dir,
        "out_dir": out_dir,
        "sample_rate": sample_rate
    }

    with multiprocessing.Pool(processes=n_processes) as pool:
        with tqdm.tqdm(total=len(zip_files), desc="Unzipping+Converting", ncols=80) as pbar:
            for _ in pool.imap_unordered(_unzip_and_convert_single, [(zf, params) for zf in zip_files]):
                pbar.update()

    print("[Unzip+Convert] Completed.")


# -------------------------------------------------
# VAD
# -------------------------------------------------
def extract_speech_timestamps(
    waveform: torch.Tensor,
    sample_rate: int,
    model,
    get_speech_timestamps,
    threshold: float = 0.3,
    min_speech_duration_ms: int = 500,
    min_silence_duration_ms: int = 500,
):
    speech_ts = get_speech_timestamps(
        waveform,
        model,
        threshold=threshold,
        sampling_rate=sample_rate,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
    )
    return [(ts["start"] / sample_rate, ts["end"] / sample_rate) for ts in speech_ts]

def apply_vad(audio_path, sample_rate=16000, min_speech_duration=0.5):
    model, utils = get_vad_model()
    get_speech_timestamps = utils[0]

    waveform, sr = torchaudio.load(audio_path)

    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        sr = 16000

    timestamps = extract_speech_timestamps(
        waveform, sr, model, get_speech_timestamps,
        threshold=0.3,
        min_speech_duration_ms=int(min_speech_duration*1000),
        min_silence_duration_ms=500
    )
    segments = [{'start': start, 'end': end} for start, end in timestamps]
    return segments

def save_vad_results(audio_path, vad_segments, output_file):
    data = {
        "audio_file": str(audio_path),
        "voice_activity": vad_segments
    }
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

def _process_vad_single(args):
    audio_path, input_dir, sample_rate, min_speech_duration = args
    try:
        segments = apply_vad(audio_path, sample_rate=sample_rate, min_speech_duration=min_speech_duration)
        rel_path = os.path.relpath(audio_path, input_dir)
        return rel_path, segments
    except Exception as e:
        print(f"Error processing VAD for {audio_path}: {e}")
        return None, None

def process_vad(input_dir, output_dir, sample_rate=16000, min_speech_duration=0.5, test_sample=None, n_processes=1):
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.mp3'):
                audio_files.append(os.path.join(root, file))
    if test_sample:
        audio_files = audio_files[:test_sample]

    print(f"Found {len(audio_files)} audio files for VAD processing.")
    os.makedirs(output_dir, exist_ok=True)

    tasks = [(audio_path, input_dir, sample_rate, min_speech_duration) for audio_path in audio_files]

    with multiprocessing.Pool(processes=n_processes) as pool:
        with tqdm.tqdm(total=len(tasks), desc="VAD Processing", ncols=80) as pbar:
            for rel_path, segments in pool.imap_unordered(_process_vad_single, tasks):
                if rel_path is not None and segments is not None:
                    json_out = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.json')
                    os.makedirs(os.path.dirname(json_out), exist_ok=True)
                    save_vad_results(os.path.join(input_dir, rel_path), segments, json_out)
                pbar.update()

    print("[VAD] Processing completed.")


# -------------------------------------------------
# 컷팅
# -------------------------------------------------
def save_cut(seq, fname, index, extension, output_dir):
    output = np.hstack(seq)
    file_name = pathlib.Path(output_dir) / fname.name / (fname.stem + f"_{index:04}{extension}")
    file_name.parent.mkdir(exist_ok=True, parents=True)
    sf.write(file_name, output, samplerate=44100)

def cut_sequence(audio_path, vad, output_dir, min_len_sec=15, max_len_sec=30, out_extension=".mp3"):
    data, samplerate = sf.read(audio_path)

    if len(data.shape) != 1:
        raise ValueError(f"{audio_path} is not mono audio.")
    if samplerate != 44100:
        raise ValueError(f"{audio_path} samplerate {samplerate} != 44100")

    to_stitch = []
    length_accumulated = 0.0
    segment_index = 0

    for segment in vad:
        start, end = segment['start'], segment['end']
        start_idx, end_idx = int(start * samplerate), int(end * samplerate)
        slice_audio = data[start_idx:end_idx]

        # max_len_sec를 초과할 것 같으면 지금까지 모은 것을 먼저 저장
        if length_accumulated + (end - start) > max_len_sec:
            if len(to_stitch) > 0:
                save_cut(to_stitch, pathlib.Path(audio_path), segment_index, out_extension, output_dir)
                segment_index += 1
            to_stitch = []
            length_accumulated = 0.0

        to_stitch.append(slice_audio)
        length_accumulated += (end - start)

    # 마지막에 남은 조각이 min_len_sec 이상이면 저장
    if to_stitch and length_accumulated >= min_len_sec:
        save_cut(to_stitch, pathlib.Path(audio_path), segment_index, out_extension, output_dir)
    else:
        if to_stitch:
            print(f"[Cut] Last segment too short ({length_accumulated:.2f}s), discarding => {audio_path}")

def _process_cut_single(args):
    json_file, vad_dir, audio_dir, output_dir, min_len_sec, max_len_sec, out_extension = args
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        vad = data.get("voice_activity", [])
        rel_path = os.path.relpath(json_file, vad_dir)
        audio_path = os.path.join(audio_dir, os.path.splitext(rel_path)[0] + '.mp3')

        if not os.path.exists(audio_path):
            print(f"Audio file {audio_path} not found for {json_file}")
            return

        out_subdir = os.path.join(output_dir, os.path.dirname(rel_path))
        cut_sequence(audio_path, vad, out_subdir, min_len_sec, max_len_sec, out_extension)
    except Exception as e:
        print(f"Error cutting {audio_path}: {e}")

def process_cut(vad_dir, audio_dir, output_dir, min_len_sec=15, max_len_sec=30, out_extension=".mp3", test_sample=None, n_processes=1):
    json_files = []
    for root, _, files in os.walk(vad_dir):
        for file in files:
            if file.lower().endswith('.json'):
                json_files.append(os.path.join(root, file))

    if test_sample:
        json_files = json_files[:test_sample]
    print(f"[Cut] Found {len(json_files)} VAD json files for cutting.")

    os.makedirs(output_dir, exist_ok=True)

    tasks = [
        (json_file, vad_dir, audio_dir, output_dir, min_len_sec, max_len_sec, out_extension)
        for json_file in json_files
    ]

    with multiprocessing.Pool(processes=n_processes) as pool:
        with tqdm.tqdm(total=len(tasks), desc="Cutting Audio", ncols=80) as pbar:
            for _ in pool.imap_unordered(_process_cut_single, tasks):
                pbar.update()

    print("[Cut] Completed cutting.")


# -------------------------------------------------
# Remove Intro Segments
# -------------------------------------------------
def remove_intro_segments(input_dir):
    """_0000.mp3 파일 제거"""
    removed_count = 0
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith("_0000.mp3"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
    print(f"[RemoveIntro] Removed {removed_count} intro segments.")


# -------------------------------------------------
# GCS Upload
# -------------------------------------------------
def upload_to_gcs_with_gsutil(local_dir, bucket_name, remote_prefix):
    gsutil_command = [
        'gsutil', '-m', 'cp', '-r',
        os.path.join(local_dir, '*'),
        f'gs://{bucket_name}/{remote_prefix}/'
    ]
    try:
        subprocess.run(gsutil_command, check=True)
        print(f"[GCS Upload] Successfully uploaded {local_dir} to gs://{bucket_name}/{remote_prefix}/")
    except subprocess.CalledProcessError as e:
        print(f"[GCS Upload] Error occurred: {e}")


# -------------------------------------------------
# 오디오 총 길이 계산 유틸
# -------------------------------------------------
def calculate_total_audio_hours(dir_path, ext=".mp3", max_workers=8):
    """dir_path 내부의 모든 mp3 파일의 총 재생시간(시간 단위)을 리턴."""
    audio_files = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith(ext):
                audio_files.append(os.path.join(root, f))

    # 병렬로 메타데이터를 얻어 합산
    total_secs = 0

    def get_length(fpath):
        try:
            info = sf.info(fpath)
            return info.frames / info.samplerate
        except:
            return 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(get_length, f) for f in audio_files]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Calc Audio Length", ncols=80):
            total_secs += future.result()

    return total_secs / 3600.0  # 초 단위 -> 시간 단위


# -------------------------------------------------
# 전체 파이프라인
# -------------------------------------------------
def run_pipeline(args):
    stage_timer = StageTimer()

    base_dir = args.base_dir
    downloads_dir = os.path.join(base_dir, "downloads")
    converted_dir = os.path.join(base_dir, "converted")
    vad_dir = os.path.join(base_dir, "vad_results")
    cut_dir = os.path.join(base_dir, "cut_segments")

    # 폴더 생성
    for d in [downloads_dir, converted_dir, vad_dir, cut_dir]:
        os.makedirs(d, exist_ok=True)

    # 1. Download
    stage_timer.start("Download")
    download_urls(
        urls_file=os.path.join(base_dir, "urls.txt"),
        output_dir=downloads_dir,
        vm_index=args.vm_index,
        num_vms=args.num_vms,
        test_sample=args.test_sample,
        n_threads=min(16, multiprocessing.cpu_count())  # 적절히 조절
    )
    stage_timer.end("Download")

    # (선택) 다운로드된 MP3의 총 길이 측정
    #  - 지금 단계에서는 ZIP만 있을 수도 있으므로, ZIP이 아니라 MP3를 직접 받는 케이스만 유효.
    #  - ZIP 파일이 대부분이면, "unzip_and_convert" 후에 계산해야 함.
    #  - 만약 "직접 mp3"를 다운로드받는 구조라면 이 시점에서 시간 계산이 가능.
    # 여기서는 'converted_dir'을 만들기 전, ZIP을 푸는 단계 이후에 실제 MP3 길이를 측정하는 것을 예시로 삼겠습니다.

    # 2. Unzip + Convert
    stage_timer.start("UnzipConvert")
    unzip_and_convert(
        in_dir=downloads_dir,
        out_dir=converted_dir,
        sample_rate=args.sample_rate,
        n_processes=args.n_processes
    )
    stage_timer.end("UnzipConvert")

    # ---> "다운로드/변환 완료 후" 오디오 총 길이 계산
    print("\n[Audio Length] Calculating total hours after conversion...")
    total_hours_after_conversion = calculate_total_audio_hours(converted_dir, ext=".mp3")
    print(f"[Audio Length] Total hours (after conversion) = {total_hours_after_conversion:.2f} hours")

    # 3. VAD
    stage_timer.start("VAD")
    process_vad(
        input_dir=converted_dir,
        output_dir=vad_dir,
        sample_rate=16000,
        min_speech_duration=args.min_speech_duration,
        test_sample=args.test_sample,
        n_processes=args.n_processes
    )
    stage_timer.end("VAD")

    # 4. Cutting
    stage_timer.start("Cut")
    process_cut(
        vad_dir=vad_dir,
        audio_dir=converted_dir,
        output_dir=cut_dir,
        out_extension=".mp3",
        test_sample=args.test_sample,
        n_processes=args.n_processes
    )
    stage_timer.end("Cut")

    # 5. Remove Intro Segments
    stage_timer.start("RemoveIntro")
    remove_intro_segments(cut_dir)
    stage_timer.end("RemoveIntro")



    # 6. (옵션) Storage Upload
    if args.gcs_bucket:
        stage_timer.start("GCS Upload")
        print("\n[Upload] Uploading cut segments to GCS...")
        upload_to_gcs_with_gsutil(
            local_dir=cut_dir,
            bucket_name=args.gcs_bucket,
            remote_prefix=args.gcs_prefix
        )
        stage_timer.end("GCS Upload")

    # 타임 리포트
    # 최종 오디오 길이 계산
    print("\n[Audio Length] Calculating total hours after cutting...")
    total_hours_final = calculate_total_audio_hours(cut_dir, ext=".mp3")
    print(f"[Audio Length] Final total hours (after cutting) = {total_hours_final:.2f} hours")
    if total_hours_after_conversion > 0:
        ratio = (total_hours_final / total_hours_after_conversion) * 100
        print(f" => Retained {ratio:.2f}% of originally converted audio.")
    stage_timer.report()
    print("\n[Pipeline] All stages completed successfully.")


# -------------------------------------------------
# CLI 인자 파싱
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Librivox End-to-End Pipeline")

    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--language", type=str, default=DEFAULT_LANGUAGE)
    parser.add_argument("--api_limit", type=int, default=DEFAULT_API_LIMIT)
    parser.add_argument("--max_urls", type=int, default=DEFAULT_MAX_URLS)
    parser.add_argument("--vm_index", type=int, default=DEFAULT_VM_INDEX)
    parser.add_argument("--num_vms", type=int, default=DEFAULT_NUM_VMS)
    parser.add_argument("--test_sample", type=int, default=None)
    parser.add_argument("--format", type=str, default=DEFAULT_FORMAT)
    parser.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--n_processes", type=int, default=DEFAULT_N_PROCESSES)
    parser.add_argument("--min_speech_duration", type=float, default=DEFAULT_MIN_SPEECH_DURATION)
    parser.add_argument("--target_len_sec", type=int, default=DEFAULT_TARGET_LEN_SEC)
    parser.add_argument("--gcs_bucket", type=str, default=None)
    parser.add_argument("--gcs_prefix", type=str, default=DEFAULT_GCS_PREFIX)
    parser.add_argument("--credentials", type=str, help="Path to Google Cloud service account key file (optional).")
    parser.add_argument("--pipeline", action="store_true", help="Run the entire pipeline.")

    args = parser.parse_args()
    if args.credentials:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = args.credentials

    return args

def main():
    args = parse_args()
    if args.pipeline:
        run_pipeline(args)
    else:
        print("Specify the --pipeline option to run the entire pipeline.")

if __name__ == "__main__":
    main()
