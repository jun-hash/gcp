#!/usr/bin/env python3
import os
import sys
import argparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import math

import subprocess
import multiprocessing
import json
import pathlib
import shutil
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
# 파이프라인 단계 정의
# -------------------------------------------------
PIPELINE_STAGES = {
    "download": "Download audio files",
    "vad":      "Voice Activity Detection",
    "cut":      "Cut segments",
    "intro":    "Remove intro segments",
    "upload":   "Upload to GCS"
}

# -------------------------------------------------
# 기본 설정값
# -------------------------------------------------
DEFAULT_BASE_DIR            = "data"
DEFAULT_LANGUAGE            = "English"
DEFAULT_SAMPLE_RATE         = 44100  # 이미 44.1kHz mp3를 받는다고 가정
DEFAULT_N_PROCESSES         = multiprocessing.cpu_count()
DEFAULT_MIN_SPEECH_DURATION = 0.5
DEFAULT_TARGET_LEN_SEC      = 30
DEFAULT_GCS_BUCKET          = "nari-librivox"
DEFAULT_GCS_PREFIX          = "v0"
DEFAULT_FORMAT              = ".mp3"

# (원하는 만큼 조정 가능)
DEFAULT_THREADS_DOWNLOAD    = 16     # 병렬 다운로드 시 스레드 개수
DEFAULT_POOL_SIZE           = 100    # requests 세션 커넥션 풀 크기

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
        self.times[stage_name]["duration"] = (
            self.times[stage_name]["end"] - self.times[stage_name]["start"]
        )

    def report(self):
        print("\n[Pipeline Stage Times]")
        for stage_name, tinfo in self.times.items():
            start_t = tinfo["start"]
            end_t = tinfo["end"]
            duration = tinfo["duration"]
            if start_t is None or end_t is None or duration is None:
                continue
            print(f"  - {stage_name:<20}: {duration:.2f} seconds ({duration/60:.2f} minutes)")


# -------------------------------------------------
# Download (MP3)
# -------------------------------------------------
def create_session(max_retries=3, backoff_factor=0.5, pool_connections=100, pool_maxsize=100):
    """
    requests 세션(Session)을 생성.
    - HTTPAdapter로 연결 풀(pool) 크기를 설정하여 동시 다운로드 시 성능을 개선
    - max_retries: 연결 오류/일시적 네트워크 문제 시 재시도 횟수
    - backoff_factor: 재시도 시 지수 백오프
    """
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

def _download_file(url, output_dir, session=None, block_size=65536):
    """
    개별 URL 하나를 다운로드하는 함수.
    session: requests.Session 객체 (연결 풀 재사용)
    block_size: 스트리밍 다운로드 시 한번에 읽을 바이트 크기
    """
    file_name = os.path.basename(url)
    save_path = os.path.join(output_dir, file_name)

    # 이미 다운로드된 파일이 있으면 스킵
    if os.path.exists(save_path):
        return file_name, True

    if session is None:
        session = create_session()

    try:
        with session.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('Content-Length', 0))

            with open(save_path, "wb") as out_file:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=block_size):
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
        return file_name, True

    except Exception as e:
        return file_name, False

def download_urls(urls_file, output_dir, test_sample=None, n_threads=16):
    """
    - urls_file: 한 줄에 하나씩 MP3 파일 URL이 들어 있는 텍스트 파일
    - output_dir: 다운로드 파일을 저장할 디렉토리
    - test_sample: 앞에서부터 N개만 다운로드 (테스트용), None이면 전체 다운로드
    - n_threads: 병렬 스레드 개수
    """
    os.makedirs(output_dir, exist_ok=True)

    # URL 목록 읽기
    with open(urls_file, "r", encoding="utf-8") as f:
        all_urls = [line.strip() for line in f if line.strip()]

    # test_sample 지정 시 일부만 다운로드
    if test_sample:
        all_urls = all_urls[:test_sample]

    print(f"[Download] {len(all_urls)} URLs to download.")

    # 세션 하나를 만들어 여러 스레드에서 공유 (HTTP Connection Pool 재사용)
    session = create_session(pool_connections=DEFAULT_POOL_SIZE, pool_maxsize=DEFAULT_POOL_SIZE)

    results = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        future_to_url = {
            executor.submit(_download_file, url, output_dir, session): url
            for url in all_urls
        }

        for future in tqdm.tqdm(as_completed(future_to_url), total=len(all_urls), desc="Downloading", ncols=80):
            url = future_to_url[future]
            try:
                file_name, status = future.result()
                results.append((file_name, status))
            except Exception as e:
                print(f"[ERROR] Download failed for {url}: {e}")
                results.append(("Unknown", False))

    success_count = sum(1 for _, s in results if s)
    fail_count = len(results) - success_count
    print(f"[Download Summary] Success={success_count}, Failed={fail_count}, Total={len(all_urls)}")


# -------------------------------------------------
# Silero VAD
# -------------------------------------------------
# 전역에서 한 번만 모델 로드
print("[VAD] Loading model...")
_VAD_MODEL, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    trust_repo=True,
    force_reload=False
)
_GET_TIMESTAMPS = utils[0]  # get_speech_timestamps 함수

def apply_vad(audio_path, sample_rate=16000, min_speech_duration=0.5):
    """
    VAD를 적용하여 (start, end) 초 단위 구간을 반환
    """
    # 전역 모델 사용
    global _VAD_MODEL, _GET_TIMESTAMPS

    # stereo mp3를 로드하되, 모노 합성을 위해 mean(dim=0)
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sr != sample_rate:
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = transform(waveform)
        sr = sample_rate

    timestamps = _GET_TIMESTAMPS(
        waveform,
        _VAD_MODEL,
        threshold=0.3,
        sampling_rate=sr,
        min_speech_duration_ms=int(min_speech_duration*1000),
        min_silence_duration_ms=500
    )
    
    segments = [{'start': ts['start'] / sr, 'end': ts['end'] / sr} for ts in timestamps]
    return segments

def save_vad_results(audio_path, vad_segments, output_file):
    data = {
        "audio_file": str(audio_path),
        "voice_activity": vad_segments
    }
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

def _process_vad_single(args):
    """VAD 처리 작업자 함수 수정"""
    audio_path, input_dir, sample_rate, min_speech_duration = args
    try:
        # 각 프로세스에서 처음 실행될 때 모델 로드
        segments = apply_vad(audio_path, sample_rate=sample_rate, min_speech_duration=min_speech_duration)
        rel_path = os.path.relpath(audio_path, input_dir)
        return rel_path, segments
    except Exception as e:
        print(f"[ERROR] VAD processing failed for {audio_path}: {str(e)}")
        return None, None

def process_vad(
    input_dir,
    output_dir,
    sample_rate=16000,
    min_speech_duration=0.5,
    test_sample=None,
    n_processes=1
):
    """
    VAD 처리 메인 함수
    """
    # 1) 입력 디렉토리 내 MP3 파일 수집
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.mp3'):
                audio_files.append(os.path.join(root, file))

    # 2) 샘플링(테스트용) 
    if test_sample:
        audio_files = audio_files[:test_sample]

    print(f"[VAD] Found {len(audio_files)} audio files.")
    os.makedirs(output_dir, exist_ok=True)

    # 3) 이미 처리된 파일 스킵하기
    tasks = []
    for audio_path in audio_files:
        rel_path = os.path.relpath(audio_path, input_dir)
        json_out = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.json')

        # 이미 결과 파일이 존재하면 스킵
        if os.path.exists(json_out):
            print(f"[VAD] Skipped already processed file: {rel_path}")
            continue

        tasks.append((audio_path, input_dir, sample_rate, min_speech_duration))

    print(f"[VAD] Processing {len(tasks)} files (others skipped).")

    # 4) 멀티프로세스를 활용한 처리
    with multiprocessing.Pool(processes=n_processes) as pool:
        with tqdm.tqdm(total=len(tasks), desc="VAD Processing", ncols=80) as pbar:
            for rel_path, segments in pool.imap_unordered(_process_vad_single, tasks):
                if rel_path is not None and segments is not None:
                    # 결과 저장
                    json_out = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.json')
                    os.makedirs(os.path.dirname(json_out), exist_ok=True)
                    save_vad_results(os.path.join(input_dir, rel_path), segments, json_out)
                    print(f"[VAD] Processed file: {rel_path}")
                pbar.update()

    print("[VAD] Processing completed.")


# -------------------------------------------------
# Cutting (FFmpeg-based)
# -------------------------------------------------
def unify_vad_segments(vad_list, min_len=15.0, max_len=30.0, eps=1e-6):
    """
    VAD로 나온 구간(초 단위 start/end)을 합쳐서
    하나의 구간이 최대 max_len(기본 30초)을 넘지 않도록 분할/병합하는 함수.
    
    - vad_list: [{"start": float, "end": float}, ...] 형태의 리스트
    - min_len: 최소 길이(이보다 짧으면 최종 구간으로 채택 X)
    - max_len: 최대 길이(이보다 길면 여러 조각으로 쪼갬)
    - eps: float 비교용 작은 값
    
    반환값: [(start_sec, end_sec), (start_sec, end_sec), ...] 형태의 리스트
    """
    final_segments = []
    current_chunk = []  # 현재 모으고 있는 segment들의 리스트 (start, end) 튜플들

    def flush_chunk():
        """current_chunk를 final_segments에 반영(필요시 분할)하고 비움"""
        nonlocal current_chunk
        if not current_chunk:
            return

        seg_start = current_chunk[0][0]
        seg_end   = current_chunk[-1][1]
        length = seg_end - seg_start

        # 혹시 청크가 max_len을 초과한다면, while문으로 쪼개기
        while length > max_len + eps:
            cut_point = seg_start + max_len
            final_segments.append((seg_start, cut_point))
            seg_start = cut_point
            length = seg_end - seg_start

        # 남은 조각이 min_len 이상이면 final_segments에 추가
        if (seg_end - seg_start) >= min_len - eps:
            final_segments.append((seg_start, seg_end))

        current_chunk = []

    for seg in vad_list:
        start = seg["start"]
        end   = seg["end"]
        seg_len = end - start

        # 1) VAD segment 자체가 max_len보다 클 때 => 통째로 쪼갠 뒤, current_chunk에 합치지 않음
        if seg_len > max_len + eps:
            # 현재까지 모은 청크를 먼저 flush
            flush_chunk()

            # seg를 여러 조각으로 분할
            t = start
            while t < end - eps:
                t_next = min(t + max_len, end)
                piece_len = t_next - t
                if piece_len >= min_len - eps:
                    final_segments.append((t, t_next))
                t = t_next
            continue

        # 2) 현재 청크에 seg를 더했을 때 max_len을 넘는지 검사
        if current_chunk:
            chunk_start = current_chunk[0][0]
            chunk_end   = current_chunk[-1][1]
            current_len = chunk_end - chunk_start

            # 새 seg를 합친 예상 길이
            if (current_len + seg_len) > max_len + eps:
                # 이미 모은 청크를 flush하고, seg는 새 청크로 시작
                flush_chunk()

        # 3) current_chunk가 비어 있으면 바로 seg 추가, 아니면 이어붙임
        if not current_chunk:
            current_chunk = [(start, end)]
        else:
            current_chunk.append((start, end))

    # 반복문 종료 후 남은 chunk도 flush
    flush_chunk()

    return final_segments

def cut_segments_ffmpeg(infile, segments, outdir, sub_batch_size=20):
    """Cuts audio segments using ffmpeg with copy codec."""
    infile = str(infile)
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    base_name = pathlib.Path(infile).stem

    base_cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-y",
        "-i", infile,
        "-map_metadata", "-1",
    ]

    for i in range(0, len(segments), sub_batch_size):
        sub = segments[i:i + sub_batch_size]
        cmd_parts = []
        for idx, (start, end) in enumerate(sub):
            duration = round(end - start, 3)
            seg_idx = i + idx
            out_path = outdir / f"{base_name}_{seg_idx:04d}.mp3"
            cmd_parts += [
                "-ss", str(round(start, 3)),
                "-t", str(duration),
                "-c:a", "copy",
                str(out_path)
            ]

        cmd = base_cmd + cmd_parts

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            # Fallback to re-encoding if copy fails
            tqdm.tqdm.write(f"[WARN] -c copy failed => re-encoding sub-batch for {infile}")
            cmd_parts_re = []
            for idx, (start, end) in enumerate(sub):
                duration = round(end - start, 3)
                seg_idx = i + idx
                out_path = outdir / f"{base_name}_{seg_idx:04d}.mp3"
                cmd_parts_re += [
                    "-ss", str(round(start, 3)),
                    "-t", str(duration),
                    "-b:a", "128k",
                    "-acodec", "libmp3lame",
                    str(out_path)
                ]
            cmd_reencode = base_cmd + cmd_parts_re
            try:
                subprocess.run(cmd_reencode, check=True)
            except subprocess.CalledProcessError as e:
                tqdm.tqdm.write(f"[ERROR] Even re-encoding failed => skip sub-batch. {e}")

def _process_cut_single(args):
    """Worker function for cutting audio files."""
    json_file, vad_dir, audio_dir, output_dir, min_len_sec, max_len_sec = args
    rel_path = os.path.relpath(json_file, vad_dir)
    audio_path = os.path.join(audio_dir, os.path.splitext(rel_path)[0] + ".mp3")

    if not os.path.exists(audio_path):
        return f"[WARN] Audio file not found for {json_file}"

    try:
        with open(json_file, "r") as f:
            data = json.load(f)
        raw_vad = data.get("voice_activity", [])

        # Merge/split into min_len~max_len segments
        merged_segments = unify_vad_segments(raw_vad, min_len_sec, max_len_sec)
        if not merged_segments:
            return f"[INFO] No valid segments in {json_file}"

        # Cut using ffmpeg
        out_dir_final = os.path.join(output_dir, os.path.dirname(rel_path))
        pathlib.Path(out_dir_final).mkdir(parents=True, exist_ok=True)

        cut_segments_ffmpeg(
            infile=audio_path,
            segments=merged_segments,
            outdir=out_dir_final,
            sub_batch_size=20
        )
        return f"[OK] {audio_path} => {len(merged_segments)} segments"

    except Exception as e:
        return f"[ERROR] Failed to process {json_file}: {e}"

def process_cut(
    vad_dir,
    audio_dir, 
    output_dir,
    min_len_sec=15,
    max_len_sec=30,
    out_extension=".mp3",
    test_sample=None,
    n_processes=1,
    batch_size=100
):
    """Main function for cutting audio segments using ffmpeg."""
    # Collect VAD JSON files
    json_files = []
    for root, _, files in os.walk(vad_dir):
        for file in files:
            if file.lower().endswith(".json"):
                json_files.append(os.path.join(root, file))

    if test_sample:
        json_files = json_files[:test_sample]
    print(f"[Cut] Found {len(json_files)} VAD json files.")

    os.makedirs(output_dir, exist_ok=True)

    # Prepare tasks
    tasks = [
        (jf, vad_dir, audio_dir, output_dir, min_len_sec, max_len_sec)
        for jf in json_files
    ]

    # Process in parallel
    with multiprocessing.Pool(processes=n_processes) as pool:
        with tqdm.tqdm(total=len(tasks), desc="Cutting Audio", ncols=80) as pbar:
            for msg in pool.imap_unordered(_process_cut_single, tasks):
                if msg.startswith("[ERROR]") or msg.startswith("[WARN]"):
                    tqdm.tqdm.write(msg)
                pbar.update()

    print("[Cut] Completed cutting.")


# -------------------------------------------------
# Remove Intro Segments (Optional)
# -------------------------------------------------
def remove_intro_segments(input_dir):
    """
    _0000.mp3 로 끝나는 (즉 첫 번째 조각) 파일 삭제
    필요 없다면 이 단계 생략 가능
    """
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
    """
    dir_path 내부의 모든 MP3 파일의 총 재생시간(시간 단위)을 리턴
    """
    audio_files = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith(ext):
                audio_files.append(os.path.join(root, f))

    total_secs = 0

    def get_length(fpath):
        try:
            info = sf.info(fpath)
            return info.frames / info.samplerate
        except:
            return 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(get_length, f) for f in audio_files]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Calc Length", ncols=80):
            total_secs += future.result()

    return total_secs / 3600.0


# -------------------------------------------------
# 파이프라인 로직
# -------------------------------------------------
def run_pipeline(args):
    stage_timer = StageTimer()
    base_dir = pathlib.Path(args.base_dir)

    # 주요 디렉토리
    download_dir = base_dir / "downloads"   # 다운로드 원본 MP3
    vad_dir      = base_dir / "vad"         # VAD JSON
    cut_dir      = base_dir / "cut"         # 최종 잘린 MP3

    # 파이프라인 실행할 단계 찾기
    start_stage_idx = list(PIPELINE_STAGES.keys()).index(args.start_stage)
    end_stage_idx   = list(PIPELINE_STAGES.keys()).index(args.end_stage)
    stages_to_run   = list(PIPELINE_STAGES.keys())[start_stage_idx : end_stage_idx + 1]

    # 필요한 디렉토리 생성
    for stg in stages_to_run:
        if stg == "download":
            download_dir.mkdir(parents=True, exist_ok=True)
        elif stg == "vad":
            vad_dir.mkdir(parents=True, exist_ok=True)
        elif stg == "cut":
            cut_dir.mkdir(parents=True, exist_ok=True)

    # 1) Download
    if "download" in stages_to_run:
        stage_timer.start("download")
        download_urls(
            urls_file=f"urls.txt",
            output_dir=download_dir,
            test_sample=args.test_sample,
            n_threads=DEFAULT_THREADS_DOWNLOAD
        )
        stage_timer.end("download")
        # 다운로드 직후 오디오 총 길이 출력
        total_hours = calculate_total_audio_hours(download_dir, ext=args.format)
        print(f"\n[Audio Length] After download = {total_hours:.2f} hours")

    # 2) VAD
    if "vad" in stages_to_run:
        stage_timer.start("vad")
        process_vad(
            input_dir=download_dir,
            output_dir=vad_dir,
            sample_rate=16000,               # VAD는 16kHz 사용
            min_speech_duration=args.min_speech_duration,
            test_sample=args.test_sample,
            n_processes=args.n_processes
        )
        stage_timer.end("vad")

    # 3) Cut
    if "cut" in stages_to_run:
        stage_timer.start("cut")
        process_cut(
            vad_dir=vad_dir,
            audio_dir=download_dir,
            output_dir=cut_dir,
            out_extension=".mp3",
            min_len_sec=15,
            max_len_sec=args.target_len_sec,
            test_sample=args.test_sample,
            n_processes=args.n_processes
        )
        stage_timer.end("cut")

        # Add segment statistics after cutting
        print("\n[Segment Statistics]")
        avg_dur, min_dur, max_dur, count = measure_segment_stats(cut_dir, ext=".mp3")
        print(f"  Total segments: {count:,}")
        print(f"  Average duration: {avg_dur:.2f} seconds")
        print(f"  Minimum duration: {min_dur:.2f} seconds")
        print(f"  Maximum duration: {max_dur:.2f} seconds")
        print(f"  Total duration: {(count * avg_dur / 3600):.2f} hours\n")

    # 4) Remove intro
    if "intro" in stages_to_run:
        stage_timer.start("intro")
        remove_intro_segments(cut_dir)
        stage_timer.end("intro")

    # 5) Upload
    if "upload" in stages_to_run and args.gcs_bucket:
        stage_timer.start("upload")
        print("\n[Upload] Uploading cut segments to GCS...")
        upload_to_gcs_with_gsutil(
            local_dir=cut_dir,
            bucket_name=args.gcs_bucket,
            remote_prefix=args.gcs_prefix
        )
        # 업로드 후 로컬 삭제
        if args.cleanup_after_upload:
            print("[Cleanup] Removing local cut files after GCS upload...")
            shutil.rmtree(cut_dir, ignore_errors=True)
            cut_dir.mkdir(exist_ok=True)
        stage_timer.end("upload")

    # 최종 컷 후 오디오 길이
    if end_stage_idx >= list(PIPELINE_STAGES.keys()).index("cut"):
        final_hours = calculate_total_audio_hours(cut_dir, ext=".mp3")
        print(f"\n[Audio Length] After cutting = {final_hours:.2f} hours")

    stage_timer.report()
    print(f"\n[Pipeline] Completed stages from {args.start_stage} to {args.end_stage} successfully.")


def should_run_stage(stage, start_stage, end_stage):
    """
    - (사용 안 함) 필요시 참조용
    """
    stages = list(PIPELINE_STAGES.keys())
    stage_idx = stages.index(stage)
    start_idx = stages.index(start_stage)
    end_idx = stages.index(end_stage)
    return start_idx <= stage_idx <= end_idx


# -------------------------------------------------
# argparse
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Librivox End-to-End Pipeline (MP3 direct download)")

    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--language", type=str, default=DEFAULT_LANGUAGE)
    parser.add_argument("--format", type=str, default=DEFAULT_FORMAT)
    parser.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE,
                        help="Expected sample rate (normally 44100)")

    parser.add_argument("--test_sample", type=int, default=None, help="For quick test: limit # of files")
    parser.add_argument("--n_processes", type=int, default=DEFAULT_N_PROCESSES)
    parser.add_argument("--min_speech_duration", type=float, default=DEFAULT_MIN_SPEECH_DURATION)
    parser.add_argument("--target_len_sec", type=int, default=DEFAULT_TARGET_LEN_SEC,
                        help="Max length for each cut segment")

    parser.add_argument("--gcs_bucket", type=str, default=None, help="GCS bucket name")
    parser.add_argument("--gcs_prefix", type=str, default=None,
                        help="GCS path prefix (folder-like)")
    parser.add_argument("--cleanup_after_upload", action="store_true")

    # 파이프라인 제어
    parser.add_argument("--start_stage", type=str, choices=list(PIPELINE_STAGES.keys()),
                       default="download", help="Start from this pipeline stage")
    parser.add_argument("--end_stage", type=str, choices=list(PIPELINE_STAGES.keys()),
                       default="cut", help="End at this pipeline stage")

    args = parser.parse_args()

    # 시작/종료 단계 검증
    if list(PIPELINE_STAGES.keys()).index(args.start_stage) > list(PIPELINE_STAGES.keys()).index(args.end_stage):
        parser.error("start_stage cannot come after end_stage")

    return args

def measure_segment_stats(dir_path, ext=".mp3", max_workers=8):
    """
    Measures audio segment statistics in dir_path:
    - Average duration (seconds)
    - Minimum duration (seconds)
    - Maximum duration (seconds) 
    - Total file count
    """
    audio_files = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith(ext):
                audio_files.append(os.path.join(root, f))

    durations = []

    def get_length(fpath):
        try:
            info = sf.info(fpath)
            return info.frames / info.samplerate
        except:
            return 0

    # Measure lengths using multithreading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(get_length, f) for f in audio_files]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), 
                              desc="Measuring segments", ncols=80):
            dur = future.result()
            if dur > 0:
                durations.append(dur)

    if not durations:
        return 0.0, 0.0, 0.0, 0  # No data

    total = sum(durations)
    count = len(durations)
    avg_dur = total / count
    min_dur = min(durations)
    max_dur = max(durations)

    return avg_dur, min_dur, max_dur, count

def main():
    args = parse_args()
    run_pipeline(args)

if __name__ == "__main__":
    main()
