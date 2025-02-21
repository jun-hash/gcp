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
import tqdm
import numpy as np
import soundfile as sf
import torch
import torchaudio
from google.cloud import storage
import shlex
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps


# Constants for default values
DEFAULT_BASE_DIR = "data"
DEFAULT_LANGUAGE = "English"
DEFAULT_API_LIMIT = 500
DEFAULT_MAX_URLS = 5
DEFAULT_VM_INDEX = 0
DEFAULT_NUM_VMS = 1
DEFAULT_FORMAT = ".mp3"
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_N_PROCESSES = multiprocessing.cpu_count()  # Use all available cores
DEFAULT_MIN_SPEECH_DURATION = 0.5
DEFAULT_TARGET_LEN_SEC = 30
DEFAULT_GCS_BUCKET = "nari-librivox-test"
DEFAULT_GCS_PREFIX = "test"



# Global variable to hold the VAD model
_VAD_MODEL = None

def get_vad_model():
    """Loads the Silero VAD model using the official API."""
    global _VAD_MODEL
    if _VAD_MODEL is None:
        _VAD_MODEL = load_silero_vad()
    return _VAD_MODEL

########################################
# 1. Download: File Download
########################################
def download_file(url, save_path):
    """Downloads a single file with a progress bar."""
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
                sys.stdout.write(f"\rDownloading {os.path.basename(save_path)}: {downloaded/total*100:.1f}%")
                sys.stdout.flush()

        sys.stdout.write("\n")
        print(f"Downloaded: {save_path}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")



def download_urls(urls_file, output_dir, vm_index=DEFAULT_VM_INDEX, num_vms=DEFAULT_NUM_VMS, test_sample=None):
    """Downloads files from URLs, distributed across VMs."""
    os.makedirs(output_dir, exist_ok=True)
    with open(urls_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    urls = [url for i, url in enumerate(urls) if i % num_vms == vm_index]

    if test_sample:
        urls = urls[:test_sample]
    print(f"VM {vm_index}/{num_vms}: {len(urls)} URLs to download.")

    for url in urls:
        file_name = os.path.basename(url)
        save_path = os.path.join(output_dir, file_name)
        if not os.path.exists(save_path):
            download_file(url, save_path)
        else:
            print(f"File already exists: {save_path}")



########################################
# 2. Unzip and Convert (Combined)
########################################

def unzip_and_convert(in_dir, out_dir, sample_rate=DEFAULT_SAMPLE_RATE, n_processes=DEFAULT_N_PROCESSES):
    """Unzips and converts to MP3 simultaneously, using multiprocessing."""
    os.makedirs(out_dir, exist_ok=True)
    zip_files = [f for f in os.listdir(in_dir) if f.lower().endswith('.zip')]
    print(f"Found {len(zip_files)} zip files to unzip and convert.")

    params = {
        "in_dir": in_dir,
        "out_dir": out_dir,
        "sample_rate": sample_rate
    }

    with multiprocessing.Pool(processes=n_processes) as pool:
        with tqdm.tqdm(total=len(zip_files), desc="Unzipping and Converting") as pbar:
            for _ in pool.imap_unordered(_unzip_and_convert_single, [(zip_file, params) for zip_file in zip_files]):
                pbar.update()

    print("Unzip and conversion completed.")


def _unzip_and_convert_single(task):
    """Unzips a single ZIP file and converts to MP3 (piped)."""
    zip_file, params = task
    zip_path = os.path.join(params["in_dir"], zip_file)
    out_base = os.path.join(params["out_dir"], os.path.splitext(zip_file)[0])
    os.makedirs(out_base, exist_ok=True)

    try:
        # ZIP 파일 내부 목록 확인
        zip_listing = subprocess.check_output(['unzip', '-qq', '-l', zip_path], text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error listing contents of {zip_path}: {e}")
        return

    # ZIP 내부의 오디오 파일만 골라내기
    audio_files = []
    for line in zip_listing.splitlines():
        parts = line.split()
        if len(parts) >= 4:
            fname = parts[-1]
            if fname.lower().endswith(('.mp3', '.wav', '.ogg', '.flac')):
                audio_files.append(fname)

    # 오디오 파일별로 MP3 변환
    for audio_file in audio_files:
        out_file = os.path.join(out_base, os.path.splitext(os.path.basename(audio_file))[0] + ".mp3")

        # 이미 변환된 파일이 있으면 스킵
        if os.path.exists(out_file):
            continue

        unzip_command = f"unzip -p {shlex.quote(zip_path)} {shlex.quote(audio_file)}"
        ffmpeg_command = (
            f"ffmpeg -i - -ac 1 -ar {params['sample_rate']} "
            f"-af 'aresample=resampler=soxr:precision=33' "
            f"-f mp3 {shlex.quote(out_file)}"
        )
        try:
            # unzip -> ffmpeg 파이프라인
            unzip_process = subprocess.Popen(unzip_command, shell=True, stdout=subprocess.PIPE)
            ffmpeg_process = subprocess.Popen(ffmpeg_command, shell=True, stdin=unzip_process.stdout,
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # 리소스 닫고 ffmpeg 변환 대기
            unzip_process.stdout.close()
            ffmpeg_process.communicate()

            if ffmpeg_process.returncode != 0:
                print(f"Error converting {audio_file} from {zip_file}. ffmpeg return code: {ffmpeg_process.returncode}")
                continue

        except Exception as e:
            print(f"Error processing {audio_file} from {zip_file}: {e}")



########################################
# 3. VAD: Apply Silero VAD and Save Results
########################################

def extract_speech_timestamps(
    waveform: torch.Tensor,
    sample_rate: int,
    model,
    get_speech_timestamps,
    threshold: float = 0.3,
    min_speech_duration_ms: int = 500,
    min_silence_duration_ms: int = 500,
) -> list[tuple[float, float]]:
    """Extracts speech timestamps using Silero VAD."""
    speech_ts = get_speech_timestamps(
        waveform,
        model,
        threshold=threshold,
        sampling_rate=sample_rate,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
    )
    return [(ts["start"] / sample_rate, ts["end"] / sample_rate) for ts in speech_ts]


def apply_vad(audio_path, sample_rate=16000, min_speech_duration=DEFAULT_MIN_SPEECH_DURATION):
    """Applies VAD to an audio file and returns speech segments."""
    model = get_vad_model()  # Load model

    # Read audio file
    waveform = read_audio(audio_path, sampling_rate=sample_rate)

    # Apply VAD
    timestamps = get_speech_timestamps(
        waveform,
        model,
        return_seconds=True  # Return timestamps in seconds
    )

    # Convert results to dictionary format
    segments = [{'start': ts[0], 'end': ts[1]} for ts in timestamps]
    return segments


def save_vad_results(audio_path, vad_segments, output_file):
    """Saves VAD results to a JSON file."""
    data = {
        "audio_file": str(audio_path),
        "voice_activity": vad_segments
    }
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


def process_vad(input_dir, output_dir, sample_rate=16000, min_speech_duration=DEFAULT_MIN_SPEECH_DURATION, test_sample=None, n_processes=DEFAULT_N_PROCESSES):
    """Applies VAD to MP3 files, using multiprocessing."""
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.mp3')):
                audio_files.append(os.path.join(root, file))

    if test_sample:
        audio_files = audio_files[:test_sample]

    print(f"Found {len(audio_files)} audio files for VAD processing.")
    os.makedirs(output_dir, exist_ok=True)


    with multiprocessing.Pool(processes=n_processes) as pool:
        with tqdm.tqdm(total=len(audio_files), desc="VAD Processing") as pbar:
            for rel_path, segments in pool.imap_unordered(_process_vad_single, [(audio_path, input_dir, sample_rate, min_speech_duration) for audio_path in audio_files]):
                if segments:  # Check if segments is not None
                    json_out = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.json')
                    os.makedirs(os.path.dirname(json_out), exist_ok=True)
                    save_vad_results(os.path.join(input_dir, rel_path), segments, json_out)  # Save with the correct path
                pbar.update()

    print("VAD processing completed.")


def _process_vad_single(args):
    """Processes VAD for a single audio file (for multiprocessing)."""
    audio_path, input_dir, sample_rate, min_speech_duration = args
    try:
        segments = apply_vad(audio_path, sample_rate=sample_rate, min_speech_duration=min_speech_duration)
        rel_path = os.path.relpath(audio_path, input_dir)
        return rel_path, segments  # Return relative path and segments
    except Exception as e:
        print(f"Error processing VAD for {audio_path}: {e}")
        return None, None  # Return None on error



########################################
# 4. Cutting: Cut Audio into Segments Based on VAD
########################################

def save_cut(seq, fname, index, extension, output_dir):
    """Saves a cut audio segment."""
    output = np.hstack(seq)
    file_name = pathlib.Path(output_dir) / fname.name / (fname.stem + f"_{index:04}{extension}")
    file_name.parent.mkdir(exist_ok=True, parents=True)
    sf.write(file_name, output, samplerate=44100)

def cut_sequence(audio_path, vad, output_dir, min_len_sec=15, max_len_sec=30, out_extension=".mp3"):
    """Cuts an audio sequence based on VAD, ensuring segment length is between min_len_sec and max_len_sec."""
    data, samplerate = sf.read(audio_path)

    if len(data.shape) != 1:
        raise ValueError(f"{audio_path} is not mono audio")

    if samplerate != 44100:
        raise ValueError(f"{audio_path} samplerate {samplerate} != 44100")

    to_stitch = []
    length_accumulated = 0.0
    segment_index = 0

    for segment in vad:
        start, end = segment['start'], segment['end']
        start_idx, end_idx = int(start * samplerate), int(end * samplerate)
        slice_audio = data[start_idx:end_idx]

        # ✅ segment가 max_len_sec을 초과하면 저장 후 새로운 segment 시작
        if length_accumulated + (end - start) > max_len_sec:
            save_cut(to_stitch, pathlib.Path(audio_path), segment_index, out_extension, output_dir)
            segment_index += 1
            to_stitch = []
            length_accumulated = 0.0

        to_stitch.append(slice_audio)
        length_accumulated += (end - start)

    # ✅ 마지막 segment 처리 (15초 이상이면 저장, 15초 미만이면 버림)
    if to_stitch and length_accumulated >= min_len_sec:
        save_cut(to_stitch, pathlib.Path(audio_path), segment_index, out_extension, output_dir)
    else:
        print(f"Last segment too short ({length_accumulated:.2f}s), discarding.")

def process_cut(vad_dir, audio_dir, output_dir, min_len_sec=15, max_len_sec=30, out_extension=DEFAULT_FORMAT, test_sample=None, n_processes=DEFAULT_N_PROCESSES):
    """Cuts audio files based on VAD results, using multiprocessing."""
    json_files = []
    for root, _, files in os.walk(vad_dir):
        for file in files:
            if file.lower().endswith('.json'):
                json_files.append(os.path.join(root, file))

    if test_sample:
        json_files = json_files[:test_sample]
    print(f"Found {len(json_files)} VAD json files for cutting.")

    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    with multiprocessing.Pool(processes=n_processes) as pool:
        with tqdm.tqdm(total=len(json_files), desc="Cutting Audio") as pbar:
            for _ in pool.imap_unordered(
                _process_cut_single,
                [(json_file, vad_dir, audio_dir, output_dir, min_len_sec, max_len_sec, out_extension) for json_file in json_files]
            ):
                pbar.update()
    print("Cutting completed.")

def _process_cut_single(args):
    """Processes a single audio file for cutting (for multiprocessing)."""
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

########################################
# 5.  Remove Intro Segments (No SNR Filtering)
########################################

def remove_intro_segments(input_dir):
    """Removes segments that start with '_0000'."""
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith("_0000.mp3"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Removed intro segment: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")

########################################
# 6. Upload to Storage: Google Cloud Storage Upload
########################################

def upload_to_gcs(bucket_name, local_dir, remote_prefix):
    """Uploads files to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            remote_path = os.path.join(remote_prefix, relative_path)

            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_path)
            print(f"Uploaded {local_path} to gs://{bucket_name}/{remote_path}")


########################################
# Integrated Pipeline Execution Function
########################################

def run_pipeline(args):
    """Runs the entire pipeline."""

    base_dir = args.base_dir
    downloads_dir = os.path.join(base_dir, "downloads")
    converted_dir = os.path.join(base_dir, "converted")
    vad_dir = os.path.join(base_dir, "vad_results")
    cut_dir = os.path.join(base_dir, "cut_segments")
    # filtered_dir = os.path.join(base_dir, "filtered")  # No longer needed

    for d in [downloads_dir, converted_dir, vad_dir, cut_dir]: # Removed filtered_dir
        os.makedirs(d, exist_ok=True)
    print("Downloading files...")
    download_urls(os.path.join(base_dir, "urls.txt"),
                  downloads_dir,
                  vm_index=args.vm_index,
                  num_vms=args.num_vms,
                  test_sample=args.test_sample)


    # 2. Unzip and Convert Stage (COMBINED)
    print("Unzipping and converting files to MP3...")
    unzip_and_convert(downloads_dir, converted_dir, sample_rate=args.sample_rate, n_processes=args.n_processes)


    # 3. VAD Stage
    print("Processing VAD on audio files...")
    process_vad(converted_dir, vad_dir, sample_rate=16000, min_speech_duration=args.min_speech_duration, test_sample=args.test_sample, n_processes=args.n_processes)

    # 4. Cutting Stage
    print("Cutting audio segments based on VAD results...")
    process_cut(vad_dir, converted_dir, cut_dir, out_extension=".mp3", test_sample=args.test_sample, n_processes=args.n_processes)


    # 5. Remove intro segments
    print("Removing intro segments...")
    remove_intro_segments(cut_dir)

    # 6. Storage Upload Stage (Optional)
    if args.gcs_bucket:
        print("Uploading cut segments to Google Cloud Storage...")
        upload_to_gcs(args.gcs_bucket, cut_dir, args.gcs_prefix)  # Upload from cut_dir

    print("Pipeline completed successfully.")


########################################
# Argument Parsing and Main
########################################

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Librivox End-to-End Pipeline")

    # General Arguments
    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR, help="Base directory for all data folders.")
    parser.add_argument("--language", type=str, default=DEFAULT_LANGUAGE, help="Target language.")
    parser.add_argument("--api_limit", type=int, default=DEFAULT_API_LIMIT, help="API limit for URL generation.")
    parser.add_argument("--max_urls", type=int, default=DEFAULT_MAX_URLS, help="Maximum number of URLs to fetch (for testing).")
    parser.add_argument("--vm_index", type=int, default=DEFAULT_VM_INDEX, help="Index of the current VM for distributed downloads.")
    parser.add_argument("--num_vms", type=int, default=DEFAULT_NUM_VMS, help="Total number of VMs for distributed downloads.")
    parser.add_argument("--test_sample", type=int, default=None, help="Number of files to process for testing at each stage.")
    parser.add_argument("--format", type=str, default=DEFAULT_FORMAT, help="Output audio format.")
    parser.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Target sample rate (for the final audio).")
    parser.add_argument("--n_processes", type=int, default=DEFAULT_N_PROCESSES, help="Number of processes for parallel operations.")
    parser.add_argument("--min_speech_duration", type=float, default=DEFAULT_MIN_SPEECH_DURATION, help="Minimum speech duration for VAD (in seconds).")
    parser.add_argument("--target_len_sec", type=int, default=DEFAULT_TARGET_LEN_SEC, help="Maximum segment length for cutting (in seconds).")
    # parser.add_argument("--snr_threshold", type=float, default=DEFAULT_SNR_THRESHOLD, help="SNR threshold for filtering (in dB).") # Removed
    parser.add_argument("--gcs_bucket", type=str, default=DEFAULT_GCS_BUCKET, help="GCS bucket name for uploads.")
    parser.add_argument("--gcs_prefix", type=str, default=DEFAULT_GCS_PREFIX, help="Path prefix for GCS uploads.")
    # Control Flow Arguments
    parser.add_argument("--pipeline", action="store_true", help="Run the entire pipeline.")
    parser.add_argument("--credentials", type=str, help="Path to the Google Cloud service account key file (optional).")

    args = parser.parse_args()

    if args.credentials:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = args.credentials

    return args


if __name__ == "__main__":
    args = parse_args()
    if args.pipeline:
        run_pipeline(args)
    else:
        print("Specify the --pipeline option to run the entire pipeline.")