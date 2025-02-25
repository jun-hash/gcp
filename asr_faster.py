#!/usr/bin/env python3
"""
Run Fast Whisper ASR model on audio files from a local folder.
Uses faster-whisper package for improved performance.
"""

import os
import torch
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List
import argparse
import time
from faster_whisper import WhisperModel

class ASRProcessor:
    def __init__(self, output_dir: str):
        """Initialize ASR processor with Fast Whisper model."""
        self.output_dir = Path(output_dir)
        
        # 시간 측정 시작
        start_time = time.time()
        print("=== Fast Whisper 초기화 시간 측정 ===")
        
        # CUDA 환경 확인 및 출력
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        print(f"PyTorch CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 장치 수: {torch.cuda.device_count()}")
            print(f"현재 CUDA 장치: {torch.cuda.current_device()}")
            print(f"CUDA 장치 이름: {torch.cuda.get_device_name(0)}")
        
        # Set device - CUDA 문제 해결을 위한 명시적 설정
        cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
        self.device = "cuda" if cuda_available else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        print(f"[1] 장치 설정: {self.device}, 계산 타입: {self.compute_type}, 시간: {time.time() - start_time:.4f}초")
        
        # Load model with optimized settings
        model_load_start = time.time()
        try:
            self.model = WhisperModel(
                "large-v3", 
                device=self.device, 
                compute_type=self.compute_type,
                cpu_threads=8 if self.device == "cpu" else 4,
                num_workers=4,
            )
            print(f"모델 로드 성공: {self.device} 모드")
        except Exception as e:
            print(f"모델 로드 오류, CPU 모드로 폴백: {e}")
            self.device = "cpu"
            self.compute_type = "int8"
            self.model = WhisperModel(
                "large-v3", 
                device="cpu", 
                compute_type="int8",
                cpu_threads=8,
                num_workers=4,
            )
        
        model_load_time = time.time() - model_load_start
        print(f"[2] 모델 로딩 시간: {model_load_time:.4f}초")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 처리 통계
        self.processed_count = 0
        self.skipped_count = 0
        
        print(f"[3] 전체 초기화 시간: {time.time() - start_time:.4f}초")

    def process_audio_file(self, audio_path: Path) -> bool:
        """Process a single audio file with Fast Whisper ASR."""
        file_id = audio_path.stem
        
        # 이미 처리된 파일인지 확인
        text_output_file = self.output_dir / f"{file_id}.txt"
        if text_output_file.exists() and text_output_file.stat().st_size > 0:
            print(f"스킵: {file_id} (이미 처리됨)")
            self.skipped_count += 1
            return True
        
        total_start = time.time()
        print(f"\n=== Fast Whisper 파일 처리: {file_id} ===")
        
        try:
            # 오디오 로딩
            audio_load_start = time.time()
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)
            audio = audio.astype(np.float32)
            audio_load_time = time.time() - audio_load_start
            print(f"[1] 오디오 로딩 시간: {audio_load_time:.4f}초")
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return False
        
        # 트랜스크립션 (API 호출)
        transcribe_start = time.time()
        segments, info = self.model.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        
        # 짧은 오디오에서는 미리 리스트로 변환하는 것이 더 빠름
        segments = list(segments)
        transcribe_time = time.time() - transcribe_start
        print(f"[2] 트랜스크립션 시간: {transcribe_time:.4f}초")
        
        # 초고속 포맷팅
        format_start = time.time()
        # 전체 텍스트 생성 (타임스탬프 없이)
        full_text = " ".join(segment.text.strip() for segment in segments)
        
        format_time = time.time() - format_start
        print(f"[3] 결과 처리 시간: {format_time:.4f}초")
        
        # 결과 저장 (최적화)
        save_start = time.time()
        
        # 텍스트 결과 저장 (추가)
        with open(text_output_file, "w", encoding="utf-8") as f:
            f.write(full_text)
        
        save_time = time.time() - save_start
        print(f"[4] 결과 저장 시간: {save_time:.4f}초")
        
        total_time = time.time() - total_start
        print(f"[5] 전체 처리 시간: {total_time:.4f}초")
        
        self.processed_count += 1
        return True

    def process_folder(self, audio_folder: str, extensions: List[str] = None):
        """Process all audio files in a folder."""
        folder_start = time.time()
        print(f"\n=== Fast Whisper 폴더 처리 시작 ===")
        
        if extensions is None:
            extensions = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]
            
        audio_folder = Path(audio_folder)
        if not audio_folder.exists():
            raise FileNotFoundError(f"Audio folder not found: {audio_folder}")
            
        # Get all audio files in the folder
        find_start = time.time()
        audio_files = []
        for ext in extensions:
            audio_files.extend(audio_folder.glob(f"*{ext}"))
        
        find_time = time.time() - find_start
        print(f"[1] 파일 검색 시간: {find_time:.4f}초 (총 {len(audio_files)}개 파일)")
            
        if not audio_files:
            print(f"No audio files found in {audio_folder} with extensions {extensions}")
            return
        
        # 이미 처리된 파일 확인
        existing_files = set(f.stem for f in self.output_dir.glob("*.txt"))
        if existing_files:
            print(f"기존 처리된 파일 수: {len(existing_files)}")
            
        # Process each audio file with progress bar
        process_start = time.time()
        for audio_file in tqdm(audio_files, desc="Processing files"):
            success = self.process_audio_file(audio_file)
            if not success:
                print(f"Failed to process {audio_file.name}")
        
        process_time = time.time() - process_start
        print(f"[2] 전체 파일 처리 시간: {process_time:.4f}초")
        print(f"[3] 폴더 전체 처리 시간: {time.time() - folder_start:.4f}초")
        
        # 처리 요약 출력
        print(f"\n=== 처리 요약 ===")
        print(f"총 파일 수: {len(audio_files)}")
        print(f"처리된 파일 수: {self.processed_count}")
        print(f"스킵된 파일 수: {self.skipped_count}")
        
        if self.processed_count > 0:
            avg_time = process_time / self.processed_count
            print(f"파일당 평균 처리 시간: {avg_time:.4f}초")


def main():
    parser = argparse.ArgumentParser(
        description="Run Fast Whisper ASR model on audio files"
    )
    parser.add_argument(
        "--audio-folder",
        type=str,
        default="./data/cut",
        help="Folder containing audio files to process",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./asr_result_text",
        help="Directory to save ASR results",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".wav", ".mp3", ".flac", ".m4a", ".ogg"],
        help="Audio file extensions to process",
    )
    args = parser.parse_args()

    overall_start = time.time()
    processor = ASRProcessor(output_dir=args.output_dir)
    processor.process_folder(args.audio_folder, args.extensions)
    print(f"\n=== 전체 실행 시간: {time.time() - overall_start:.4f}초 ===")


if __name__ == "__main__":
    main() 