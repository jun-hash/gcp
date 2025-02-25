#!/usr/bin/env python3
"""
Optimized Whisper ASR processor for audio files.
Uses single process with threaded I/O and batch processing for maximum performance.
"""

import os
import json
import torch
import asr
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any
import argparse
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import gc


def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def load_audio(audio_path: Path) -> Tuple[Optional[np.ndarray], str]:
    """
    Load an audio file and return its data and ID.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Tuple of (audio_data, file_id)
    """
    try:
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        # Ensure audio is float32
        audio = audio.astype(np.float32)
        return audio, audio_path.stem
    except Exception as e:
        logging.error(f"Error loading audio file {audio_path}: {e}")
        return None, audio_path.stem


def save_result(result: Dict, output_dir: Path) -> bool:
    """Save ASR result to output directory."""
    try:
        file_id = result.get("file_id")
        formatted_results = result.get("results", [])
        
        if formatted_results and file_id:
            output_file = output_dir / f"{file_id}.json"
            with open(output_file, "w") as f:
                serializable_results = json.loads(
                    json.dumps(formatted_results, default=convert_to_serializable)
                )
                json.dump(serializable_results, f, indent=2)
            return True
        return False
    except Exception as e:
        logging.error(f"Error saving results for {result.get('file_id')}: {e}")
        return False


class OptimizedASRProcessor:
    def __init__(
        self,
        output_dir: str,
        batch_size: int = 4,
        num_workers: int = 4,
        buffer_size: int = 16,
    ):
        """
        Initialize ASR processor with optimized performance settings.

        Args:
            output_dir: Directory to save ASR results
            batch_size: Number of files to process in a GPU batch
            num_workers: Number of threads for I/O operations
            buffer_size: Size of the loading queue buffer
        """
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "asr_processing.log"),
                logging.StreamHandler()
            ]
        )
        
        # Initialize device and model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # Set appropriate torch settings for speed
        torch.set_grad_enabled(False)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Load model (just once)
        logging.info("Loading Whisper model...")
        self.model = asr.load_model("turbo").to(self.device)
        logging.info("Model loaded successfully")

    def process_audio(self, audio: np.ndarray, file_id: str) -> Dict:
        """
        Process a single audio file with Whisper ASR.
        
        Args:
            audio: Audio data as numpy array
            file_id: ID of the audio file
            
        Returns:
            Dictionary with ASR results
        """
        try:
            # Process the audio file
            result = self.model.transcribe(
                audio,
                word_timestamps=True,
                condition_on_previous_text=False,
                language="en",
            )
            
            if not result.get("segments"):
                logging.warning(f"No segments returned for audio file {file_id}")
                return {"file_id": file_id, "results": []}
                
            # Process segments and their timestamps
            formatted_results = []
            for segment in result["segments"]:
                # Extract word-level information if available
                words_info = []
                if "words" in segment:
                    for word in segment["words"]:
                        words_info.append(
                            {
                                "word": word["word"],
                                "start": float(word["start"]),
                                "end": float(word["end"]),
                                "probability": word["probability"],
                            }
                        )

                # Format segment data
                segment_data = {
                    "text": segment["text"],
                    "start": f"{float(segment['start']):.3f}",
                    "end": f"{float(segment['end']):.3f}",
                    "prob": np.exp(segment.get("avg_logprob", 0.0)),
                    "words": words_info,
                }
                formatted_results.append(segment_data)
            
            return {"file_id": file_id, "results": formatted_results}
            
        except Exception as e:
            logging.error(f"Error processing {file_id}: {e}")
            return {"file_id": file_id, "results": [], "error": str(e)}

    def process_batch(self, batch: List[Tuple[np.ndarray, str]]) -> List[Dict]:
        """
        Process a batch of audio files.
        
        Args:
            batch: List of (audio_data, file_id) tuples
            
        Returns:
            List of dictionaries with ASR results
        """
        results = []
        valid_items = [(audio, file_id) for audio, file_id in batch if audio is not None]
        
        if not valid_items:
            return results
            
        for audio, file_id in valid_items:
            result = self.process_audio(audio, file_id)
            results.append(result)
            
            # Save result immediately
            save_result(result, self.output_dir)
        
        return results

    def process_folder(self, audio_folder: str, extensions: List[str] = None):
        """
        Process all audio files in a folder with optimized threading.
        
        Args:
            audio_folder: Path to folder containing audio files
            extensions: List of audio file extensions to process
        """
        if extensions is None:
            extensions = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]
            
        audio_folder = Path(audio_folder)
        if not audio_folder.exists():
            raise FileNotFoundError(f"Audio folder not found: {audio_folder}")
            
        # Collect all audio files
        audio_files = []
        for ext in extensions:
            audio_files.extend(audio_folder.glob(f"*{ext}"))
            
        if not audio_files:
            logging.warning(f"No audio files found in {audio_folder} with extensions {extensions}")
            return
            
        logging.info(f"Found {len(audio_files)} audio files to process")
        
        # Process files with loading in separate threads but sequential GPU processing
        total_processed = 0
        start_time = time.time()
        
        # Create audio loading queue with ThreadPoolExecutor
        audio_queue = Queue(maxsize=self.buffer_size)
        done_loading = False
        
        def load_audio_files():
            nonlocal done_loading
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for result in executor.map(load_audio, audio_files):
                    audio_queue.put(result)
            done_loading = True
        
        # Start loading thread
        import threading
        loading_thread = threading.Thread(target=load_audio_files)
        loading_thread.daemon = True
        loading_thread.start()
        
        # Process audio files from queue
        with tqdm(total=len(audio_files), desc="Processing files") as pbar:
            while not (done_loading and audio_queue.empty()):
                # Collect a batch of files
                batch = []
                while len(batch) < self.batch_size:
                    try:
                        item = audio_queue.get(timeout=0.1)
                        batch.append(item)
                    except Empty:
                        if done_loading:
                            break
                        continue
                
                if not batch:
                    continue
                
                # Process the batch
                results = self.process_batch(batch)
                total_processed += len(results)
                
                # Update progress bar
                pbar.update(len(batch))
                
                # Log progress stats
                elapsed = time.time() - start_time
                files_per_second = total_processed / elapsed if elapsed > 0 else 0
                logging.info(f"Processed {total_processed}/{len(audio_files)} files "
                            f"({files_per_second:.2f} files/sec)")
        
        # Final stats
        total_time = time.time() - start_time
        logging.info(f"Completed processing {total_processed} files in {total_time:.2f} seconds "
                    f"({total_processed/total_time:.2f} files/sec)")


def main():
    parser = argparse.ArgumentParser(
        description="Optimized Whisper ASR processor for audio files"
    )
    parser.add_argument(
        "--audio-folder",
        type=str,
        default="./data/v12_sample",
        help="Folder containing audio files to process",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./asr_result",
        help="Directory to save ASR results",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".wav", ".mp3", ".flac", ".m4a", ".ogg"],
        help="Audio file extensions to process",
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=16,
        help="Number of files to process in a single GPU batch"
    )
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=6,
        help="Number of worker threads for loading audio files"
    )
    parser.add_argument(
        "--buffer-size", 
        type=int, 
        default=8,
        help="Size of the loading queue buffer"
    )
    args = parser.parse_args()

    # Set higher priority for this process
    try:
        os.nice(-10)  # Linux/Unix: higher priority
    except:
        pass  # Ignore if not available

    processor = OptimizedASRProcessor(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        buffer_size=args.buffer_size
    )
    processor.process_folder(args.audio_folder, args.extensions)


if __name__ == "__main__":
    main() 