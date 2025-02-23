#!/usr/bin/env python3
# generate_urls.py

import os
import argparse
import requests
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import statistics
import numpy as np

# 상수 정의
DEFAULT_LANGUAGE = "English"
DEFAULT_API_LIMIT = 50
TARGET_TOTAL_DURATION_HOURS = 36526.27
MAX_READER_DURATION = 14 * 3600  # 각 화자의 최대 녹음 시간: 12시간(21600초)
MAX_RETRIES = 1

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_page(base_url, params, retries=MAX_RETRIES):
    """단일 페이지의 데이터를 가져옵니다."""
    for attempt in range(retries):
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            logging.warning(f"시도 {attempt + 1} 실패: {e}")
            time.sleep(2 ** attempt)  # 지수 백오프
    logging.error("모든 재시도가 실패했습니다.")
    return None


def collect_audiobooks(language, limit, target_duration):
    """
    오디오북 데이터를 수집합니다.
    - Librivox API 페이지네이션 방식
    - 총 수집 재생 시간이 target_duration(초) 이상이 되면 중단
    - 각 화자별 누적 녹음 시간이 12시간(MAX_READER_DURATION)을 넘지 않도록 제한
    """
    base_url = "https://librivox.org/api/feed/audiobooks"
    params = {
        "format": "json",
        "extended": "1",
        "limit": limit,
        "offset": 0,
        "sort_field": "id",
        "sort_order": "desc",
        "language": language
    }

    total_duration = 0
    audiobooks = []
    reader_durations = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        with tqdm(total=target_duration, unit='sec', desc='총 수집된 재생 시간') as pbar:
            while total_duration < target_duration:
                future = executor.submit(fetch_page, base_url, params)
                result = future.result()
                if not result or 'books' not in result or not result['books']:
                    logging.info("더 이상 가져올 데이터가 없습니다.")
                    break

                for book in result['books']:
                    if total_duration >= target_duration:
                        break

                    book_duration = int(book.get('totaltimesecs', 0))
                    if book_duration == 0:
                        continue

                    readers = book.get('authors', [])
                    if not readers:
                        continue

                    # 각 화자의 녹음 시간이 최대치를 넘지 않도록 확인
                    can_add_book = True
                    for reader in readers:
                        reader_id = reader.get('id')
                        if not reader_id:
                            continue
                        current_duration = reader_durations.get(reader_id, 0)
                        if current_duration + book_duration > MAX_READER_DURATION:
                            can_add_book = False
                            break

                    if not can_add_book:
                        continue

                    # 오디오북 추가 및 화자 녹음 시간 업데이트
                    audiobooks.append({
                        "id": book.get('id'),
                        "title": book.get('title'),
                        "url_zip_file": book.get('url_zip_file'),
                        "duration": book_duration,
                        "reader_ids": [reader.get('id') for reader in readers],
                        # 원본 API 필드를 보존하기 위해 authors 자체도 저장해둘 수 있음
                        "authors": book.get('authors', [])
                    })
                    total_duration += book_duration
                    pbar.update(book_duration)
                    for reader in readers:
                        reader_id = reader.get('id')
                        if reader_id:
                            reader_durations[reader_id] = reader_durations.get(reader_id, 0) + book_duration

                params['offset'] += limit

    return audiobooks, total_duration, reader_durations


def filter_books(books):
    """
    (A) 필터링 로직:
    - 메타데이터(title, authors)가 비정상인 경우 제외
    - duration(초)이 0이면 제외
    - 화자(authors) 배열 길이가 1을 초과(=여러 화자)하는 책 제외
    - title에 "Dramatic" 등의 키워드가 있으면 제외(예시)
    """
    filtered = []
    for book in books:
        title = book.get('title')
        authors = book.get('authors', [])
        totalsecs = int(book.get('duration', 0))  # 여기선 이미 "duration" 키에 totalsecs 저장됨

        # 최소한의 메타데이터 검증
        if not title or not authors or totalsecs == 0:
            continue

        # 다중화자(길이가 1을 초과)
        if len(authors) > 1:
            continue

        # "Dramatic" 단어가 제목에 들어가면 제외(원하면 조건 수정)
        if "Dramatic" in title:
            continue

        filtered.append(book)

    return filtered


def keep_only_latest_version(books):
    """
    (B) 동일 (title + author_id)에 대해 여러 버전이 있을 경우,
        id가 가장 큰(최신) 것만 남깁니다.
    """
    unique_map = {}
    for book in books:
        title = (book['title'] or "").strip().lower()
        authors = book.get('authors', [])
        author_id = authors[0].get('id') if authors else None
        key = (title, author_id)

        if key not in unique_map:
            unique_map[key] = book
        else:
            if book['id'] > unique_map[key]['id']:
                unique_map[key] = book

    return list(unique_map.values())


def recalc_reader_durations(books):
    """
    (C) 필터 후, 화자별 녹음 시간을 다시 계산합니다.
        books 리스트에는 각 아이템에 "duration"과 "reader_ids"가 있습니다.
    """
    new_reader_durations = {}
    total_duration = 0
    for bk in books:
        dur = bk.get('duration', 0)
        total_duration += dur
        for r_id in bk.get('reader_ids', []):
            new_reader_durations[r_id] = new_reader_durations.get(r_id, 0) + dur
    return new_reader_durations, total_duration


def split_urls(audiobooks, num_vms):
    """수집한 오디오북의 URL을 VM 수에 따라 분할합니다."""
    urls = [book['url_zip_file'] for book in audiobooks if book.get('url_zip_file')]
    avg = len(urls) / float(num_vms) if num_vms > 0 else len(urls)
    split_result = []
    last = 0.0

    while last < len(urls):
        split_result.append(urls[int(last):int(last + avg)])
        last += avg

    return split_result


def save_urls(split_urls_list, base_dir):
    """분할된 URL을 파일로 저장합니다."""
    os.makedirs(base_dir, exist_ok=True)
    for idx, urls in enumerate(split_urls_list):
        file_path = os.path.join(base_dir, f"urls_vm_{idx + 1}.txt")
        with open(file_path, 'w') as f:
            for url in urls:
                f.write(f"{url}\n")
        logging.info(f"{file_path}에 {len(urls)}개의 URL이 저장되었습니다.")

def calculate_speaker_stats(reader_durations):
    """화자 통계를 계산합니다."""
    num_speakers = len(reader_durations)
    if num_speakers == 0:
        return 0, 0.0, 0.0, 0.0, 0.0

    durations = list(reader_durations.values())
    avg_duration = statistics.mean(durations)
    median_duration = statistics.median(durations)
    p95 = np.percentile(durations, 95)
    std_duration = statistics.stdev(durations)

    return num_speakers, avg_duration, median_duration, p95, std_duration


def main():
    parser = argparse.ArgumentParser(description="LibriVox 오디오북 URL 생성기 (고도화 필터 반영)")
    parser.add_argument("--language", type=str, default=DEFAULT_LANGUAGE, help="대상 언어")
    parser.add_argument("--limit", type=int, default=DEFAULT_API_LIMIT, help="한 번의 API 요청당 가져올 항목 수")
    parser.add_argument("--target_hours", type=int, default=TARGET_TOTAL_DURATION_HOURS, help="목표 총 재생 시간(시간 단위)")
    parser.add_argument("--num_vms", type=int, default=8, help="VM의 수")
    parser.add_argument("--output_dir", type=str, default="output_gcp_urls", help="URL 파일을 저장할 디렉토리")
    args = parser.parse_args()

    target_duration = args.target_hours * 3600  # 시간 단위를 초 단위로 변환

    # 1) 책 수집 (원본 로직)
    audiobooks, total_duration, original_reader_durations = collect_audiobooks(
        args.language,
        args.limit,
        target_duration
    )

    logging.info(f"[수집 완료] 총 {len(audiobooks)}권, 총 재생 시간 {total_duration/3600:.2f}h, 화자 수 {len(original_reader_durations)}명")

    # 2) (A) 필터링: 멀티화자/메타데이터 이상/드라마틱 등
    before_filter_count = len(audiobooks)
    audiobooks = filter_books(audiobooks)
    after_filter_count = len(audiobooks)
    logging.info(f"[필터] 적용 전 {before_filter_count}권 -> 적용 후 {after_filter_count}권")

    # 3) (B) 중복 버전 제거: 동일 (title+author_id) 중 최신만
    before_latest_count = len(audiobooks)
    audiobooks = keep_only_latest_version(audiobooks)
    after_latest_count = len(audiobooks)
    logging.info(f"[최신 버전] 적용 전 {before_latest_count}권 -> 적용 후 {after_latest_count}권")

    # 4) (C) 필터 결과에 따른 화자/전체 시간 재계산
    reader_durations, final_total_duration = recalc_reader_durations(audiobooks)

    # 5) 화자 통계
    num_speakers, avg_duration, median_duration, p95, std_duration = calculate_speaker_stats(reader_durations)
    logging.info(f"최종 화자 수: {num_speakers}")
    logging.info(f"화자당 평균 녹음 시간: {avg_duration/3600:.2f} 시간")
    logging.info(f"화자당 중간값: {median_duration/3600:.2f} 시간")
    logging.info(f"화자당 표준편차: {std_duration / 3600:.2f} 시간")
    logging.info(f"화자당 95% 백분위수: {p95/3600:.2f} 시간")
    logging.info(f"최종 총 재생 시간: {final_total_duration/3600:.2f} 시간")

    # 6) URL 분할
    logging.info("URL 분할 시작...")
    splitted = split_urls(audiobooks, args.num_vms)
    logging.info("URL 분할 완료.")

    # 7) 분할된 URL 저장
    logging.info("URL 저장 시작...")
    try:
        save_urls(splitted, args.output_dir)
        logging.info("URL 저장 완료.")
    except Exception as e:
        logging.error(f"URL 저장 중 오류 발생: {e}")


if __name__ == "__main__":
    main()
