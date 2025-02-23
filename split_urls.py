from pathlib import Path
from typing import List
import math

def read_urls(file_path: str) -> List[str]:
    """파일에서 URL 목록을 읽어옵니다."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def split_urls(urls: List[str], num_parts: int = 20) -> List[List[str]]:
    """URL 목록을 지정된 수만큼 균등하게 분할합니다."""
    urls_per_part = math.ceil(len(urls) / num_parts)
    return [urls[i:i + urls_per_part] for i in range(0, len(urls), urls_per_part)]

def save_url_parts(url_parts: List[List[str]], base_filename: str) -> None:
    """분할된 URL들을 각각의 파일로 저장합니다."""
    print("\n=== 분할 결과 ===")
    print(f"총 분할 파일 수: {len(url_parts)}")
    
    for idx, urls in enumerate(url_parts, 1):
        output_path = f"{Path(base_filename).stem}_part_{idx}.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(urls))
        print(f"파일: {output_path}")
        print(f"  - URL 개수: {len(urls)}개")
        print(f"  - 첫 번째 URL: {urls[0]}")
        print(f"  - 마지막 URL: {urls[-1]}\n")

def main():
    input_file = "merged_not_processed_urls.txt"
    
    # URL 목록 읽기
    urls = read_urls(input_file)
    total_urls = len(urls)
    expected_per_file = math.ceil(total_urls / 16)
    
    print("\n=== 전체 요약 ===")
    print(f"총 URL 개수: {total_urls:,}개")
    print(f"예상 파일당 URL 개수: {expected_per_file:,}개")
    print("=" * 30)
    
    # URL 목록 16등분하기
    url_parts = split_urls(urls)
    
    # 분할된 URL 저장
    save_url_parts(url_parts, input_file)

if __name__ == "__main__":
    main() 