def merge_processed_urls():
    """여러 processed URL 파일들을 하나의 파일로 병합합니다."""
    output_file = 'merged_not_processed_urls.txt'
    input_files = [
        'url_not_processed_1.txt',
        'url_not_processed_2.txt',
        'url_not_processed_3.txt'
    ]
    
    seen_urls = set()  # 중복 URL 제거를 위한 set
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for input_file in input_files:
            try:
                with open(input_file, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        url = line.strip()
                        if url and url not in seen_urls:  # 빈 줄이 아니고 중복되지 않은 경우만 추가
                            seen_urls.add(url)
                            outfile.write(url + '\n')
            except FileNotFoundError:
                print(f"경고: {input_file}을 찾을 수 없습니다.")

if __name__ == '__main__':
    merge_processed_urls() 