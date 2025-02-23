import os

def main():
    # 1. VAD 폴더 설정
    vad_dir = "vad_2"

    # 2. VAD 폴더에서 .json 파일의 스템(파일명 .json 제외 부분)만 추출
    vad_filenames = set()
    for file in os.listdir(vad_dir):
        if file.endswith(".json"):
            # 예: 'startreader_01_smith.json' -> 'startreader_01_smith'
            stem = os.path.splitext(file)[0]
            vad_filenames.add(stem)

    # 3. urls.txt 읽기
    urls_txt = "urls/final_mp3_url2.txt"
    with open(urls_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 4. 처리됨 / 처리 안 됨 분류 리스트
    processed = []
    unprocessed = []

    for line in lines:
        url = line.strip()
        if not url:
            continue

        # URL에서 mp3 파일명 추출
        # 예: http://example.com/path/startreader_01_smith.mp3 -> startreader_01_smith.mp3
        filename = os.path.basename(url)

        # 파일명에서 .mp3 등 확장자 떼어내어 스템 추출
        # 예: startreader_01_smith.mp3 -> startreader_01_smith
        stem = os.path.splitext(filename)[0]

        # 5. VAD json 파일이 있으면 처리된 것으로, 없으면 처리 안 된 것으로 분류
        if stem in vad_filenames:
            processed.append(url)
        else:
            unprocessed.append(url)

    # 6. 결과를 각각 txt 파일로 저장
    with open("url_processed.txt", "w", encoding="utf-8") as f:
        for url in processed:
            f.write(url + "\n")

    with open("url_not_processed.txt", "w", encoding="utf-8") as f:
        for url in unprocessed:
            f.write(url + "\n")

    # 완료 정보 출력
    print("처리 완료!")
    print(f"  - 처리된 파일 수: {len(processed)}")
    print(f"  - 처리되지 않은 파일 수: {len(unprocessed)}")

if __name__ == "__main__":
    main()
