import os

def extract_urls_by_vad_directory():
    """
    Extracts URLs from merged_processed_urls.txt and groups them by which VAD directory
    contains the corresponding JSON file.
    """
    # VAD directories to check
    vad_dirs = ["vad_00_1", "vad_00_2", "vad_00_3"]
    
    # Get the stems of all JSON files in each VAD directory
    vad_stems = {}
    for vad_dir in vad_dirs:
        vad_stems[vad_dir] = set()
        if os.path.exists(vad_dir):
            for file in os.listdir(vad_dir):
                if file.endswith(".json"):
                    stem = os.path.splitext(file)[0]
                    vad_stems[vad_dir].add(stem)
    
    # Read all URLs from the merged file
    with open("merged_processed_urls.txt", "r", encoding="utf-8") as f:
        all_urls = [line.strip() for line in f if line.strip()]
    
    # Group URLs by VAD directory
    url_groups = {vad_dir: [] for vad_dir in vad_dirs}
    unmatched = []
    
    for url in all_urls:
        # Extract the file name stem from the URL
        filename = os.path.basename(url)
        stem = os.path.splitext(filename)[0]
        
        # Check which VAD directory contains this stem
        found = False
        for vad_dir in vad_dirs:
            if stem in vad_stems[vad_dir]:
                url_groups[vad_dir].append(url)
                found = True
                break
        
        if not found:
            unmatched.append(url)
    
    # Write each group to a separate file
    for vad_dir in vad_dirs:
        output_file = f"url_{vad_dir.replace('vad_', '')}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for url in url_groups[vad_dir]:
                f.write(f"{url}\n")
        print(f"{output_file}: {len(url_groups[vad_dir])} URLs")
    
    if unmatched:
        with open("url_unmatched.txt", "w", encoding="utf-8") as f:
            for url in unmatched:
                f.write(f"{url}\n")
        print(f"url_unmatched.txt: {len(unmatched)} URLs")
    
    # Print totals
    print(f"\nTotal URLs processed: {len(all_urls)}")
    matched = sum(len(urls) for urls in url_groups.values())
    print(f"Total URLs matched: {matched}")
    print(f"Total URLs unmatched: {len(unmatched)}")

if __name__ == "__main__":
    extract_urls_by_vad_directory() 