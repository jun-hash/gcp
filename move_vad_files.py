from pathlib import Path
import shutil

def move_vad_files():
    """VAD1, VAD2, VAD3 폴더의 파일들을 vad_total 폴더로 통합합니다."""
    # 소스 폴더들과 대상 폴더 설정
    source_folders = ['vad_list/vad_1', 'vad_list/vad_2', 'vad_list/vad_3']
    target_folder = Path('vad_total')
    
    # vad_total 폴더 생성
    target_folder.mkdir(exist_ok=True)
    
    print("\n=== 파일 이동 시작 ===")
    total_moved = 0
    
    # 각 소스 폴더에서 파일 이동
    for folder in source_folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"{folder} 폴더가 존재하지 않습니다.")
            continue
            
        moved_count = 0
        for file_path in folder_path.glob('*.json'):
            target_path = target_folder / file_path.name
            shutil.move(str(file_path), str(target_path))
            moved_count += 1
            total_moved += 1
        
        print(f"{folder} -> {moved_count}개 파일 이동 완료")
    
    # 최종 결과 출력
    final_count = len(list(target_folder.glob('*.json')))
    print(f"\n=== 이동 완료 ===")
    print(f"총 이동된 파일 개수: {total_moved}개")
    print(f"vad_total 폴더 최종 파일 개수: {final_count}개")

if __name__ == "__main__":
    move_vad_files() 