import os
import shutil
import random

def remove_files_keeping_percentage(folder_path, percentage_to_keep=30):
    """
    폴더 내 파일 중 지정된 비율(%)만 남기고 나머지를 제거합니다.

    Args:
        folder_path (str): 파일 제거를 수행할 폴더 경로.
        percentage_to_keep (int, optional): 유지할 파일의 비율 (0~100). 기본값은 30%.
    """

    if not os.path.exists(folder_path):
        print(f"오류: 폴더 '{folder_path}'를 찾을 수 없습니다.")
        return

    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    if not files:
        print(f"폴더 '{folder_path}'에 파일이 없습니다.")
        return

    num_files = len(files)
    files_to_keep_count = int(num_files * (percentage_to_keep / 100.0))

    if files_to_keep_count >= num_files:
        print(f"유지 비율이 너무 높거나 파일 수가 적어 삭제할 파일이 없습니다. 현재 파일 수: {num_files}, 유지할 파일 수: {files_to_keep_count}")
        return

    files_to_remove_count = num_files - files_to_keep_count

    print(f"총 {num_files}개 파일 중 {percentage_to_keep}%인 {files_to_keep_count}개 파일을 유지하고, {files_to_remove_count}개 파일을 삭제합니다.")
    input("계속하려면 Enter 키를 누르세요. 취소하려면 Ctrl+C를 누르세요...")  # 사용자 확인

    # 파일을 무작위로 섞어서 유지할 파일을 선택 (또는 다른 기준으로 선택 가능)
    random.shuffle(files)
    files_to_remove = files[files_to_keep_count:]

    removed_count = 0
    failed_count = 0

    for filename in files_to_remove:
        file_path = os.path.join(folder_path, filename)
        try:
            os.remove(file_path)
            removed_count += 1
            print(f"삭제됨: {filename}") # 삭제된 파일명 출력 (선택 사항)
        except Exception as e:
            print(f"오류: '{filename}' 삭제 실패 - {e}")
            failed_count += 1

    print(f"\n작업 완료: 총 {removed_count}개 파일 삭제됨. {failed_count}개 파일 삭제 실패.")
    print(f"폴더 '{folder_path}'에 {len(os.listdir(folder_path))}개 파일이 남았습니다.") # 최종 파일 개수 확인


if __name__ == "__main__":
    folder_path = input("파일을 정리할 폴더 경로를 입력하세요: ")
    percentage_str = input("유지할 파일 비율을 입력하세요 (기본값: 30%): ")

    percentage_to_keep = 30  # 기본값
    if percentage_str:
        try:
            percentage_to_keep = int(percentage_str)
            if not 0 <= percentage_to_keep <= 100:
                print("오류: 유지 비율은 0에서 100 사이의 값이어야 합니다. 기본값 30%를 사용합니다.")
                percentage_to_keep = 30
        except ValueError:
            print("오류: 유효하지 않은 비율 형식입니다. 기본값 30%를 사용합니다.")
            percentage_to_keep = 30

    remove_files_keeping_percentage(folder_path, percentage_to_keep)