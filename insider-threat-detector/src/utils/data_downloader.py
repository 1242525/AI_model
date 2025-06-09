"""
파일 경로: C:/Projects/AISecApp/01/insider-threat-detector/src/utils/data_downloader.py
파일명: data_downloader.py
설명: 데이터셋 자동 다운로드 및 설정
"""

import requests
import zipfile
import os
from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm
import shutil
from typing import Dict

class DatasetDownloader:
    """데이터셋 다운로드 및 관리 클래스"""

    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / "raw"
        self.processed_dir = self.data_root / "processed"

        # 디렉터리 생성
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, save_path: Path, chunk_size: int = 8192) -> bool:
        """파일을 다운로드합니다."""
        try:
            print(f"다운로드 시작: {url}")

            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(save_path, 'wb') as file:
                if total_size > 0:
                    with tqdm(desc=save_path.name, total=total_size, unit='B', 
                             unit_scale=True, unit_divisor=1024) as progress_bar:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                file.write(chunk)
                                progress_bar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            file.write(chunk)

            print(f"다운로드 완료: {save_path}")
            return True

        except Exception as e:
            print(f"다운로드 실패: {e}")
            return False

    def download_cmu_keystroke_dataset(self) -> bool:
        """CMU 키스트로크 데이터셋 다운로드"""
        keystroke_dir = self.raw_dir / "keystroke"
        keystroke_dir.mkdir(exist_ok=True)

        # CMU 데이터셋 URL들
        urls = {
            "DSL-StrongPasswordData.txt": "https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.txt",
        }

        success = True
        for filename, url in urls.items():
            save_path = keystroke_dir / filename

            if save_path.exists():
                print(f"파일이 이미 존재합니다: {save_path}")
                continue

            if not self.download_file(url, save_path):
                success = False
                # 다운로드 실패 시 더미 파일 생성
                self._create_dummy_keystroke_file(save_path)

        return success

    def download_balabit_mouse_dataset(self) -> bool:
        """Balabit 마우스 데이터셋 다운로드"""
        mouse_dir = self.raw_dir / "mouse"
        mouse_dir.mkdir(exist_ok=True)
        
        # GitHub 리포지토리에서 직접 다운로드
        urls = {
            "training_files": "https://github.com/balabit/Mouse-Dynamics-Challenge/archive/master.zip",
        }

        success = True
        for filename, url in urls.items():
            save_path = mouse_dir / f"{filename}.zip"
            
            if save_path.exists():
                print(f"파일이 이미 존재합니다: {save_path}")
                continue

            if not self.download_file(url, save_path):
                success = False
            else:
                # ZIP 파일 압축 해제
                with zipfile.ZipFile(save_path, 'r') as zip_ref:
                    zip_ref.extractall(mouse_dir)
                print(f"압축 해제 완료: {save_path}")

        return success

    def _create_dummy_mouse_file(self, file_path: Path):
        """더미 마우스 파일 생성"""
        print(f"더미 마우스 파일 생성: {file_path}")

        header = "user_id,session_id,timestamp,event_type,x,y,button,velocity"
        dummy_lines = [
            "user_1,1,1000.0,move,400,300,,45.2",
            "user_1,1,1100.0,move,445,310,,38.1", 
            "user_1,1,1200.0,click,445,310,left,0.0",
            "user_1,1,1300.0,move,450,315,,5.2",
            "user_2,1,2000.0,move,500,400,,52.1",
            "user_2,1,2100.0,click,500,400,left,0.0"
        ]

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(header + "\n")
            for line in dummy_lines:
                f.write(line + "\n")

    def download_all_datasets(self):
        """모든 데이터셋 다운로드"""
        print("데이터셋 다운로드를 시작합니다...")

        # CMU 키스트로크 데이터셋
        print("\n=== CMU 키스트로크 데이터셋 다운로드 ===")
        keystroke_success = self.download_cmu_keystroke_dataset()

        # Balabit 마우스 데이터셋  
        print("\n=== Balabit 마우스 데이터셋 다운로드 ===")
        mouse_success = self.download_balabit_mouse_dataset()

        print("\n=== 다운로드 완료 ===")
        print(f"키스트로크 데이터셋: {'성공' if keystroke_success else '실패 (더미 데이터 사용)'}")
        print(f"마우스 데이터셋: {'성공' if mouse_success else '실패 (더미 데이터 사용)'}")

        return keystroke_success and mouse_success

    def check_datasets(self) -> Dict[str, bool]:
        """데이터셋 존재 여부 확인"""
        keystroke_file = self.raw_dir / "keystroke" / "DSL-StrongPasswordData.txt"
        mouse_file = self.raw_dir / "mouse" / "mouse_data.csv"

        status = {
            "keystroke": keystroke_file.exists(),
            "mouse": mouse_file.exists(),
        }

        print("데이터셋 상태:")
        for dataset, exists in status.items():
            status_text = "존재" if exists else "없음"
            print(f"  {dataset}: {status_text}")

        return status


def setup_datasets(data_root: str = "data") -> bool:
    """데이터셋 설정 메인 함수"""
    downloader = DatasetDownloader(data_root)

    # 기존 데이터셋 확인
    status = downloader.check_datasets()

    # 없는 데이터셋 다운로드
    if not all(status.values()):
        return downloader.download_all_datasets()
    else:
        print("모든 데이터셋이 이미 존재합니다.")
        return True


if __name__ == "__main__":
    # 데이터셋 설정 실행
    setup_datasets()