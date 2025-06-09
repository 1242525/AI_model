"""
파일 경로: C:\Projects\AISecApp\01\insider-threat-detector\main.py
파일명: main.py
설명: 내부자 위협 탐지 시스템 메인 실행 파일
"""

import sys
import torch
import argparse
import multiprocessing
import numpy as np
from pathlib import Path

# 프로젝트 모듈 import
sys.path.append(str(Path(__file__).parent))
from config import *
from src.models.siamese import SiameseNetwork
from src.preprocessing.keystroke_preprocessor import preprocess_keystroke_data
from src.preprocessing.mouse_preprocessor import preprocess_mouse_data
from src.utils.data_downloader import setup_datasets

def load_model(model_path: str) -> SiameseNetwork:
    """훈련된 모델 로드"""
    try:
        if Path(model_path).exists():
            print(f"모델 로드 중: {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu')

            model = SiameseNetwork(**checkpoint['config']['model'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            print("모델 로드 완료!")
            return model
        else:
            print(f"모델 파일이 없습니다: {model_path}")
            print("더미 모델을 생성합니다.")
            model = SiameseNetwork()
            model.eval()
            return model

    except Exception as e:
        print(f"모델 로드 실패: {e}")
        print("기본 모델을 생성합니다.")
        model = SiameseNetwork()
        model.eval()
        return model

def test_inference(model: SiameseNetwork):
    """추론 테스트"""
    print("\n=== 추론 테스트 ===")

    # 더미 데이터로 테스트 (차원 64로 수정)
    batch_size = 4
    input_dim = 64  # 모델 입력 차원과 일치

    # 랜덤 입력 생성
    x1 = torch.randn(batch_size, input_dim)
    x2 = torch.randn(batch_size, input_dim)

    with torch.no_grad():
        # 모델 추론
        output1, output2 = model(x1, x2)

        # 유클리디안 거리 계산
        distances = torch.nn.functional.pairwise_distance(output1, output2)

        print(f"입력 형태: {x1.shape}")
        print(f"출력 형태: {output1.shape}")
        print(f"거리: {distances}")

        # 임계값 기반 분류
        threshold = 0.5
        predictions = (distances > threshold).float()

        print(f"예측 (임계값 {threshold}): {predictions}")
        print("추론 테스트 완료!")

def run_data_preprocessing():
    """데이터 전처리 실행"""
    print("\n=== 데이터 전처리 ===")

    # 데이터셋 설정
    print("데이터셋 다운로드/확인 중...")
    setup_datasets()

    # 키스트로크 데이터 전처리
    print("키스트로크 데이터 전처리 중...")
    keystroke_features, keystroke_labels = preprocess_keystroke_data(
        str(KEYSTROKE_DATA_PATH),
        str(KEYSTROKE_CSV_PATH)
    )

    # 마우스 데이터 전처리
    print("마우스 데이터 전처리 중...")
    mouse_features, mouse_labels = preprocess_mouse_data(
        None,  # 더미 데이터 사용
        str(MOUSE_CSV_PATH)
    )

    print(f"키스트로크 데이터: {keystroke_features.shape}")
    print(f"마우스 데이터: {mouse_features.shape}")
    print("데이터 전처리 완료!")

    return keystroke_features, keystroke_labels, mouse_features, mouse_labels

def run_training():
    """훈련 실행"""
    print("\n=== 모델 훈련 ===")

    try:
        # 훈련 스크립트 실행
        import subprocess
        result = subprocess.run([sys.executable, "train.py"], 
                              capture_output=True, text=True)

        print("훈련 스크립트 출력:")
        print(result.stdout)

        if result.stderr:
            print("오류:")
            print(result.stderr)

        if result.returncode == 0:
            print("훈련 완료!")
        else:
            print("훈련 실패!")

    except Exception as e:
        print(f"훈련 실행 중 오류: {e}")

def run_gui():
    """GUI 실행"""
    print("\n=== GUI 시작 ===")

    try:
        # GUI 스크립트 실행
        import subprocess
        result = subprocess.run([sys.executable, "gui.py"])

    except Exception as e:
        print(f"GUI 실행 중 오류: {e}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="내부자 위협 탐지 시스템")

    parser.add_argument("--mode", choices=["train", "gui", "test", "preprocess"], 
                       default="gui", help="실행 모드")
    parser.add_argument("--model-path", default=str(BEST_MODEL_PATH), 
                       help="모델 파일 경로")

    args = parser.parse_args()

    print("=" * 50)
    print("내부자 위협 탐지 시스템")
    print("=" * 50)

    # 필요한 디렉터리 생성
    from config import create_directories
    create_directories()

    if args.mode == "preprocess":
        # 데이터 전처리만 실행
        run_data_preprocessing()

    elif args.mode == "train":
        # 훈련 실행
        run_training()

    elif args.mode == "test":
        # 모델 테스트
        model = load_model(args.model_path)
        test_inference(model)

    elif args.mode == "gui":
        # GUI 실행
        run_gui()

    else:
        print(f"알 수 없는 모드: {args.mode}")

if __name__ == "__main__":
    # Windows에서의 멀티프로세싱 이슈 해결
    multiprocessing.freeze_support()
    main()
