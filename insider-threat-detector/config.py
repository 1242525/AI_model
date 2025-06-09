"""
파일 경로: C:\Projects\AISecApp\01\insider-threat-detector\config.py
파일명: config.py
설명: 프로젝트 전체 설정 및 하이퍼파라미터 관리
"""

import os
from pathlib import Path

# 프로젝트 루트 디렉터리
PROJECT_ROOT = Path(r"C:/Projects/AISecApp/01/insider-threat-detector")

# 데이터 경로 설정
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
MODEL_DIR = DATA_ROOT / "models"

# 키스트로크 데이터셋 경로
KEYSTROKE_RAW_DIR = RAW_DATA_DIR / "keystroke"
KEYSTROKE_DATA_PATH = KEYSTROKE_RAW_DIR / "DSL-StrongPasswordData.txt"
KEYSTROKE_CSV_PATH = PROCESSED_DATA_DIR / "keystroke_processed.csv"

# 마우스 데이터셋 경로
MOUSE_RAW_DIR = RAW_DATA_DIR / "mouse"
MOUSE_DATA_PATH = MOUSE_RAW_DIR / "mouse_data.csv"
MOUSE_CSV_PATH = PROCESSED_DATA_DIR / "mouse_processed.csv"

# 모델 저장 경로
MODEL_SAVE_PATH = MODEL_DIR / "siamese_model.pth"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"

# (임시 추가)
MODEL_PATH = r'C:\Projects\AISecApp\01\insider-threat-detector\data\models\siamese_model.pth'  # r 접두사 추가
DATA_PATH = 'C:/Projects/AISecApp/01/insider-threat-detector/data'  # 슬래시 사용

# 데이터셋 다운로드 URL
DATASETS = {
    "cmu_keystroke": {
        "url": "https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.txt",
        "filename": "DSL-StrongPasswordData.txt",
        "description": "CMU Keystroke Dynamics Dataset"
    },
    "balabit_mouse": {
        "url": "https://github.com/balabit/Mouse-Dynamics-Challenge/archive/master.zip",
        "filename": "balabit_mouse.zip",
        "description": "Balabit Mouse Dynamics Challenge Dataset"
    }
}

# 모델 하이퍼파라미터
MODEL_CONFIG = {
    "input_dim": 128,
    "hidden_dim": 256,
    "embedding_dim": 128,
    "num_layers": 2,
    "dropout": 0.3
}

# 훈련 하이퍼파라미터
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 100,
    "margin": 1.0,
    "early_stopping_patience": 10,
    "validation_split": 0.2,
    "test_split": 0.1,
    "seed": 42
}

# 데이터 전처리 설정
PREPROCESSING_CONFIG = {
    "sequence_length": 100,  # 시퀀스 길이
    "keystroke_features": 64,  # 키스트로크 특징 수
    "mouse_features": 64,     # 마우스 특징 수
    "normalization": "minmax", # 정규화 방법
    "window_size": 5,         # 슬라이딩 윈도우 크기
}

# 실시간 데이터 수집 설정
COLLECTION_CONFIG = {
    "sampling_rate": 100,     # Hz
    "buffer_size": 1000,      # 버퍼 크기
    "session_timeout": 300,   # 세션 타임아웃 (초)
    "min_events": 50,         # 최소 이벤트 수
}

# GUI 설정
GUI_CONFIG = {
    "window_title": "내부자 위협 탐지 시스템",
    "window_size": (800, 600),
    "theme": "dark",
    "update_interval": 1000,  # ms
}

# 로깅 설정
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": PROJECT_ROOT / "logs" / "app.log",
}

# 이상 탐지 임계값
DETECTION_CONFIG = {
    "threshold": 0.5,         # 이상 탐지 임계값
    "confidence_threshold": 0.8,  # 신뢰도 임계값
    "alert_threshold": 3,     # 연속 이상 탐지 횟수
}

# 디렉터리 생성 함수
def create_directories():
    """필요한 디렉터리들을 생성합니다."""
    directories = [
        DATA_ROOT,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        KEYSTROKE_RAW_DIR,
        MOUSE_RAW_DIR,
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "outputs",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"디렉터리 생성: {directory}")

if __name__ == "__main__":
    create_directories()
    print("설정 파일 로드 완료")