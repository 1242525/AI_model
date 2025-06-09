"""
파일 경로: C:\Projects\AISecApp\01\insider-threat-detector\train.py
파일명: train.py
설명: Siamese Network 훈련 메인 스크립트
"""

from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import logging
import json
import sys
import os
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional
from torch.utils.data import DataLoader, Dataset, random_split

# 프로젝트 모듈 import
sys.path.append(str(Path(__file__).parent))
from config import *
from src.models.siamese import SiameseNetwork, ContrastiveLoss
from src.preprocessing.keystroke_preprocessor import preprocess_keystroke_data
from src.preprocessing.mouse_preprocessor import preprocess_mouse_data
from src.utils.data_downloader import setup_datasets

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BehaviorDataset(Dataset):
    """행동 패턴 데이터셋"""

    def __init__(self, keystroke_features: np.ndarray, mouse_features: np.ndarray, 
                 keystroke_labels: np.ndarray, mouse_labels: np.ndarray):
        self.keystroke_features = keystroke_features
        self.mouse_features = mouse_features
        self.keystroke_labels = keystroke_labels
        self.mouse_labels = mouse_labels

        # 데이터 정합성 확인
        assert len(keystroke_features) == len(keystroke_labels)
        assert len(mouse_features) == len(mouse_labels)

        # 사용자 매핑
        self.unique_users = np.unique(np.concatenate([keystroke_labels, mouse_labels]))
        self.user_to_idx = {user: idx for idx, user in enumerate(self.unique_users)}

        # 페어 생성
        self.pairs = self._create_pairs()
        
        # 차원 검증 코드 추가
        assert self.keystroke_features.shape[1] == 64, f"Keystroke features must be 64, got {self.keystroke_features.shape[1]}"
        assert self.mouse_features.shape[1] == 64, f"Mouse features must be 64, got {self.mouse_features.shape[1]}"

    def _create_pairs(self) -> List[Tuple]:
        """훈련용 positive/negative 페어 생성"""
        pairs = []
        user_samples = defaultdict(list)
        
        # 사용자별 샘플 그룹화
        for i in range(len(self.keystroke_features)):
            user = self.keystroke_labels[i]
            user_samples[user].append(i)
        
        # 각 사용자당 100개 페어만 생성
        for user, indices in user_samples.items():
            for _ in range(100):
                if len(indices) >= 2:
                    i, j = random.sample(indices, 2)
                    pairs.append((i, j, 0))  # Positive pair
                
                # Negative pair
                other_users = [u for u in user_samples.keys() if u != user]
                if other_users:
                    other_user = random.choice(other_users)
                    j = random.choice(user_samples[other_user])
                    pairs.append((i, j, 1))
        
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j, label = self.pairs[idx]  # 인덱스 추출
        return (
            torch.tensor(self.keystroke_features[i], dtype=torch.float32),
            torch.tensor(self.keystroke_features[j], dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )

class Trainer:
    """Siamese Network 훈련 클래스"""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"사용 디바이스: {self.device}")

        # 모델 초기화
        allowed_keys = ['input_dim', 'hidden_dim', 'embedding_dim', 'dropout']
        model_params = {k: v for k, v in config['model'].items() if k in allowed_keys}
        self.model = SiameseNetwork(**model_params).to(self.device)
        self.criterion = ContrastiveLoss(margin=config['training']['margin'])
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['training']['learning_rate']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

        # 훈련 기록
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """데이터 로드 및 전처리"""
        logger.info("데이터 로드 시작...")
        setup_datasets()

        # 키스트로크 데이터 전처리
        keystroke_path = KEYSTROKE_DATA_PATH
        if not keystroke_path.exists():
            logger.warning("키스트로크 데이터가 없습니다. 더미 데이터를 사용합니다.")

        keystroke_features, keystroke_labels = preprocess_keystroke_data(
            str(keystroke_path), 
            str(KEYSTROKE_CSV_PATH),
            feature_dim=64
        )

        # 마우스 데이터 전처리
        mouse_path = MOUSE_DATA_PATH
        mouse_features, mouse_labels = preprocess_mouse_data(
            str(mouse_path) if mouse_path.exists() else None,
            str(MOUSE_CSV_PATH),
            feature_dim=64
        )

        logger.info(f"키스트로크 데이터: {keystroke_features.shape}")
        logger.info(f"마우스 데이터: {mouse_features.shape}")

        # 데이터셋 생성
        dataset = BehaviorDataset(
            keystroke_features, mouse_features,
            keystroke_labels, mouse_labels
        )

        # 훈련/검증 분할
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config['training']['seed'])
        )

        # 데이터 로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=0,
            persistent_workers=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=0,
            persistent_workers=False
        )

        logger.info(f"훈련 데이터: {len(train_dataset)} 샘플")
        logger.info(f"검증 데이터: {len(val_dataset)} 샘플")

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader) -> float:
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, (x1, x2, labels) in enumerate(progress_bar):
            x1, x2, labels = x1.to(self.device), x2.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            out1, out2 = self.model(x1, x2)
            loss = self.criterion(out1, out2, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Progress bar 업데이트
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
            })

        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> float:
        """검증"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for x1, x2, labels in val_loader:
                x1, x2, labels = x1.to(self.device), x2.to(self.device), labels.to(self.device)

                out1, out2 = self.model(x1, x2)
                loss = self.criterion(out1, out2, labels)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """전체 훈련 과정"""
        logger.info("훈련 시작...")

        patience = self.config['training']['early_stopping_patience']
        patience_counter = 0

        for epoch in range(self.config['training']['num_epochs']):
            logger.info(f"Epoch {epoch + 1}/{self.config['training']['num_epochs']}")

            # 훈련
            train_loss = self.train_epoch(train_loader)

            # 검증
            val_loss = self.validate(val_loader)

            # 학습률 스케줄링
            self.scheduler.step(val_loss)

            # 기록
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(BEST_MODEL_PATH)
                patience_counter = 0
                logger.info("새로운 최고 모델 저장!")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # 최종 모델 저장
        self.save_model(MODEL_SAVE_PATH)
        self.plot_training_history()

    def save_model(self, path: Path):
        """모델 저장"""
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }

        torch.save(checkpoint, path)
        logger.info(f"모델 저장 완료: {path}")

    def plot_training_history(self):
        """훈련 기록 플롯"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)

        # 플롯 저장
        plot_path = PROJECT_ROOT / "outputs" / "training_history.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
        plt.show()

        logger.info(f"훈련 기록 플롯 저장: {plot_path}")

def main():
    """메인 훈련 함수"""
    # 시드 설정
    torch.manual_seed(TRAINING_CONFIG['seed'])
    np.random.seed(TRAINING_CONFIG['seed'])
    random.seed(TRAINING_CONFIG['seed'])

    # 설정 생성
    config = {
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'preprocessing': PREPROCESSING_CONFIG,
    }

    logger.info("내부자 위협 탐지 모델 훈련 시작")
    logger.info(f"설정: {json.dumps(config, indent=2, ensure_ascii=False)}")

    # 필요한 디렉터리 생성
    from config import create_directories
    create_directories()

    # 훈련 시작
    trainer = Trainer(config)
    train_loader, val_loader = trainer.load_data()
    trainer.train(train_loader, val_loader)

    logger.info("훈련 완료!")

if __name__ == "__main__":
    main()
