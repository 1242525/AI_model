# train.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from model.siamese import SiameseNetwork
from model.utils import ContrastiveLoss
from config import *

from preprocess.preprocess_keystroke import preprocess_keystrokes
from preprocess.preprocess_mouse import preprocess_mouse

class BehaviorDataset(Dataset):
    def __init__(self, keystroke_df, mouse_df):
        self.keystroke_df = keystroke_df
        self.mouse_df = mouse_df

        self.users = keystroke_df['user_id'].unique()
        self.pairs = []
        # 모든 사용자별로 (positive + negative) 쌍 생성
        for i, user_a in enumerate(self.users):
            idx_a = keystroke_df[keystroke_df['user_id'] == user_a].index.tolist()
            for j, user_b in enumerate(self.users):
                idx_b = keystroke_df[keystroke_df['user_id'] == user_b].index.tolist()
                for ia in idx_a:
                    for ib in idx_b:
                        label = 0 if user_a == user_b else 1
                        self.pairs.append((ia, ib, label))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ia, ib, label = self.pairs[idx]

        key_i = preprocess_keystrokes(self.keystroke_df.loc[ia])
        key_j = preprocess_keystrokes(self.keystroke_df.loc[ib])
        mouse_i = preprocess_mouse(self.mouse_df.loc[ia])
        mouse_j = preprocess_mouse(self.mouse_df.loc[ib])

        x1 = np.concatenate([key_i, mouse_i])
        x2 = np.concatenate([key_j, mouse_j])

        return torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def train():
    keystroke_df = pd.read_csv(KEYSTROKE_PATH)
    mouse_df = pd.read_csv(MOUSE_PATH)

    dataset = BehaviorDataset(keystroke_df, mouse_df)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SiameseNetwork(input_dim=INPUT_DIM)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(10):
        total_loss = 0
        for x1, x2, label in dataloader:
            optimizer.zero_grad()
            out1, out2 = model(x1, x2)
            loss = criterion(out1, out2, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("모델 학습 완료 및 저장 완료")

if __name__ == "__main__":
    train()
