# C:\Projects\AISecApp\01\insider-threat-detector\src\models\advanced.py
"""
파일 경로: C:\Projects\AISecApp\01\insider-threat-detector\src\models\advanced.py
파일명: advanced.py
설명: 고급 멀티모달 모델 아키텍처
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class MultiModalModel(nn.Module):
    """멀티모달 행동 인식 모델"""
    
    def __init__(self, keystroke_dim=64, mouse_dim=64, hidden_dim=256, 
                 num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        # 키스트로크 처리 네트워크 (Transformer)
        self.keystroke_embedding = nn.Linear(keystroke_dim, hidden_dim)
        self.keystroke_pos_encoding = PositionalEncoding(hidden_dim, dropout)
        
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.keystroke_net = TransformerEncoder(encoder_layer, num_layers)
        
        # 마우스 처리 네트워크 (1D CNN)
        self.mouse_net = nn.Sequential(
            nn.Conv1d(mouse_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, hidden_dim)
        )
        
        # 크로스 어텐션 융합 레이어
        self.fusion_layer = CrossAttention(hidden_dim, num_heads)
        
        # 최종 분류 레이어
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, keystroke_data, mouse_data):
        # 키스트로크 특징 추출
        keystroke_emb = self.keystroke_embedding(keystroke_data)
        keystroke_emb = self.keystroke_pos_encoding(keystroke_emb)
        keystroke_features = self.keystroke_net(keystroke_emb)
        keystroke_pooled = torch.mean(keystroke_features, dim=1)
        
        # 마우스 특징 추출
        mouse_features = self.mouse_net(mouse_data.transpose(1, 2))
        
        # 크로스 어텐션 융합
        fused_features = self.fusion_layer(keystroke_pooled, mouse_features)
        
        # 최종 예측
        combined = torch.cat([keystroke_pooled, mouse_features], dim=1)
        output = self.classifier(combined)
        
        return output

class PositionalEncoding(nn.Module):
    """위치 인코딩"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class CrossAttention(nn.Module):
    """크로스 어텐션 메커니즘"""
    
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, query, key_value):
        # query: keystroke features, key_value: mouse features
        query = query.unsqueeze(1)  # Add sequence dimension
        key_value = key_value.unsqueeze(1)
        
        attn_output, _ = self.multihead_attn(query, key_value, key_value)
        output = self.norm(query + attn_output)
        
        return output.squeeze(1)
