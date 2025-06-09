# C:\Projects\AISecApp\01\insider-threat-detector\src\detection\threat_detector.py
"""
파일 경로: C:\Projects\AISecApp\01\insider-threat-detector\src\detection\threat_detector.py
파일명: threat_detector.py
설명: 실시간 위협 탐지 로직
"""

import torch
import torch.nn.functional as F
import queue
from typing import Tuple
import numpy as np

class ThreatDetector:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold
        self.alert_queue = queue.Queue()
    
    def detect_anomaly(self, output1: torch.Tensor, output2: torch.Tensor) -> bool:
        """이상 행동 탐지"""
        # 거리 계산
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        # 이상치 판별
        if euclidean_distance > self.threshold:
            self.alert_queue.put("이상 행동 감지!")
            return True
        return False
    
    def get_risk_score(self, distance: float) -> float:
        """위험도 점수 계산"""
        return min(distance / self.threshold, 1.0)
