"""
파일 경로: C:\Projects\AISecApp\01\insider-threat-detector\src\preprocessing\mouse_preprocessor.py
파일명: mouse_preprocessor.py
설명: Balabit 마우스 데이터셋 전처리 클래스 (확장자 없는 파일/원본 컬럼명 지원)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import StandardScaler
import math
import warnings
warnings.filterwarnings('ignore')

class MousePreprocessor:
    """Balabit 마우스 데이터셋 전처리 클래스"""

    def __init__(self, sequence_length: int = 100, feature_dim: int = 64):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.scaler = None

    def extract_movement_features(self, trajectory: np.ndarray) -> Dict[str, float]:
        if len(trajectory) < 2:
            return self._get_default_movement_features()
        x_coords = trajectory[:, 0]
        y_coords = trajectory[:, 1]
        distances = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
        angles = []
        for i in range(1, len(trajectory)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            angle = math.atan2(dy, dx)
            angles.append(angle)
        angles = np.array(angles)
        direction_changes = []
        for i in range(1, len(angles)):
            change = abs(angles[i] - angles[i-1])
            if change > np.pi:
                change = 2 * np.pi - change
            direction_changes.append(change)
        direction_changes = np.array(direction_changes)
        features = {
            'total_distance': np.sum(distances),
            'avg_distance': np.mean(distances),
            'max_distance': np.max(distances),
            'distance_std': np.std(distances),
            'avg_velocity': np.mean(distances) if len(distances) > 0 else 0,
            'max_velocity': np.max(distances) if len(distances) > 0 else 0,
            'velocity_std': np.std(distances),
            'trajectory_length': len(trajectory),
            'avg_direction_change': np.mean(direction_changes) if len(direction_changes) > 0 else 0,
            'max_direction_change': np.max(direction_changes) if len(direction_changes) > 0 else 0,
            'direction_change_std': np.std(direction_changes) if len(direction_changes) > 0 else 0,
            'linearity': self._calculate_linearity(trajectory),
            'tremor': self._calculate_tremor(distances),
        }
        return features

    def _get_default_movement_features(self) -> Dict[str, float]:
        return {
            'total_distance': 0, 'avg_distance': 0, 'max_distance': 0,
            'distance_std': 0, 'avg_velocity': 0, 'max_velocity': 0,
            'velocity_std': 0, 'trajectory_length': 0, 'avg_direction_change': 0,
            'max_direction_change': 0, 'direction_change_std': 0,
            'linearity': 0, 'tremor': 0,
        }

    def _calculate_linearity(self, trajectory: np.ndarray) -> float:
        if len(trajectory) < 2:
            return 0.0
        start_point = trajectory[0]
        end_point = trajectory[-1]
        straight_distance = np.sqrt(np.sum((end_point - start_point)**2))
        actual_distance = 0
        for i in range(1, len(trajectory)):
            actual_distance += np.sqrt(np.sum((trajectory[i] - trajectory[i-1])**2))
        if actual_distance == 0:
            return 1.0
        return straight_distance / actual_distance

    def _calculate_tremor(self, distances: np.ndarray) -> float:
        if len(distances) < 3:
            return 0.0
        accelerations = np.diff(distances)
        tremor = np.std(accelerations) if len(accelerations) > 0 else 0.0
        return tremor

    def extract_click_features(self, click_events):
        intervals = np.asarray([e['rtime'] for e in click_events], dtype=np.float64)
        if intervals.size >= 2:
            intervals = np.diff(intervals) * 1000
        else:
            intervals = np.array([])
        return {
            'click_count': len(click_events),
            'avg_click_interval': intervals.mean() if intervals.size > 0 else 0.0,
            'std_click_interval': intervals.std() if intervals.size > 0 else 0.0,
            'max_click_interval': intervals.max() if intervals.size > 0 else 0.0
        }

    def create_session_features(self, session_data: pd.DataFrame) -> np.ndarray:
        move_events = session_data[session_data['state'] == 'Move']
        click_events = session_data[session_data['state'].isin(['Pressed', 'Released'])]
        drag_events = session_data[session_data['state'] == 'Dragging']

        if len(move_events) > 0:
            trajectory = move_events[['x', 'y']].values.astype(float)
            movement_features = self.extract_movement_features(trajectory)
        else:
            movement_features = self._get_default_movement_features()

        click_features = self.extract_click_features(click_events.to_dict('records'))

        session_duration = session_data['rtime'].max() - session_data['rtime'].min()
        total_events = len(session_data)
        session_features = {
            'session_duration': session_duration,
            'total_events': total_events,
            'events_per_second': total_events / session_duration if session_duration > 0 else 0,
            'move_ratio': len(move_events) / total_events if total_events > 0 else 0,
            'click_ratio': len(click_events) / total_events if total_events > 0 else 0,
            'drag_ratio': len(drag_events) / total_events if total_events > 0 else 0,
        }

        all_features = {**movement_features, **click_features, **session_features}
        feature_vector = list(all_features.values())
        if len(feature_vector) < self.feature_dim:
            padding = [0.0] * (self.feature_dim - len(feature_vector))
            feature_vector += padding
        else:
            feature_vector = feature_vector[:self.feature_dim]

        return np.array(feature_vector, dtype=np.float32)

    def process_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Balabit 마우스 데이터셋 처리"""
        # 데이터 경로 설정
        balabit_path = Path(r"C:\Projects\AISecApp\01\insider-threat-detector\data\raw\mouse\Mouse-Dynamics-Challenge\training_files")
        
        if not balabit_path.exists():
            raise FileNotFoundError(f"Balabit 데이터셋 경로를 찾을 수 없습니다: {balabit_path}")

        # 세션 데이터 로드
        sessions = []
        for user_dir in balabit_path.iterdir():
            if user_dir.is_dir() and user_dir.name.startswith('user'):
                print(f"사용자 디렉터리 처리 중: {user_dir.name}")
                session_files = list(user_dir.glob('*'))
                
                for session_file in session_files:
                    try:
                        # 컬럼명 명시적 지정 및 정제
                        df = pd.read_csv(session_file, names=[
                            'record timestamp',
                            'client timestamp',
                            'button',
                            'state',
                            'x',
                            'y'
                        ])
    
                        # 컬럼명 정제
                        df = df.rename(columns={
                            'record timestamp': 'rtime',
                            'client timestamp': 'ctime'
                        })
                        
                        df['user_id'] = user_dir.name
                        sessions.append(df)
                        print(f"세션 로드 완료: {session_file.name}")
                    except Exception as e:
                        print(f"경고: {session_file} 처리 실패 - {str(e)}")
                        continue

        if not sessions:
            error_msg = """
            처리된 세션 데이터가 없습니다. 다음 사항을 확인하세요:
            1. CSV 파일 존재 여부
            2. 파일 컬럼 구조: record timestamp,x,y,state,button
            3. 사용자 디렉터리 명명 규칙(user로 시작)
            """
            raise FileNotFoundError(error_msg)

        df = pd.concat(sessions, ignore_index=True)
        
        # 데이터 정제
        df = df.sort_values(by=['user_id', 'rtime'])  # ✅ 컬럼명 변경
        df = df.dropna()
        df = df.reset_index(drop=True)

        # 특징 추출
        features_list = []
        labels_list = []

        user_label_map = {user: idx for idx, user in enumerate(df['user_id'].unique())}

        for user_id in df['user_id'].unique():
            user_data = df[df['user_id'] == user_id]
            
            # 세션 구분 (60초 간격으로 세션 분할)
            time_diff = df['rtime'].diff().gt(60).cumsum()  # ✅ 컬럼명 변경
            for session_id, session_data in user_data.groupby(time_diff):
                if len(session_data) >= 5:
                    session_features = self.create_session_features(session_data)
                    features_list.append(session_features)
                    labels_list.append(user_label_map[user_id])

        # 특징 배열 변환
        features = np.array(features_list)
        labels = np.array(labels_list)

        # 정규화
        if self.scaler is None:
            self.scaler = StandardScaler()
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)

        print(f"마우스 데이터셋 처리 완료: {features.shape[0]} 샘플, {features.shape[1]} 특징")
        return features, labels


    def save_processed_data(self, features: np.ndarray, labels: np.ndarray, save_path: str):
        processed_data = {
            'features': features,
            'labels': labels,
            'user_mapping': {i: f'user_{i}' for i in range(len(np.unique(labels)))},
            'feature_dim': self.feature_dim,
        }
        np.savez(save_path, **processed_data)
        print(f"처리된 마우스 데이터 저장 완료: {save_path}")

def preprocess_mouse_data(output_path: str = "mouse_processed.npz", feature_dim: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    preprocessor = MousePreprocessor(feature_dim=feature_dim)
    features, labels = preprocessor.process_dataset()
    preprocessor.save_processed_data(features, labels, output_path)
    return features, labels

if __name__ == "__main__":
    features, labels = preprocess_mouse_data()
    print(f"특징 형태: {features.shape}")
    print(f"레이블 형태: {labels.shape}")
    print(f"고유 사용자 수: {len(np.unique(labels))}")
