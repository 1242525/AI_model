"""
파일 경로: C:\Projects\AISecApp\01\insider-threat-detector\src\preprocessing\mouse_preprocessor.py
파일명: mouse_preprocessor.py
설명: 마우스 동작 데이터 전처리 및 특징 추출
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math
import warnings
warnings.filterwarnings('ignore')

class MousePreprocessor:
    """마우스 동작 데이터 전처리 클래스"""

    def __init__(self, sequence_length: int = 100, feature_dim: int = 64):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.scaler = None

    def _generate_dummy_mouse_data(self) -> pd.DataFrame:
        """더미 마우스 데이터 생성 (테스트용)"""
        print("더미 마우스 데이터 생성 중...")

        num_users = 10
        sessions_per_user = 100
        events_per_session = 500

        data = []

        for user_id in range(1, num_users + 1):
            for session in range(1, sessions_per_user + 1):
                user_speed_factor = 1 + (user_id - 1) * 0.2
                user_precision = 0.1 + (user_id - 1) * 0.05
                session_start_time = session * 3600
                current_x, current_y = 400, 300

                for event_id in range(events_per_session):
                    timestamp = session_start_time + event_id * np.random.exponential(0.1)
                    event_type = np.random.choice(['move', 'click', 'scroll'], p=[0.8, 0.15, 0.05])

                    if event_type == 'move':
                        movement_distance = np.random.exponential(50 * user_speed_factor)
                        angle = np.random.uniform(0, 2 * np.pi)
                        new_x = current_x + movement_distance * np.cos(angle)
                        new_y = current_y + movement_distance * np.sin(angle)
                        new_x = max(0, min(1920, new_x))
                        new_y = max(0, min(1080, new_y))
                        current_x, current_y = new_x, new_y

                        data.append({
                            'user_id': f'user_{user_id}',
                            'session_id': session,
                            'timestamp': timestamp,
                            'event_type': 'move',
                            'x': current_x,
                            'y': current_y,
                            'button': None,
                            'velocity': movement_distance / 0.1,
                        })

                    elif event_type == 'click':
                        button = np.random.choice(['left', 'right'], p=[0.9, 0.1])
                        data.append({
                            'user_id': f'user_{user_id}',
                            'session_id': session,
                            'timestamp': timestamp,
                            'event_type': 'click',
                            'x': current_x,
                            'y': current_y,
                            'button': button,
                            'velocity': 0,
                        })

                    elif event_type == 'scroll':
                        scroll_direction = np.random.choice(['up', 'down'])
                        data.append({
                            'user_id': f'user_{user_id}',
                            'session_id': session,
                            'timestamp': timestamp,
                            'event_type': 'scroll',
                            'x': current_x,
                            'y': current_y,
                            'button': scroll_direction,
                            'velocity': 0,
                        })

        df = pd.DataFrame(data)
        print(f"더미 마우스 데이터 생성 완료: {len(df)} 이벤트")
        return df

    def extract_movement_features(self, trajectory: np.ndarray) -> Dict[str, float]:
        """마우스 이동 궤적에서 특징을 추출합니다."""
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
        """기본 이동 특징 반환"""
        return {
            'total_distance': 0, 'avg_distance': 0, 'max_distance': 0,
            'distance_std': 0, 'avg_velocity': 0, 'max_velocity': 0,
            'velocity_std': 0, 'trajectory_length': 0, 'avg_direction_change': 0,
            'max_direction_change': 0, 'direction_change_std': 0,
            'linearity': 0, 'tremor': 0,
        }

    def _calculate_linearity(self, trajectory: np.ndarray) -> float:
        """궤적의 선형성 계산"""
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
        """떨림 정도 계산"""
        if len(distances) < 3:
            return 0.0
        accelerations = np.diff(distances)
        tremor = np.std(accelerations) if len(accelerations) > 0 else 0.0
        return tremor

    def extract_click_features(self, click_events):
        intervals = np.asarray([e['timestamp'] for e in click_events], dtype=np.float64)
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

    def extract_scroll_features(self, scroll_data: List[Dict]) -> Dict[str, float]:
        """스크롤 패턴 특징 추출"""
        # ----------- [여기서부터 수정/추가] -----------
        if not scroll_data:
            return {
                'scroll_frequency': 0, 'avg_scroll_interval': 0,
                'scroll_direction_changes': 0, 'scroll_speed': 0,
            }
        timestamps = np.array([event['timestamp'] for event in scroll_data], dtype=np.float64)
        if timestamps.size < 2:
            return {
                'scroll_frequency': len(scroll_data),
                'avg_scroll_interval': 0,
                'scroll_direction_changes': 0,
                'scroll_speed': 0,
            }
        intervals = np.diff(timestamps)
        directions = [event.get('button', 'up') for event in scroll_data]
        direction_changes = 0
        for i in range(1, len(directions)):
            if directions[i] != directions[i-1]:
                direction_changes += 1
        features = {
            'scroll_frequency': len(scroll_data),
            'avg_scroll_interval': np.mean(intervals) if intervals.size > 0 else 0,
            'scroll_direction_changes': direction_changes,
            'scroll_speed': len(scroll_data) / (timestamps[-1] - timestamps[0]) if (timestamps[-1] - timestamps[0]) > 0 else 0,
        }
        return features
        # ----------- [여기까지 수정/추가] -----------

    def create_session_features(self, session_data: pd.DataFrame) -> np.ndarray:
        """세션 데이터에서 특징 벡터 생성"""
        # 이벤트 타입별 분리
        move_events = session_data[session_data['event_type'] == 'move']
        click_events = session_data[session_data['event_type'] == 'click']
        scroll_events = session_data[session_data['event_type'] == 'scroll']

        # 이동 궤적 추출
        if len(move_events) > 0:
            trajectory = move_events[['x', 'y']].values
            movement_features = self.extract_movement_features(trajectory)
        else:
            movement_features = self._get_default_movement_features()

        # 클릭 특징 추출
        click_features = self.extract_click_features(click_events.to_dict('records'))

        # ----------- [여기서부터 수정/추가] -----------
        # 스크롤 이벤트 유효성 검증 추가
        if scroll_events.empty:
            scroll_features = {
                'scroll_frequency': 0,
                'avg_scroll_interval': 0,
                'scroll_direction_changes': 0,
                'scroll_speed': 0
            }
        else:
            timestamps = scroll_events['timestamp'].values
            if timestamps.size < 2:
                scroll_features = {
                    'scroll_frequency': len(scroll_events),
                    'avg_scroll_interval': 0,
                    'scroll_direction_changes': 0,
                    'scroll_speed': 0
                }
            else:
                scroll_features = self.extract_scroll_features(scroll_events.to_dict('records'))
        # ----------- [여기까지 수정/추가] -----------

        # 세션 전체 특징
        session_duration = session_data['timestamp'].max() - session_data['timestamp'].min()
        total_events = len(session_data)
        session_features = {
            'session_duration': session_duration,
            'total_events': total_events,
            'events_per_second': total_events / session_duration if session_duration > 0 else 0,
            'move_ratio': len(move_events) / total_events if total_events > 0 else 0,
            'click_ratio': len(click_events) / total_events if total_events > 0 else 0,
            'scroll_ratio': len(scroll_events) / total_events if total_events > 0 else 0,
        }

        # 모든 특징 결합
        all_features = {**movement_features, **click_features, **scroll_features, **session_features}
        feature_vector = list(all_features.values())
        if len(feature_vector) < self.feature_dim:
            feature_vector.extend([0] * (self.feature_dim - len(feature_vector)))
        else:
            feature_vector = feature_vector[:self.feature_dim]
        return np.array(feature_vector, dtype=np.float32)

    def process_dataset(self, data_path: Optional[str] = None, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        마우스 데이터셋을 처리합니다.
        """
        if data_path and Path(data_path).exists():
            df = pd.read_csv(data_path)
            print(f"데이터 로드 완료: {len(df)} 행")
        else:
            print("데이터 파일이 없어 더미 데이터를 생성합니다.")
            df = self._generate_dummy_mouse_data()

        features_list = []
        labels_list = []

        user_label_map = {user: idx for idx, user in enumerate(df['user_id'].unique())}

        for user_id in df['user_id'].unique():
            user_data = df[df['user_id'] == user_id]
            for session_id in user_data['session_id'].unique():
                session_data = user_data[user_data['session_id'] == session_id]
                if len(session_data) >= 1:
                    session_features = self.create_session_features(session_data)
                    features_list.append(session_features)
                    labels_list.append(user_label_map[user_id])

        # 수정: 빈 데이터셋 검증 추가
        if len(features_list) == 0:
            raise ValueError("처리된 데이터가 없습니다. 데이터셋 구조를 확인하세요.")

        features = np.array(features_list)
        labels = np.array(labels_list)

        if normalize:
            if self.scaler is None:
                self.scaler = StandardScaler()
                features = self.scaler.fit_transform(features)
            else:
                features = self.scaler.transform(features)

        print(f"마우스 데이터셋 처리 완료: {features.shape[0]} 샘플, {features.shape[1]} 특징")
        return features, labels

    def save_processed_data(self, features: np.ndarray, labels: np.ndarray, save_path: str):
        """처리된 데이터를 저장합니다."""
        processed_data = {
            'features': features,
            'labels': labels,
            'user_mapping': {i: f'user_{i}' for i in range(len(np.unique(labels)))},
            'feature_dim': self.feature_dim,
        }
        np.savez(save_path, **processed_data)
        print(f"처리된 마우스 데이터 저장 완료: {save_path}")

def preprocess_mouse_data(data_path: Optional[str] = None, output_path: str = "mouse_processed.npz", feature_dim: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """마우스 데이터 전처리 메인 함수"""
    preprocessor = MousePreprocessor(feature_dim=feature_dim)
    features, labels = preprocessor.process_dataset(data_path)
    preprocessor.save_processed_data(features, labels, output_path)
    return features, labels

if __name__ == "__main__":
    features, labels = preprocess_mouse_data()
    print(f"특징 형태: {features.shape}")
    print(f"레이블 형태: {labels.shape}")
    print(f"고유 사용자 수: {len(np.unique(labels))}")
