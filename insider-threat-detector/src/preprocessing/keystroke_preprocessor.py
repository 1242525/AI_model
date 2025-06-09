"""
파일 경로: C:\Projects\AISecApp\01\insider-threat-detector\src\preprocessing\keystroke_preprocessor.py
파일명: keystroke_preprocessor.py
설명: CMU 키스트로크 데이터셋 전처리 및 특징 추출
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class KeystrokePreprocessor:
    """CMU Keystroke Dynamics Dataset 전처리 클래스"""

    def __init__(self, sequence_length: int = 100, feature_dim: int = 64):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.scaler = None
        self.user_stats = {}

    def parse_cmu_dataset(self, file_path: str) -> pd.DataFrame:
        """CMU 데이터 정확한 고정 너비 파싱"""
        try:
            # 정확한 컬럼 위치 지정 (CMU 데이터셋 구조에 맞춤)
            colspecs = [
                (0, 5), (6, 8), (9, 12),   # subject, sessionIndex, rep
                (13, 20), (21, 28), (29, 36),  # H.period, DD.period.t, UD.period.t
                (37, 44), (45, 52), (53, 60),  # H.t, DD.t.i, UD.t.i
                (61, 68), (69, 76), (77, 84),  # H.i, DD.i.e, UD.i.e
                (85, 92), (93, 100), (101, 108), # H.e, DD.e.five, UD.e.five
                (109, 116), (117, 124), (125, 132), # H.five, DD.five.Shift.r, UD.five.Shift.r
                (133, 140), (141, 148), (149, 156), # H.Shift.r, DD.Shift.r.o, UD.Shift.r.o
                (157, 164), (165, 172), (173, 180), # H.o, DD.o.a, UD.o.a
                (181, 188), (189, 196), (197, 204), # H.a, DD.a.n, UD.a.n
                (205, 212), (213, 220), (221, 228)  # H.n, DD.n.l, UD.n.l
            ]
            
            # 정확한 컬럼 이름 (31개)
            columns = [
                'subject', 'sessionIndex', 'rep',
                'H.period', 'DD.period.t', 'UD.period.t',
                'H.t', 'DD.t.i', 'UD.t.i',
                'H.i', 'DD.i.e', 'UD.i.e',
                'H.e', 'DD.e.five', 'UD.e.five',
                'H.five', 'DD.five.Shift.r', 'UD.five.Shift.r',
                'H.Shift.r', 'DD.Shift.r.o', 'UD.Shift.r.o',
                'H.o', 'DD.o.a', 'UD.o.a',
                'H.a', 'DD.a.n', 'UD.a.n',
                'H.n', 'DD.n.l', 'UD.n.l'
            ]
            
            df = pd.read_fwf(
                file_path,
                colspecs=colspecs,
                header=0,
                names=columns,
                dtype={'subject': str}
            )
            
            # 데이터 타입 변환
            numeric_cols = df.columns.drop(['subject'])
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            return df
            
        except Exception as e:
            print(f"파일 파싱 오류: {e}")
            raise

    def _generate_dummy_keystroke_data(self) -> pd.DataFrame:
        """더미 키스트로크 데이터 생성 (테스트용)"""
        print("더미 키스트로크 데이터 생성 중...")

        # 51명의 사용자, 각각 50개의 세션
        num_users = 51
        sessions_per_user = 50

        data = []

        for user_id in range(1, num_users + 1):
            for session in range(1, sessions_per_user + 1):
                # 키 타이밍 특징 생성 (더 현실적인 값들)
                base_timing = np.random.normal(0.15, 0.05)  # 기본 키 누름 시간

                row = {
                    'subject': f's{user_id:03d}',
                    'sessionIndex': session,
                    'rep': 1,
                }

                # 키별 Hold Time (H.), Down-Down Time (DD.), Up-Down Time (UD.) 생성
                keys = ['.', 't', 'i', 'e', '5', 'R', 'o', 'a', 'n', 'l']

                for i, key in enumerate(keys):
                    # 사용자별 고유한 패턴 생성
                    user_factor = 1 + (user_id - 1) * 0.1

                    # Hold time
                    h_time = max(0.05, np.random.normal(base_timing * user_factor, 0.02))
                    row[f'H.{key}'] = h_time

                    if i < len(keys) - 1:  # 마지막 키가 아닌 경우
                        next_key = keys[i + 1]

                        # Down-Down time
                        dd_time = max(0.1, np.random.normal(0.3 * user_factor, 0.05))
                        row[f'DD.{key}.{next_key}'] = dd_time

                        # Up-Down time
                        ud_time = dd_time - h_time + np.random.normal(0, 0.01)
                        row[f'UD.{key}.{next_key}'] = max(0, ud_time)

                data.append(row)

        df = pd.DataFrame(data)
        print(f"더미 데이터 생성 완료: {len(df)} 행")
        return df

    def extract_timing_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """타이밍 특징을 추출합니다."""
        print("타이밍 특징 추출 중...")

        users = df['subject'].unique()
        features_dict = {}

        for user in users:
            user_data = df[df['subject'] == user]

            # Hold times
            h_cols = [col for col in df.columns if col.startswith('H.')]
            hold_times = user_data[h_cols].values

            # Down-Down times
            dd_cols = [col for col in df.columns if col.startswith('DD.')]
            dd_times = user_data[dd_cols].values

            # Up-Down times
            ud_cols = [col for col in df.columns if col.startswith('UD.')]
            ud_times = user_data[ud_cols].values

            # 특징 결합
            all_features = np.concatenate([hold_times, dd_times, ud_times], axis=1)

            # NaN 값 처리
            all_features = np.nan_to_num(all_features)

            features_dict[user] = all_features

        print(f"특징 추출 완료: {len(users)} 사용자")
        return features_dict

    def create_statistical_features(self, timing_data: np.ndarray) -> np.ndarray:
        """통계적 특징을 생성합니다."""
        if timing_data.size == 0:
            return np.zeros(self.feature_dim)

        features = []

        # 기본 통계량
        features.extend([
            np.mean(timing_data),
            np.std(timing_data),
            np.median(timing_data),
            np.min(timing_data),
            np.max(timing_data),
        ])

        # 백분위수
        percentiles = [25, 75, 90, 95]
        for p in percentiles:
            features.append(np.percentile(timing_data, p))

        # 분포 특성
        features.extend([
            np.var(timing_data),
            self._calculate_skewness(timing_data),
            self._calculate_kurtosis(timing_data),
        ])

        # 시계열 특성
        if len(timing_data) > 1:
            diff = np.diff(timing_data)
            features.extend([
                np.mean(diff),
                np.std(diff),
            ])
        else:
            features.extend([0, 0])

        # 패딩 또는 자르기
        features = np.array(features)
        if len(features) < self.feature_dim:
            # 패딩
            padding = np.zeros(self.feature_dim - len(features))
            features = np.concatenate([features, padding])
        else:
            # 자르기
            features = features[:self.feature_dim]

        return features

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """왜도 계산"""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """첨도 계산"""
        if len(data) < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3

    

    def process_dataset(self, file_path: str, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """전체 데이터셋 처리"""
        # 데이터 파싱
        df = self.parse_cmu_dataset(file_path)

        # 특징 추출
        timing_features = self.extract_timing_features(df)

        # 특징 및 레이블 배열 생성
        features_list = []
        labels_list = []

        for user_id, (user, user_timings) in enumerate(timing_features.items()):
            for session_data in user_timings:
                # 통계적 특징 생성
                stat_features = self.create_statistical_features(session_data)
                features_list.append(stat_features)
                labels_list.append(user_id)

        features = np.array(features_list)
        labels = np.array(labels_list)

        # 정규화
        if normalize:
            if self.scaler is None:
                self.scaler = StandardScaler()
                features = self.scaler.fit_transform(features)
            else:
                features = self.scaler.transform(features)

        print(f"데이터셋 처리 완료: {features.shape[0]} 샘플, {features.shape[1]} 특징")
        return features, labels

    def save_processed_data(self, features: np.ndarray, labels: np.ndarray, 
                           save_path: str):
        """처리된 데이터를 저장합니다."""
        processed_data = {
            'features': features,
            'labels': labels,
            'user_mapping': {i: f'user_{i}' for i in range(len(np.unique(labels)))},
            'feature_dim': self.feature_dim,
            'sequence_length': self.sequence_length
        }

        np.savez(save_path, **processed_data)
        print(f"처리된 데이터 저장 완료: {save_path}")


def preprocess_keystroke_data(data_path: str, output_path: str, 
                             feature_dim: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """키스트로크 데이터 전처리 메인 함수"""

    preprocessor = KeystrokePreprocessor(feature_dim=feature_dim)

    # 데이터 처리
    features, labels = preprocessor.process_dataset(data_path)

    # 저장
    preprocessor.save_processed_data(features, labels, output_path)

    return features, labels


if __name__ == "__main__":
    # 테스트
    import os

    # 더미 데이터로 테스트
    preprocessor = KeystrokePreprocessor()
    dummy_df = preprocessor._generate_dummy_keystroke_data()

    # 임시 파일 저장
    temp_file = "temp_keystroke.csv"
    dummy_df.to_csv(temp_file, index=False)

    # 전처리 테스트
    features, labels = preprocess_keystroke_data(temp_file, "keystroke_processed.npz")

    print(f"특징 형태: {features.shape}")
    print(f"레이블 형태: {labels.shape}")
    print(f"고유 사용자 수: {len(np.unique(labels))}")

    # 임시 파일 삭제
    if os.path.exists(temp_file):
        os.remove(temp_file)