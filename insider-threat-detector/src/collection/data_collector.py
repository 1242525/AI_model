# C:\Projects\AISecApp\01\insider-threat-detector\src\collection\data_collector.py
"""
파일 경로: C:\Projects\AISecApp\01\insider-threat-detector\src\collection\data_collector.py
파일명: data_collector.py
설명: 실시간 키보드/마우스 데이터 수집
"""

import time
import sqlite3
import os
from pynput import keyboard, mouse
from pathlib import Path

class RealTimeCollector:
    def __init__(self, db_path="data/user_behavior.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.mouse_listener = mouse.Listener(on_move=self.on_mouse_move)
        
    def init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                event_type TEXT,
                key_data TEXT,
                x_coord INTEGER,
                y_coord INTEGER,
                user_id TEXT
            )
        ''')
        conn.commit()
        conn.close()
        
    def on_press(self, key):
        """키 입력 이벤트 처리"""
        self.save_to_database({
            'timestamp': time.time(),
            'event_type': 'keypress',
            'key_data': str(key),
            'x_coord': None,
            'y_coord': None,
            'user_id': self.get_current_user()
        })
    
    def on_mouse_move(self, x, y):
        """마우스 이동 이벤트 처리"""
        self.save_to_database({
            'timestamp': time.time(),
            'event_type': 'mouse_move',
            'key_data': None,
            'x_coord': x,
            'y_coord': y,
            'user_id': self.get_current_user()
        })
        
    def save_to_database(self, event_data):
        """이벤트 데이터를 데이터베이스에 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_events 
            (timestamp, event_type, key_data, x_coord, y_coord, user_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            event_data['timestamp'],
            event_data['event_type'],
            event_data['key_data'],
            event_data['x_coord'],
            event_data['y_coord'],
            event_data['user_id']
        ))
        
        conn.commit()
        conn.close()
    
    def get_current_user(self):
        """현재 사용자 ID 반환"""
        return os.getenv('USERNAME', 'unknown_user')
    
    def start_collection(self):
        """데이터 수집 시작"""
        self.listener.start()
        self.mouse_listener.start()
    
    def stop_collection(self):
        """데이터 수집 중지"""
        self.listener.stop()
        self.mouse_listener.stop()
