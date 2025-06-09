"""
파일 경로: C:\Projects\AISecApp\01\insider-threat-detector\gui.py
파일명: gui.py
설명: 내부자 위협 탐지 시스템 GUI
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                           QProgressBar, QTabWidget, QGridLayout, QGroupBox,
                           QFileDialog, QMessageBox, QSlider, QLCDNumber)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QFont, QPalette, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pynput import keyboard, mouse
import time
import queue
import threading

# 프로젝트 모듈
sys.path.append(str(Path(__file__).parent))
from config import *
from src.models.siamese import SiameseNetwork
from src.preprocessing.keystroke_preprocessor import KeystrokePreprocessor
from src.preprocessing.mouse_preprocessor import MousePreprocessor

class DataCollector(QThread):
    """실시간 데이터 수집 스레드"""

    data_received = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.running = False
        self.keystroke_data = []
        self.mouse_data = []

    def start_collection(self):
        """데이터 수집 시작"""
        self.running = True
        self.start()

    def stop_collection(self):
        """데이터 수집 중지"""
        self.running = False

    def run(self):
        """데이터 수집 실행"""
        # 키보드 리스너
        def on_key_press(key):
            if self.running:
                self.keystroke_data.append({
                    'timestamp': time.time(),
                    'key': str(key),
                    'event': 'press'
                })

        def on_key_release(key):
            if self.running:
                self.keystroke_data.append({
                    'timestamp': time.time(),
                    'key': str(key),
                    'event': 'release'
                })

        # 마우스 리스너
        def on_mouse_move(x, y):
            if self.running:
                self.mouse_data.append({
                    'timestamp': time.time(),
                    'x': x,
                    'y': y,
                    'event': 'move'
                })

        def on_mouse_click(x, y, button, pressed):
            if self.running:
                self.mouse_data.append({
                    'timestamp': time.time(),
                    'x': x,
                    'y': y,
                    'button': str(button),
                    'pressed': pressed,
                    'event': 'click'
                })

        # 리스너 시작
        keyboard_listener = keyboard.Listener(
            on_press=on_key_press,
            on_release=on_key_release
        )
        mouse_listener = mouse.Listener(
            on_move=on_mouse_move,
            on_click=on_mouse_click
        )

        keyboard_listener.start()
        mouse_listener.start()

        # 주기적으로 데이터 전송
        while self.running:
            time.sleep(1)  # 1초마다

            if len(self.keystroke_data) > 0 or len(self.mouse_data) > 0:
                data = {
                    'keystroke': self.keystroke_data.copy(),
                    'mouse': self.mouse_data.copy()
                }
                self.data_received.emit(data)

                # 데이터 초기화 (최근 100개만 유지)
                self.keystroke_data = self.keystroke_data[-100:]
                self.mouse_data = self.mouse_data[-100:]

        keyboard_listener.stop()
        mouse_listener.stop()

class ThreatDetector(QThread):
    """위협 탐지 스레드"""

    threat_detected = pyqtSignal(float, str)  # 위험도, 메시지

    def __init__(self):
        super().__init__()
        self.model = None
        self.keystroke_preprocessor = KeystrokePreprocessor()
        self.mouse_preprocessor = MousePreprocessor()
        self.data_buffer = []
        self.running = False

    def load_model(self, model_path: str):
        """모델 로드"""
        try:
            if Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location='cpu')
                self.model = SiameseNetwork(**checkpoint['config']['model'])
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                return True
            else:
                # 더미 모델 생성 (테스트용)
                self.model = SiameseNetwork()
                self.model.eval()
                return True
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return False

    def analyze_data(self, data: dict):
        """데이터 분석 및 위협 탐지"""
        try:
            # 간단한 휴리스틱 기반 탐지 (실제로는 모델 사용)
            keystroke_count = len(data.get('keystroke', []))
            mouse_count = len(data.get('mouse', []))

            # 비정상적인 활동 패턴 감지
            threat_score = 0.0
            message = "정상"

            if keystroke_count > 50:  # 너무 많은 키 입력
                threat_score += 0.3
                message = "비정상적인 키보드 활동 감지"

            if mouse_count > 100:  # 너무 많은 마우스 이벤트
                threat_score += 0.3
                message = "비정상적인 마우스 활동 감지"

            # 랜덤 요소 추가 (시연용)
            import random
            if random.random() < 0.1:  # 10% 확률로 위협 감지
                threat_score += 0.5
                message = "의심스러운 행동 패턴 감지"

            self.threat_detected.emit(threat_score, message)

        except Exception as e:
            print(f"데이터 분석 실패: {e}")

    def add_data(self, data: dict):
        """데이터 버퍼에 추가"""
        self.data_buffer.append(data)

        # 최근 10개 데이터만 유지
        if len(self.data_buffer) > 10:
            self.data_buffer.pop(0)

        # 데이터 분석
        if len(self.data_buffer) >= 3:  # 최소 3개 데이터가 있을 때 분석
            combined_data = {
                'keystroke': [],
                'mouse': []
            }
            for d in self.data_buffer:
                combined_data['keystroke'].extend(d.get('keystroke', []))
                combined_data['mouse'].extend(d.get('mouse', []))

            self.analyze_data(combined_data)

class PlotWidget(FigureCanvas):
    """실시간 플롯 위젯"""

    def __init__(self):
        self.figure = Figure(figsize=(10, 6), facecolor='none')
        super().__init__(self.figure)
        self.ax = self.figure.add_subplot(111, facecolor='none')
        
        # 축 라벨 색상 설정
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        
        # 틱 마커 색상 설정
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')

    def update_plot(self, threat_score: float):
        """플롯 업데이트"""
        current_time = time.time()
        self.threat_scores.append(threat_score)
        self.timestamps.append(current_time)

        # 최근 100개 데이터만 유지
        if len(self.threat_scores) > 100:
            self.threat_scores.pop(0)
            self.timestamps.pop(0)

        # 플롯 업데이트
        self.ax.clear()
        if len(self.threat_scores) > 0:
            # 시간을 상대적으로 변환
            relative_times = [(t - self.timestamps[0]) for t in self.timestamps]
            self.ax.plot(relative_times, self.threat_scores, 'b-', linewidth=2)
            self.ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='위험 임계값')
            self.ax.fill_between(relative_times, self.threat_scores, alpha=0.3)
            self.ax.legend()

        self.ax.set_title('실시간 위험도 모니터링')
        self.ax.set_xlabel('시간 (초)')
        self.ax.set_ylabel('위험도')
        self.ax.set_ylim(0, 1)
        self.ax.grid(True, alpha=0.3)

        self.draw()

class MainWindow(QMainWindow):
    """메인 윈도우"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("내부자 위협 탐지 시스템")
        self.setGeometry(100, 100, 1200, 800)

        # 컴포넌트 초기화
        self.data_collector = DataCollector()
        self.threat_detector = ThreatDetector()
        self.current_threat_level = 0.0

        # UI 설정
        self.setup_ui()
        self.setup_connections()

        # 타이머 설정
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_status)
        self.update_timer.start(1000)  # 1초마다 업데이트

    def setup_ui(self):
        """UI 초기화"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 메인 레이아웃
        main_layout = QVBoxLayout(central_widget)

        # 상단 제어 패널
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        # 탭 위젯
        tab_widget = QTabWidget()

        # 실시간 모니터링 탭
        monitor_tab = self.create_monitor_tab()
        tab_widget.addTab(monitor_tab, "실시간 모니터링")

        # 설정 탭
        settings_tab = self.create_settings_tab()
        tab_widget.addTab(settings_tab, "설정")

        # 로그 탭
        log_tab = self.create_log_tab()
        tab_widget.addTab(log_tab, "로그")

        main_layout.addWidget(tab_widget)

        # 하단 상태바
        self.statusBar().showMessage("시스템 준비")

    def create_control_panel(self) -> QGroupBox:
        """제어 패널 생성"""
        group = QGroupBox("제어 패널")
        layout = QHBoxLayout(group)

        # 시작/정지 버튼
        self.start_button = QPushButton("모니터링 시작")
        self.start_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")

        self.stop_button = QPushButton("모니터링 정지")
        self.stop_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
        self.stop_button.setEnabled(False)

        # 모델 로드 버튼
        self.load_model_button = QPushButton("모델 로드")

        # 현재 위험도 표시
        self.threat_lcd = QLCDNumber()
        self.threat_lcd.setSegmentStyle(QLCDNumber.Flat)
        self.threat_lcd.setDigitCount(4)
        self.threat_lcd.display("0.00")

        threat_label = QLabel("현재 위험도:")
        threat_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.load_model_button)
        layout.addStretch()
        layout.addWidget(threat_label)
        layout.addWidget(self.threat_lcd)

        return group

    def create_monitor_tab(self) -> QWidget:
        """모니터링 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 위험도 플롯
        self.plot_widget = PlotWidget()
        layout.addWidget(self.plot_widget)

        # 상태 정보
        status_group = QGroupBox("시스템 상태")
        status_layout = QGridLayout(status_group)

        self.keystroke_count_label = QLabel("키보드 이벤트: 0")
        self.mouse_count_label = QLabel("마우스 이벤트: 0")
        self.threat_status_label = QLabel("상태: 정상")
        self.last_update_label = QLabel("마지막 업데이트: -")

        status_layout.addWidget(self.keystroke_count_label, 0, 0)
        status_layout.addWidget(self.mouse_count_label, 0, 1)
        status_layout.addWidget(self.threat_status_label, 1, 0)
        status_layout.addWidget(self.last_update_label, 1, 1)

        layout.addWidget(status_group)

        return widget

    def create_settings_tab(self) -> QWidget:
        """설정 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 임계값 설정
        threshold_group = QGroupBox("탐지 임계값")
        threshold_layout = QVBoxLayout(threshold_group)

        threshold_label = QLabel("위험도 임계값:")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(10)
        self.threshold_slider.setMaximum(90)
        self.threshold_slider.setValue(50)

        self.threshold_value_label = QLabel("0.50")

        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_value_label)

        layout.addWidget(threshold_group)
        layout.addStretch()

        return widget

    def create_log_tab(self) -> QWidget:
        """로그 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # 로그 제어 버튼
        log_control_layout = QHBoxLayout()
        clear_log_button = QPushButton("로그 지우기")
        save_log_button = QPushButton("로그 저장")

        log_control_layout.addWidget(clear_log_button)
        log_control_layout.addWidget(save_log_button)
        log_control_layout.addStretch()

        layout.addLayout(log_control_layout)

        # 연결
        clear_log_button.clicked.connect(self.clear_log)
        save_log_button.clicked.connect(self.save_log)

        return widget

    def setup_connections(self):
        """시그널-슬롯 연결"""
        self.start_button.clicked.connect(self.start_monitoring)
        self.stop_button.clicked.connect(self.stop_monitoring)
        self.load_model_button.clicked.connect(self.load_model)

        self.data_collector.data_received.connect(self.on_data_received)
        self.threat_detector.threat_detected.connect(self.on_threat_detected)

        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)

    def start_monitoring(self):
        """모니터링 시작"""
        self.data_collector.start_collection()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.statusBar().showMessage("모니터링 중...")
        self.add_log("모니터링을 시작했습니다.")

    def stop_monitoring(self):
        """모니터링 정지"""
        self.data_collector.stop_collection()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.statusBar().showMessage("모니터링 정지됨")
        self.add_log("모니터링을 정지했습니다.")

    def load_model(self, model_path: str):
        """모델 로드 개선"""
        try:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"모델 파일 없음: {model_path}")
                
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model = SiameseNetwork(**checkpoint['config']['model'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # 성공 메시지 표시
            QMessageBox.information(self, "성공", 
                f"모델 버전: {checkpoint.get('version','1.0')}\n"
                f"훈련 정확도: {checkpoint.get('accuracy',0):.2f}%")
                
            return True
        except Exception as e:
            QMessageBox.critical(self, "오류", f"모델 로드 실패: {str(e)}")
            return False

    def on_data_received(self, data: dict):
        """데이터 수신 처리"""
        keystroke_count = len(data.get('keystroke', []))
        mouse_count = len(data.get('mouse', []))

        self.keystroke_count_label.setText(f"키보드 이벤트: {keystroke_count}")
        self.mouse_count_label.setText(f"마우스 이벤트: {mouse_count}")

        # 위협 탐지기에 데이터 전달
        self.threat_detector.add_data(data)

    def on_threat_detected(self, threat_score: float, message: str):
        """위협 탐지 처리"""
        self.current_threat_level = threat_score
        self.threat_lcd.display(f"{threat_score:.2f}")

        # 위험도에 따른 색상 변경
        if threat_score > 0.7:
            self.threat_lcd.setStyleSheet("QLCDNumber { color: red; }")
            self.threat_status_label.setText(f"상태: 위험 - {message}")
            self.add_log(f"[위험] {message} (위험도: {threat_score:.2f})")
        elif threat_score > 0.5:
            self.threat_lcd.setStyleSheet("QLCDNumber { color: orange; }")
            self.threat_status_label.setText(f"상태: 주의 - {message}")
            self.add_log(f"[주의] {message} (위험도: {threat_score:.2f})")
        else:
            self.threat_lcd.setStyleSheet("QLCDNumber { color: green; }")
            self.threat_status_label.setText("상태: 정상")

        # 플롯 업데이트
        self.plot_widget.update_plot(threat_score)

    def on_threshold_changed(self, value: int):
        """임계값 변경"""
        threshold = value / 100.0
        self.threshold_value_label.setText(f"{threshold:.2f}")

    def update_status(self):
        """상태 업데이트"""
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.last_update_label.setText(f"마지막 업데이트: {current_time}")

    def add_log(self, message: str):
        """로그 추가"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_text.append(log_entry)

    def clear_log(self):
        """로그 지우기"""
        self.log_text.clear()

    def save_log(self):
        """로그 저장"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "로그 저장", "threat_detection_log.txt", "텍스트 파일 (*.txt)"
        )
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.log_text.toPlainText())
            QMessageBox.information(self, "성공", f"로그가 저장되었습니다: {filename}")

def main():
    """GUI 애플리케이션 실행"""
    app = QApplication(sys.argv)

    # 다크 테마 적용
    app.setStyleSheet("""
        QMainWindow { background-color: #2b2b2b; color: white; }
        QGroupBox { font-weight: bold; color: white; }
        QLabel { color: white; }
        QPushButton { 
            padding: 8px; 
            border: 1px solid #555; 
            border-radius: 4px; 
            background-color: #404040; 
            color: white; 
        }
        QPushButton:hover { background-color: #505050; }
        QTextEdit { background-color: #1e1e1e; color: white; border: 1px solid #555; }
        QTabWidget::pane { border: 1px solid #555; background-color: #2b2b2b; }
        QTabBar::tab { 
            background-color: #404040; 
            color: white; 
            padding: 8px; 
            margin-right: 2px; 
        }
        QTabBar::tab:selected { background-color: #555; }
    """)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()