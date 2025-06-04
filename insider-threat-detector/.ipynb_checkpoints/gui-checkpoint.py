# gui.py

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog
from main import main

class DetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("내부자 위협 탐지 AI")
        self.resize(350, 150)

        layout = QVBoxLayout()

        self.label = QLabel("데이터 및 모델 준비 후 탐지 시작")
        layout.addWidget(self.label)

        self.btn = QPushButton("탐지 실행")
        self.btn.clicked.connect(self.run_detection)
        layout.addWidget(self.btn)

        self.setLayout(layout)

    def run_detection(self):
        try:
            main()
            self.label.setText("탐지 완료! 결과는 콘솔을 확인하세요.")
        except Exception as e:
            self.label.setText(f"오류 발생: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DetectionApp()
    window.show()
    sys.exit(app.exec_())
