# main.py

from config import *
from dataset.keystroke_loader import load_keystroke_data
from dataset.mouse_loader import load_mouse_data
from preprocess.preprocess_keystroke import preprocess_keystrokes
from preprocess.preprocess_mouse import preprocess_mouse
from model.siamese import SiameseNetwork
from inference import run_inference

import torch

def main():
    keystroke_data = load_keystroke_data(KEYSTROKE_PATH)
    mouse_data = load_mouse_data(MOUSE_PATH)

    # 임의 행 0 기준 테스트
    key_seq = preprocess_keystrokes(keystroke_data.loc[0])
    mouse_seq = preprocess_mouse(mouse_data.loc[0])

    model = SiameseNetwork(input_dim=INPUT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    run_inference(model, key_seq, mouse_seq)

if __name__ == "__main__":
    main()
