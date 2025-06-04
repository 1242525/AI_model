# dataset/keystroke_loader.py

import pandas as pd

def load_keystroke_data(path):
    df = pd.read_csv(path)
    return df
