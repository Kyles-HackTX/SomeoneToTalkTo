import torch
import torchaudio as ta
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram


from Loader import load_data

class Train:
    def __init__(self):
        # self.data_val = load_data('./data/val/', 3, 128)
    
    def extract_features(self):
        #Extract features from audio using MFCC

trainer = Train()