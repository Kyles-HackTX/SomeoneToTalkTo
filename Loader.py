import torch
import torchaudio as ta
from torch.utils.data import Dataset, DataLoader

AUDIO_ONLY = '03'

class Loader:
    def __init__(self, path):
        from glob import glob
        from os import path
        self.files = []
        for audio_f in glob(path.join(path, '*.wav')):
            self.files.append(audio_f.replace('.wav', ''))

        self.files = list(filter(lambda f: f[:2] == AUDIO_ONLY, self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx, separate_weak_and_strong=True):
        f_name = self.files[idx]
        label = f_name.split('-')
        audio_tensor = ta.load(f_name + '.wav')

        lbl = int(label[2]) - 1
        if separate_weak_and_strong:
            lbl = (2 * lbl) - 1
            strength = int(label[3]) - 1
            lbl += strength
            if lbl > 1:
                lbl -= 1

        return audio_tensor, lbl


"""
    Recommended: num_workers = 6 for training, 3 for validiation/test
    batch_size = 128, 64, or 32 - the larger, the better usually
    Example usage:
        ```
        # load from whatever path
        data_train = load_data("data/train", 6, 128)
        data_val = load_data("data/val", 3, 128)
        for _ in range(n_epochs):
            for signal, label in data_train:
                ...
        ```
"""
def load_data(path, num_workers=0, batch_size=32):
    loader = Loader(path)
    return DataLoader(loader, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)