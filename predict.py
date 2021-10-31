import torch
import librosa
import librosa.display
from librosa.feature.spectral import mfcc
import numpy as np
from model import EmotionClassifier

class Predict():
    def __init__(self):
        model = EmotionClassifier()
        model.load_state_dict(torch.load('./mvp.zip'))
        model.eval()

    def __call__(self, wav_file):
        X, sample_rate = librosa.load(wav_file,
                                      res_type='kaiser_fast',
                                      duration=2.5,
                                      sr=44100,
                                      offset=0.5
        )
        sample_rate = np.array(sample_rate)

        mfccs = np.mean(
            librosa.feature.mfcc(y=X,
                                 sr=sample_rate,
                                 n_mfcc=13),
            axis=0
        )

        example = torch.from_numpy(mfccs)
        example = torch.unsqueeze(torch.unsqueeze(example, dim=0), dim=0)
        print(example.shape, example)
        outputs = self.model(example.float())
        print(outputs.data)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        # print(le_name_mapping[max(predicted)])

        return name_le_mapping[predicted.item()]
