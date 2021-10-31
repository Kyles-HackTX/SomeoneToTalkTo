import torch
import librosa
import librosa.display
from librosa.feature.spectral import mfcc
import numpy as np
from model import EmotionClassifier


class Predict():
    def __init__(self):
        self.model = EmotionClassifier()
        self.model.load_state_dict(torch.load('./mvp.zip'))
        self.model.eval()

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

        name_le_mapping = {0: 'female_angry',
    1: 'female_calm',
 2: 'female_disgust',
 3: 'female_fear',
 4: 'female_happy',
 5: 'female_neutral',
 6: 'female_sad',
 7: 'female_surprise',
 8: 'male_angry',
 9: 'male_calm',
 10: 'male_disgust',
 11: 'male_fear',
 12: 'male_happy',
 13: 'male_neutral',
 14: 'male_sad',
     15: 'male_surprise'}

        return name_le_mapping[predicted.item()]
