from glob import glob
import os
from pathlib import Path

from librosa.feature.spectral import mfcc
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import librosa
import librosa.display
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from preprocess import pre_train, Loader


load_path = "./data/train/RAVDESS/"

dir_list = os.listdir(load_path)
dir_list.sort()

emotion = []
gender = []
path = []

for i in dir_list:
    fname = os.listdir(load_path + i)

    for f in fname:
        part = f.split('.')[0].split('-')

        if(part[0] == '03'):
            emotion.append(int(part[2]))
            temp = int(part[6])

            if temp%2 == 0:
                temp = "female"

            else:
                temp = "male"

            gender.append(temp)
            path.append(load_path + i + '/' + f)


RAV_df = pd.DataFrame(emotion)
RAV_df = RAV_df.replace({
    1:'neutral',
    2:'calm',
    3:'happy',
    4:'sad',
    5:'angry',
    6:'fear',
    7:'disgust',
    8:'surprise'
})

RAV_df = pd.concat([pd.DataFrame(gender), RAV_df], axis=1)
RAV_df.columns = ['gender','emotion']
RAV_df['labels'] = RAV_df.gender + '_' + RAV_df.emotion
RAV_df['source'] = 'RAVDESS'
RAV_df = pd.concat(
    [RAV_df,pd.DataFrame(path, columns = ['path'])],
    axis=1
)
RAV_df = RAV_df.drop(['gender', 'emotion'], axis=1)
wav_df = RAV_df


X_traintest, X_pred, y_traintest, y_pred = train_test_split(
    wav_df,
    wav_df.labels,
    test_size=0.20,
    shuffle=True,
    random_state=20
)

traintest_df = pd.DataFrame(X_traintest)

X_train, X_test, y_train, y_test = pre_train(traintest_df)

le = LabelEncoder()
le.fit(y_train)
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
name_le_mapping = {v:k for k, v in le_name_mapping.items()}

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

X_train = X_train.transpose((0,2,1))
X_test = X_test.transpose((0,2,1))

train_loader_obj = Loader(X_train, y_train, le)
test_loader_obj = Loader(X_test, y_test, le)

training_loader = DataLoader(
    train_loader_obj,
    num_workers=6,
    batch_size=32,
    shuffle=True,
    drop_last=True
)

test_loader = DataLoader(
    test_loader_obj,
    num_workers=6,
    batch_size=32,
    shuffle=True,
    drop_last=True
)
