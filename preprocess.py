from types import prepare_class
from librosa.feature.spectral import mfcc
import torch
from torch._C import dtype
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import librosa
import librosa.display
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def pre_process(wav_df):

    mfcc_df = pd.DataFrame(columns=['feature'])
    counter = 0

    for index, value in enumerate(wav_df.path):

        X, sample_rate = librosa.load(value
                                    , res_type='kaiser_fast'
                                    ,duration=2.5
                                    ,sr=44100
                                    ,offset=0.5
                                    )
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(
            librosa.feature.mfcc(y=X,
                                sr=sample_rate,
                                n_mfcc=13),
                                axis=0
                            )

        mfcc_df.loc[counter] = [mfccs]
        counter += 1
    print(mfcc_df.head())
    print(f"unique values pre-concat: {wav_df.labels.value_counts()}")

    mfcc_df = pd.concat([wav_df.reset_index(drop=True),pd.DataFrame(mfcc_df['feature'].values.tolist()).reset_index(drop=True)],axis=1)
    print(f"Head Before FillNA: {mfcc_df.head()}")

    print(mfcc_df.loc[12])
    print(mfcc_df.loc[14])
    print(mfcc_df.loc[16])
    print(mfcc_df.loc[19])
    print(mfcc_df.loc[36])
    print(mfcc_df.loc[1128])

    mfcc_df = mfcc_df.fillna(0)

    # print(mfcc_df.head())


    return mfcc_df




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

    #return DataLoader(loader, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

def pre_train(traintest_df, normalize = True):
        # pass the load path to PreProcessor
    pre_processed_df = pre_process(traintest_df)

    print(f"unique values in pre-processdf: {pre_processed_df.labels.value_counts()}")

    print(pre_processed_df.loc[pre_processed_df.labels == 0])

    X_train, X_test, y_train, y_test = train_test_split(pre_processed_df.drop(['path','labels','source'],axis=1)
                                                , pre_processed_df.labels
                                                , test_size=0.25
                                                , shuffle=True
                                                , random_state=42
                                                )

    print(y_train.value_counts())
    print(y_test.value_counts())

    if(normalize):
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)

        X_train = (X_train - mean)/std
        X_test = (X_test - mean)/std

    print(type(X_train), type(X_test), type(y_test), type(y_train))
    return X_train, X_test, y_train, y_test

class Loader():
    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]

    def __init__(self, X, y, label_encoder):
        print(X)
        print(len(X))

        # labels = self.to_categorical(label_encoder.transform(y), len(label_encoder.classes_))
        label = label_encoder.transform(y)

        self.data = torch.from_numpy(X)
        self.labels = torch.tensor(label, dtype=torch.float)
        # self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
