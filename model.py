import torch
import torch.nn as nn
import torch.nn.functional as F


# input shape transpose: transpose((0,2,1))

class EmotionClassifier(nn.Module):
    def __init(self):
        super(EmotionClassifier, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1D(1, 256, 8, padding='same'),
            nn.ReLU(),
            nn.Conv1D(1, 256, 8, padding='same'),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool1d(8),
            nn.Conv1D(1, 128, 8, padding='same'),
            nn.ReLU(),
            nn.Conv1D(1, 128, 8, padding='same'),
            nn.ReLU(),
            nn.Conv1D(1, 128, 8, padding='same'),
            nn.ReLU(),
            nn.Conv1D(1, 128, 8, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool1d(8),
            nn.Conv1D(1, 64, 8, padding='same'),
            nn.ReLU(),
            nn.Conv1D(1, 64, 8, padding='same'),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(14),
            nn.Softmax()
        )

    def forward(self, x):
        logits = self.net(x)

        return logits
