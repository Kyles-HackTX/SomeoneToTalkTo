import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# input shape transpose: transpose((0,2,1))

class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 256, 8, padding='same'),
            nn.ReLU(),
            nn.Conv1d(256, 256, 8, padding='same'),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool1d(8),
            nn.Conv1d(256, 128, 8, padding='same'),
            nn.ReLU(),
            nn.Conv1d(128, 128, 8, padding='same'),
            nn.ReLU(),
            nn.Conv1d(128, 128, 8, padding='same'),
            nn.ReLU(),
            nn.Conv1d(128, 128, 8, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool1d(8),
            nn.Conv1d(128, 64, 8, padding='same'),
            nn.ReLU(),
            nn.Conv1d(64, 64, 8, padding='same'),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(192, 16),
            nn.Softmax()
        )

    def forward(self, x):
        logits = self.net(x)

        return logits


def get_optimizer_and_loss():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
