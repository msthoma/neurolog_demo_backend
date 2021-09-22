import torch
import torch.nn as nn


class Sum2NN(nn.Module):
    def __init__(self, N=10):
        super(Sum2NN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, N),
            nn.Softmax(1)
        )

    def forward(self, vector: torch.Tensor):
        x = self.encoder(vector)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x
