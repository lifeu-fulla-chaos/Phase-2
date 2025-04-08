import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, hidden_size=128, layers=2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, 6)
        self.hidden_size = hidden_size
        self.layers = layers

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x
