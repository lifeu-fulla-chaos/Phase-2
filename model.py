import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, hidden_size, layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=3, hidden_size=hidden_size, num_layers=layers, batch_first=True
        )
        self.fc_initial_conditions = nn.Linear(hidden_size, 3)
        self.fc_parameters = nn.Linear(hidden_size, 3)
        self.hidden_size = hidden_size
        self.layers = layers

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x1 = self.fc_initial_conditions(x)
        x2 = self.fc_parameters(x)
        return torch.cat((x1, x2), dim=1)
