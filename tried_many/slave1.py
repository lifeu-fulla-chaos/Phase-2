import os
import numpy as np
import torch
import torch.nn as nn
import warnings
import time
import socket
import pickle
import queue

from tried_many.lorentz import LorenzParameters, LorenzSystem

# PyTorch model for parameter estimation
class LorenzParameterModel(nn.Module):
    def __init__(self, hidden_size=128, layers=2):
        super(LorenzParameterModel, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_size, 
                          num_layers=layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, 6)
        self.hidden_size = hidden_size
        self.layers = layers

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x