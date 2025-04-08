from model import LSTM
from lorenz import LorenzParameters, LorenzSystem
import numpy as np
import socket
import torch
import pickle

socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
port = 3000
socket.connect(("localhost", port))

model = LSTM(hidden_size=128, layers=2)
model.load_state_dict(torch.load("92.pth"))
model.eval()


def receive_and_process_states(sock, model):
    """Receive states from the master, process them, and send back the result"""
    data_size = int.from_bytes(sock.recv(4), byteorder="big")

    data = b""
    while len(data) < data_size:
        packet = sock.recv(data_size - len(data))
        if not packet:
            raise ConnectionError("Socket connection broken")
        data += packet

    states = pickle.loads(data)
    print("Slave: Received states from Master")

    preds = predict(model, states)
    print(preds)
    print("Slave: Sent processed data back to Master")


def predict(model, states):
    """Predict the next state using the model"""
    states = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction = model(states)
    return prediction.squeeze(0).cpu().numpy()


receive_and_process_states(socket, model)

