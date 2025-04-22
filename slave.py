from model import LSTM
from lorenz import LorenzParameters, LorenzSystem
from rossler import RosslerParameters, RosslerSystem
from controller import BacksteppingController, dynamics
import numpy as np
import socket
import torch
import pickle

socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
port = 3000
socket.connect(("localhost", port))

model = LSTM(hidden_size=256, layers=8)
model.load_state_dict(torch.load("model.pth"))
model.eval()

master_parameters = LorenzParameters(
    sigma=10.0,
    rho=28.0,
    beta=8.0 / 3.0,
)

slave_parameters = RosslerParameters(
    a=0.2,
    b=0.2,
    c=5.7,
)

IC_RANGE = (-30, 30)

slave_system = RosslerSystem(
    initial_state=np.random.uniform(*IC_RANGE, size=3),
    params=slave_parameters,
    dt=0.001,
)


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
    print("predicted initial conditions", states[-1])
    print("Slave: Sent processed data back to Master")
    return states[-1]


def predict(model, states):
    """Predict the next state using the model"""
    states = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction = model(states)
    return prediction.squeeze(0).cpu().numpy()


def main():
    initial_conditions = receive_and_process_states(socket, model)
    master_copy = LorenzSystem(
        initial_state=initial_conditions, params=master_parameters, dt=0.001
    )
    controller = BacksteppingController(
        p=(
            master_parameters.sigma,
            master_parameters.rho,
            master_parameters.beta,
            slave_parameters.a,
            slave_parameters.b,
            slave_parameters.c,
        ),
        k=(1.0, 1.0, 1.0),
    )
    t_span = (0, master_copy.dt * 50)
    t_eval = np.linspace(*t_span, 50)
    

