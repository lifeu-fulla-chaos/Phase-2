import matplotlib.pyplot as plt
from model import LSTM
from lorenz import LorenzParameters, LorenzSystem
from rossler import RosslerParameters, RosslerSystem
from controller import BacksteppingController, dynamics
from scipy.integrate import solve_ivp
import numpy as np
import socket
import torch
import pickle
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
port = 3000
sock.connect(("localhost", port))

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


def send_ack(sock):
    """Send an acknowledgment to the master"""
    ack = b"ACK"
    ack = pickle.dumps(ack)
    sock.sendall(len(ack).to_bytes(4, byteorder="big"))
    sock.sendall(ack)
    print("Slave: Sent ACK to Master")


def synchronize_states(u0, t_span, t_eval, controller):
    solution = solve_ivp(
        fun=lambda t, u: dynamics(t, u, controller),
        t_span=t_span,
        y0=u0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-8,
    )
    return solution.t, solution.y[:3].T, solution.y[3:].T


def main():
    initial_conditions = receive_and_process_states(sock, model)
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
        k=(5.0, 5.0, 5.0),
    )
    t_span = (0, master_copy.dt * 10000)
    t_eval = np.linspace(*t_span, 10000)
    u0 = np.concatenate(
        (
            master_copy.initial_state,
            slave_system.initial_state,
        )
    )
    timesteps, master_trajectory, slave_trajectory = synchronize_states(
        u0=u0,
        t_eval=t_eval,
        t_span=t_span,
        controller=controller,
    )
    print(master_trajectory[-1], slave_trajectory[-1])
    plot_trajectories(timesteps, master_trajectory, slave_trajectory)
    send_ack(sock)
    time.sleep(6)
    t_span = (0, master_copy.dt * 50)
    t_eval = np.linspace(*t_span, 50)
    with open("master-copy.txt", "w") as f:
        while True:
            data_size = int.from_bytes(sock.recv(4), byteorder="big")
            data = b""
            while len(data) < data_size:
                packet = sock.recv(data_size - len(data))
                if not packet:
                    raise ConnectionError("Socket connection broken")
                data += packet

            ack = pickle.loads(data)
            if ack == b"ACK":
                u0 = np.concatenate(
                    (
                        master_trajectory[-1],
                        slave_trajectory[-1],
                    )
                )
                _, master_trajectory, slave_trajectory = synchronize_states(
                    u0, t_span, t_eval, controller
                )
                f.write(f"{master_trajectory[-1]}\n")
                send_ack(sock)


def plot_trajectories(time, master_trajectory, slave_trajectory):
    """Overlay the master and slave trajectories, separated by axis."""
    plt.figure(figsize=(15, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time, master_trajectory[:, 0], label="x (Master)", color="blue")
    plt.plot(time, slave_trajectory[:, 0], label="x (Slave)", color="orange")
    plt.title("X-Axis Trajectory")
    plt.xlabel("Time")
    plt.ylabel("x")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time, master_trajectory[:, 1], label="y (Master)", color="blue")
    plt.plot(time, slave_trajectory[:, 1], label="y (Slave)", color="orange")
    plt.title("Y-Axis Trajectory")
    plt.xlabel("Time")
    plt.ylabel("y")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time, master_trajectory[:, 2], label="z (Master)", color="blue")
    plt.plot(time, slave_trajectory[:, 2], label="z (Slave)", color="orange")
    plt.title("Z-Axis Trajectory")
    plt.xlabel("Time")
    plt.ylabel("z")
    plt.legend()

    plt.tight_layout()
    plt.savefig("trajectories.png")


main()
