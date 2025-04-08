import numpy as np
import socket
from lorenz import LorenzParameters, LorenzSystem
import pickle
import random

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
max_retries = 5
retry_delay = 5
port = 3000

RHO_RANGE = (20, 100)
SIGMA_RANGE = (5, 45)
BETA_RANGE = (0.1, 5)
IC_RANGE = (-30, 30)

parameters = LorenzParameters(
    random.uniform(*SIGMA_RANGE),
    random.uniform(*RHO_RANGE),
    random.uniform(*BETA_RANGE),
)

initial_conditions = np.random.uniform(*IC_RANGE, size=3)
print(
    f"Initial conditions: {initial_conditions, parameters.sigma, parameters.rho, parameters.beta}"
)
lorenz_system = LorenzSystem(
    initial_state=initial_conditions, params=parameters, dt=0.01
)


def run_initial_trajectory(lorenz_system, steps=60):
    """Run the system for a specified number of steps"""
    print(f"Master: Running initial trajectory for {steps} steps...")
    state_history = lorenz_system.run_steps(steps)
    print(f"Master: Initial trajectory complete with {len(state_history)} states")
    return state_history


def send_and_receive_states(state_history, conn):
    """Send the last 50 states to the slave and receive processed data"""
    last_states = state_history[-50:]

    try:
        data = pickle.dumps(last_states)
        conn.sendall(len(data).to_bytes(4, byteorder="big"))
        conn.sendall(data)
        print("Master: Sent states to Slave")

        data_size = int.from_bytes(conn.recv(4), byteorder="big")

        data = b""
        while len(data) < data_size:
            packet = conn.recv(data_size - len(data))
            if not packet:
                raise ConnectionError("Socket connection broken")
            data += packet

        processed_data = pickle.loads(data)
        print("Master: Received processed data from Slave")
        return processed_data

    except Exception as e:
        print(f"Master: Error during communication with Slave: {e}")
        return None


def start_master_server(port):
    """Start the master server to listen for connections"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("localhost", port))
        server_sock.listen(1)
        print(f"Master: Listening for connections on port {port}...")

        conn, addr = server_sock.accept()
        with conn:
            print(f"Master: Connected to {addr}")
            state_history = run_initial_trajectory(lorenz_system, steps=60)
            processed_data = send_and_receive_states(state_history, conn)
            print(f"Master: Final processed data: {processed_data}")


start_master_server(port)
