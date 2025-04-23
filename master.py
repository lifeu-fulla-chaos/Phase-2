import numpy as np
import socket
from lorenz import LorenzParameters, LorenzSystem
import pickle
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
max_retries = 5
retry_delay = 5
port = 3000

IC_RANGE = (-30, 30)

parameters = LorenzParameters(
    sigma=10.0,
    rho=28.0,
    beta=8.0 / 3.0,
)

initial_conditions = np.round(np.random.uniform(*IC_RANGE, size=3), 4)

print(f"Initial conditions: {initial_conditions}")
lorenz_system = LorenzSystem(
    initial_state=initial_conditions, params=parameters, dt=0.001
)


def run_initial_trajectory(lorenz_system, steps=60):
    """Run the system for a specified number of steps"""
    print(f"Master: Running initial trajectory for {steps} steps...")
    state_history = lorenz_system.run_steps(0, steps)
    print(f"Master: Initial trajectory complete with {len(state_history)} states")
    return state_history


def send_and_receive_states(state_history, conn, init=False):
    """Send the last 50 states to the slave and receive processed data"""
    last_states = state_history[-50:]
    if init:
        last_states = np.append(
            last_states, np.expand_dims(state_history[0], axis=0), axis=0
        )

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
        print("Master: Received ACK from Slave")
        return processed_data

    except Exception as e:
        print(f"Master: Error during communication with Slave: {e}")
        return None


def send_ack(conn, text=None):
    try:
        if text is not None:
            payload = (b"ACK", text)
            data = pickle.dumps(payload)
            conn.sendall(len(data).to_bytes(4, byteorder="big"))
            conn.sendall(data)
            print("Master: Sent states and encrypted text to Slave")
        else:
            ack = b"ACK"
            ack = pickle.dumps(ack)
            conn.sendall(len(ack).to_bytes(4, byteorder="big"))
            conn.sendall(ack)
            print("Master: Sent ACK to Slave")
    except Exception as e:
        print(f"Master: Error in send_ack: {e}")


def receive_ack(conn):
    """Receive an acknowledgment from the master"""
    data_size = int.from_bytes(conn.recv(4), byteorder="big")
    data = b""
    while len(data) < data_size:
        packet = conn.recv(data_size - len(data))
        if not packet:
            raise ConnectionError("Socket connection broken")
        data += packet

    ack = pickle.loads(data)
    print("Slave: Received ACK from Master")
    return ack


def encrypt(text, state):
    """Encrypt the text using the state"""
    cipher = int(sum(state))
    encrypted_text = []
    for i, char in enumerate(text):
        encrypted_char = ord(char) ^ cipher
        encrypted_text.append(encrypted_char)
    print("Encrypted text:", encrypted_text)
    return encrypted_text


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
            state_history = run_initial_trajectory(lorenz_system, steps=10000)
            print(lorenz_system.initial_state)
            processed_data = send_and_receive_states(state_history, conn, True)

            if processed_data == b"ACK":
                with open("master.txt", "w") as f:
                    while True:
                        text = input("Enter text: ").strip()
                        state_history = lorenz_system.run_steps(0, 50)
                        if text != "":
                            encrypted_text = encrypt(text, state_history[-1])
                            send_ack(conn, encrypted_text)
                        else:
                            send_ack(conn)
                        f.write(f"{state_history[-1]}\n")
                        ack = receive_ack(conn)
                        if ack != b"ACK":
                            break


start_master_server(port)
