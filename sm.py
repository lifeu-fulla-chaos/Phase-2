import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
import psutil

# Process monitoring
process = psutil.Process(os.getpid())

# Metrics storage
metrics = {
    "sync_times": [],
    "message_receive_times": [],
    "decryption_times": [],
    "prediction_times": [],
    "cpu_usage": [],
    "memory_usage": [],
    "throughput": [],
    "synchronization_error": []
}

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
port = 3000
sock.connect(("localhost", port))

model = LSTM(hidden_size=256, layers=8)
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
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

def record_metrics(metric_type, value):
    """Record a metric to the metrics dictionary"""
    if metric_type in metrics:
        metrics[metric_type].append(value)
    else:
        metrics[metric_type] = [value]

def record_resource_usage():
    """Record CPU and memory usage"""
    cpu_percent = process.cpu_percent()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
    
    record_metrics("cpu_usage", cpu_percent)
    record_metrics("memory_usage", memory_mb)
    
    return cpu_percent, memory_mb

def calculate_synchronization_error(master_trajectory, slave_trajectory):
    """Calculate the synchronization error between master and slave"""
    errors = np.linalg.norm(master_trajectory - slave_trajectory, axis=1)
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    record_metrics("synchronization_error", mean_error)
    return mean_error, max_error

def receive_and_process_states(sock, model):
    """Receive states from the master, process them, and send back the result"""
    start_time = time.time()
    data_size = int.from_bytes(sock.recv(4), byteorder="big")

    data = b""
    while len(data) < data_size:
        packet = sock.recv(data_size - len(data))
        if not packet:
            raise ConnectionError("Socket connection broken")
        data += packet

    recv_time = time.time() - start_time
    data_size_kb = len(data) / 1024
    throughput = data_size_kb / recv_time if recv_time > 0 else 0
    
    record_metrics("message_receive_times", recv_time)
    record_metrics("throughput", throughput)
    
    print(f"Slave: Received states from Master in {recv_time:.4f} seconds")
    print(f"Slave: Throughput: {throughput:.2f} KB/s")
    
    states = pickle.loads(data)
    
    # Record resource usage after receiving data
    record_resource_usage()
    
    # Predict using the model
    prediction_start = time.time()
    preds = predict(model, states)
    prediction_time = time.time() - prediction_start
    record_metrics("prediction_times", prediction_time)
    
    print(f"Slave: Predicted initial conditions in {prediction_time:.4f} seconds")
    print(f"Predicted initial conditions: {states[-1]}")
    
    return states[-1]

def predict(model, states):
    """Predict the next state using the model"""
    states = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction = model(states)
    return prediction.squeeze(0).cpu().numpy()

def send_ack(sock):
    """Send an acknowledgment to the master"""
    start_time = time.time()
    ack = b"ACK"
    ack = pickle.dumps(ack)
    sock.sendall(len(ack).to_bytes(4, byteorder="big"))
    sock.sendall(ack)
    
    elapsed = time.time() - start_time
    record_metrics("message_receive_times", elapsed)  # Reusing this metric for send time too
    
    print(f"Slave: Sent ACK to Master in {elapsed:.4f} seconds")
    record_resource_usage()

def synchronize_states(u0, t_span, t_eval, controller):
    start_time = time.time()
    solution = solve_ivp(
        fun=lambda t, u: dynamics(t, u, controller),
        t_span=t_span,
        y0=u0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-9,
        atol=1e-9,
    )
    
    sync_time = time.time() - start_time
    record_metrics("sync_times", sync_time)
    
    print(f"Slave: Synchronized states in {sync_time:.4f} seconds")
    
    return solution.t, solution.y[:3].T, solution.y[3:].T

def decrypt_text(encrypted_text, state):
    """Decrypt the text using the state"""
    start_time = time.time()
    cipher = int(sum(state))
    decrypted_text = []
    for i, char in enumerate(encrypted_text):
        decrypted_char = chr(char ^ cipher)
        decrypted_text.append(decrypted_char)
    
    decryption_time = time.time() - start_time
    record_metrics("decryption_times", decryption_time)
    
    decrypted_result = "".join(decrypted_text)
    print(f"Slave: Decrypted text in {decryption_time:.6f} seconds")
    
    return decrypted_result

def save_metrics():
    """Save collected metrics to a file"""
    with open("slave_metrics.txt", "w") as f:
        f.write("=== SLAVE PERFORMANCE METRICS ===\n\n")
        
        f.write("=== TIMING METRICS ===\n")
        f.write(f"Average synchronization time: {np.mean(metrics['sync_times']):.4f} seconds\n")
        f.write(f"Average message receive time: {np.mean(metrics['message_receive_times']):.4f} seconds\n")
        f.write(f"Average decryption time: {np.mean(metrics['decryption_times']):.6f} seconds\n")
        f.write(f"Average prediction time: {np.mean(metrics['prediction_times']):.4f} seconds\n")
        
        f.write("\n=== RESOURCE USAGE ===\n")
        f.write(f"Average CPU usage: {np.mean(metrics['cpu_usage']):.2f}%\n")
        f.write(f"Average memory usage: {np.mean(metrics['memory_usage']):.2f} MB\n")
        
        f.write("\n=== NETWORK METRICS ===\n")
        f.write(f"Average throughput: {np.mean(metrics['throughput']):.2f} KB/s\n")
        
        f.write("\n=== SYNCHRONIZATION QUALITY ===\n")
        f.write(f"Average synchronization error: {np.mean(metrics['synchronization_error']):.6f}\n")
        f.write(f"Max synchronization error: {np.max(metrics['synchronization_error']):.6f}\n")
        f.write(f"Min synchronization error: {np.min(metrics['synchronization_error']):.6f}\n")
        
        # Detailed logs
        f.write("\n=== DETAILED LOGS ===\n")
        for metric_name, values in metrics.items():
            f.write(f"\n{metric_name}:\n")
            for i, value in enumerate(values):
                f.write(f"  {i}: {value}\n")

def plot_trajectories(time, master_trajectory, slave_trajectory):
    """Overlay the master and slave trajectories, separated by axis."""
    plt.figure(figsize=(15, 12))

    # Plot each axis
    plt.subplot(4, 1, 1)
    plt.plot(time, master_trajectory[:, 0], label="x (Master)", color="blue")
    plt.plot(time, slave_trajectory[:, 0], label="x (Slave)", color="orange")
    plt.title("X-Axis Trajectory")
    plt.xlabel("Time")
    plt.ylabel("x")
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(time, master_trajectory[:, 1], label="y (Master)", color="blue")
    plt.plot(time, slave_trajectory[:, 1], label="y (Slave)", color="orange")
    plt.title("Y-Axis Trajectory")
    plt.xlabel("Time")
    plt.ylabel("y")
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(time, master_trajectory[:, 2], label="z (Master)", color="blue")
    plt.plot(time, slave_trajectory[:, 2], label="z (Slave)", color="orange")
    plt.title("Z-Axis Trajectory")
    plt.xlabel("Time")
    plt.ylabel("z")
    plt.legend()
    
    # Add a plot for synchronization error
    plt.subplot(4, 1, 4)
    errors = np.linalg.norm(master_trajectory - slave_trajectory, axis=1)
    plt.plot(time, errors, label="Synchronization Error", color="red")
    plt.title("Synchronization Error")
    plt.xlabel("Time")
    plt.ylabel("Error (Euclidean distance)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("trajectories_with_error.png")
    print("Trajectory plot saved with synchronization error")

def main():
    try:
        # Record initial resource usage
        record_resource_usage()
        
        # Receive initial conditions from master
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
        
        # Initial synchronization
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
        
        # Calculate synchronization error
        mean_error, max_error = calculate_synchronization_error(master_trajectory, slave_trajectory)
        print(f"Synchronization complete. Mean error: {mean_error:.6f}, Max error: {max_error:.6f}")
        print(f"Final states - Master: {master_trajectory[-1]}, Slave: {slave_trajectory[-1]}")
        
        # Plot trajectories with error
        plot_trajectories(timesteps, master_trajectory, slave_trajectory)
        
        send_ack(sock)
        
        # Setup for message exchange
        t_span = (0, master_copy.dt * 1000)
        t_eval = np.linspace(*t_span, 1000)
        
        message_count = 0
        with open("slave-copy.txt", "w") as f:
            while message_count < 10:  # Limit to 10 messages for testing
                try:
                    start_time = time.time()
                    data_size = int.from_bytes(sock.recv(4), byteorder="big")
                    data = b""
                    while len(data) < data_size:
                        packet = sock.recv(data_size - len(data))
                        if not packet:
                            raise ConnectionError("Socket connection broken")
                        data += packet
                    
                    recv_time = time.time() - start_time
                    data_size_kb = len(data) / 1024
                    throughput = data_size_kb / recv_time if recv_time > 0 else 0
                    
                    record_metrics("message_receive_times", recv_time)
                    record_metrics("throughput", throughput)
                    
                    record_resource_usage()

                    try:
                        payload = pickle.loads(data)
                        
                        # Synchronize states for each message
                        u0 = np.concatenate(
                            (
                                master_trajectory[-1],
                                slave_trajectory[-1],
                            )
                        )
                        _, master_trajectory, slave_trajectory = synchronize_states(
                            u0, t_span, t_eval, controller
                        )
                        
                        # Calculate synchronization error
                        mean_error, _ = calculate_synchronization_error(master_trajectory, slave_trajectory)
                        
                        if isinstance(payload, tuple) and len(payload) == 2:
                            _, encrypted_text = payload
                            print(f"Slave: Received states and encrypted text from Master")
                            decrypted_text = decrypt_text(encrypted_text, master_trajectory[-1])
                            print(f"Slave: Decrypted text: {decrypted_text}")
                            message_count += 1

                        f.write(f"{master_trajectory[-1]} | Error: {mean_error:.6f}\n")
                        send_ack(sock)
                    except pickle.UnpicklingError:
                        print("Slave: Received non-pickle data, ignoring.")
                except Exception as e:
                    print(f"Error in main loop: {e}")
                    break
    finally:
        # Save metrics before exiting
        save_metrics()
        print("Performance metrics saved to slave_metrics.txt")

main()