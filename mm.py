import numpy as np
import socket
from lorenz import LorenzParameters, LorenzSystem
import pickle
import time
import psutil
import os
import hashlib

# Process monitoring
process = psutil.Process(os.getpid())

# Metrics storage
metrics = {
    "sync_times": [],
    "message_send_times": [],
    "encryption_times": [],
    "cpu_usage": [],
    "memory_usage": [],
    "throughput": [],
    "entropy_values": []
}

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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

def calculate_entropy(data):
    """Calculate Shannon entropy of the data to measure encryption strength"""
    if not data:
        return 0
    
    # Convert to bytes if it's a string
    if isinstance(data, str):
        data_bytes = data.encode('utf-8')
    elif isinstance(data, list):
        data_bytes = bytes(data)
    else:
        data_bytes = data
        
    # Count byte occurrences
    byte_counts = {}
    for byte in data_bytes:
        if byte in byte_counts:
            byte_counts[byte] += 1
        else:
            byte_counts[byte] = 1
    
    # Calculate entropy
    entropy = 0
    for count in byte_counts.values():
        probability = count / len(data_bytes)
        entropy -= probability * np.log2(probability)
    
    return entropy

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

def run_initial_trajectory(lorenz_system, steps=60):
    """Run the system for a specified number of steps"""
    print(f"Master: Running initial trajectory for {steps} steps...")
    start_time = time.time()
    state_history = lorenz_system.run_steps(0, steps)
    sync_time = time.time() - start_time
    record_metrics("sync_times", sync_time)
    print(f"Master: Initial trajectory complete with {len(state_history)} states in {sync_time:.4f} seconds")
    record_resource_usage()
    return state_history

def send_and_receive_states(state_history, conn, init=False):
    """Send the last 50 states to the slave and receive processed data"""
    last_states = state_history[-50:]
    if init:
        last_states = np.append(
            last_states, np.expand_dims(state_history[0], axis=0), axis=0
        )

    try:
        start_time = time.time()
        data = pickle.dumps(last_states)
        conn.sendall(len(data).to_bytes(4, byteorder="big"))
        conn.sendall(data)
        
        # Calculate throughput
        data_size_kb = len(data) / 1024
        
        data_size = int.from_bytes(conn.recv(4), byteorder="big")
        data = b""
        while len(data) < data_size:
            packet = conn.recv(data_size - len(data))
            if not packet:
                raise ConnectionError("Socket connection broken")
            data += packet
            
        end_time = time.time()
        elapsed = end_time - start_time
        
        throughput = data_size_kb / elapsed if elapsed > 0 else 0
        record_metrics("throughput", throughput)
        record_metrics("message_send_times", elapsed)
        
        print(f"Master: Sent states to Slave and received response in {elapsed:.4f} seconds")
        print(f"Master: Throughput: {throughput:.2f} KB/s")
        
        record_resource_usage()
        processed_data = pickle.loads(data)
        return processed_data

    except Exception as e:
        print(f"Master: Error during communication with Slave: {e}")
        return None

def encrypt(text, state):
    """Encrypt the text using the state"""
    start_time = time.time()
    cipher = int(sum(state))
    encrypted_text = []
    for i, char in enumerate(text):
        encrypted_char = ord(char) ^ cipher
        encrypted_text.append(encrypted_char)
    
    encryption_time = time.time() - start_time
    record_metrics("encryption_times", encryption_time)
    
    # Calculate encryption strength metrics
    entropy = calculate_entropy(encrypted_text)
    record_metrics("entropy_values", entropy)
    
    print(f"Encrypted text in {encryption_time:.6f} seconds, Entropy: {entropy:.4f}")
    return encrypted_text

def send_ack(conn, text=None):
    try:
        start_time = time.time()
        if text is not None:
            payload = (b"ACK", text)
            data = pickle.dumps(payload)
            conn.sendall(len(data).to_bytes(4, byteorder="big"))
            conn.sendall(data)
            
            # Calculate throughput
            data_size_kb = len(data) / 1024
            elapsed = time.time() - start_time
            throughput = data_size_kb / elapsed if elapsed > 0 else 0
            
            record_metrics("message_send_times", elapsed)
            record_metrics("throughput", throughput)
            
            print(f"Master: Sent states and encrypted text to Slave in {elapsed:.4f} seconds")
            print(f"Master: Throughput: {throughput:.2f} KB/s")
        else:
            ack = b"ACK"
            ack = pickle.dumps(ack)
            conn.sendall(len(ack).to_bytes(4, byteorder="big"))
            conn.sendall(ack)
            elapsed = time.time() - start_time
            record_metrics("message_send_times", elapsed)
            print(f"Master: Sent ACK to Slave in {elapsed:.4f} seconds")
            
        record_resource_usage()
    except Exception as e:
        print(f"Master: Error in send_ack: {e}")

def receive_ack(conn):
    """Receive an acknowledgment from the master"""
    start_time = time.time()
    data_size = int.from_bytes(conn.recv(4), byteorder="big")
    data = b""
    while len(data) < data_size:
        packet = conn.recv(data_size - len(data))
        if not packet:
            raise ConnectionError("Socket connection broken")
        data += packet

    elapsed = time.time() - start_time
    record_metrics("message_send_times", elapsed)
    
    ack = pickle.loads(data)
    print(f"Master: Received ACK from Slave in {elapsed:.4f} seconds")
    record_resource_usage()
    return ack

def save_metrics():
    """Save collected metrics to a file"""
    with open("master_metrics.txt", "w") as f:
        f.write("=== MASTER PERFORMANCE METRICS ===\n\n")
        
        f.write("=== TIMING METRICS ===\n")
        f.write(f"Average synchronization time: {np.mean(metrics['sync_times']):.4f} seconds\n")
        f.write(f"Average message send/receive time: {np.mean(metrics['message_send_times']):.4f} seconds\n")
        f.write(f"Average encryption time: {np.mean(metrics['encryption_times']):.4f} seconds\n")
        
        f.write("\n=== RESOURCE USAGE ===\n")
        f.write(f"Average CPU usage: {np.mean(metrics['cpu_usage']):.2f}%\n")
        f.write(f"Average memory usage: {np.mean(metrics['memory_usage']):.2f} MB\n")
        
        f.write("\n=== NETWORK METRICS ===\n")
        f.write(f"Average throughput: {np.mean(metrics['throughput']):.2f} KB/s\n")
        
        f.write("\n=== ENCRYPTION METRICS ===\n")
        f.write(f"Average entropy: {np.mean(metrics['entropy_values']):.4f} bits\n")
        f.write(f"Max entropy: {np.max(metrics['entropy_values']):.4f} bits\n")
        f.write(f"Min entropy: {np.min(metrics['entropy_values']):.4f} bits\n")
        
        # Detailed logs
        f.write("\n=== DETAILED LOGS ===\n")
        for metric_name, values in metrics.items():
            f.write(f"\n{metric_name}:\n")
            for i, value in enumerate(values):
                f.write(f"  {i}: {value}\n")

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
            start_time = time.time()
            state_history = run_initial_trajectory(lorenz_system, steps=10000)
            print(f"Initial state: {lorenz_system.initial_state}")
            processed_data = send_and_receive_states(state_history, conn, True)

            if processed_data == b"ACK":
                try:
                    with open("master.txt", "w") as f:
                        message_count = 0
                        while message_count < 10:  # Limit to 10 messages for testing
                            text = input("Enter text (or 'exit' to quit): ").strip()
                            if text.lower() == 'exit':
                                break
                                
                            start_time = time.time()
                            state_history = lorenz_system.run_steps(0, 1000)
                            sync_time = time.time() - start_time
                            record_metrics("sync_times", sync_time)
                            
                            f.write(f"{state_history[-1]}\n")
                            
                            if text != "":
                                encrypted_text = encrypt(text, state_history[-1])
                                send_ack(conn, encrypted_text)
                                message_count += 1
                            else:
                                send_ack(conn)
                                
                            ack = receive_ack(conn)
                            if ack != b"ACK":
                                break
                finally:
                    # Save metrics at the end
                    save_metrics()
                    print("Performance metrics saved to master_metrics.txt")

start_master_server(port)