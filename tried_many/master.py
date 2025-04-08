import numpy as np
import time
import socket
import pickle
import threading
import queue

# Import from lorentz.py
from tried_many.lorentz import LorenzParameters, LorenzSystem, encrypt_message

class Master:
    def __init__(self, params=None, initial_state=None, dt=0.01, port_slave1=5000, port_slave2=5001):
        # Initialize Lorenz system
        self.lorenz_system = LorenzSystem(initial_state, params, dt)
        # Communication ports
        self.port_slave1 = port_slave1
        self.port_slave2 = port_slave2
        # Message queue for incoming messages
        self.message_queue = queue.Queue()
        
    def run_initial_trajectory(self, steps=60):
        """Run the system for a specified number of steps"""
        print(f"Master: Running initial trajectory for {steps} steps...")
        state_history = self.lorenz_system.run_steps(steps)
        print(f"Master: Initial trajectory complete with {len(state_history)} states")
        return state_history
    
    def send_states_to_slave1(self, state_history):
        """Send the last 50 states to slave 1"""
        # Extract the last 50 states (or all if less than 50)
        last_states = state_history[-50:] if len(state_history) >= 50 else state_history
        
        # Create a socket connection to slave 1
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        max_retries = 5
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                sock.connect(('localhost', self.port_slave1))
                # Serialize and send the states
                data = pickle.dumps(last_states)
                sock.sendall(len(data).to_bytes(4, byteorder='big'))  # Send data size first
                sock.sendall(data)  # Send the actual data
                print(f"Master: Sent {len(last_states)} states to Slave 1")
                return True
            except ConnectionRefusedError:
                if attempt < max_retries - 1:
                    print(f"Master: Connection to Slave 1 refused. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Master: Failed to connect to Slave 1 after multiple attempts")
                    return False
            except Exception as e:
                print(f"Master: Error sending states to Slave 1: {e}")
                return False
            finally:
                sock.close()