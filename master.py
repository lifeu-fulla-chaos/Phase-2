import numpy as np
import time
import socket
import pickle

from lorentz import LorenzParameters, LorenzSystem

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
max_retries = 5
retry_delay = 5
port 
parameters = LorenzParameters(sigma=10.0, rho=28.0, beta=8/3)
lorenz_system = LorenzSystem(initial_state=np.array([1.0, 1.0, 1.0]), params=parameters, dt=0.01)

def run_initial_trajectory(steps=60):
        """Run the system for a specified number of steps"""
        print(f"Master: Running initial trajectory for {steps} steps...")
        state_history = lorenz_system.run_steps(steps)
        print(f"Master: Initial trajectory complete with {len(state_history)} states")
        return state_history
    
def send_states_to_slave1(state_history, sock, port):
        """Send the last 50 states to slave 1"""
        last_states = state_history[-50:] 
        
        for attempt in range(max_retries):
            try:
                sock.connect(('localhost', port))
                data = pickle.dumps(last_states)
                sock.sendall(len(data).to_bytes(4, byteorder='big'))  
                sock.sendall(data)
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