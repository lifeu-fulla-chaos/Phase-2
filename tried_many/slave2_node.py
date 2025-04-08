import numpy as np
from scipy.integrate import RK45
import time
import socket
import pickle
import threading
import queue

from tried_many.slave2_common import controlled_lorenz_system, backstepping_controller, decrypt_message, LorenzParameters

class Slave2:
    def __init__(self, port_slave1=5001, port_master=5000, dt=0.01):
        """Initialize Slave2 node in the Lorenz communication network"""
        self.dt = dt
        self.params = LorenzParameters()
        self.state = np.array([0.1, 0.1, 0.1])  # Initial state
        self.master_state = None
        self.slave1_state = None
        self.control = np.zeros(3)
        self.sync_error = float('inf')
        
        # Communication setup
        self.port_slave1 = port_slave1
        self.port_master = port_master
        self.message_queue = queue.Queue()
        self.running = True
        
        # Threads for communication and simulation
        self.slave1_thread = None
        self.master_thread = None
        self.simulation_thread = None
        
    def start(self):
        """Start all threads"""
        self.slave1_thread = threading.Thread(target=self._connect_to_slave1)
        self.master_thread = threading.Thread(target=self._connect_to_master)
        self.simulation_thread = threading.Thread(target=self._run_simulation)
        
        self.slave1_thread.start()
        self.master_thread.start()
        self.simulation_thread.start()
        
    def stop(self):
        """Stop all threads"""
        self.running = False
        if self.slave1_thread and self.slave1_thread.is_alive():
            self.slave1_thread.join()
        if self.master_thread and self.master_thread.is_alive():
            self.master_thread.join()
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join()
        
    def _connect_to_slave1(self):
        """Connect to Slave1 to receive state information"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect(("localhost", self.port_slave1))
            print(f"Connected to Slave1 on port {self.port_slave1}")
            
            while self.running:
                data = sock.recv(1024)
                if not data:
                    break
                self.slave1_state = pickle.loads(data)
        except Exception as e:
            print(f"Error connecting to Slave1: {e}")
        finally:
            sock.close()
            
    def _connect_to_master(self):
        """Connect to Master to receive state and encrypted messages"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect(("localhost", self.port_master))
            print(f"Connected to Master on port {self.port_master}")
            
            while self.running:
                data = sock.recv(4096)  # Larger buffer for messages
                if not data:
                    break
                    
                # Unpack the received data
                try:
                    payload = pickle.loads(data)
                    # Check if we received a state update or an encrypted message
                    if isinstance(payload, tuple) and len(payload) == 2:
                        # We received a message
                        master_state, encrypted_message = payload
                        self.master_state = master_state
                        self.message_queue.put(encrypted_message)
                    else:
                        # Just a state update
                        self.master_state = payload
                except Exception as e:
                    print(f"Error unpacking data: {e}")
                    
        except Exception as e:
            print(f"Error connecting to Master: {e}")
        finally:
            sock.close()
    
    def _run_simulation(self):
        """Run the Lorenz system simulation with control input"""
        t = 0.0
        
        while self.running:
            # Update control if we have master state
            if self.master_state is not None:
                # Compute control using backstepping approach
                self.control = backstepping_controller(self.master_state, self.state, self.params)
                
                # Update system state with control input
                solver = RK45(
                    lambda t, y: controlled_lorenz_system(t, y, self.params, self.control),
                    t, self.state, t + self.dt
                )
                solver.step()
                
                # Update state and time
                self.state = solver.y
                t = solver.t
                
                # Calculate synchronization error
                self.sync_error = np.linalg.norm(self.master_state - self.state)
                
                # Process any pending messages in the queue
                self._process_messages()
                
            time.sleep(self.dt)
            
    def _process_messages(self):
        """Process any encrypted messages in the queue"""
        while not self.message_queue.empty():
            encrypted_message = self.message_queue.get()
            
            # Decrypt message using current state
            decrypted_message = decrypt_message(encrypted_message, self.state)
            
            try:
                message_text = decrypted_message.decode('utf-8')
                print(f"Decrypted message: {message_text}")
                
                # Here you can add additional processing for the decrypted message
                # For example, responding to commands or storing data
                
            except UnicodeDecodeError:
                print("Failed to decode message - synchronization might not be complete")
                
    def get_status(self):
        """Return current status information"""
        return {
            "state": self.state,
            "master_state": self.master_state,
            "sync_error": self.sync_error,
            "control": self.control
        }