import numpy as np
from scipy.integrate import RK45
import time
import socket
import pickle
import threading

from tried_many.slave2_common import lorenz_system, LorenzParameters

class MasterNode:
    def __init__(self, port=5000, dt=0.01):
        """Initialize Master node in the Lorenz communication network"""
        self.dt = dt
        self.params = LorenzParameters()
        self.state = np.array([1.0, 1.0, 1.0])  # Different initial state
        self.clients = []
        self.port = port
        self.running = True
        
        # Threads for server and simulation
        self.server_thread = None
        self.simulation_thread = None
        
    def start(self):
        """Start all threads"""
        self.server_thread = threading.Thread(target=self._run_server)
        self.simulation_thread = threading.Thread(target=self._run_simulation)
        
        self.server_thread.start()
        self.simulation_thread.start()
        
    def stop(self):
        """Stop all threads"""
        self.running = False
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join()
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join()
        
    def _run_server(self):
        """Run the server to accept connections and send state updates"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind(("0.0.0.0", self.port))
            server_socket.listen(5)
            print(f"Master server listening on port {self.port}")
            
            # Set a timeout to check running flag periodically
            server_socket.settimeout(1.0)
            
            while self.running:
                try:
                    client_socket, addr = server_socket.accept()
                    print(f"New connection from {addr}")
                    
                    # Start a new thread to handle each client
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, addr)
                    )
                    client_thread.start()
                    self.clients.append((client_socket, addr, client_thread))
                    
                except socket.timeout:
                    continue
                    
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            server_socket.close()
            
    def _handle_client(self, client_socket, addr):
        """Handle communication with a connected client"""
        try:
            while self.running:
                # Send current state to the client
                data = pickle.dumps(self.state)
                client_socket.sendall(data)
                time.sleep(self.dt)  # Send updates at simulation rate
                
        except Exception as e:
            print(f"Error handling client {addr}: {e}")
        finally:
            client_socket.close()
            # Remove client from list
            self.clients = [(s, a, t) for s, a, t in self.clients if a != addr]
            
    def _run_simulation(self):
        """Run the Lorenz system simulation"""
        t = 0.0
        
        while self.running:
            # Update system state
            solver = RK45(
                lambda t, y: lorenz_system(t, y, self.params),
                t, self.state, t + self.dt
            )
            solver.step()
            
            # Update state and time
            self.state = solver.y
            t = solver.t
            
            time.sleep(self.dt)
            
    def send_encrypted_message(self, message):
        """Encrypt and send a message to all connected clients"""
        # Convert message to bytes if it's a string
        if isinstance(message, str):
            message_bytes = message.encode('utf-8')
        else:
            message_bytes = message
            
        # Create key from current state
        key_bytes = np.abs(np.sin(self.state * 1000)).astype(np.uint8)
        
        # Expand key if message is longer
        key_bytes = np.tile(key_bytes, (len(message_bytes) // len(key_bytes)) + 1)[:len(message_bytes)]
        
        # XOR encryption
        encrypted = bytearray([message_bytes[i] ^ key_bytes[i % len(key_bytes)] for i in range(len(message_bytes))])
        
        # Send encrypted message with current state to all clients
        payload = pickle.dumps((self.state, encrypted))
        
        for client_socket, _, _ in self.clients:
            try:
                client_socket.sendall(payload)
            except Exception as e:
                print(f"Error sending encrypted message: {e}")
                
    def get_status(self):
        """Return current status information"""
        return {
            "state": self.state,
            "clients": len(self.clients),
            "params": self.params.__dict__
        }