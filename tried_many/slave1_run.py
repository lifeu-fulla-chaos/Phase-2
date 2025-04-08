import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import numpy as np
from scipy.integrate import RK45
import queue

from tried_many.lorentz import LorenzParameters, LorenzSystem
from tried_many.slave1_estimator import ParameterEstimator
from tried_many.slave1_controller import backstepping_controller, controlled_lorenz_system
from tried_many.slave1_communication import listen_for_master_states, send_parameters_to_slave2, listen_for_acknowledgment

class Slave1Runner:
    def __init__(self, port=5000, port_slave2=5001, dt=0.01, model_path='92.pth'):
        # Initialize with default parameters that will be updated later
        self.params = LorenzParameters()
        self.initial_state = np.array([0.0, 0.0, 0.0])  # Will be estimated
        self.port = port  # Port to listen for master data
        self.port_slave2 = port_slave2  # Port to send data to slave 2
        self.dt = dt
        
        # State and tracking
        self.current_state = self.initial_state.copy()
        self.current_time = 0.0
        self.state_history = []
        self.master_states = None
        
        # Parameter estimator
        self.estimator = ParameterEstimator(model_path)
        
        # Message queue for incoming messages
        self.message_queue = queue.Queue()
        
        # Control input (for synchronization)
        self.control_input = np.zeros(3)
        
        # Integrator
        self.reset_integrator()
    
    def reset_integrator(self):
        """Reset the RK45 integrator with current state"""
        self.integrator = RK45(
            lambda t, y: controlled_lorenz_system(t, y, self.params, self.control_input),
            self.current_time,
            self.current_state,
            t_bound=float('inf'),
            rtol=1e-6,
            atol=1e-6
        )
    
    def step(self, master_state=None):
        """Take a single time step using RK45, with optional synchronization to master"""
        # Update control input if master state is provided
        if master_state is not None:
            self.control_input = backstepping_controller(master_state, self.current_state, self.params)
        else:
            self.control_input = np.zeros(3)
        
        # Advance the integrator
        target_time = self.current_time + self.dt
        while self.integrator.t < target_time:
            self.integrator.step()
        
        self.current_state = self.integrator.y
        self.current_time = self.integrator.t
        self.state_history.append(self.current_state.copy())
        return self.current_state
    
    def run_synchronization(self, steps=100):
        """Run the synchronization phase with the master"""
        print(f"Slave 1: Running synchronization for {steps} steps...")
        
        # We don't have direct access to master states in real-time,
        # but in a real system we would receive synchronization signals
        # For simulation purposes, we'll just run our own dynamics
        for i in range(steps):
            self.step()
            if i % 10 == 0:
                print(f"Slave 1: Synchronization step {i}/{steps}")
            time.sleep(0.1)  # Small delay to simulate real-time operation
        
        print("Slave 1: Synchronization complete")
    
    def run(self):
        """Main run loop for Slave 1"""
        try:
            # Phase 1: Listen for master states
            self.master_states = listen_for_master_states(self.port)
            if self.master_states is None:
                print("Slave 1: Failed to receive master states. Exiting.")
                return
            
            # Phase 2: Estimate parameters
            print("Slave 1: Estimating parameters from master states...")
            self.params, self.initial_state = self.estimator.predict_parameters(self.master_states)
            print(f"Slave 1: Estimated parameters: sigma={self.params.sigma:.4f}, rho={self.params.rho:.4f}, beta={self.params.beta:.4f}")
            print(f"Slave 1: Estimated initial state: {self.initial_state}")
            
            # Reset current state to estimated initial state
            self.current_state = self.initial_state.copy()
            self.current_time = 0.0
            self.state_history = []
            self.reset_integrator()
            
            # Phase 3: Send parameters to Slave 2
            print("Slave 1: Waiting for Slave 2 to be ready...")
            time.sleep(3)  # Give Slave 2 time to start listening
            
            if not send_parameters_to_slave2(self.params, self.initial_state, self.port_slave2):
                print("Slave 1: Failed to send parameters to Slave 2. Exiting.")
                return
            
            # Phase 4: Wait for acknowledgment
            ack = listen_for_acknowledgment(self.port_slave2)
            if not ack:
                print("Slave 1: Did not receive acknowledgment from Slave 2. Exiting.")
                return
            
            print(f"Slave 1: Processing acknowledgment: {ack}")
            
            # Phase 5: Run synchronization
            self.run_synchronization(steps=100)
            
            print("Slave 1: Process complete. Exiting.")
                
        except Exception as e:
            print(f"Slave 1: Error in run loop: {e}")
            import traceback
            traceback.print_exc()

def run_slave1():
    print("\n--- SLAVE 1 SYSTEM STARTING ---\n")
    # Try to find model path, but don't fail if it doesn't exist
    model_path = "92.pth" if os.path.exists("92.pth") else None
    if model_path is None:
        print("Slave 1: Warning - Model file 'model.pth' not found. Using fallback parameter estimation.")
    
    slave1 = Slave1Runner(model_path=model_path)
    slave1.run()

if __name__ == "__main__":
    # Ensure compatibility with PyTorch on macOS
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    run_slave1()