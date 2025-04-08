import random
import time
import numpy as np
from tried_many.lorentz import LorenzParameters, LorenzSystem
from tried_many.master import Master
from tried_many.master_communication import listen_for_acknowledgment, send_encrypted_message

def run_master():
    """Main run function for the master"""
    # Create and run the master
    master_params = LorenzParameters(sigma=10.0, rho=28.0, beta=8.0/3.0)
    initial_state = np.array([1.0, 1.0, 1.0])
    lorenz_system = LorenzSystem(initial_state, master_params)
    
    print("\n--- MASTER SYSTEM STARTING ---\n")
    master = Master(master_params, initial_state)
    
    try:
        # Phase 1: Run initial trajectory and send states to Slave 1
        state_history = master.run_initial_trajectory(steps=60)
        
        print("Master: Waiting for Slave 1 to start...")
        time.sleep(2)  # Give Slave 1 a moment to start up
        
        if not master.send_states_to_slave1(state_history):
            print("Master: Failed to send states to Slave 1. Exiting.")
            return
        
        # Wait for acknowledgment from Slave 2
        print("Master: Waiting for acknowledgment from Slave 2...")
        ack = listen_for_acknowledgment(master.port_slave2)
        if not ack:
            print("Master: Did not receive acknowledgment from Slave 2 in time. Exiting.")
            return
        
        # Phase 2: Restart and synchronize
        print("Master: Both slaves acknowledged. Starting synchronization phase.")
        lorenz_system.reset()
        
        # Run for a few steps to let slaves synchronize
        print("Master: Running initial synchronization steps...")
        for _ in range(5):
            lorenz_system.step()
            time.sleep(0.5)
        
        # Phase 3: Send an encrypted message
        message = "This is a secret message from the master system!"
        random_timestep = random.randint(10, 30)  # Choose a random timestep
        print(f"Master: Will encrypt message at timestep {random_timestep}")
        
        if not send_encrypted_message(lorenz_system, message, random_timestep, master.port_slave2):
            print("Master: Failed to send encrypted message. Exiting.")
            return
        
        # Continue running to maintain synchronization
        print("Master: Continuing to run for synchronization...")
        for i in range(100):  # Run for additional steps
            lorenz_system.step()
            if i % 10 == 0:
                print(f"Master: Still running... step {i}/100")
            time.sleep(0.1)  # Small delay to simulate real-time operation
        
        print("Master: Process complete. Exiting.")
            
    except Exception as e:
        print(f"Master: Error in run loop: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_master()