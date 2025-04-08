import os
import time
from tried_many.slave2_node import Slave2
from tried_many.master_node import MasterNode

# Set environment variable to avoid OpenMP runtime issues
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def run_slave2():
    """Run the Slave 2 node only"""
    print("\n--- SLAVE 2 SYSTEM STARTING ---\n")
    slave = Slave2(port_slave1=5001, port_master=5000)
    slave.start()
    
    try:
        print("Slave 2: Running and waiting for data...")
        while True:
            time.sleep(5)
            status = slave.get_status()
            print(f"Slave 2: Sync error: {status['sync_error']:.6f}")
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        slave.stop()

def run_master():
    """Run the Master node only"""
    print("\n--- MASTER NODE STARTING ---\n")
    master = MasterNode(port=5000)
    master.start()
    
    try:
        print("Master: Running and waiting for connections...")
        while True:
            time.sleep(5)
            status = master.get_status()
            print(f"Master: Clients connected: {status['clients']}")
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        master.stop()

def run_example():
    """Run both Master and Slave2 for demonstration"""
    print("\n--- LORENZ SYSTEM DEMO STARTING ---\n")
    
    # Start the master node
    master = MasterNode(port=5000)
    master.start()
    
    # Wait for server to initialize
    time.sleep(1)
    
    # Start slave node
    slave = Slave2(port_slave1=5001, port_master=5000)
    slave.start()
    
    try:
        # Run for a while to establish synchronization
        for i in range(30):
            time.sleep(1)
            # Print synchronization status
            status = slave.get_status()
            print(f"Sync error: {status['sync_error']:.6f}")
            
        # Send an encrypted message
        master.send_encrypted_message("Hello from the Lorenz system!")
        
        # Continue running to process the message
        time.sleep(5)
        
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # Stop all threads
        slave.stop()
        master.stop()

if __name__ == "__main__":
    # Choose which component to run
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "slave":
            run_slave2()
        elif sys.argv[1].lower() == "master":
            run_master()
        else:
            print("Unknown argument. Use 'slave', 'master', or no argument for demo.")
    else:
        # Run full example by default
        run_example()