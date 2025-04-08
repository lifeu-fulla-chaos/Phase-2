import socket
import pickle
import time

def listen_for_master_states(port):
    """Listen for state data from the master"""
    # Create a socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('localhost', port))
    server_socket.listen(1)
    server_socket.settimeout(120)  # 120 seconds timeout (2 minutes)
    
    print("Slave 1: Listening for master states...")
    try:
        conn, addr = server_socket.accept()
        print(f"Slave 1: Received connection from {addr}")
        # Receive the data size first
        size_bytes = conn.recv(4)
        size = int.from_bytes(size_bytes, byteorder='big')
        
        # Receive the actual data
        data = b''
        while len(data) < size:
            packet = conn.recv(size - len(data))
            if not packet:
                break
            data += packet
        
        # Deserialize the data
        master_states = pickle.loads(data)
        print(f"Slave 1: Received {len(master_states)} states from master")
        
        conn.close()
        return master_states
    except socket.timeout:
        print("Slave 1: Timeout waiting for master states (2 minutes)")
        return None
    finally:
        server_socket.close()

def send_parameters_to_slave2(params, initial_state, port):
    """Send the estimated parameters and initial state to Slave 2"""
    # Create a socket connection to slave 2
    max_retries = 10
    retry_delay = 3
    
    for attempt in range(max_retries):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect(('localhost', port))
            # Prepare the parameter data
            param_data = {
                'params': params,
                'initial_state': initial_state
            }
            data = pickle.dumps(param_data)
            sock.sendall(len(data).to_bytes(4, byteorder='big'))  # Send data size first
            sock.sendall(data)  # Send the actual data
            print(f"Slave 1: Sent parameters to Slave 2")
            return True
        except ConnectionRefusedError:
            if attempt < max_retries - 1:
                print(f"Slave 1: Connection to Slave 2 refused. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Slave 1: Failed to connect to Slave 2 after multiple attempts")
                return False
        except Exception as e:
            print(f"Slave 1: Error sending parameters to Slave 2: {e}")
            return False
        finally:
            sock.close()

def listen_for_acknowledgment(port):
    """Listen for acknowledgment from Slave 2"""
    # Create a socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('localhost', port + 5))  # Use a different port
    server_socket.listen(1)
    server_socket.settimeout(120)  # 120 seconds timeout (2 minutes)
    
    print("Slave 1: Listening for acknowledgment from Slave 2...")
    try:
        conn, addr = server_socket.accept()
        print(f"Slave 1: Received acknowledgment connection from {addr}")
        # Receive the acknowledgment
        size_bytes = conn.recv(4)
        size = int.from_bytes(size_bytes, byteorder='big')
        data = b''
        while len(data) < size:
            packet = conn.recv(size - len(data))
            if not packet:
                break
            data += packet
        
        ack = pickle.loads(data)
        print(f"Slave 1: Received acknowledgment from Slave 2: {ack}")
        conn.close()
        return ack
    except socket.timeout:
        print("Slave 1: Timeout waiting for acknowledgment (2 minutes)")
        return None
    finally:
        server_socket.close()