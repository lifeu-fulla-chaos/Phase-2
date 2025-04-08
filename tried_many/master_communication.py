import socket
import pickle
import time
from tried_many.lorentz import encrypt_message
def listen_for_acknowledgment(port):
    """Listen for acknowledgment from Slave 2"""
    # Create a socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('localhost', port + 10))  # Use a different port for master to listen
    server_socket.listen(1)
    server_socket.settimeout(120)  # 120 seconds timeout (2 minutes)
    
    print("Master: Listening for acknowledgment from Slave 2...")
    try:
        conn, addr = server_socket.accept()
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
        print(f"Master: Received acknowledgment from Slave 2: {ack}")
        conn.close()
        return ack
    except socket.timeout:
        print("Master: Timeout waiting for acknowledgment from Slave 2 (2 minutes)")
        return None
    finally:
        server_socket.close()

def send_encrypted_message(lorenz_system, message, timestep, port):
    """Encrypt a message using state at specified timestep and send to Slave 2"""
    # Reset the system
    lorenz_system.reset()
    
    # Run until the specified timestep
    print(f"Master: Running to timestep {timestep} for encryption")
    state_history = lorenz_system.run_steps(timestep)
    current_state = state_history[-1]
    
    # Use the current state for encryption
    encrypted_msg = encrypt_message(message, current_state)
    print(f"Master: Original message: '{message}'")
    print(f"Master: Current state for encryption: {current_state}")
    print(f"Master: Encrypted message (bytes): {encrypted_msg[:20]}... (showing first 20 bytes)")
    
    # Try multiple times to send to Slave 2
    max_retries = 10
    retry_delay = 3
    
    for attempt in range(max_retries):
        # Send to Slave 2
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect(('localhost', port + 20))  # Use a different port for encrypted messages
            # Create a message with the timestep and encrypted content
            msg_data = {
                'timestep': timestep,
                'encrypted_message': encrypted_msg
            }
            data = pickle.dumps(msg_data)
            sock.sendall(len(data).to_bytes(4, byteorder='big'))  # Send data size first
            sock.sendall(data)  # Send the actual data
            print(f"Master: Sent encrypted message for timestep {timestep}")
            return True
        except ConnectionRefusedError:
            if attempt < max_retries - 1:
                print(f"Master: Connection to Slave 2 refused. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Master: Failed to connect to Slave 2 after multiple attempts")
                return False
        except Exception as e:
            print(f"Master: Error sending encrypted message: {e}")
            return False
        finally:
            sock.close()