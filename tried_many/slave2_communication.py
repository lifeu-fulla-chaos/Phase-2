import socket
import pickle
import time

def send_acknowledgment_to_slave1(message, port):
    """Send acknowledgment message to Slave 1"""
    # Create a socket connection to slave 1
    max_retries = 10
    retry_delay = 3
    
    for attempt in range(max_retries):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect(('localhost', port + 5))  # Use the port Slave 1 is listening on for acknowledgments
            data = pickle.dumps(message)
            sock.sendall(len(data).to_bytes(4, byteorder='big'))  # Send data size first
            sock.sendall(data)  # Send the actual data
            print(f"Slave 2: Sent acknowledgment to Slave 1: {message}")
            return True
        except ConnectionRefusedError:
            if attempt < max_retries - 1:
                print(f"Slave 2: Connection to Slave 1 refused. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Slave 2: Failed to connect to Slave 1 after multiple attempts")
                return False
        except Exception as e:
            print(f"Slave 2: Error sending acknowledgment: {e}")
            return False
        finally:
            sock.close()

def send_acknowledgment_to_master(message, port):
    """Send acknowledgment message to Master"""
    # Create a socket connection to master
    max_retries = 10
    retry_delay = 3
    
    for attempt in range(max_retries):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect(('localhost', port + 10))  # Use the port Master is listening on for acknowledgments
            data = pickle.dumps(message)
            sock.sendall(len(data).to_bytes(4, byteorder='big'))  # Send data size first
            sock.sendall(data)  # Send the actual data
            print(f"Slave 2: Sent acknowledgment to Master: {message}")
            return True
        except ConnectionRefusedError:
            if attempt < max_retries - 1:
                print(f"Slave 2: Connection to Master refused. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Slave 2: Failed to connect to Master after multiple attempts")
                return False
        except Exception as e:
            print(f"Slave 2: Error sending acknowledgment to Master: {e}")
            return False
        finally:
            sock.close()

def listen_for_parameters(port):
    """Listen for parameter data from Slave 1"""
    # Create a socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('localhost', port))
    server_socket.listen(1)
    server_socket.settimeout(120)  # 120 seconds timeout (2 minutes)
    
    print("Slave 2: Listening for parameters from Slave 1...")
    try:
        conn, addr = server_socket.accept()
        print(f"Slave 2: Received connection from {addr}")
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
        param_data = pickle.loads(data)
        print(f"Slave 2: Received parameters from Slave 1")
        
        conn.close()
        return param_data
    except socket.timeout:
        print("Slave 2: Timeout waiting for parameters (2 minutes)")
        return None
    finally:
        server_socket.close()

def listen_for_encrypted_message(port):
    """Listen for encrypted messages from Master"""
    # Create a socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('localhost', port + 20))  # Use port+20 for encrypted messages
    server_socket.listen(1)
    server_socket.settimeout(120)  # 120 seconds timeout (2 minutes)
    
    print("Slave 2: Listening for encrypted messages from Master...")
    try:
        conn, addr = server_socket.accept()
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
        msg_data = pickle.loads(data)
        print(f"Slave 2: Received encrypted message from Master")
        
        conn.close()
        return msg_data
    except socket.timeout:
        print("Slave 2: Timeout waiting for encrypted message (2 minutes)")
        return None
    finally:
        server_socket.close()