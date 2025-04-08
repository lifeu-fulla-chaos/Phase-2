import numpy as np
from scipy.integrate import RK45

class LorenzParameters:
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

def lorenz_system(t, state, params):
    """Lorenz system equations"""
    x, y, z = state
    dx_dt = params.sigma * (y - x)
    dy_dt = x * (params.rho - z) - y
    dz_dt = x * y - params.beta * z
    return np.array([dx_dt, dy_dt, dz_dt])

# Controller for backstepping synchronization
def backstepping_controller(master_state, slave_state, params, gain=5.0):
    """Backstepping controller for synchronization"""
    e = master_state - slave_state
    # Simple proportional controller with gain
    u = gain * e
    return u

# Modified Lorenz system with control input
def controlled_lorenz_system(t, state, params, control):
    """Lorenz system with control input for synchronization"""
    x, y, z = state
    dx_dt = params.sigma * (y - x) + control[0]
    dy_dt = x * (params.rho - z) - y + control[1]
    dz_dt = x * y - params.beta * z + control[2]
    return np.array([dx_dt, dy_dt, dz_dt])

def decrypt_message(encrypted_message, key_state):
    """Decrypt a message using Lorenz state as key"""
    # Simple XOR decryption using the state values
    key_bytes = np.abs(np.sin(key_state * 1000)).astype(np.uint8)
    # Expand key if message is longer
    key_bytes = np.tile(key_bytes, (len(encrypted_message) // len(key_bytes)) + 1)[:len(encrypted_message)]
    # XOR decryption
    decrypted = bytearray([encrypted_message[i] ^ key_bytes[i % len(key_bytes)] for i in range(len(encrypted_message))])
    return decrypted