import numpy as np
from scipy.integrate import RK45

class LorenzParameters:
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

def lorenz_equations(t, state, params):
    x, y, z = state
    dx = params.sigma * (y - x)
    dy = x * (params.rho - z) - y
    dz = x * y - params.beta * z
    return np.array([dx, dy, dz])

class LorenzSystem:
    def __init__(self, initial_state=None, params=None, dt=0.01):
        self.params = params if params else LorenzParameters()
        self.initial_state = initial_state if initial_state is not None else np.array([1.0, 1.0, 1.0])
        self.dt = dt
        self.current_state = self.initial_state.copy()
        self.current_time = 0.0
        self.state_history = []
        self._setup_integrator()

    def _setup_integrator(self):
        self.integrator = RK45(
            lambda t, y: lorenz_equations(t, y, self.params),
            self.current_time,
            self.current_state,
            t_bound=float('inf'),
            rtol=1e-6,
            atol=1e-6
        )

    def step(self):
        target_time = self.current_time + self.dt
        while self.integrator.t < target_time:
            self.integrator.step()
        self.current_state = self.integrator.y
        self.current_time = self.integrator.t
        self.state_history.append(self.current_state.copy())
        return self.current_state

    def run_steps(self, steps):
        self.state_history = []
        for _ in range(steps):
            self.step()
        return np.array(self.state_history)

    def reset(self):
        self.current_state = self.initial_state.copy()
        self.current_time = 0.0
        self.state_history = []
        self._setup_integrator()

def encrypt_message(message, key_state):
    key_bytes = np.abs(np.sin(key_state * 1000)).astype(np.uint8)
    key_bytes = np.tile(key_bytes, (len(message) // len(key_bytes)) + 1)[:len(message)]
    message_bytes = message.encode('utf-8') if isinstance(message, str) else message
    return bytearray([message_bytes[i] ^ key_bytes[i % len(key_bytes)] for i in range(len(message_bytes))])
