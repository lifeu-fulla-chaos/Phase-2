import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class LorenzParameters:
    def __init__(self, sigma, rho, beta):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta


class LorenzSystem:
    def __init__(self, initial_state, params, dt=0.001):
        self.params = params
        self.initial_state = initial_state
        self.dt = dt
        self.state_history = []

    def lorenz_equations(self, t, state):
        x, y, z = state
        dx = self.params.sigma * (y - x)
        dy = x * (self.params.rho - z) - y
        dz = x * y - self.params.beta * z
        return [dx, dy, dz]

    def run_steps(self, t, steps):
        t_span = (t, t + self.dt * steps)
        t_eval = np.linspace(*t_span, steps)

        solution = solve_ivp(
            fun=self.lorenz_equations,
            t_span=t_span,
            y0=self.initial_state,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
        )
        self.state_history = solution.y.T
        return self.state_history

    def reset(self):
        self.state_history = []


# # Example usage:
# p = LorenzParameters(sigma=10, rho=28, beta=8 / 3)
# a = LorenzSystem(initial_state=np.array([1.0, 1.0, 1.0]), params=p, dt=0.001)

# # Run the Lorenz system for 1000 steps
# steps = 100000
# trajectory = a.run_steps(steps)

# # Extract x, y, z coordinates from the trajectory
# x = trajectory[:, 0]
# y = trajectory[:, 1]
# z = trajectory[:, 2]

# # Plot the Lorenz attractor
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot(x, y, z, lw=0.5)
# ax.set_title("Lorenz Attractor (solve_ivp)")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# # Save the plot to a file
# plt.savefig("lorenz_attractor.png")
# plt.show()
