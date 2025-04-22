import numpy as np
from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt


class RosslerParameters:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c


class RosslerSystem:
    def __init__(self, initial_state, params, dt=0.001):
        self.params = params
        self.initial_state = initial_state
        self.dt = dt
        self.state_history = []

    def rossler_equations(self, t, state):
        x, y, z = state
        dx = -y - z
        dy = x + self.params.a * y
        dz = self.params.b + z * (x - self.params.c)
        return [dx, dy, dz]

    def run_steps(self, t, steps):
        t_span = (t, t + self.dt * steps)
        t_eval = np.linspace(*t_span, steps)

        solution = solve_ivp(
            fun=self.rossler_equations,
            t_span=t_span,
            y0=self.initial_state,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
        )
        self.state_history = solution.y.T
        self.initial_state = self.state_history[-1] 
        return self.state_history

    def reset(self):
        self.state_history = []


# p = RosslerParameters(a=0.2, b=0.2, c=5.7)
# a = RosslerSystem(initial_state=np.array([1.0, 1.0, 1.0]), params=p, dt=0.001)

# steps = 1000000
# trajectory = a.run_steps(0, steps)

# x = trajectory[:, 0]
# y = trajectory[:, 1]
# z = trajectory[:, 2]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot(x, y, z, lw=0.5)
# ax.set_title("Lorenz Attractor (solve_ivp)")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# plt.savefig("lorenz_attractor.png")
# plt.show()
