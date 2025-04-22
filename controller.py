import numpy as np


class BacksteppingController:
    def __init__(self, p, k):
        self.sigma, self.rho, self.beta, self.d, self.e, self.f = p
        self.k1, self.k2, self.k3 = k

    def compute_control(self, x, y):
        x1, x2, x3 = x
        y1, y2, y3 = y

        e1 = y1 - x1
        e2 = y2 - x2
        e3 = y3 - x3

        u1 = self.sigma * (x2 - x1) + y2 + y3 + e2
        u2 = -y1 - self.d * y2 + self.rho * x1 - x2 - x1 * x3 + e3
        u3 = (
            -self.e
            - y3 * (y1 - self.f)
            + x1 * x2
            - self.beta * x3
            - ((3 + 2 * self.k1) * e1)
            - ((5 + 2 * self.k1) * e2)
            - ((3 + self.k1) * e3)
        )

        return np.array([u1, u2, u3])


def dynamics(t, u, controller):
    x = u[:3]
    y = u[3:]

    sigma, rho, beta = controller.sigma, controller.rho, controller.beta
    x1, x2, x3 = x
    dx1 = sigma * (x2 - x1)
    dx2 = x1 * (rho - x3) - x2
    dx3 = x1 * x2 - beta * x3
    dx = np.array([dx1, dx2, dx3])

    u_control = controller.compute_control(x, y)

    y1, y2, y3 = y
    u1, u2, u3 = u_control
    dy1 = -y2 - y3 + u1
    dy2 = y1 + controller.d * y2 + u2
    dy3 = controller.e + y3 * (y1 - controller.f) + u3
    dy = np.array([dy1, dy2, dy3])

    return np.concatenate((dx, dy))
