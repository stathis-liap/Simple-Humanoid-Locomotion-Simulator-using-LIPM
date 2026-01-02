import numpy as np
from .interfaces import DynamicsModel

class LIPMContinuousEuler(DynamicsModel):
    """
    Option A: Simple Euler Integration.
    x_next = x + (Ac*x + Bc*u) * dt
    """
    def __init__(self, omega, dt):
        self.omega = omega
        self.dt = dt
        # Continuous Matrices
        # Ac = [[0, 1], [w^2, 0]]
        self.Ac = np.array([
            [0, 1.0],
            [self.omega**2, 0]
        ])
        # Bc = [[0], [-w^2]]
        self.Bc = np.array([
            [0],
            [-self.omega**2]
        ])

    def propagate(self, x: np.ndarray, u: float, dt: float) -> np.ndarray:
        x_dot = self.Ac @ x + (self.Bc * u).flatten()
        return x + x_dot * dt
    
    def get_AB(self):
        # Approximate discrete A, B for LQR usage
        # A approx I + Ac*dt
        A = np.eye(2) + self.Ac * self.dt
        B = self.Bc * self.dt
        return A, B


class LIPMDiscreteLinearAB(DynamicsModel):
    """
    Option B: Exact discretization using analytical closed-form solution.
    Reference: x_{k+1} = A x_k + B u_k
    """
    def __init__(self, omega, dt):
        self.omega = omega
        self.dt = dt
        
        
        w_dt = self.omega * self.dt
        c = np.cosh(w_dt)
        s = np.sinh(w_dt)
        
        # A matrix: [[cosh, sinh/w], [w*sinh, cosh]] (precomputed)
        self.A = np.array([
            [c, s / self.omega],
            [self.omega * s, c]
        ])
        
        # B matrix: [[1 - cosh], [-w*sinh]] (precomputed)
        self.B = np.array([
            [1.0 - c],
            [-self.omega * s]
        ])

    def propagate(self, x: np.ndarray, u: float, dt: float) -> np.ndarray:
        # x_next = A @ x + B * u
        return self.A @ x + (self.B * u).flatten()

    def get_AB(self):
        return self.A, self.B

