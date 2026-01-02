import numpy as np
from .interfaces import ControlPolicy

class CapturePointPolicy(ControlPolicy):
    """
    Stabilizes the robot by placing the foot at the Capture Point.
    xi = p + p_dot / omega
    """
    def __init__(self, omega, u_min=-0.5, u_max=0.5):
        self.omega = omega
        self.u_min = u_min
        self.u_max = u_max

    def compute_control(self, x: np.ndarray, t: float) -> float:
        p, v = x
        xi = p + (v / self.omega)
        return float(np.clip(xi, self.u_min, self.u_max))

class LeastSquaresPolicy(ControlPolicy):
    """
    Least Squares (One-Step Lookahead):
    Finds u that minimizes ||x_{k+1} - x_ref||^2.
    
    Since we have 2 states (p, v) and 1 control (u), we cannot 
    perfectly reach (0,0) in one step. This finds the best geometric compromise.
    """
    def __init__(self, A, B, target_vel=0.0):
        self.A = A
        # Ensure B is a column vector (2, 1) for correct matrix math
        self.B = B.reshape(-1, 1) 
        self.target_vel = target_vel

        # Precompute the "Pseudo-Inverse" of B
        # Formula: pinv = (B.T @ B)^-1 @ B.T
        
        # 1. B_transpose @ B is a scalar (dot product) because B is (2,1)
        # B.T is (1,2), B is (2,1) -> result is (1,1)
        self.B_dot_B = (self.B.T @ self.B).item()
        
        # 2. Check for division by zero (unlikely in LIPM unless dt=0)
        if np.abs(self.B_dot_B) < 1e-9:
            raise ValueError("Matrix B is too close to zero. Check dynamics.")

    def compute_control(self, x: np.ndarray, t: float) -> float:
        """
        Solves: min || (Ax + Bu) - x_ref ||^2
        Result: u = (B^T B)^-1 B^T (x_ref - Ax)
        """
        # 1. Define Target State (x_ref)
        # We want position to be 0, velocity to be target_vel
        x_ref = np.array([0.0, self.target_vel])
        
        # 2. Calculate the "Drift" (where we would go if u=0)
        drift = self.A @ x
        
        # 3. Calculate the "Desired Correction" (Y)
        # B * u = x_ref - Ax
        Y = x_ref - drift
        
        # 4. Solve for u using the Normal Equation
        # u = (B.T * Y) / (B.T * B)
        numerator = self.B.T @ Y
        u = numerator.item() / self.B_dot_B
        
        return u