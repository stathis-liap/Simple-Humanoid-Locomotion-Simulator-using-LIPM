import numpy as np
from .interfaces import ControlPolicy

class CapturePointPolicy(ControlPolicy):
    """
    Stabilizes the robot by placing the foot at the Capture Point.
    xi = p + p_dot / omega
    with tracking
    """
    def __init__(self, omega, u_min=-0.5, u_max=0.5, v_walk=0.0, kp=1.0):
        self.omega = omega
        self.u_min = u_min
        self.u_max = u_max
        self.v_walk = v_walk
        self.kp = kp # proportional gain for tracking

    def compute_control(self, x: np.ndarray, t: float) -> float:
        p, v = x
        xi = p + (v / self.omega)
        
        # calculate reference targets
        p_star = self.v_walk * t
        xi_star = p_star + (self.v_walk / self.omega)
        
        # base stability (xi) + error correction (xi - xi_star)
        # basically if the robot is behind the target CP, this pushes the foot behind xi, 
        # causing the robot to accelerate forward.
        u_target = xi + self.kp * (xi - xi_star)

        return u_target

class LeastSquaresPolicy(ControlPolicy):
    """
    Least Squares with tracking.
    """
    def __init__(self, A, B, v_walk=0.0):
        self.A = A
        self.B = B.reshape(-1, 1) 
        self.v_walk = v_walk
        self.B_dot_B = (self.B.T @ self.B).item()

    def compute_control(self, x: np.ndarray, t: float) -> float:
        # define moving target state (x_ref)
        # position moves forward over time - velocity is constant
        x_ref = np.array([self.v_walk * t, self.v_walk])
        
        drift = self.A @ x
        Y = x_ref - drift
        
        numerator = self.B.T @ Y
        u = numerator / self.B_dot_B
        
        return u