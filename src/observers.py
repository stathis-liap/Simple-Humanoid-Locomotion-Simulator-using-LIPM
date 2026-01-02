from .interfaces import Observer
import numpy as np

class StateLoggerObserver(Observer):
    def __init__(self):
        self.history = []

    def update(self, data: dict):
        self.history.append({
            "t": data["t"],
            "p": data["x"][0],
            "v": data["x"][1],
            "u": data["u"]
        })
    
    def get_arrays(self):
        import numpy as np
        ts = [h["t"] for h in self.history]
        ps = [h["p"] for h in self.history]
        vs = [h["v"] for h in self.history]
        us = [h["u"] for h in self.history]
        return np.array(ts), np.array(ps), np.array(vs), np.array(us)

class FallCounterObserver(Observer):
    def __init__(self, h: float, limit_factor: float = 1.2):
        self.h = h
        self.limit_factor = limit_factor
        self.falls = 0
        self.just_fell = False  # The flag for the visualizer to check

    def update(self, data: dict):
        """
        Checks if leg length exceeds safe limits.
        data: {'t': t, 'x': [p, v], 'u': u, 'x_next': ...}
        """
        # 1. Reset flag (we only want to signal a fall ONCE per step)
        self.just_fell = False
        
        p = data["x"][0]
        u = data["u"]
        
        # 2. The Math: Calculate physical leg length
        # leg_length^2 = (horizontal_dist)^2 + (height)^2
        leg_length = np.sqrt((p - u)**2 + self.h**2)
        
        # 3. The Judgment
        if leg_length > (self.h * self.limit_factor):
            self.falls += 1
            self.just_fell = True