import numpy as np
from typing import List
from .interfaces import DynamicsModel, ControlPolicy, Observer

class BaseSimulator:
    """
    Template Method pattern: Defines the skeleton of step().
    """
    def __init__(self, dynamics: DynamicsModel, policy: ControlPolicy, dt: float = 0.01):
        self.dynamics = dynamics
        self.policy = policy
        self.dt = dt
        self.observers: List[Observer] = []
        
        self.x = np.zeros(2) # [p, p_dot]
        self.t = 0.0

    def attach(self, observer: Observer):
        self.observers.append(observer)

    def step(self):
        # 1. Compute Control (Strategy)
        u = self.policy.compute_control(self.x, self.t)
        
        # 2. Hook: Constraints (can be overridden)
        u = self._enforce_constraints(u)

        # 3. Propagate Dynamics (Strategy)
        x_next = self.dynamics.propagate(self.x, u, self.dt)
        
        # 4. Hook: Disturbances
        x_next = self._apply_disturbances(x_next)

        # 5. Notify Observers (Observer)
        self._notify_observers(self.x, u, x_next)

        self.x = x_next
        self.t += self.dt

    def run(self, steps: int):
        for _ in range(steps):
            self.step()

    def _notify_observers(self, x_prev, u, x_next):
        data = {"t": self.t, "x": x_prev, "u": u, "x_next": x_next}
        for obs in self.observers:
            obs.update(data)

    # Hooks
    def _enforce_constraints(self, u): return u
    def _apply_disturbances(self, x): return x

class ScenarioSimulator(BaseSimulator):
    """
    Concrete implementation adding constraints and random pushes.
    """
    def __init__(self, dynamics, policy, dt, u_min, u_max, push_prob=0.0):
        super().__init__(dynamics, policy, dt)
        self.u_min = u_min
        self.u_max = u_max
        self.push_prob = push_prob
        self.rng = np.random.default_rng(42)

    def _enforce_constraints(self, u):
        return float(np.clip(u, self.u_min, self.u_max))

    def _apply_disturbances(self, x):
        # Random velocity kick
        if self.rng.random() < self.push_prob:
            kick = self.rng.uniform(-0.5, 0.5)
            x[1] += kick
        return x