import abc
import numpy as np

class DynamicsModel(abc.ABC):
    @abc.abstractmethod
    def propagate(self, x: np.ndarray, u: float, dt: float) -> np.ndarray:
        """Calculates next state x_{k+1}."""
        pass

    @abc.abstractmethod
    def get_AB(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the A and B matrices (2x2, 2x1)."""
        pass

class ControlPolicy(abc.ABC):
    @abc.abstractmethod
    def compute_control(self, x: np.ndarray, t: float) -> float:
        """Calculates control input u."""
        pass

class Observer(abc.ABC):
    @abc.abstractmethod
    def update(self, data: dict):
        """Receives simulation data."""
        pass