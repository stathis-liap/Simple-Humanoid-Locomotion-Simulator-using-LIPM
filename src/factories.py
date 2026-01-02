from .dynamics import LIPMDiscreteLinearAB, LIPMContinuousEuler
from .policies import CapturePointPolicy, LeastSquaresPolicy
from .simulator import ScenarioSimulator
import numpy as np

class SimulatorFactory:
    @staticmethod
    def create(config: dict):
        # 1. Create Dynamics
        omega = np.sqrt(config["g"] / config["h"])
        dt = config["dt"]
        
        if config["dynamics_type"] == "discrete":
            dynamics = LIPMDiscreteLinearAB(omega, dt)
        elif config["dynamics_type"] == "continuous":
            dynamics = LIPMContinuousEuler(omega, dt)
        else:
            raise ValueError("Unknown dynamics type")
            
        # 2. Create Policy
        if config["policy_type"] == "capture_point":
            policy = CapturePointPolicy(omega, config["u_min"], config["u_max"])
        elif config["policy_type"] == "least_square":
            A, B = dynamics.get_AB()
            policy = LeastSquaresPolicy(A, B)
        else:
            raise ValueError("Unknown policy")

        # 3. Create Simulator
        sim = ScenarioSimulator(
            dynamics, 
            policy, 
            dt, 
            config["u_min"], 
            config["u_max"],
            config.get("push_prob", 0.0)
        )
        
        return sim