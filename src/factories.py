from .dynamics import LIPMDiscreteLinearAB, LIPMContinuousEuler
from .policies import CapturePointPolicy, LeastSquaresPolicy
from .simulator import ScenarioSimulator
import numpy as np

class SimulatorFactory:
    @staticmethod
    def create(config: dict):
        # Create Dynamics
        omega = np.sqrt(config["g"] / config["h"])
        dt = config["dt"]
        v_walk = config.get("v_walk", 0.0)

        if config["dynamics_type"] == "discrete":
            dynamics = LIPMDiscreteLinearAB(omega, dt)
        elif config["dynamics_type"] == "continuous":
            dynamics = LIPMContinuousEuler(omega, dt)
        else:
            raise ValueError("Unknown dynamics type")
            
        # Create Policy
        if config["policy_type"] == "capture_point":
            policy = CapturePointPolicy(omega, config["u_min"], config["u_max"], v_walk=v_walk)
        elif config["policy_type"] == "least_square":
            A, B = dynamics.get_AB()
            policy = LeastSquaresPolicy(A, B, v_walk=v_walk)
        else:
            raise ValueError("Unknown policy")

        # Create Simulator
        sim = ScenarioSimulator(
            dynamics, 
            policy, 
            dt, 
            config["u_min"], 
            config["u_max"],
            config.get("push_prob", 0.0),
            config.get("step_time", 0.3)
        )
        
        return sim