import numpy as np
import matplotlib.pyplot as plt
from src.factories import SimulatorFactory
from src.visualizer import LIPMVisualizer
from src.observers import StateLoggerObserver, FallCounterObserver

def run_demo():
    # 1. Configuration
    config = {
        "g": 9.81,
        "h": 0.6, #leg height         
        "dt": 0.01, 
        "dynamics_type": "continuous", # or "discreet"
        "policy_type": "least_square", # or "lqr"
        "u_min": -0.3, #min length of step
        "u_max": 0.3, #max length of step
        "push_prob": 0.1 #probability of getting pushed each step
    }

    print("Initializing Simulation...")
    sim = SimulatorFactory.create(config)
    
    #Attach the loggers
    logger = StateLoggerObserver()
    sim.attach(logger)
    
    referee = FallCounterObserver(h=config["h"], limit_factor=1.2)
    sim.attach(referee)
    
    print("-> Observers attached: Logger & FallCounter")
    
    # Starting pose (a little of balance to have initial movement)
    sim.x = np.array([0.1, 0.0])

    print("Starting Animation...")
    viz = LIPMVisualizer(sim, config)
    viz.show() 

    print("\n" + "="*40)
    print("       MISSION REPORT       ")
    print("="*40)
    
    # Get data from the Referee
    print(f"Total Falls Detected: {referee.falls}")
    
    # Get data from the Historian
    t_log, p_log, v_log, u_log = logger.get_arrays()
    print(f"Total Steps Simulated: {len(t_log)}")
    print(f"Max Velocity Reached:  {np.max(np.abs(v_log)):.4f} m/s")
    

if __name__ == "__main__":
    run_demo()