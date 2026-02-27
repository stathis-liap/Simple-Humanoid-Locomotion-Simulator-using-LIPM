***Humanoid Locomotion Simulator (LIPM + Capture Point)***

Author: Stathis Liapodimitris

Affiliation: Embodied Intelligence Robotics Club (Christmas Project)


**📌 Overview**

This repository contains a modular, 2D sagittal-plane locomotion simulation engine for a simplified humanoid model. The primary goal of this project is to explore the theoretical foundations of humanoid balance (using the Linear Inverted Pendulum Model) while strictly enforcing professional software architecture through classic design patterns.

**🧠 Theoretical Foundations**

To make a robot walk, we must first understand how to keep it from falling over. This project implements the math required to predict falls and safely place the robot's feet to maintain balance.

**1. The Physics: Linear Inverted Pendulum Model (LIPM)**

Modeling a full humanoid robot with dozens of joints is incredibly complex. Instead, we use the LIPM, which simplifies the robot into a single point mass (the Center of Mass, or COM) balancing on a massless leg.

To make the math linear and solvable, the model assumes the robot's COM stays at a constant height h. The motion of the robot is entirely dictated by where its COM (p) is relative to its foot / support point (u).

The core dynamic equation governing this is:

  $p^¨(t)=ω^2(p(t)−u(t))$

Where $ω=hg​​$ is the natural frequency of the pendulum.

Intuition: If the robot's body (p) is in front of its foot (u), gravity pulls it forward, and it accelerates into a fall. If the robot steps in front of its body (u>p), it brakes.

**2. The Stabilizer: Capture Point (CP)**

The LIPM is inherently unstable—like balancing a broomstick on your hand. If you get pushed, you fall exponentially faster. To prevent this, we use a metric called the Capture Point (or Divergent Component of Motion).

The Capture Point (ξ) predicts exactly where the robot will end up if it doesn't take a step, factoring in both its current position and velocity:

  $ξ=p+ωp˙$​​

How to balance: To stop the robot from falling instantly, the controller simply calculates the Capture Point and places the foot exactly on top of it (u=ξ).

How to walk: To walk forward, the controller places the foot slightly behind the moving target Capture Point, deliberately "falling" forward into the next step.

**3. Alternative Control: Least Squares**

As an alternative to the pure Capture Point policy, this project also implements a Least Squares one-step lookahead policy. It uses the discrete state-space matrices (A and B) to find the optimal foot placement u that minimizes the error between the robot's predicted next state and a desired target state (like x=[0,vwalk​]).

**🏗️ Software Architecture (Design Pattern Map)**

For this project, the goal was to build a locomotion simulator that is designed like a properly structured program. To make the code clean, modular, and easy to modify later, I implemented the mandatory design patterns that were suggested in the explanation of the project. Below is a breakdown of how I used each pattern and the justification for each one.

**1. Strategy Pattern**

  Classes: DynamicsModel (LIPMContinuousEuler, LIPMDiscreteLinearAB), ControlPolicy (CapturePointPolicy, LeastSquaresPolicy).

  Justification: This pattern lets me easily swap out the physics engine and the policy of the robot. Instead of hardcoding one specific way to do the math, the simulator just talks to a generic interface. This means I can switch between simple Euler integration and exact discrete math, or change the walking logic from a Capture Point strategy to a Least Squares approach, without having to rewrite any of the main simulation code.

**2. Factory Method Pattern**

  Classes: SimulatorFactory

  Justification: Instead of manually creating my physics models and control policies inside the main script, I built a Factory class to handle it. The factory reads a configuration dictionary and builds the correct objects for me automatically. This way I never directly call constructors outside of the factory. It makes testing easier since I can run completely different scenarios just by editing a simple config dictionary.

**3. Template Method Pattern**

  Classes: BaseSimulator, ScenarioSimulator

  Justification: The BaseSimulator defines a strict, step-by-step routine for the simulation loop inside the step() method: compute control, apply constraints, calculate physics, add pushes, and log data. I locked this sequence down so the core math order cannot be accidentally messed up. If I want to change how a specific part works (like I did when adding stepping functionality with a type of sample and hold logic), I use the ScenarioSimulator subclass to safely override just that specific hook.

**4. Observer Pattern**

  Classes: Observer (Interface), StateLoggerObserver, FallCounterObserver, BaseSimulator (Subject).

Justification: I wanted to ensure the physics engine focuses only on doing math, not on printing text, saving arrays, or drawing graphs. The Observer pattern solves this. Every time the simulator finishes a step, it simply broadcasts the new state. My observers (like the logger or the referee that counts falls) listen for this broadcast and record what they need. This keeps the core simulation loop fast and minimal.

**🚀 Running the Project**

_Prerequisites_

This project relies purely on standard math and plotting libraries to avoid "black-box" locomotion frameworks.

Bash

    pip install numpy matplotlib

_Running the Demo_

To launch the live Matplotlib animation of the robot balancing and walking:
Bash

    python demo.py

_Configuration_

You can easily tweak the robot's physical traits or the simulation rules by editing the config dictionary inside demo.py:

Python
    
    config = {
        "g": 9.81,
        "h": 0.6,                 # robot center of mass height
        "dt": 0.01,               # simulation timestep
        "dynamics_type": "continuous", # "discrete" or "continuous"
        "policy_type": "capture_point", # "capture_point" or "least_square"
        "u_min": -0.3,            # minimum relative step length
        "u_max": 0.3,             # maximum relative step length
        "push_prob": 0.1,         # probability of a random disturbance
        "step_time": 0.3,         # time between foot placements
        "v_walk": 0.5             # target walking velocity
    }
