# IAEML — Autonomous Driving with Reinforcement Learning

This repository contains my assignments for the **IAEML (Introduction to Artificial Intelligence and Machine Learning)** course at Innopolis University.

The project focuses on autonomous driving in a simple 2D environment. The goal is to build an agent that can navigate a vehicle from a starting point to a destination while avoiding static and moving obstacles.

The project started with a basic environment and was later extended with a more realistic vehicle model and a reinforcement learning agent trained using PPO.

## Project Overview

The environment represents a simplified road scene containing:
- An agent (vehicle) with a starting position and direction.
- A destination that the agent must reach.
- Static obstacles.
- Moving obstacles.

All objects in the original environment are modeled as circles with radius `r = 1`.

The project is built around **JAX**, which allows environment simulation, path planning, and agent training to run efficiently using JIT compilation and vectorized operations.

## Environment

The original environment is defined in `base.py`. It provides the main functionality needed for autonomous navigation:

- **Perception:** The agent uses `M` evenly distributed rays to detect nearby obstacles. The rays start at the center of the agent and cover its field of view.
- **Navigation:** A dynamic programming algorithm calculates a path from the current position to the destination. The computation is performed using JAX operations.
- **Simulation:** The environment updates the agent's state, moves dynamic obstacles, checks collisions, and calculates rewards.

The main environment interface includes:

- `init_params()` — creates environment parameters and the initial state.
- `reset()` — initializes the environment state.
- `step()` — performs one simulation step.
- `get_observation()` — returns information about the agent and its surroundings.

`BaseEnv` is implemented as a collection of static functions operating on `BaseEnvState` and `BaseEnvParams`. The environment parameters are passed as static arguments to JIT-compiled functions using `jax.jit()`.

### Task Sampling

The task sampling functionality is implemented in `env/tasks.py`.

It allows the environment to generate different navigation scenarios with different starting positions, destinations, and obstacles. This can also be useful for experiments with meta-learning, where an agent needs to adapt to different tasks.

One current limitation is that sampling a new task can trigger JAX recompilation if the environment parameters or state shapes change. Using maps with fixed dimensions would help avoid some of these recompilations.

## Vehicle Dynamics

The original environment used a simplified point-mass movement model. In the first assignment, I extended it with a more realistic vehicle model based on the **Kinematic Bicycle Model**.

This implementation is available in `env/robotaxi.py`.

### State and Action Spaces

The vehicle state includes additional information:
- Current speed.
- Heading angle.
- Steering angle.

The agent has two continuous actions, both in the range `[-1, 1]`:

- Acceleration control.
- Steering control.

### Movement Model

The vehicle dynamics are calculated using a simulation time step of `1/60` seconds.

The model accounts for two resistive forces:

- Rolling friction: `F_friction = c_fric * v`
- Aerodynamic drag: `F_drag = c_drag * v * |v|`

The resulting acceleration updates the vehicle's speed, which is limited by predefined minimum and maximum values.

The heading angle is updated using the bicycle model:

`theta_next = theta + (v_next / L) * tan(delta) * dt`

where:
- `L = 0.65` is the wheelbase.
- `delta` is the steering angle.
- `dt` is the simulation time step.

The vehicle cannot move sideways and needs enough space to make turns. This makes obstacle avoidance and path following more challenging than in the original point-mass environment.

## Observation Space and Reward Function

The vehicle receives information about its current state and the surrounding environment.

The observation includes:
- Distance from the vehicle to the planned path.
- Current speed.
- Heading error represented by sine and cosine.
- The direction and position of a lookahead target in the vehicle's local coordinate system.
- The direction and position of the final goal.
- Distances measured by 16 lidar rays.
- Vehicle position.

The observations are normalized before being passed to the neural network.

### Reward Function

The reward function combines several components to encourage smooth and safe navigation:

- **Progress reward (`+0.5`):** Encourages movement towards a dynamically selected lookahead point on the planned path.
- **Alignment reward (`+0.15`):** Encourages the vehicle to point in the direction of the target.
- **Cross-track penalty (`-0.1`):** Penalizes deviations from the planned path. It is disabled near the destination to make the final turn easier.
- **Step penalty (`-0.02`):** Slightly penalizes every simulation step.
- **Goal reward (`+200`):** Given when the vehicle reaches the destination.
- **Collision penalty (`-35`):** Given when the vehicle hits an obstacle.

Using several reward components instead of only a final success reward makes it easier for the agent to learn useful driving behavior.

## Reinforcement Learning

The second assignment focuses on training an autonomous driving agent using **Proximal Policy Optimization (PPO)**.

The implementation is available in `algorithms/ppo.py` and `train.py`. It uses JAX, Equinox, and Optax.

### Actor-Critic Architecture

The agent uses an Actor-Critic architecture with two separate MLP networks:

- **Actor:** Predicts continuous actions for acceleration and steering.
- **Critic:** Estimates the value of the current state.

Both networks have two hidden layers with 128 neurons each and use `tanh` activation.

The Actor outputs the mean of a Gaussian action distribution. A trainable `log_std` parameter controls the exploration level.

### Training Process

Training is divided into two main stages.

**1. Behavioral Cloning (Pre-training)**

Before using reinforcement learning, the Actor learns to imitate an expert controller.

The expert uses:
- Pure pursuit for path following.
- Lidar-based potential fields for obstacle avoidance.

The Actor is trained using Mean Squared Error (MSE) between its predicted actions and the expert actions. The Critic is also initialized during this stage using the expert trajectories.

This gives the agent a reasonable starting policy before PPO training begins.

**2. PPO Fine-tuning**

After pre-training, the agent continues learning using PPO.

The implementation uses:
- The clipped PPO objective.
- Generalized Advantage Estimation (GAE).
- Discounted returns for value estimation.
- Gradient clipping and the Adam optimizer.

This stage improves the initial driving policy through interaction with the environment.

### Training Optimizations

Several JAX features are used to make training faster:

- `jax.vmap` runs 16 environments in parallel instead of processing them one by one.
- `jax.lax.scan` collects experience over multiple simulation steps without relying on a Python loop for every step.
- `equinox.filter_jit` compiles rollout collection, GAE calculation, and PPO updates.

These optimizations reduced the training time from approximately 25 minutes to around 1.5 minutes in my experiments. Actual performance depends on the hardware and JAX configuration.

## Repository Structure

```text
IAEML-assignment/
│
├── src/
│   ├── algorithms/
│   │   └── ppo.py          # PPO and Actor-Critic implementation
│   │
│   ├── env/
│   │   ├── base.py         # Base 2D environment
│   │   ├── robotaxi.py     # Vehicle dynamics and observations
│   │   └── tasks.py        # Task and map sampling
│   │
│   ├── utils/
│   │   ├── autoreset.py    # Automatic environment reset
│   │   └── ...             # Other helper modules
│   │
│   ├── main.py             # Manual control and agent evaluation
│   └── train.py            # PPO training pipeline
│
├── checkpoints/
│   └── ppo_robotaxi.eqx    # Trained agent checkpoint
│
├── instructions/           # Assignment instructions
├── videos/                 # Recorded simulation videos
│   ├── stage1_manual.mp4
|   └── stage2_ppo_agent.mp4
├── assignment_1_report.md  # Environment and vehicle dynamics
├── assignment_2_report.md  # PPO implementation and training
├── requirements.txt
└── readme.md
```

## Installation

The project was tested with Python 3.11.

Clone the repository:

```bash
git clone https://github.com/alinkaPestoletik/IAEML-assignment.git
cd IAEML-assignment
```

Install the required packages:

```bash
pip install -r requirements.txt
```

The main dependencies include:
- JAX
- Chex
- Optax
- Equinox
- Flax
- NumPy
- Pygame

## How to Run

All commands below assume that you are inside the `src` directory:

```bash
cd src
```

### 1. Manual Control

To start the environment and control the vehicle manually:

```bash
python main.py --mode manual --map_id 0
```

Use `W/A/S/D` or the arrow keys to control the vehicle and navigate towards the green destination.

To record a video:

```bash
python main.py --mode manual --map_id 0 --record_video ../videos/stage1_manual.mp4
```

### 2. Train the PPO Agent

To train the agent from scratch:

```bash
python train.py
```

The training process first performs Behavioral Cloning and then continues with PPO fine-tuning.

The best model is saved to:

```text
../checkpoints/ppo_robotaxi.eqx
```

### 3. Evaluate the Trained Agent

After training, run the environment in evaluation mode:

```bash
python main.py --mode eval --map_id 0 --checkpoint ../checkpoints/ppo_robotaxi.eqx
```

The trained agent controls the vehicle using its learned policy.

To record the evaluation:

```bash
python main.py --mode eval --map_id 0 --checkpoint ../checkpoints/ppo_robotaxi.eqx --record_video ../videos/stage2_ppo_agent.mp4
```

## Reports

More details about the implementation and experiments can be found in the assignment reports:

- [Assignment 1 — Environment and Vehicle Dynamics](assignment_1_report.md)
- [Assignment 2 — PPO and Reinforcement Learning](assignment_2_report.md)
