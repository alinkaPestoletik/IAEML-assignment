# Assignment 2: ML

**Author:** Alina Pestova

## 1. PPO Implementation

For the second part, I implemented the Proximal Policy Optimization
algorithm using JAX and Equinox.

### Model Architecture

I used an Actor-Critic architecture. Both networks are MLPs with two
hidden layers, each containing 128 neurons. I used `tanh` as the
activation function.

Since the action space is continuous, the Actor outputs the mean of a
Gaussian distribution. There is also a separate trainable `log_std`
parameter that controls the amount of exploration.

### Training Process

I divided the training into two parts, as required by the project
instructions.

1.  **Behavioral Cloning (Pre-training)**

    First, I trained the Actor to imitate an expert controller. I used
    MSE loss to compare the actions predicted by the Actor with the
    expert's actions.

    The expert controller uses pure pursuit and lidar-based potential
    fields to navigate around obstacles.

    At the same time, I trained the Critic using the discounted returns
    collected from the expert's trajectories.

2.  **PPO Fine-Tuning**

    After pre-training, I continued training the model with PPO. I used
    the clipped surrogate objective and Generalized Advantage Estimation.

## 2. JAX Optimizations

I also made some changes to speed up training. Without them, the
training process took much longer.

-   **Parallel environments:** Instead of running
    environments one by one, I used `jax.vmap` to run 16 environments at
    the same time. I applied it to the step and reset functions.

-   **Rollout collection:** I used `lax.scan` to
    collect experience over 256 steps. It allows JAX to compile the
    whole process instead of executing a Python loop for every step.

-   **JIT compilation:** I used `@equinox.filter_jit` for rollout
    collection, GAE calculation, and PPO updates.

These changes reduced the training time from about 25 minutes to around
1.5 minutes.

## 3. Problems I Encountered

Training the agent was not very stable at first. I had to fix several problems with the reward function, lidar observations, and training process

1. **Reward based on distance to the goal**

At first, the agent received a progress reward when the Euclidean distance to the goal became smaller. It caused a problem near the first obstacle: to go around it, the agent had to move slightly away from the goal, so the distance increased. As a result, the agent often got stuck.

I fixed this by using a moving lookahead_point from the planned path. The reward is now based on how much of the car's speed is directed towards this point.

2. **Lidar and coordinate systems**

The agent often crashed into walls while turning. The problem was that lidar ray directions were given in global coordinates. The network could not easily understand where an obstacle was relative to the car.

I converted the lidar directions and the vector to the goal into the car's local coordinate system. I also changed the front lidar detection to use a cone of five rays. It helped the agent understand the position of obstacles around it.

3. **Reaching the goal**

Near the finish, the lidar rays sometimes pushed the right side of the car away from the bottom curb. Because of this, the car could not move down into the green goal area and sometimes passed it.

I fixed this by disabling the cross_track_penalty when the car is within 4 cells of the goal. This gives the agent more freedom to make the final turn and reach the target.

4. **Catastrophic Forgetting**

During the first training attempts, the Critic network was not trained yet and produced unreliable value estimates. This led to large errors in GAE and changed the Actor's policy too much. As a result, the Actor lost the useful behavior it had learned before PPO.

To solve this, I split the training into two stages: Stage 2.1 (Pre-training) and Stage 2.2 (PPO Fine-Tuning). First, the agent learns to imitate a simple expert controller using Behavioral Cloning. After that, PPO starts and fine-tunes the pre-trained policy.

## 4. How to Run

To train the agent, run:

``` bash
python train.py
```

Training takes around 1.5 minutes. The best model is saved to:

`checkpoints/ppo_robotaxi.eqx`

To evaluate the trained model and record a video, run:

``` bash
python main.py --mode eval --map_id 0 --checkpoint ../checkpoints/ppo_robotaxi.eqx --record_video ../videos/stage2_ppo_agent.mp4
```
