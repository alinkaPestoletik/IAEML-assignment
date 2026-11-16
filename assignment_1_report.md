# Assignment 1: Environment

**Author:** Alina Pestova

## 1. Vehicle Dynamics Model

In this assignment, I changed the original point-mass movement model to
a more realistic car model. I used the Kinematic Bicycle Model and
implemented it in RobotaxiEnv.

### State and Action Spaces

I added three new values to the state: car speed, heading angle, and steering angle.

The agent has two continuous actions, both in the range [-1, 1]. One
controls acceleration, and the other sets the steering angle.

### Vehicle Dynamics

At each simulation step ($\Delta t = 1/60$ s), the vehicle experiences longitudinal resistive forces:

Rolling Friction: $F_{\text{friction}} = c_{\text{fric}} \cdot v$

Aerodynamic Drag: $F_{\text{drag}} = c_{\text{drag}} \cdot v \vert{}v\vert{}
$

The net acceleration updates the velocity, ensuring it stays within the bounds of [min_speed, max_speed]. The heading is updated based on the bicycle kinematics:$$\theta_{t+1} = \theta_t + \frac{v_{t+1}}{L} \tan(\delta) \Delta t$$


Here, (L=0.65) is the wheelbase.

This model makes the car behave more like a real vehicle. It cannot move
sideways, and it needs some space to turn. Because of this, avoiding
obstacles and following the path become more difficult.

## 2. Observation Space & Reward Function

The agent receives a 24-dimensional observation vector. It contains
information about the car and its surroundings:

-   Distance from the car to the path.
-   Current speed.
-   Heading error, represented by sine and cosine.
-   Vectors pointing to the target, converted to the car's coordinate
    system.
-   Distances from 16 lidar rays.

I normalized and flattened the observations before passing them to the
agent.

### Reward Function

I decided to use several reward components instead of giving a reward
only when the agent reaches the goal. This helps the agent understand
whether its current movement is good or bad.

The reward consists of the following parts:

Progress Reward ($+0.5 \cdot v_{\text{lookahead}}$): Rewards the agent for moving along the vector pointing towards a dynamically calculated "lookahead" point on the path.

Alignment Reward ($+0.15 \cdot v \cdot \cos(\alpha)$): Encourages the agent's heading to match the target direction.

Cross-Track Penalty ($-0.1 \cdot d_{\text{path}}$): Penalizes drifting away from the dynamic programming path. To allow the agent to park smoothly, this penalty is disabled when the agent is near the goal ($x > 10.5$).

Terminal Rewards: $+200$ for reaching the goal and $-35$ for collisions.

## 3. Problems I Encountered

Creating the basic environment went quite smoothly because I could use well-known kinematics equations.

In the original code, all objects had the same fixed radius: `RADIUS=1.0`. However, when I displayed the car in Pygame, its size was only half a cell (`CELL_SIZE // 2 = 0.5`).

Because of this difference, the car sometimes collided with objects even when it looked like there was enough free space. It seemed to crash in an empty area, before it visually reached a wall.

I fixed this by adding a separate parameter, `COLLISION_RADIUS = 0.58`. This made the collision area match the car's visual size much better and allowed it to pass through narrow turns.

## 4. How to Run

To test the environment manually:

1.  Install the required packages:

    ``` bash
    pip install -r requirements.txt
    ```

2.  Run the manual mode:

    ``` bash
    python main.py --mode manual --map_id 0 --record_video ../videos/stage1_manual.mp4
    ```

3.  Use `W/A/S/D` or the arrow keys to control the car and drive to the
    green goal.

The recorded video will be saved in the videos directory.
