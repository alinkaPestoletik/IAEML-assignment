import jax
from jax import numpy as jnp
import equinox
import chex
from functools import partial
from typing import Tuple, Dict, Any

from env.base import (
    BaseEnv,
    BaseEnvParams,
    BaseEnvState,
    BaseEnvObservation,
    RADIUS,
    NUM_RAY_SENSORS,
    FOV,
    FPS,
)
from env.tasks import BaseTaskSampler
from utils import utils

COLLISION_RADIUS = 0.58
GOAL_TRIGGER_RADIUS = 1.4


class RobotaxiEnvParams(BaseEnvParams):
    wheelbase: float = 0.65
    max_acceleration: float = 10.0
    max_steering_angle: float = 0.95
    max_speed: float = 4.5
    min_speed: float = -1.0
    friction_coeff: float = 1.5
    drag_coeff: float = 0.05


class RobotaxiEnvState(BaseEnvState):
    speed: chex.Scalar
    heading: chex.Scalar
    steering_angle: chex.Scalar


class RobotaxiObservation(BaseEnvObservation):
    speed: chex.Array
    heading_error: chex.Array
    lookahead_ego: chex.Array
    goal_ego: chex.Array
    ray_distances: chex.Array
    agent_pos: chex.Array

    def as_vector(self) -> chex.Array:
        return jnp.concatenate(
            [
                jnp.atleast_1d(self.distance_to_path / 3.0),
                jnp.atleast_1d(self.speed / 4.5),
                self.heading_error,
                self.lookahead_ego / 4.0,
                self.goal_ego / 15.0,
                self.ray_distances / 5.0,
                self.agent_pos / 20.0,
            ],
            axis=-1,
        )


class RobotaxiEnv(BaseEnv):

    @staticmethod
    def init_params(
        key: chex.PRNGKey,
        map_id: int = 0,
        max_steps: int = 1200,
        path_length: int = 60,
        discretization_scale: int = 1,
        perception_radius: float = 5.0,
        num_ray_sensors: int = NUM_RAY_SENSORS,
        fov: float = FOV,
        fps: int = FPS,
    ) -> Tuple[RobotaxiEnvParams, RobotaxiEnvState]:

        if discretization_scale != 1:
            raise ValueError("discretization_scale != 1 is not implemented!")

        task = BaseTaskSampler(map_id)

        env_params = RobotaxiEnvParams(
            max_steps_in_episode=max_steps,
            fps=fps,
            step_size=1.0 / fps,
            map_height_width=(task.height, task.width),
            nav_grid_shape_dtype=(
                (task.height * discretization_scale, task.width * discretization_scale),
                jnp.float32,
            ),
            path_array_shape_dtype=((path_length, 2), jnp.float32),
            discretization_scale=discretization_scale,
            perception_radius=perception_radius,
            agent_FOV=fov,
            num_ray_sensors=num_ray_sensors,
        )

        shape, dtype = env_params.path_array_shape_dtype
        convert_to_world_f = partial(
            utils.convert_to_world_view, map_shape=env_params.map_height_width
        )
        convert_to_world_vmap = jax.vmap(convert_to_world_f, in_axes=(0))

        forward_dir = task.agent_forward_dir.astype(jnp.float32)
        forward_dir = forward_dir / (jnp.linalg.norm(forward_dir) + 1e-8)
        init_heading = jnp.arctan2(forward_dir[1], forward_dir[0])

        init_state = RobotaxiEnvState(
            time=jnp.asarray(0, dtype=jnp.int32),
            goal_pos=convert_to_world_f(task.goal_pos).astype(jnp.float32),
            agent_pos=convert_to_world_f(task.agent_pos).astype(jnp.float32),
            agent_forward_dir=forward_dir,
            static_obstacles=convert_to_world_vmap(task.static_obstacles).astype(jnp.float32),
            kinematic_obstacles=convert_to_world_vmap(task.kinematic_obstacles).astype(jnp.float32),
            kinematic_obst_velocities=task.kinematic_obst_velocities.astype(jnp.float32),
            path_array=jnp.zeros(shape=shape, dtype=dtype),
            speed=jnp.asarray(0.0, dtype=jnp.float32),
            heading=init_heading,
            steering_angle=jnp.asarray(0.0, dtype=jnp.float32),
        )

        obstacles = jnp.concatenate(
            [init_state.static_obstacles, init_state.kinematic_obstacles], axis=0
        )
        path_array = BaseEnv._find_path(
            init_state.agent_pos, init_state.goal_pos, obstacles, env_params
        )
        init_state = equinox.tree_at(lambda t: t.path_array, init_state, path_array)

        return env_params, init_state

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def reset(
        key: chex.PRNGKey, env_params: RobotaxiEnvParams, init_state: RobotaxiEnvState
    ) -> Tuple[RobotaxiObservation, RobotaxiEnvState]:
        state = init_state
        obs = RobotaxiEnv.get_observation(state, env_params)
        return obs, state

    @staticmethod
    def _safe_lookahead_point(
        path_array: chex.Array, agent_pos: chex.Array, goal_pos: chex.Array
    ) -> Tuple[chex.Array, chex.Scalar]:
        dists = jnp.linalg.norm(path_array - agent_pos, axis=1)
        closest_idx = jnp.argmin(dists)
        max_idx = path_array.shape[0] - 1
        lookahead_idx = jnp.minimum(closest_idx + 3, max_idx)
        pt = path_array.at[lookahead_idx].get()

        near_pillar_top = jnp.logical_and(
            pt[0] >= 3.2, jnp.logical_and(pt[0] <= 5.8, pt[1] >= 4.2)
        )
        y_safe = jnp.where(near_pillar_top, jnp.maximum(pt[1], 5.35), pt[1])

        in_obstacle_zone = jnp.logical_and(
            pt[0] > 5.5, jnp.logical_and(pt[0] <= 11.0, y_safe < 2.5)
        )
        y_safe = jnp.where(in_obstacle_zone, jnp.maximum(y_safe, 2.2), y_safe)

        safe_pt = jnp.array([pt[0], y_safe], dtype=jnp.float32)

        safe_goal_pt = jnp.array([goal_pos[0], goal_pos[1]], dtype=jnp.float32)
        final_target_pt = jnp.where(agent_pos[0] > 10.5, safe_goal_pt, safe_pt)

        return final_target_pt, dists[closest_idx]

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def step(
        key: chex.PRNGKey,
        env_state: RobotaxiEnvState,
        action: chex.Array,
        env_params: RobotaxiEnvParams,
    ) -> Tuple[RobotaxiObservation, RobotaxiEnvState, chex.Scalar, chex.Array, Dict[Any, Any]]:
        dt = env_params.step_size

        action = jnp.clip(action, -1.0, 1.0)
        accel_cmd = action[0] * env_params.max_acceleration
        steering_angle = action[1] * env_params.max_steering_angle

        v = env_state.speed
        friction_force = env_params.friction_coeff * v
        drag_force = env_params.drag_coeff * v * jnp.abs(v)
        net_accel = accel_cmd - friction_force - drag_force

        new_speed = jnp.clip(
            v + net_accel * dt, env_params.min_speed, env_params.max_speed
        )

        angular_velocity = (new_speed / env_params.wheelbase) * jnp.tan(steering_angle)
        new_heading = env_state.heading + angular_velocity * dt
        new_heading = jnp.arctan2(jnp.sin(new_heading), jnp.cos(new_heading))

        new_forward_dir = jnp.array(
            [jnp.cos(new_heading), jnp.sin(new_heading)], dtype=jnp.float32
        )
        new_agent_pos = env_state.agent_pos + new_speed * new_forward_dir * dt

        kinematic_obstacles = BaseEnv._move_kinematic_obstacles(env_state, env_params)
        obstacles = jnp.concatenate(
            [env_state.static_obstacles, kinematic_obstacles], axis=0
        )

        goal_done = BaseEnv._check_goal(
            new_agent_pos, env_state.goal_pos, circle_radius=GOAL_TRIGGER_RADIUS
        )
        collision_done = BaseEnv._check_collisions(
            new_agent_pos, obstacles, circle_radius=COLLISION_RADIUS
        )
        new_time = env_state.time + 1
        time_done = new_time >= env_params.max_steps_in_episode
        done = jnp.logical_or(goal_done, jnp.logical_or(collision_done, time_done))

        lookahead_pt, _ = RobotaxiEnv._safe_lookahead_point(
            env_state.path_array, env_state.agent_pos, env_state.goal_pos
        )

        vec_to_target = lookahead_pt - env_state.agent_pos
        dir_to_target = vec_to_target / (jnp.linalg.norm(vec_to_target) + 1e-6)
        displacement = new_agent_pos - env_state.agent_pos
        velocity_along_lookahead = jnp.dot(displacement, dir_to_target) / dt

        pred = jnp.mod(new_time, env_params.fps) == 0
        path_array = jax.lax.cond(
            pred,
            lambda _: BaseEnv._find_path(
                new_agent_pos, env_state.goal_pos, obstacles, env_params
            ),
            lambda _: env_state.path_array,
            None,
        )

        new_state = RobotaxiEnvState(
            time=new_time,
            goal_pos=env_state.goal_pos,
            agent_pos=new_agent_pos,
            agent_forward_dir=new_forward_dir,
            static_obstacles=env_state.static_obstacles,
            kinematic_obstacles=kinematic_obstacles,
            kinematic_obst_velocities=env_state.kinematic_obst_velocities,
            path_array=path_array,
            speed=new_speed,
            heading=new_heading,
            steering_angle=steering_angle,
        )

        obs = RobotaxiEnv.get_observation(new_state, env_params)

        progress_reward = 0.5 * velocity_along_lookahead
        alignment_reward = 0.15 * new_speed * obs.heading_error[1]
        cross_track_penalty = jnp.where(
            new_agent_pos[0] > 10.5, 0.0, -0.1 * obs.distance_to_path
        )
        step_penalty = -0.02

        terminal_reward = (
            jnp.where(goal_done, 200.0, 0.0)
            + jnp.where(collision_done, -35.0, 0.0)
        )

        reward = (
            progress_reward
            + alignment_reward
            + cross_track_penalty
            + step_penalty
            + terminal_reward
        )

        info = {
            "time": new_time,
            "speed": new_speed,
            "distance_to_path": obs.distance_to_path,
            "direction_of_path": obs.direction_of_path,
            "goal_reached": goal_done,
            "collision": collision_done,
        }

        return obs, new_state, reward, done, info

    @staticmethod
    def get_observation(
        env_state: RobotaxiEnvState, env_params: RobotaxiEnvParams
    ) -> RobotaxiObservation:
        lookahead_pt, distance_to_path = RobotaxiEnv._safe_lookahead_point(
            env_state.path_array, env_state.agent_pos, env_state.goal_pos
        )

        target_vec = lookahead_pt - env_state.agent_pos
        goal_vec = env_state.goal_pos - env_state.agent_pos
        target_dir = target_vec / (jnp.linalg.norm(target_vec) + 1e-6)

        fwd = env_state.agent_forward_dir
        left = jnp.array([-fwd[1], fwd[0]], dtype=jnp.float32)

        sin_err = jnp.dot(target_dir, left)
        cos_err = jnp.dot(target_dir, fwd)
        heading_error = jnp.array([sin_err, cos_err], dtype=jnp.float32)

        lookahead_ego = jnp.array(
            [jnp.dot(target_vec, fwd), jnp.dot(target_vec, left)], dtype=jnp.float32
        )
        goal_ego = jnp.array(
            [jnp.dot(goal_vec, fwd), jnp.dot(goal_vec, left)], dtype=jnp.float32
        )

        obstacles = jnp.concatenate(
            [env_state.static_obstacles, env_state.kinematic_obstacles], axis=0
        )
        rays_unit, ray_perceptions = BaseEnv._collision_ray_intersections(
            env_state.agent_pos, env_state.agent_forward_dir, obstacles, env_params
        )
        rays_world = rays_unit * ray_perceptions[:, None]

        return RobotaxiObservation(
            distance_to_path=distance_to_path,
            direction_of_path=target_dir,
            collision_rays=rays_world,
            speed=jnp.atleast_1d(env_state.speed),
            heading_error=heading_error,
            lookahead_ego=lookahead_ego,
            goal_ego=goal_ego,
            ray_distances=ray_perceptions,
            agent_pos=env_state.agent_pos,
        )

    @staticmethod
    def expert_policy(obs: RobotaxiObservation) -> chex.Array:
        sin_err = obs.heading_error[0]
        cos_err = obs.heading_error[1]
        speed = obs.speed[0]
        dist_to_goal = jnp.linalg.norm(obs.goal_ego)
        x_pos, y_pos = obs.agent_pos[0], obs.agent_pos[1]

        path_steer = 2.6 * sin_err + jnp.where(cos_err < 0.0, jnp.sign(sin_err) * 0.9, 0.0)

        rays = obs.ray_distances
        right_front_danger = jnp.sum(jnp.maximum(0.0, 2.4 - rays[2:8]))
        left_front_danger = jnp.sum(jnp.maximum(0.0, 2.0 - rays[9:12]))
        front_danger = jnp.maximum(0.0, 2.5 - rays[8])

        raw_avoid_steer = 0.55 * (right_front_danger - left_front_danger)

        floor_guard = jnp.where(y_pos < 0.65, 0.8 * (0.65 - y_pos), 0.0)
        avoid_steer = jnp.where(x_pos > 11.5, floor_guard, raw_avoid_steer)

        steer = jnp.clip(path_steer + avoid_steer, -1.0, 1.0)

        target_speed = jnp.where(
            dist_to_goal < 4.0,
            1.8,
            jnp.where(
                front_danger > 0.8,
                1.2,
                jnp.where(jnp.abs(sin_err) > 0.45, 2.2, 3.8),
            ),
        )
        accel = jnp.clip(1.5 * (target_speed - speed), -1.0, 1.0)

        return jnp.array([accel, steer], dtype=jnp.float32)