import os
import argparse
import jax
import equinox as eqx

from env.robotaxi import RobotaxiEnv
from utils.renderer import PygameFrontend
from utils.autoreset import AutoResetWrapper
from algorithms.ppo import ActorCritic


def main():
    parser = argparse.ArgumentParser(description="Robotaxi Environment & RL Evaluation")
    parser.add_argument(
        "--mode",
        type=str,
        default="manual",
        choices=["manual", "eval"],
        help="Режим запуска: 'manual' (управление с клавиатуры W/A/S/D) или 'eval' (обученный агент PPO)",
    )
    parser.add_argument("--map_id", type=int, default=0, help="ID карты (0, 1 или 2)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../checkpoints/ppo_robotaxi.eqx",
        help="Путь к сохраненному чекпоинту модели",
    )
    parser.add_argument(
        "--record_video",
        type=str,
        default=None,
        help="Путь для сохранения видео",
    )
    args = parser.parse_args()

    key = jax.random.PRNGKey(0)
    env_params, init_state = RobotaxiEnv.init_params(
        key=key,
        map_id=args.map_id,
        max_steps=3600,
        discretization_scale=1,
        path_length=60,
        fps=60,
        perception_radius=5.0,
        num_ray_sensors=16,
    )

    env = AutoResetWrapper(RobotaxiEnv, env_params, init_state)

    eval_mode = args.mode == "eval"
    agent_fn = None

    if eval_mode:
        sample_obs, _ = env.reset(key)
        obs_dim = sample_obs.as_vector().shape[0]
        model = ActorCritic(obs_dim=obs_dim, action_dim=2, hidden_dim=128, key=key)

        if os.path.exists(args.checkpoint):
            model = eqx.tree_deserialise_leaves(args.checkpoint, model)
            print(f"Загружен чекпоинт модели из {args.checkpoint}")
        else:
            print(f"Чекпоинт {args.checkpoint} не найден! Запустите сначала train.py.")

        @eqx.filter_jit
        def policy_fn(obs, state):
            return model.act_deterministic(obs.as_vector())

        agent_fn = policy_fn

    if args.record_video:
        os.makedirs(os.path.dirname(args.record_video), exist_ok=True)

    frontend = PygameFrontend(
        env=env,
        env_params=env_params,
        init_state=init_state,
        eval_mode=eval_mode,
        agent_fn=agent_fn,
        record_video_path=args.record_video,
        max_record_frames=3600,
    )
    frontend.run()


if __name__ == "__main__":
    main()