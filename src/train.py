import os
import time
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from env.robotaxi import RobotaxiEnv
from utils.autoreset import AutoResetWrapper
from algorithms.ppo import ActorCritic, Transition, compute_gae, ppo_update_epoch


def train(
    map_id: int = 0,
    num_envs: int = 16,
    num_steps: int = 450,
    pretrain_steps: int = 200,
    total_updates: int = 60,
    num_epochs: int = 4,
    num_minibatches: int = 4,
    learning_rate: float = 2e-4,
    seed: int = 42,
    checkpoint_path: str = "../checkpoints/ppo_robotaxi.eqx",
):
    print(f"=== Инициализация RobotaxiEnv (map_id={map_id}, num_envs={num_envs}) ===")
    key = jax.random.PRNGKey(seed)
    key, env_key, model_key = jax.random.split(key, 3)

    env_params, init_state = RobotaxiEnv.init_params(
        key=env_key,
        map_id=map_id,
        max_steps=800,
        discretization_scale=1,
        path_length=60,
        fps=60,
        perception_radius=5.0,
        num_ray_sensors=16,
    )

    env = AutoResetWrapper(RobotaxiEnv, env_params, init_state)

    sample_obs, _ = env.reset(env_key)
    obs_dim = sample_obs.as_vector().shape[0]
    action_dim = 2
    print(f"Размерность наблюдений (obs_dim): {obs_dim}, Размерность действий (action_dim): {action_dim}")

    model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=128, key=model_key)
    model = eqx.tree_at(lambda m: m.log_std, model, jnp.full((action_dim,), -1.2, dtype=jnp.float32))

    # PRE-TRAINING
    print("\nЗапуск быстрого Pre-training на навигационных траекториях")
    pre_opt = optax.adam(3e-3)
    pre_opt_state = pre_opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def collect_expert_rollout(obs_batch, states, rng):
        def step_fn(carry, _):
            curr_obs, curr_states, k = carry
            k, step_k = jax.random.split(k)
            expert_acts = jax.vmap(RobotaxiEnv.expert_policy)(curr_obs)
            noisy_acts = jnp.clip(
                expert_acts + 0.12 * jax.random.normal(k, expert_acts.shape), -1.0, 1.0
            )
            step_keys = jax.random.split(step_k, num_envs)
            next_obs, next_states, _, _, infos = jax.vmap(env.step)(
                step_keys, curr_states, noisy_acts
            )
            obs_vec = jax.vmap(lambda o: o.as_vector())(curr_obs)
            return (next_obs, next_states, k), (obs_vec, expert_acts, infos["goal_reached"])

        (_, _, _), (obs_vecs, target_acts, goals) = jax.lax.scan(
            step_fn, (obs_batch, states, rng), None, length=450
        )
        return obs_vecs.reshape(-1, obs_dim), target_acts.reshape(-1, action_dim), jnp.sum(goals)

    @eqx.filter_jit
    def bc_train_step(m: ActorCritic, opt_st, x_batch, y_batch):
        def loss_fn(net: ActorCritic):
            pred_actions = jax.vmap(net.act_deterministic)(x_batch)
            return jnp.mean((pred_actions - y_batch) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
        updates, new_opt_st = pre_opt.update(grads, opt_st, m)
        new_m = eqx.apply_updates(m, updates)
        return new_m, new_opt_st, loss

    reset_keys = jax.random.split(env_key, num_envs)
    obs_batch, env_states = jax.vmap(env.reset)(reset_keys)

    key, exp_key = jax.random.split(key)
    exp_obs, exp_acts, exp_goals = collect_expert_rollout(obs_batch, env_states, exp_key)

    for step in range(1, pretrain_steps + 1):
        model, pre_opt_state, bc_loss = bc_train_step(model, pre_opt_state, exp_obs, exp_acts)
        if step % 50 == 0 or step == 1:
            print(f"  Pre-train step {step:3d}/{pretrain_steps} | BC MSE Loss: {float(bc_loss):.5f}")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    eqx.tree_serialise_leaves(checkpoint_path, model)

    # REINFORCEMENT LEARNING
    print("\nЗапуск обучения PPO")
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=learning_rate, eps=1e-5),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    obs_batch, env_states = jax.vmap(env.reset)(reset_keys)
    obs_vec_batch = jax.vmap(lambda o: o.as_vector())(obs_batch)

    @eqx.filter_jit
    def collect_rollout(model: ActorCritic, obs_vec, states, key):
        def step_fn(carry, _):
            curr_obs_vec, curr_states, rng = carry
            rng, act_rng, step_rng = jax.random.split(rng, 3)

            act_keys = jax.random.split(act_rng, num_envs)
            actions, log_probs, values = jax.vmap(model.get_action_and_value)(
                curr_obs_vec, act_keys
            )

            step_keys = jax.random.split(step_rng, num_envs)
            next_obs, next_states, rewards, dones, infos = jax.vmap(env.step)(
                step_keys, curr_states, actions
            )
            next_obs_vec = jax.vmap(lambda o: o.as_vector())(next_obs)

            transition = Transition(
                obs=curr_obs_vec,
                action=actions,
                log_prob=log_probs,
                reward=rewards,
                done=dones.astype(jnp.float32),
                value=values,
            )
            return (next_obs_vec, next_states, rng), (transition, infos)

        (final_obs_vec, final_states, final_key), (transitions, infos) = jax.lax.scan(
            step_fn, (obs_vec, states, key), None, length=num_steps
        )
        _, _, last_val = jax.vmap(model)(final_obs_vec)
        return final_obs_vec, final_states, final_key, transitions, last_val, infos

    start_time = time.time()
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches
    best_goals = -1
    best_reward = -1e9

    for update in range(1, total_updates + 1):
        obs_vec_batch, env_states, key, transitions, last_val, infos = collect_rollout(
            model, obs_vec_batch, env_states, key
        )

        advantages, returns = jax.vmap(
            compute_gae, in_axes=(1, 0), out_axes=1
        )(transitions, last_val)

        b_obs = transitions.obs.reshape(-1, obs_dim)
        b_actions = transitions.action.reshape(-1, action_dim)
        b_log_probs = transitions.log_prob.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        mean_rew = float(jnp.mean(jnp.sum(transitions.reward, axis=0)))
        goals_reached = int(jnp.sum(infos["goal_reached"]))
        collisions = int(jnp.sum(infos["collision"]))

        if goals_reached > best_goals or (goals_reached == best_goals and mean_rew > best_reward):
            best_goals = goals_reached
            best_reward = mean_rew
            eqx.tree_serialise_leaves(checkpoint_path, model)

        curr_clip = 0.05 if update <= 5 else 0.15

        for _ in range(num_epochs):
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, batch_size)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = perm[start : start + minibatch_size]
                model, opt_state, loss, _ = ppo_update_epoch(
                    model,
                    opt_state,
                    optimizer,
                    b_obs[mb_idx],
                    b_actions[mb_idx],
                    b_log_probs[mb_idx],
                    b_advantages[mb_idx],
                    b_returns[mb_idx],
                    clip_eps=curr_clip,
                    ent_coef=0.001,
                )

        if update % 5 == 0 or update == 1:
            elapsed = time.time() - start_time
            print(
                f"Update {update:2d}/{total_updates} | "
                f"Mean Reward: {mean_rew:7.2f} | "
                f"Goals: {goals_reached:2d} | "
                f"Collisions: {collisions:2d} | "
                f"Time: {elapsed:.1f}s"
            )

    print(f"\nОбучение успешно завершено! Лучший чекпоинт (Goals={best_goals}) сохранен в: {checkpoint_path}")


if __name__ == "__main__":
    train()