import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import chex
from typing import Tuple, NamedTuple


class Transition(NamedTuple):
    obs: chex.Array
    action: chex.Array
    log_prob: chex.Array
    reward: chex.Array
    done: chex.Array
    value: chex.Array


class ActorCritic(eqx.Module):
    actor_mlp: eqx.nn.MLP
    critic_mlp: eqx.nn.MLP
    log_std: jax.Array

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128, *, key: chex.PRNGKey):
        k1, k2 = jax.random.split(key)
        actor_mlp = eqx.nn.MLP(
            in_size=obs_dim,
            out_size=action_dim,
            width_size=hidden_dim,
            depth=2,
            activation=jax.nn.tanh,
            key=k1,
        )
        last_layer = actor_mlp.layers[-1]
        new_weight = last_layer.weight * 0.01
        new_bias = jnp.array([0.5, 0.0], dtype=jnp.float32)
        actor_mlp = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias),
            actor_mlp,
            (new_weight, new_bias),
        )

        self.actor_mlp = actor_mlp
        self.critic_mlp = eqx.nn.MLP(
            in_size=obs_dim,
            out_size=1,
            width_size=hidden_dim,
            depth=2,
            activation=jax.nn.tanh,
            key=k2,
        )

        self.log_std = jnp.full((action_dim,), -0.7, dtype=jnp.float32)

    def __call__(self, obs: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
        mean = jnp.tanh(self.actor_mlp(obs))
        std = jnp.exp(jnp.clip(self.log_std, -2.0, 0.2))
        value = jnp.squeeze(self.critic_mlp(obs), axis=-1)
        return mean, std, value

    def get_action_and_value(
        self, obs: chex.Array, key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        mean, std, value = self(obs)
        noise = jax.random.normal(key, shape=mean.shape)
        action = mean + std * noise
        log_prob = self._gaussian_log_prob(mean, std, action)
        return jnp.clip(action, -1.0, 1.0), log_prob, value

    def evaluate_actions(
        self, obs: chex.Array, action: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        mean, std, value = self(obs)
        log_prob = self._gaussian_log_prob(mean, std, action)
        entropy = jnp.sum(0.5 + 0.5 * jnp.log(2 * jnp.pi) + jnp.log(std), axis=-1)
        return log_prob, entropy, value

    def act_deterministic(self, obs: chex.Array) -> chex.Array:
        mean, _, _ = self(obs)
        return jnp.clip(mean, -1.0, 1.0)

    @staticmethod
    def _gaussian_log_prob(mean: chex.Array, std: chex.Array, action: chex.Array) -> chex.Array:
        var = std ** 2
        log_scale = jnp.log(std)
        return jnp.sum(
            -((action - mean) ** 2) / (2 * var) - log_scale - 0.5 * jnp.log(2 * jnp.pi),
            axis=-1,
        )


def compute_gae(
    transitions: Transition,
    last_val: chex.Array,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[chex.Array, chex.Array]:
    def _get_advantages(carry, transition: Transition):
        gae, next_value = carry
        delta = (
            transition.reward
            + gamma * next_value * (1.0 - transition.done)
            - transition.value
        )
        gae = delta + gamma * gae_lambda * (1.0 - transition.done) * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        transitions,
        reverse=True,
    )
    returns = advantages + transitions.value
    return advantages, returns


@eqx.filter_jit
def ppo_update_epoch(
    model: ActorCritic,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch_obs: chex.Array,
    batch_actions: chex.Array,
    batch_log_probs: chex.Array,
    batch_advantages: chex.Array,
    batch_returns: chex.Array,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.005,
):
    def loss_fn(m: ActorCritic):
        new_log_prob, entropy, new_value = jax.vmap(m.evaluate_actions)(
            batch_obs, batch_actions
        )
        logratio = new_log_prob - batch_log_probs
        ratio = jnp.exp(logratio)

        adv = (batch_advantages - jnp.mean(batch_advantages)) / (
            jnp.std(batch_advantages) + 1e-8
        )

        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        pg_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

        v_loss = 0.5 * jnp.mean((new_value - batch_returns) ** 2)
        ent_loss = jnp.mean(entropy)

        total_loss = pg_loss + vf_coef * v_loss - ent_coef * ent_loss
        return total_loss, (pg_loss, v_loss, ent_loss)

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss, metrics