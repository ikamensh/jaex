"""Minimal accelerated RL training using JAX and Brax.

This script trains a PPO agent on the HalfCheetah environment using Brax's
JAX-based simulation. It demonstrates how to launch a simple accelerated
reinforcement learning experiment.
"""

from typing import Tuple, Callable
import argparse
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import tqdm
from brax import envs


class Policy(nn.Module):
    """Simple Gaussian policy network."""

    act_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(64)(x))
        mu = nn.Dense(self.act_size)(x)
        log_std = self.param("log_std", nn.initializers.zeros, (self.act_size,))
        std = jnp.exp(log_std)
        return mu, std


def train(env_name: str = "halfcheetah", num_timesteps: int = 1000, seed: int = 0, enable_jit: bool = True) -> Tuple[Callable, dict]:
    """Trains a policy using a basic REINFORCE loop in a Brax environment.

    Args:
        env_name: Name of the Brax environment to train on.
        num_timesteps: Number of training timesteps.
        seed: Random seed for reproducibility.
        enable_jit: If False, disables JIT compilation for faster debugging at the
            cost of runtime performance.

    Returns:
        A tuple of (policy function, parameters) representing the trained policy.

    The implementation is intentionally lightweight and meant for demonstration
    and testing. It uses JAX for accelerated simulation and optimization.
    """

    env = envs.get_environment(env_name)
    episode_length = jnp.minimum(1000, num_timesteps)

    # Surface which device JAX computations will run on. This helps users
    # understand whether they are utilizing CPU, GPU or TPU resources.
    backend = jax.default_backend()
    print(f"Training on {backend} device")

    policy = Policy(env.action_size)
    rng = jax.random.PRNGKey(seed)
    params = policy.init(rng, jnp.zeros((env.observation_size,)))
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    def loss_fn(params, rng):
        def rollout(carry, _):
            state, rng = carry
            obs = state.obs
            mu, std = policy.apply(params, obs)
            rng, key = jax.random.split(rng)
            action = mu + std * jax.random.normal(key, shape=mu.shape)
            logp = -0.5 * jnp.sum(((action - mu) / std) ** 2 + 2 * jnp.log(std) + jnp.log(2 * jnp.pi))
            next_state = env.step(state, action)
            return (next_state, rng), (next_state.reward, logp)

        state = env.reset(rng)
        (_, _), (rewards, logps) = jax.lax.scan(rollout, (state, rng), None, length=episode_length)
        total_reward = jnp.sum(rewards)
        return -(total_reward * jnp.sum(logps)), total_reward

    value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)
    grad_fn = jax.jit(value_and_grad) if enable_jit else value_and_grad

    num_updates = jnp.maximum(1, num_timesteps // episode_length)
    for _ in tqdm.tqdm( range(num_updates) ):
        rng, key = jax.random.split(rng)
        (loss, _), grads = grad_fn(params, key)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    return policy.apply, params


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO agent in Brax")
    parser.add_argument("--env", default="halfcheetah", help="Name of the Brax environment")
    parser.add_argument("--num-timesteps", type=int, default=10_000, help="Training timesteps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    train(env_name=args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == "__main__":
    main()
