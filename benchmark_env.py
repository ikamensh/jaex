"""Benchmark Brax environment rollout speed with random actions."""

import argparse
import time

import jax
import jax.numpy as jnp
from brax import envs


def time_rollout(rollout, state, key):
    """Times a single compilation + execution and a subsequent execution.

    Parameters
    ----------
    rollout: Callable
        A jitted rollout function accepting ``(state, key)``.
    state: Any
        Initial environment state.
    key: jax.random.PRNGKey
        Random key for action sampling.

    Returns
    -------
    compile_time: float
        Time for the first rollout call which includes compilation.
    exec_time: float
        Time for a subsequent rollout call after compilation.
    state, key: Any, jax.random.PRNGKey
        Updated state and key after the second rollout.
    """
    # First call triggers compilation and runs once.
    t0 = time.perf_counter()
    state, key, _ = rollout(state, key)
    jax.block_until_ready(state)
    first_run = time.perf_counter() - t0

    # Second call measures steady-state execution time.
    t1 = time.perf_counter()
    state, key, _ = rollout(state, key)
    jax.block_until_ready(state)
    exec_time = time.perf_counter() - t1

    # Compilation is the extra cost paid during the first run.
    compile_time = first_run - exec_time

    return compile_time, exec_time, state, key, first_run


def make_rollout(env, num_steps):
    """Creates a jitted rollout function for the environment."""

    def rollout(state, key):
        def step_fn(carry, _):
            state, key = carry
            key, subkey = jax.random.split(key)
            action = jax.random.uniform(subkey, (env.action_size,), minval=-1.0, maxval=1.0)
            state = env.step(state, action)
            return (state, key), state.reward

        (state, key), rewards = jax.lax.scan(step_fn, (state, key), None, length=num_steps)
        return state, key, jnp.sum(rewards)

    return jax.jit(rollout)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Brax rollout speed with random actions")
    parser.add_argument("--env", default="halfcheetah", help="Name of the Brax environment")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to roll out")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    backend = jax.default_backend()
    print(f"Benchmarking {args.env} for {args.steps} steps on {backend}")

    env = envs.get_environment(args.env)
    key = jax.random.PRNGKey(args.seed)
    state = env.reset(key)

    rollout = make_rollout(env, args.steps)

    compile_time, exec_time, state, key, _ = time_rollout(rollout, state, key)

    steps_per_sec = args.steps / exec_time if exec_time > 0 else float("inf")
    print(f"Compilation time: {compile_time:.4f} s")
    print(f"Execution time: {exec_time:.4f} s")
    print(f"Throughput: {steps_per_sec:.2f} steps/sec")


if __name__ == "__main__":
    main()
