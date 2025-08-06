import os
import sys
import time

import jax
import jax.numpy as jnp
import pytest

# Ensure project root on path for direct module imports.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import benchmark_env


def test_compile_time_excludes_execution(monkeypatch):
    """Ensure benchmark separates compilation and execution costs.

    Previously ``time_rollout`` returned the duration of the first call as the
    "compile" time, which meant the reported value also included the time taken
    to actually run the rollout once. This test simulates a rollout where the
    first invocation is artificially slower to represent compilation, and
    verifies that the measured compilation time equals that extra cost alone.
    """

    class FakeRollout:
        def __init__(self, compile_delay: float, exec_delay: float):
            self.compile_delay = compile_delay
            self.exec_delay = exec_delay
            self.calls = 0

        def __call__(self, state, key):
            self.calls += 1
            if self.calls == 1:
                time.sleep(self.compile_delay + self.exec_delay)
            else:
                time.sleep(self.exec_delay)
            # Return JAX arrays so block_until_ready is meaningful.
            return jnp.array(state), key, jnp.array(0.0)

    # Avoid JAX backend synchronization in the test environment.
    monkeypatch.setattr(jax, "block_until_ready", lambda x: x)

    state = jnp.zeros(1)
    key = jax.random.PRNGKey(0)

    # Warm up JAX so that initialization cost does not skew timing.
    jnp.zeros(1)

    # Manually measure first and second call durations with a fresh rollout.
    manual = FakeRollout(compile_delay=0.02, exec_delay=0.01)
    start = time.perf_counter()
    manual(state, key)
    first = time.perf_counter() - start
    start = time.perf_counter()
    manual(state, key)
    second = time.perf_counter() - start

    # Now run time_rollout on another fresh rollout instance.
    rollout = FakeRollout(compile_delay=0.02, exec_delay=0.01)
    compile_time, exec_time, _, _, first_run = benchmark_env.time_rollout(rollout, state, key)

    assert exec_time == pytest.approx(second, rel=0.5)
    assert compile_time == pytest.approx(first_run - exec_time, rel=0.5)
