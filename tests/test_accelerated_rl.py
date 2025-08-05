"""High level smoke tests for the accelerated RL training script.

These tests aim to ensure that the JAX/Brax setup can run without runtime
errors. They do not check for learning quality but simply validate that the
training loop executes for a couple of steps, which is useful for catching
missing dependencies or major API changes.
"""

import os
import sys

# Ensure the project root is on the Python path so the training module can be
# imported when tests are executed from within the tests directory.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from using_brax import train
import jax


def test_train_runs_without_error():
    """Training for a handful of steps should complete without raising.

    If this test fails, it typically indicates that the environment is missing
    required dependencies, or that the Brax/JAX APIs have changed in an
    incompatible way. A successful run confirms that the accelerated training
    stack is wired correctly.
    """
    # Keep the number of timesteps small to make the test fast yet meaningful.
    train(num_timesteps=32, enable_jit=False)


def test_train_reports_device(capsys):
    """Ensure the training loop announces which device is being used.

    Prior to this change the training function ran without revealing whether
    JAX executed on CPU, GPU or TPU. This can lead to confusion when
    debugging performance issues. We run a single training step and capture
    stdout to verify that the selected backend is mentioned. Without the
    fix this assertion fails because no such message is printed.
    """

    # A minimal run keeps the test quick while still exercising the print.
    train(num_timesteps=1, enable_jit=False)

    out, _ = capsys.readouterr()

    # The print should include the backend name (e.g. "cpu" or "gpu").
    assert jax.default_backend() in out
