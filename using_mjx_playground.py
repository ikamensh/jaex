import os
from pathlib import Path

import mujoco
mujoco.MjModel.from_xml_string('<mujoco/>')
print('Installation successful.')

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=100)

from datetime import datetime
import functools
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from matplotlib import pyplot as plt

from mujoco_playground import wrapper
from mujoco_playground import registry


print(registry.locomotion.ALL_ENVS)

env_name = 'Go1JoystickFlatTerrain'
env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)

print(env_cfg)


from mujoco_playground.config import locomotion_params
ppo_params = locomotion_params.brax_ppo_config(env_name)
print(ppo_params)


registry.get_domain_randomizer(env_name)

"""### Train

The policy takes 7 minutes to train on an RTX 4090.
Measured it to take 50 minutes on a GTX 1080
"""
from rich.progress import (
    Progress,
    BarColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    TextColumn,
)

# ──────────────────── configure once ────────────────────
_MAX_STEPS: int = int(ppo_params["num_timesteps"])  # set before import
_PLOT_PATH   = Path("learning_curve.png")           # final PNG

# ──────────────────── live progress bar ────────────────
_bar = Progress(
    "[progress.percentage]{task.percentage:>3.0f}%",
    BarColumn(bar_width=40),
    "• steps:",
    "{task.completed:,}/{task.total:,}",
    "• reward:",
    TextColumn("{task.fields[reward]:.3f}"),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    auto_refresh=True,
    transient=True,   # bar disappears when we stop() -> keeps log clean
)
_task = _bar.add_task("PPO", total=_MAX_STEPS, reward=0.0)
_bar.start()

# ──────────────────── data we will plot at the end ─────
_x: list[int]          = []
_y: list[float]        = []
_yerr: list[float]     = []
_time: list[datetime]  = [datetime.now()]

def progress(num_steps: int, metrics: dict[str, float]) -> None:
    """
    Terminal-only training monitor.

    Parameters
    ----------
    num_steps
        Environment steps completed so far.
    metrics
        Must contain "eval/episode_reward" and "eval/episode_reward_std".
    """
    _x.append(num_steps)
    _y.append(metrics["eval/episode_reward"])
    _yerr.append(metrics["eval/episode_reward_std"])
    _time.append(datetime.now())
    _bar.update(_task, completed=num_steps, reward=_y[-1])

def finalize_progress() -> None:
    """Call once after ppo.train() finishes."""
    _bar.stop()

    # plot and save the learning curve
    fig, ax = plt.subplots()
    ax.errorbar(_x, _y, yerr=_yerr, color="blue", capsize=2)
    ax.set_xlabel("# environment steps")
    ax.set_ylabel("reward per episode")
    ax.set_title(f"final reward = {_y[-1]:.3f}")
    ax.set_xlim(0, _MAX_STEPS * 1.25)
    fig.savefig(_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {_PLOT_PATH.resolve()}")

randomizer = registry.get_domain_randomizer(env_name)
ppo_training_params = dict(ppo_params)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
  del ppo_training_params["network_factory"]
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      **ppo_params.network_factory
  )

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    randomization_fn=randomizer,
    progress_fn=progress
)

make_inference_fn, params, metrics = train_fn(
    environment=env,
    eval_env=registry.load(env_name, config=env_cfg),
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print(f"time to jit: {_time[1] - _time[0]}")
print(f"time to train: {_time[-1] - _time[1]}")
