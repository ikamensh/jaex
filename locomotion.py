import os
import mujoco
mujoco.MjModel.from_xml_string('<mujoco/>')


print('Installation successful.')

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import numpy as np

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

# @title Import MuJoCo, MJX, and Brax
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
"""
import matplotlib

# ──────────────────── choose a backend ────────────────────
# If you ssh into a headless box with no $DISPLAY, fall back to a non-GUI backend
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")  # writes PNGs; no live window
else:                      # macOS / local run
    matplotlib.use("TkAgg")  # any GUI backend is fine

plt.ion()  # interactive mode ⟹ fig.canvas.draw() updates in place

# ──────────────────── globals used by progress() ────────────────────
x_data: list[int] = []
y_data: list[float] = []
y_dataerr: list[float] = []
times: list[datetime] = [datetime.now()]

_fig, _ax = plt.subplots()
_line = _ax.errorbar([], [], yerr=[], color="blue", capsize=2)[0]  # keep handle
def progress(num_steps: int, metrics: dict[str, float]) -> None:
    """
    Live-update the learning curve inside a terminal script.

    Parameters
    ----------
    num_steps
        Environment steps completed so far.
    metrics
        Dict coming from the PPO training loop.
        Expected keys: "eval/episode_reward", "eval/episode_reward_std".
    """
    # accumulate series
    print(num_steps)
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])

    # redraw
    _ax.clear()
    _ax.set_xlim(0, int(ppo_params["num_timesteps"] * 1.25))
    _ax.set_xlabel("# environment steps")
    _ax.set_ylabel("reward per episode")
    _ax.set_title(f"y = {y_data[-1]:.3f}")
    _ax.errorbar(x_data, y_data, yerr=y_dataerr, color="blue", capsize=2)

    _fig.canvas.draw()
    _fig.canvas.flush_events()          # guarantees on-screen refresh
    # Optional: persist a frame every N calls for later inspection
    # if len(x_data) % 50 == 0:
    #     _fig.savefig(f"curve_{num_steps:07d}.png", dpi=150)

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
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")
