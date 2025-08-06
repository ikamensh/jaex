# jaex

try to put together a fast rl pipeline

## Installation

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the training script which uses JAX and Brax to train a tiny policy on the
HalfCheetah environment:

```bash
python using_brax.py --env halfcheetah --num-timesteps 1000
```

### Benchmarking

Measure raw environment rollout speed using random actions. By default the
script runs the HalfCheetah environment and reports compilation and execution
times along with throughput:

```bash
python benchmark_env.py --env halfcheetah --steps 1000
```

## Testing

Run the smoke test to ensure the training script executes without errors:

```bash
pytest
```
