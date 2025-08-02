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
python accelerated_rl.py --env halfcheetah --num-timesteps 1000
```

## Testing

Run the smoke test to ensure the training script executes without errors:

```bash
pytest
```
