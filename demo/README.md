# RLVR Demo

Proof-of-concept: train a small language model with Reinforcement Learning from Verifiable Rewards on GSM8K math problems.

## Setup

```bash
pip install torch transformers datasets trl accelerate
```

## Quick start

```bash
# Run the full demo (baseline eval -> RLVR training -> post-training eval)
python rlvr_demo.py

# Just evaluate a model (no training)
python rlvr_demo.py --eval-only

# Use a different model
python rlvr_demo.py --model allenai/OLMo-1B

# Run the verifiers standalone
python verifiers.py
```

## What it does

1. Loads a small language model (default: Qwen2.5-1.5B-Instruct)
2. Evaluates it on 50 GSM8K test problems (baseline)
3. Trains it with PPO using verifiable rewards on 200 GSM8K train problems
4. Evaluates again and shows before/after comparison

## Hardware requirements

- **1B model**: ~16GB VRAM (e.g., A10, RTX 4090)
- **8B model**: ~40GB VRAM (e.g., A100 40GB)
- Training 200 steps takes ~15-30 minutes on A100

## Files

- `rlvr_demo.py` — Main demo script
- `verifiers.py` — Verifier functions (GSM8K, MATH, IF constraints)
- `eval_before_after.py` — Standalone evaluation comparison script
