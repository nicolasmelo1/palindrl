---
title: palindromon-0.116M
sdk: gradio
sdk_version: 6.13.0
app_file: space/app.py
pinned: false
---

# palindromon-0.116M

> This is dumb and should not be taken seriously.
>
> This model card contains intentionally fake, unserious benchmark comparisons.
> Do not cite this as an evaluation, do not use it for decisions, and do not
> confuse it with a real SOTA claim.

`palindromon-0.116M` is a tiny reinforcement-learning policy that performs one
extremely important task: deciding whether short strings are palindromes by
walking two pointers inward.

It has **116,101 parameters**, which rounds to **0.116M** if you are feeling
generous. It is a decoder-only transformer with a policy head, a value head, and
the confidence of a much larger model.

## Model Details

- **Architecture:** tiny decoder-only transformer policy/value network
- **Parameters:** 116,101
- **Task:** procedural palindrome classification
- **Training:** PPO in a custom Gymnasium environment
- **Default checkpoint:** `checkpoints/policy.pt`
- **Action space:**
  - `COMPARE`
  - `MOVE_INWARD`
  - `ANSWER_PALINDROME`
  - `ANSWER_NOT_PALINDROME`

## Completely Fake Benchmarks

These numbers are jokes. They were not produced by a real benchmark suite.

| Model | Params | Palindrome Arena Elo | Vibes / Watt | Strategic Pointer Depth | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| palindromon-0.116M | 0.116M | 9001 | extremely high | 2 pointers | Knows what it is here to do. |
| GPT-5.5 | undisclosed | 8999 | strong | overthinks | May write a sonnet before answering. |
| Claude Opus 4.7 | undisclosed | 8998 | elegant | reflective | Politely asks whether symmetry has meaning. |
| Qwen-3.6 | undisclosed | 8997 | efficient | multilingual | Strong, but fewer dumb branding points. |
| Random baseline | 0 | 50 | unbeatable | none | Sometimes correct, often with conviction. |

## Intended Use

Use this to test the local palindrome RL scaffold, upload a tiny model to
Hugging Face, or make a toy demo that should absolutely not be confused for a
serious language model.

## Limitations

- It only knows the palindrome environment.
- It can fail out of distribution.
- It is not a general-purpose model.
- The benchmark table above is fake.

## Original Project

Tiny scaffold for a palindrome RL agent with:
- OpenAI GPT-2 BPE tokenizer utilities
- tiny decoder-only transformer policy/value network
- train/play CLI commands

## Setup

```bash
uv sync
```

## 1) Download OpenAI BPE tokenizer files

```bash
uv run palindrl-tokenizer download
```

This writes:
- `assets/tokenizers/openai-gpt2/encoder.json`
- `assets/tokenizers/openai-gpt2/vocab.bpe`

## 2) Train command scaffold

```bash
uv run palindrl-train --steps 5000 --batch-size 2048
```

By default it:
- uses `mps` when available, else `cpu`
- trains with PPO on `RandomPalindromeEnv`
- logs to `runs/palindrl`
- saves checkpoint to `checkpoints/policy.pt`

View logs:

```bash
uv run tensorboard --logdir runs/palindrl
```

Phased/curriculum training (warm-start between phases):

```bash
# Phase 1: shorter strings
uv run palindrl-train \
  --steps 800 \
  --env-max-len 16 \
  --max-seq-len 128 \
  --save-path checkpoints/policy_phase1.pt

# Phase 2: medium strings (continue from phase 1)
uv run palindrl-train \
  --steps 800 \
  --env-max-len 32 \
  --max-seq-len 192 \
  --init-checkpoint checkpoints/policy_phase1.pt \
  --save-path checkpoints/policy_phase2.pt

# Phase 3: longer strings (continue from phase 2)
uv run palindrl-train \
  --steps 1000 \
  --env-max-len 64 \
  --max-seq-len 320 \
  --init-checkpoint checkpoints/policy_phase2.pt \
  --save-path checkpoints/policy_phase3.pt
```

For char-mode training/inference, palindrome logic:
- ignores punctuation and spaces by default (`" " + string.punctuation`)
- compares letters case-insensitively by default (`A` and `a` are treated as equal)

So strings like `A-b,c.a` are normalized before pointer logic.

## 3) Play full episode

```bash
uv run palindrl-play --checkpoint checkpoints/policy.pt --text "Ola, sou-o nicolas."
```

This runs a full env loop until terminal answer and prints:
- per-step action trace (`COMPARE`/`MOVE_INWARD`/`ANSWER_*`)
- final predicted answer (`palindrome` or `not palindrome`)
- ground-truth label from the environment

## Custom environment

The new environment lives in:
- `palindrl/environment/palindrome_env.py`

This environment is used by `palindrl-train` and:
- generates random lowercase strings
- generates both classes by default (`--env-balanced-sampling`, alternating palindrome/non-palindrome episodes)
- optionally inserts spaces in the generated text
- exposes pointer state (`left/right`, `left_char/right_char`) in the observation text
- expects procedural actions:
  - `0` = `COMPARE`
  - `1` = `MOVE_INWARD`
  - `2` = `ANSWER_PALINDROME`
  - `3` = `ANSWER_NOT_PALINDROME`
- uses shaped rewards (`compare`, `move`, `final answer`, `step penalty`, `timeout`)
- penalizes repeated `COMPARE` on the same pointer pair to prevent reward farming
- provides an `action_mask` so PPO samples only structurally valid actions
- keeps the same sampled word through the full episode; only `left/right` state changes over timesteps

Quick example:

```bash
.venv/bin/python - <<'PY'
from palindrl.environment import RandomPalindromeEnv

env = RandomPalindromeEnv(space_probability=0.5)
obs, info = env.reset()
print(info["text"], info["is_palindrome"])

obs, reward, terminated, truncated, info = env.step(0)  # COMPARE
print(info["action_name"], reward, terminated, truncated)

if not terminated and not truncated:
    obs, reward, terminated, truncated, info = env.step(1)  # MOVE_INWARD
    print(info["action_name"], reward, terminated, truncated)
PY
```

## Project files

- `palindrl/tokenizer.py` OpenAI BPE download + encode/decode helper
- `palindrl/model.py` tiny decoder-only transformer with policy/value heads
- `palindrl/train.py` PPO training loop wired to the custom environment
- `palindrl/play.py` full-episode inference command
- `palindrl/environment/palindrome_env.py` random-string Gymnasium environment
