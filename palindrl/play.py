from __future__ import annotations

import argparse
import inspect
import string
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from palindrl.environment import RandomPalindromeEnv, normalize_text
from palindrl.model import TinyTransformerConfig, TinyTransformerPolicy


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a full palindrome episode loop and emit final answer."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/policy.pt"),
        help="Checkpoint created by palindrl-train",
    )
    parser.add_argument("--text", type=str, required=True, help="Raw input phrase.")
    parser.add_argument(
        "--ignored-chars",
        type=str,
        default=None,
        help="Override ignored chars for normalization. Default comes from checkpoint/env.",
    )
    parser.add_argument(
        "--sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sample actions from policy instead of greedy argmax.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser


def build_env_kwargs(
    env_config: dict, text: str, ignored_chars_override: str | None
) -> dict:
    signature = inspect.signature(RandomPalindromeEnv.__init__)
    kwargs: dict = {}
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if name in env_config:
            kwargs[name] = env_config[name]
        elif parameter.default is not inspect.Parameter.empty:
            kwargs[name] = parameter.default

    if ignored_chars_override is not None:
        kwargs["ignored_characters"] = ignored_chars_override

    ignored_chars = set(str(kwargs.get("ignored_characters", " " + string.punctuation)))
    normalized_text = normalize_text(text, ignored_chars)
    if not normalized_text:
        raise ValueError(
            "Input text becomes empty after ignoring characters. "
            "Pass --ignored-chars with a less strict set."
        )

    # Ensure observation limits are large enough for user-provided text.
    kwargs["max_len"] = max(int(kwargs.get("max_len", 12)), len(normalized_text))
    kwargs["balanced_sampling"] = False
    kwargs["space_probability"] = 0.0
    return kwargs


def prepare_model_input(
    obs: dict[str, np.ndarray],
    vocab_size: int,
    max_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids_np = obs["input_ids"][:max_seq_len].astype(np.int64, copy=True)
    oov = input_ids_np >= int(vocab_size)
    if np.any(oov):
        input_ids_np[oov] = 0
    input_ids = (
        torch.from_numpy(input_ids_np).unsqueeze(0).to(device=device, dtype=torch.long)
    )

    action_mask_np = obs["action_mask"].astype(np.bool_, copy=False)
    action_mask = (
        torch.from_numpy(action_mask_np)
        .unsqueeze(0)
        .to(device=device, dtype=torch.bool)
    )
    return input_ids, action_mask


def main() -> None:
    args = build_parser().parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")

    device = resolve_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = TinyTransformerConfig(**checkpoint["model_config"])

    model = TinyTransformerPolicy(config=config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    env_config = checkpoint.get("env_config", {})
    env_kwargs = build_env_kwargs(
        env_config=env_config,
        text=args.text,
        ignored_chars_override=args.ignored_chars,
    )
    env = RandomPalindromeEnv(**env_kwargs)
    obs, info = env.reset(options={"text": args.text})

    n_actions = int(obs["action_mask"].shape[0])
    if int(config.n_actions) != n_actions:
        env.close()
        raise ValueError(
            f"Checkpoint n_actions={config.n_actions} but env produced mask size {n_actions}. "
            "Use a checkpoint trained with the current environment."
        )

    max_steps = args.max_steps if args.max_steps is not None else env.max_steps
    total_reward = 0.0
    last_info = info
    last_action_name = "N/A"

    print(f"Input: {args.text}")
    print(f"Normalized: {info['normalized_text']}")
    print(f"Ignored chars: {''.join(sorted(env.ignored_characters))!r}")
    print(f"Case sensitive: {env.case_sensitive}")
    train_max_len = env_config.get("max_len")
    if train_max_len is not None and len(info["normalized_text"]) > int(train_max_len):
        print(
            "Warning: normalized input length is larger than training max_len "
            f"({len(info['normalized_text'])} > {train_max_len}); this is OOD and may fail."
        )
    print("Trace:")

    for step_idx in range(1, max_steps + 1):
        input_ids, action_mask = prepare_model_input(
            obs=obs,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            device=device,
        )
        with torch.no_grad():
            output = model(input_ids)
            masked_logits = output.action_logits.masked_fill(~action_mask, -1e9)
            if args.sample:
                dist = Categorical(logits=masked_logits / args.temperature)
                action = int(dist.sample().item())
            else:
                action = int(torch.argmax(masked_logits, dim=-1).item())
            probs = torch.softmax(masked_logits, dim=-1)[0].cpu().tolist()

        next_obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += float(reward)
        last_info = step_info
        last_action_name = step_info["action_name"]

        print(
            f"{step_idx:02d}. action={step_info['action_name']:<20} "
            f"reward={reward:+.3f} "
            f"L={step_info['left']}({step_info['left_char']}) "
            f"R={step_info['right']}({step_info['right_char']}) "
            f"probs={[round(p, 3) for p in probs]}"
        )

        if terminated or truncated:
            break
        obs = next_obs

    env.close()

    answered = bool(last_info.get("answered", False))
    if answered:
        predicted = (
            "palindrome"
            if last_action_name == "ANSWER_PALINDROME"
            else "not palindrome"
        )
        correct = last_info.get("final_correct")
    else:
        predicted = "unknown (no terminal answer)"
        correct = None

    print("Result:")
    print(f"- predicted: {predicted}")
    print(
        f"- ground_truth: {'palindrome' if last_info['is_palindrome'] else 'not palindrome'}"
    )
    print(f"- correct: {correct}")
    print(f"- total_reward: {total_reward:.3f}")
    print(f"- steps: {last_info['steps']}")


if __name__ == "__main__":
    main()
