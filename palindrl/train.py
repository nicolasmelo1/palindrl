from __future__ import annotations

import argparse
import random
import string
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from palindrl.environment import ENV_ACTION_NAMES, RandomPalindromeEnv
from palindrl.model import TinyTransformerConfig, TinyTransformerPolicy


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor
    action_masks: torch.Tensor
    actions: torch.Tensor
    old_logprobs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    mean_episode_return: float
    mean_episode_len: float
    answer_accuracy: float
    answer_rate: float
    timeout_rate: float
    invalid_action_rate: float
    repeat_compare_rate: float
    episodes_finished: int


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_rollout_batch(
    env: RandomPalindromeEnv,
    model: TinyTransformerPolicy,
    rollout_steps: int,
    device: torch.device,
    gamma: float,
    gae_lambda: float,
    normalize_advantages: bool,
) -> RolloutBatch:
    model.eval()

    input_ids_list: list[torch.Tensor] = []
    action_masks_list: list[torch.Tensor] = []
    actions_list: list[torch.Tensor] = []
    old_logprobs_list: list[torch.Tensor] = []
    values_list: list[torch.Tensor] = []
    rewards_list: list[torch.Tensor] = []
    done_list: list[torch.Tensor] = []

    episode_returns: list[float] = []
    episode_lens: list[int] = []
    answer_correct: list[float] = []
    answered_episodes = 0
    timed_out_episodes = 0
    invalid_actions = 0
    repeat_compares = 0

    obs, _ = env.reset()
    running_episode_return = 0.0
    running_episode_len = 0

    for _ in range(rollout_steps):
        input_ids = torch.tensor(obs["input_ids"], dtype=torch.long, device=device)
        action_mask = torch.tensor(obs["action_mask"], dtype=torch.bool, device=device)

        with torch.no_grad():
            output = model(input_ids.unsqueeze(0))
            logits = output.action_logits.squeeze(0)
            masked_logits = logits.masked_fill(~action_mask, -1e9)
            dist = Categorical(logits=masked_logits)
            action = dist.sample()
            logprob = dist.log_prob(action)
            value = output.state_value.squeeze(0)

        next_obs, reward, terminated, truncated, info = env.step(int(action.item()))
        done = terminated or truncated
        invalid_actions += int(bool(info.get("invalid_action", False)))
        repeat_compares += int(bool(info.get("repeat_compare", False)))

        input_ids_list.append(input_ids)
        action_masks_list.append(action_mask)
        actions_list.append(action)
        old_logprobs_list.append(logprob)
        values_list.append(value)
        rewards_list.append(torch.tensor(reward, dtype=torch.float32, device=device))
        done_list.append(torch.tensor(float(done), dtype=torch.float32, device=device))

        running_episode_return += float(reward)
        running_episode_len += 1

        if done:
            episode_returns.append(running_episode_return)
            episode_lens.append(running_episode_len)
            if bool(info.get("answered", False)):
                answered_episodes += 1
            if info.get("final_correct") is not None:
                answer_correct.append(float(info["final_correct"]))
            if bool(info.get("timed_out", False)):
                timed_out_episodes += 1
            running_episode_return = 0.0
            running_episode_len = 0
            obs, _ = env.reset()
        else:
            obs = next_obs

    with torch.no_grad():
        if running_episode_len == 0:
            next_value = torch.zeros((), dtype=torch.float32, device=device)
        else:
            last_input_ids = torch.tensor(
                obs["input_ids"], dtype=torch.long, device=device
            )
            next_value = model(last_input_ids.unsqueeze(0)).state_value.squeeze(0)

    values = torch.stack(values_list, dim=0)
    rewards = torch.stack(rewards_list, dim=0)
    dones = torch.stack(done_list, dim=0)
    old_logprobs = torch.stack(old_logprobs_list, dim=0)
    actions = torch.stack(actions_list, dim=0)
    input_ids_batch = torch.stack(input_ids_list, dim=0)
    action_masks_batch = torch.stack(action_masks_list, dim=0)

    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=torch.float32, device=device)
    next_val = next_value
    for t in reversed(range(rollout_steps)):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * non_terminal - values[t]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[t] = gae
        next_val = values[t]

    returns = advantages + values
    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

    mean_episode_return = (
        sum(episode_returns) / len(episode_returns)
        if episode_returns
        else float(rewards.mean().item())
    )
    mean_episode_len = (
        sum(episode_lens) / len(episode_lens) if episode_lens else float(rollout_steps)
    )
    accuracy = sum(answer_correct) / len(answer_correct) if answer_correct else 0.0
    episodes_finished = len(episode_returns)
    answer_rate = answered_episodes / episodes_finished if episodes_finished else 0.0
    timeout_rate = timed_out_episodes / episodes_finished if episodes_finished else 0.0
    invalid_action_rate = invalid_actions / rollout_steps
    repeat_compare_rate = repeat_compares / rollout_steps

    return RolloutBatch(
        input_ids=input_ids_batch,
        action_masks=action_masks_batch,
        actions=actions,
        old_logprobs=old_logprobs,
        returns=returns,
        advantages=advantages,
        mean_episode_return=mean_episode_return,
        mean_episode_len=mean_episode_len,
        answer_accuracy=accuracy,
        answer_rate=answer_rate,
        timeout_rate=timeout_rate,
        invalid_action_rate=invalid_action_rate,
        repeat_compare_rate=repeat_compare_rate,
        episodes_finished=episodes_finished,
    )


def print_debug_rollouts(
    env: RandomPalindromeEnv,
    model: TinyTransformerPolicy,
    device: torch.device,
    episodes: int,
) -> None:
    model.eval()
    for episode_idx in range(1, episodes + 1):
        obs, info = env.reset()
        total_reward = 0.0
        print(
            f"debug episode {episode_idx}: "
            f"text={info['text']!r} norm={info['normalized_text']!r} "
            f"label={'palindrome' if info['is_palindrome'] else 'not palindrome'}"
        )

        for step_idx in range(1, env.max_steps + 1):
            input_ids = torch.tensor(obs["input_ids"], dtype=torch.long, device=device)
            action_mask = torch.tensor(
                obs["action_mask"], dtype=torch.bool, device=device
            )

            with torch.no_grad():
                output = model(input_ids.unsqueeze(0))
                logits = output.action_logits.squeeze(0)
                masked_logits = logits.masked_fill(~action_mask, -1e9)
                action = int(torch.argmax(masked_logits, dim=-1).item())

            obs, reward, terminated, truncated, step_info = env.step(action)
            total_reward += float(reward)
            print(
                f"  {step_idx:02d} {step_info['action_name']:<20} "
                f"L={step_info['left']}({step_info['left_char']}) "
                f"R={step_info['right']}({step_info['right_char']}) "
                f"reward={reward:+.3f}"
            )

            if terminated or truncated:
                break

        answered = bool(step_info.get("answered", False))
        if answered:
            predicted = (
                "palindrome"
                if step_info["action_name"] == "ANSWER_PALINDROME"
                else "not palindrome"
            )
            correct = step_info.get("final_correct")
        else:
            predicted = "unknown"
            correct = None
        print(
            f"  result predicted={predicted} correct={correct} "
            f"total_reward={total_reward:.3f} steps={step_info['steps']}"
        )


def ppo_update(
    model: TinyTransformerPolicy,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    ppo_epochs: int,
    minibatch_size: int,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
) -> dict[str, float]:
    model.train()
    batch_size = batch.input_ids.shape[0]
    if minibatch_size <= 0:
        minibatch_size = batch_size
    minibatch_size = min(minibatch_size, batch_size)

    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    approx_kls: list[float] = []
    clip_fracs: list[float] = []
    total_losses: list[float] = []

    for _ in range(ppo_epochs):
        permutation = torch.randperm(batch_size, device=batch.input_ids.device)
        for start in range(0, batch_size, minibatch_size):
            idx = permutation[start : start + minibatch_size]

            output = model(batch.input_ids[idx])
            masked_logits = output.action_logits.masked_fill(
                ~batch.action_masks[idx], -1e9
            )
            dist = Categorical(logits=masked_logits)
            new_logprobs = dist.log_prob(batch.actions[idx])
            entropy = dist.entropy().mean()
            values = output.state_value

            ratios = torch.exp(new_logprobs - batch.old_logprobs[idx])
            unclipped = ratios * batch.advantages[idx]
            clipped = (
                torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps)
                * batch.advantages[idx]
            )
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, batch.returns[idx])
            total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (batch.old_logprobs[idx] - new_logprobs).mean()
                clip_frac = ((ratios - 1.0).abs() > clip_eps).float().mean()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))
            approx_kls.append(float(approx_kl.item()))
            clip_fracs.append(float(clip_frac.item()))
            total_losses.append(float(total_loss.item()))

    return {
        "loss/total": sum(total_losses) / len(total_losses),
        "loss/policy": sum(policy_losses) / len(policy_losses),
        "loss/value": sum(value_losses) / len(value_losses),
        "policy/entropy": sum(entropies) / len(entropies),
        "policy/approx_kl": sum(approx_kls) / len(approx_kls),
        "policy/clip_frac": sum(clip_fracs) / len(clip_fracs),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the tiny transformer with PPO on RandomPalindromeEnv."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=300, help="Number of PPO updates")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Transitions per rollout update (not episodes).",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-path", type=Path, default=Path("checkpoints/policy.pt"))
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to warm-start model (and optimizer unless reset).",
    )
    parser.add_argument(
        "--reset-optimizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, do not load optimizer state from --init-checkpoint.",
    )
    parser.add_argument("--log-dir", type=Path, default=Path("runs/palindrl"))
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--debug-rollouts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print greedy model episodes during training.",
    )
    parser.add_argument(
        "--debug-rollout-every",
        type=int,
        default=10,
        help="When --debug-rollouts is enabled, print traces every N PPO updates.",
    )
    parser.add_argument(
        "--debug-rollout-episodes",
        type=int,
        default=1,
        help="Number of episodes to print for each debug rollout.",
    )

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--no-adv-norm", action="store_true", help="Disable adv normalization"
    )

    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--n-actions", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--ff-multiplier", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--env-min-len", type=int, default=3)
    parser.add_argument("--env-max-len", type=int, default=12)
    parser.add_argument("--env-palindrome-prob", type=float, default=0.5)
    parser.add_argument(
        "--env-balanced-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Alternate palindrome/non-palindrome episodes to guarantee mix.",
    )
    parser.add_argument("--env-space-prob", type=float, default=0.5)
    parser.add_argument("--env-separator-chars", type=str, default=" -,.")
    parser.add_argument(
        "--env-ignored-chars",
        type=str,
        default=" " + string.punctuation,
        help="Characters removed before palindrome checks (and pointer logic).",
    )
    parser.add_argument(
        "--env-case-sensitive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When false (default), letter comparisons are case-insensitive (A==a).",
    )
    parser.add_argument("--env-max-steps", type=int, default=None)

    parser.add_argument("--env-final-correct-reward", type=float, default=1.0)
    parser.add_argument("--env-final-incorrect-reward", type=float, default=-1.0)
    parser.add_argument("--env-invalid-answer-penalty", type=float, default=-0.20)
    parser.add_argument("--env-repeat-compare-penalty", type=float, default=-0.05)
    parser.add_argument("--env-compare-match-reward", type=float, default=0.05)
    parser.add_argument("--env-compare-mismatch-reward", type=float, default=0.05)
    parser.add_argument("--env-move-valid-reward", type=float, default=0.03)
    parser.add_argument("--env-move-invalid-penalty", type=float, default=-0.10)
    parser.add_argument("--env-step-penalty", type=float, default=-0.01)
    parser.add_argument("--env-timeout-penalty", type=float, default=-0.50)
    parser.add_argument(
        "--env-ignore-spaces-for-label",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    device = resolve_device()
    print(f"Using device: {device}")

    env = RandomPalindromeEnv(
        min_len=args.env_min_len,
        max_len=args.env_max_len,
        palindrome_probability=args.env_palindrome_prob,
        balanced_sampling=args.env_balanced_sampling,
        space_probability=args.env_space_prob,
        separator_chars=args.env_separator_chars,
        ignore_spaces_for_label=args.env_ignore_spaces_for_label,
        ignored_characters=args.env_ignored_chars,
        case_sensitive=args.env_case_sensitive,
        final_correct_reward=args.env_final_correct_reward,
        final_incorrect_reward=args.env_final_incorrect_reward,
        invalid_answer_penalty=args.env_invalid_answer_penalty,
        repeat_compare_penalty=args.env_repeat_compare_penalty,
        compare_match_reward=args.env_compare_match_reward,
        compare_mismatch_reward=args.env_compare_mismatch_reward,
        move_valid_reward=args.env_move_valid_reward,
        move_invalid_penalty=args.env_move_invalid_penalty,
        step_penalty=args.env_step_penalty,
        timeout_penalty=args.env_timeout_penalty,
        max_steps=args.env_max_steps,
    )

    env_vocab_size = len(env.char_to_id)
    vocab_size = args.vocab_size if args.vocab_size is not None else env_vocab_size
    if vocab_size < env_vocab_size:
        raise ValueError(
            f"--vocab-size ({vocab_size}) must be >= env vocab size ({env_vocab_size})"
        )

    env_n_actions = int(env.action_space.n)
    if args.n_actions is not None and args.n_actions != env_n_actions:
        raise ValueError(
            f"--n-actions ({args.n_actions}) must match env action space size ({env_n_actions})"
        )
    n_actions = env_n_actions

    if args.max_seq_len < env.max_observation_len:
        raise ValueError(
            f"--max-seq-len ({args.max_seq_len}) must be >= env max observation len "
            f"({env.max_observation_len})"
        )

    config = TinyTransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        n_actions=n_actions,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_multiplier=args.ff_multiplier,
        dropout=args.dropout,
    )
    model = TinyTransformerPolicy(config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.init_checkpoint is not None:
        if not args.init_checkpoint.exists():
            raise FileNotFoundError(
                f"init checkpoint not found: {args.init_checkpoint}"
            )
        init_ckpt = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(init_ckpt["state_dict"])
        loaded_optimizer = False
        if not args.reset_optimizer and "optimizer_state_dict" in init_ckpt:
            optimizer.load_state_dict(init_ckpt["optimizer_state_dict"])
            loaded_optimizer = True
        print(
            f"Loaded init checkpoint: {args.init_checkpoint} "
            f"(optimizer_loaded={loaded_optimizer})"
        )

    args.log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.log_dir))

    for step in range(1, args.steps + 1):
        if args.debug_rollouts and (
            step == 1 or step % args.debug_rollout_every == 0 or step == args.steps
        ):
            print_debug_rollouts(
                env=env,
                model=model,
                device=device,
                episodes=args.debug_rollout_episodes,
            )

        batch = collect_rollout_batch(
            env=env,
            model=model,
            rollout_steps=args.batch_size,
            device=device,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            normalize_advantages=not args.no_adv_norm,
        )
        update_metrics = ppo_update(
            model=model,
            optimizer=optimizer,
            batch=batch,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            clip_eps=args.clip_eps,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
        )

        writer.add_scalar("env/episode_return_mean", batch.mean_episode_return, step)
        writer.add_scalar("env/episode_len_mean", batch.mean_episode_len, step)
        writer.add_scalar("env/answer_accuracy", batch.answer_accuracy, step)
        writer.add_scalar("env/answer_rate", batch.answer_rate, step)
        writer.add_scalar("env/timeout_rate", batch.timeout_rate, step)
        writer.add_scalar("env/invalid_action_rate", batch.invalid_action_rate, step)
        writer.add_scalar("env/repeat_compare_rate", batch.repeat_compare_rate, step)
        writer.add_scalar("env/episodes_finished", batch.episodes_finished, step)
        for metric_name, metric_value in update_metrics.items():
            writer.add_scalar(metric_name, metric_value, step)

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            print(
                f"step={step} "
                f"ep_return={batch.mean_episode_return:.4f} "
                f"ep_len={batch.mean_episode_len:.2f} "
                f"acc={batch.answer_accuracy:.4f} "
                f"ans_rate={batch.answer_rate:.4f} "
                f"timeout={batch.timeout_rate:.4f} "
                f"invalid={batch.invalid_action_rate:.4f} "
                f"repeat_cmp={batch.repeat_compare_rate:.4f} "
                f"episodes={batch.episodes_finished} "
                f"loss={update_metrics['loss/total']:.4f} "
                f"policy={update_metrics['loss/policy']:.4f} "
                f"value={update_metrics['loss/value']:.4f} "
                f"entropy={update_metrics['policy/entropy']:.4f}"
            )

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": config.to_dict(),
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "action_names": ENV_ACTION_NAMES,
            "env_config": {
                "min_len": args.env_min_len,
                "max_len": args.env_max_len,
                "palindrome_probability": args.env_palindrome_prob,
                "balanced_sampling": args.env_balanced_sampling,
                "space_probability": args.env_space_prob,
                "separator_chars": args.env_separator_chars,
                "ignore_spaces_for_label": args.env_ignore_spaces_for_label,
                "ignored_characters": args.env_ignored_chars,
                "case_sensitive": args.env_case_sensitive,
                "max_steps": args.env_max_steps,
                "final_correct_reward": args.env_final_correct_reward,
                "final_incorrect_reward": args.env_final_incorrect_reward,
                "invalid_answer_penalty": args.env_invalid_answer_penalty,
                "repeat_compare_penalty": args.env_repeat_compare_penalty,
                "compare_match_reward": args.env_compare_match_reward,
                "compare_mismatch_reward": args.env_compare_mismatch_reward,
                "move_valid_reward": args.env_move_valid_reward,
                "move_invalid_penalty": args.env_move_invalid_penalty,
                "step_penalty": args.env_step_penalty,
                "timeout_penalty": args.env_timeout_penalty,
            },
        },
        args.save_path,
    )
    writer.close()
    env.close()

    print(f"Saved checkpoint to: {args.save_path}")


if __name__ == "__main__":
    main()
