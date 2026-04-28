from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import gradio as gr
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from palindrl.environment import RandomPalindromeEnv  # noqa: E402
from palindrl.model import TinyTransformerConfig, TinyTransformerPolicy  # noqa: E402
from palindrl.play import build_env_kwargs, prepare_model_input  # noqa: E402


CHECKPOINT_PATH = ROOT / "checkpoints" / "policy.pt"


def load_model() -> tuple[TinyTransformerPolicy, TinyTransformerConfig, dict]:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    config = TinyTransformerConfig(**checkpoint["model_config"])
    model = TinyTransformerPolicy(config=config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config, checkpoint.get("env_config", {})


MODEL, CONFIG, ENV_CONFIG = load_model()


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def predict(message: str, history: list[dict[str, str]] | None = None) -> str:
    text = strip_accents(message).strip()
    if not text:
        return "False"

    try:
        env_kwargs = build_env_kwargs(
            env_config=ENV_CONFIG,
            text=text,
            ignored_chars_override=None,
        )
        env = RandomPalindromeEnv(**env_kwargs)
        obs, _ = env.reset(options={"text": text})
    except Exception:
        return "False"

    last_action_name = ""
    last_info = None

    try:
        for _ in range(env.max_steps):
            input_ids, action_mask = prepare_model_input(
                obs=obs,
                vocab_size=CONFIG.vocab_size,
                max_seq_len=CONFIG.max_seq_len,
                device=torch.device("cpu"),
            )

            with torch.no_grad():
                output = MODEL(input_ids)
                masked_logits = output.action_logits.masked_fill(~action_mask, -1e9)
                action = int(torch.argmax(masked_logits, dim=-1).item())

            obs, _, terminated, truncated, info = env.step(action)
            last_action_name = info["action_name"]
            last_info = info

            if terminated or truncated:
                break
    finally:
        env.close()

    if not last_info or not last_info.get("answered", False):
        return "False"

    return "True" if last_action_name == "ANSWER_PALINDROME" else "False"


demo = gr.ChatInterface(
    fn=predict,
    title="palindromon-0.116M",
    description="Send text. It answers whether the text is a palindrome.",
    examples=[
        "This is an example",
        "Anotaram a data da maratona",
        "A cara rajada da jararaca",
        "Roma me tem amor",
        "A base do teto desaba",
        "racecar",
        "A man, a plan, a canal: Panama!",
        "not a palindrome",
    ],
)


if __name__ == "__main__":
    demo.launch()
