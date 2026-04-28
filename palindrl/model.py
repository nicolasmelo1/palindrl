from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TinyTransformerConfig:
    vocab_size: int = 50_257
    max_seq_len: int = 64
    n_actions: int = 4
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 2
    ff_multiplier: int = 4
    dropout: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass
class PolicyOutput:
    action_logits: Tensor
    state_value: Tensor
    hidden_states: Tensor


class TransformerBlock(nn.Module):
    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        ff_hidden = config.d_model * config.ff_multiplier

        self.norm_attn = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        attn_in = self.norm_attn(x)
        attn_out, _ = self.attn(
            query=attn_in,
            key=attn_in,
            value=attn_in,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = x + attn_out

        ff_in = self.norm_ff(x)
        x = x + self.ff(ff_in)
        return x


class TinyTransformerPolicy(nn.Module):
    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config=config) for _ in range(config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.policy_head = nn.Linear(config.d_model, config.n_actions)
        self.value_head = nn.Linear(config.d_model, 1)

    def forward(self, input_ids: Tensor) -> PolicyOutput:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} is bigger than max_seq_len "
                f"{self.config.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=input_ids.device).expand(
            batch_size, seq_len
        )
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        attn_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.final_norm(x)
        final_state = x[:, -1, :]
        action_logits = self.policy_head(final_state)
        state_value = self.value_head(final_state).squeeze(-1)
        return PolicyOutput(
            action_logits=action_logits,
            state_value=state_value,
            hidden_states=x,
        )
