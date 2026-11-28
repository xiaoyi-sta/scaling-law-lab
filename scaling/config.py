"""Model/training configuration for the scaling-law lab.

Single source of truth: every run is fully described by a YAML file that maps
1:1 onto these dataclasses (see configs/).  Nothing about a run may live only
in a shell command.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

VOCAB_SIZE = 8192
SEQ_LEN = 1024


@dataclass
class ModelConfig:
    name: str
    d_model: int
    n_layer: int
    n_head: int
    seq_len: int = SEQ_LEN
    vocab_size: int = VOCAB_SIZE
    ffn_mult: int = 4            # 4x GeLU, deliberately NOT SwiGLU
    rope_theta: float = 10_000.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError(f"{self.name}: d_model % n_head != 0")
        head_dim = self.d_model // self.n_head
        if head_dim != 64:
            raise ValueError(
                f"{self.name}: head_dim={head_dim}, spec requires 64 for 同构"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def n_params_nonemb(self) -> int:
        """Chinchilla-style N: attention + MLP + norms, embeddings excluded.

        12*d^2*L  (attn 4d^2 + mlp 8d^2)  +  d*(2L+1)  (2 RMSNorm per block + final)
        No bias terms, so nothing else is added.
        """
        return 12 * self.d_model**2 * self.n_layer + self.d_model * (2 * self.n_layer + 1)

    @property
    def n_params_emb(self) -> int:
        return self.vocab_size * self.d_model

    @property
    def n_params_total(self) -> int:
        return self.n_params_nonemb + self.n_params_emb

    def to_dict(self) -> dict:
        d = asdict(self)
        d["head_dim"] = self.head_dim
        d["n_params_nonemb"] = self.n_params_nonemb
        d["n_params_total"] = self.n_params_total
        return d


# All sizes use 64-dimensional attention heads.
PRESETS: dict[str, ModelConfig] = {
    "8M": ModelConfig(name="8M", d_model=256, n_layer=10, n_head=4),
    "20M": ModelConfig(name="20M", d_model=384, n_layer=11, n_head=6),
    "25M": ModelConfig(name="25M", d_model=384, n_layer=14, n_head=6),
    "50M": ModelConfig(name="50M", d_model=512, n_layer=16, n_head=8),
}

# Total tokens each size is trained for (drives the LR schedule horizon).
TOKEN_BUDGET = {
    "8M": 500_000_000, "20M": 1_000_000_000, "25M": 620_756_992, "50M": 1_000_000_000,
}


@dataclass
class TrainConfig:
    """Everything that must be held constant across runs (spec §3)."""
    tokens_per_step: int = 131_072      # == 128 sequences x 1024
    micro_seqs: int = 16                # gradient-accumulation chunk, NOT part of the science
    lr: float = 2e-3
    min_lr: float = 5e-5
    warmup_frac: float = 0.02
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    seed: int = 0
    data_dir: str = "data"
    log_dir: str = "logs"
    ckpt_dir: str = "ckpts"

    def __post_init__(self) -> None:
        chunk = self.micro_seqs * SEQ_LEN
        if self.tokens_per_step % chunk != 0:
            raise ValueError(
                f"tokens_per_step={self.tokens_per_step} is not a multiple of "
                f"micro_seqs*seq_len={chunk}: gradient accumulation would be ragged, "
                f"and the global batch is the one thing that must be identical across runs"
            )

    def steps_for(self, model: str) -> int:
        """Total optimizer steps. Floors the nominal budget, so a run ends slightly short."""
        if model not in TOKEN_BUDGET:
            raise KeyError(f"unknown model {model!r}, expected one of {list(TOKEN_BUDGET)}")
        return TOKEN_BUDGET[model] // self.tokens_per_step

    def tokens_at(self, step: int) -> int:
        """ACTUAL tokens seen after `step` optimizer steps.

        This -- never TOKEN_BUDGET[model] -- is the D that goes into the fit.
        """
        return step * self.tokens_per_step

    def total_tokens(self, model: str) -> int:
        return self.tokens_at(self.steps_for(model))

    def breakpoints(self, model: str, fractions=(0.25, 0.5, 0.75, 1.0)) -> list[int]:
        """Eval steps, locked to step boundaries so the last one is exactly the final step.

        Defined as fractions of the TOTAL STEP COUNT, not of the nominal token
        budget: 1.25e8 is not a multiple of 131072, so a token-defined breakpoint
        would either overshoot or never fire.
        """
        total = self.steps_for(model)
        steps = sorted({min(total, max(1, int(round(f * total)))) for f in fractions})
        if steps[-1] != total:
            steps.append(total)
        return steps

    def breakpoint_tokens(self, model: str, fractions=(0.25, 0.5, 0.75, 1.0)) -> list[int]:
        """Actual token counts at each breakpoint -- log these, they are the D values."""
        return [self.tokens_at(s) for s in self.breakpoints(model, fractions)]
