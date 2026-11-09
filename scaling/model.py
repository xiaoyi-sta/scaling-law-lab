"""A deliberately boring GPT: RMSNorm + RoPE + 4x GeLU MLP + tied embeddings.

No SwiGLU, no GQA, no biases, no dropout.  Boring is the point: the independent
variable is model size, so the architecture must not move with it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PRESETS, ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    """Half-split rotary embedding, cos/sin cached as non-persistent buffers."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)          # (T, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)   # (T, head_dim) matches rotate_half
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        T = q.shape[-2]
        cos = self.cos[:T].to(q.dtype)[None, None]   # (1,1,T,head_dim)
        sin = self.sin[:T].to(q.dtype)[None, None]
        return (q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.d_head = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = RoPE(cfg.head_dim, cfg.seq_len, cfg.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        q, k = self.rope(q, k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = cfg.ffn_mult * cfg.d_model
        self.fc = nn.Linear(cfg.d_model, hidden, bias=False)
        self.proj = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        # scaled init on residual-output projections (GPT-2 style)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layer) ** 0.5)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """Raw forward.  `targets` MUST already be shifted (targets[t] = idx[t+1]).

        Passing the same tensor for both arguments is a silent catastrophe: causal
        self-attention lets position t attend to idx[t], so the model can copy the
        answer.  Measured on the 8M model that drops loss from 9.04 to 7.87 nats
        while the training curve still looks perfectly plausible.  Use
        `forward_lm()` unless you know exactly why you are not.
        """
        if targets is not None and targets is idx:
            raise ValueError(
                "targets is idx: labels leak through causal self-attention "
                "(loss drops ~1.2 nats below ln V, curve looks fine, result is garbage). "
                "Use GPT.forward_lm()."
            )
        B, T = idx.shape
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1), reduction="mean"
        )
        return logits, loss

    def forward_lm(self, tokens: torch.Tensor):
        """The only correct training entry point: shift happens in here, once.

        tokens: (B, T) packed token ids.  Returns (logits, loss), predicting t+1
        from <= t.  Because the slicing is internal, the label-aliasing bug that
        forward() guards against cannot be triggered by a caller.
        """
        return self(tokens[:, :-1], targets=tokens[:, 1:])

    @torch.no_grad()
    def per_token_loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """(B, T) loss, used by eval for document-level aggregation."""
        B, T = idx.shape
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
        ).view(B, T)

    @property
    def n_params_nonemb(self) -> int:
        emb = self.tok_emb.weight.numel()
        # parameters() de-duplicates shared tensors, so the tied weight counts once
        return sum(p.numel() for p in self.parameters()) - emb


def build_model(name: str, **overrides) -> GPT:
    base = PRESETS[name]
    fields = set(ModelConfig.__dataclass_fields__)
    unknown = set(overrides) - fields
    if unknown:
        raise TypeError(f"unknown ModelConfig fields: {sorted(unknown)}")
    cfg = ModelConfig(**{**{k: getattr(base, k) for k in fields}, **overrides})
    return GPT(cfg)
