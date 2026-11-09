"""Validation loss with the uncertainty decomposition the study needs.

Two different standard errors, deliberately kept separate:

    SE_doc   std(doc means) / sqrt(n_docs)   -- how precisely we measured THIS run
    sigma_run                                 -- how much a rerun would move the number

SE_doc is ~1e-3 nats and is measured here.  sigma_run is ~1e-2 nats, is a property
of the training procedure, and can only be estimated from replicate seeds
(see scripts/sigma_run.py).  Using SE_doc as a fit weight -- the v2 mistake --
understates the true error by an order of magnitude.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch

from .data import iter_sequential


def autocast_ctx(device: str, dtype: str):
    if dtype == "bf16" and device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if dtype == "bf16" and device == "cpu":
        return torch.autocast("cpu", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, val_bin: str, doc_lens: np.ndarray, seq_len: int,
             device: str, batch_seqs: int = 8, dtype: str = "bf16",
             max_tokens: int | None = None) -> dict:
    """max_tokens caps the scan (smoke tests); production runs must leave it None."""
    """Per-token loss over the held-out shard, aggregated to document means."""
    model.eval()
    mm = np.memmap(val_bin, dtype=np.uint16, mode="r")
    n = int(mm.size)
    tok_loss = np.full(n, np.nan, dtype=np.float64)

    pos = 0
    for batch in iter_sequential(val_bin, seq_len, batch_seqs):
        if max_tokens is not None and pos >= max_tokens:
            break
        x = torch.as_tensor(batch, dtype=torch.long, device=device)
        B, T = x.shape
        with autocast_ctx(device, dtype):
            per_tok = model.per_token_loss(x[:, :-1], x[:, 1:])   # (B, T-1)
        pt = per_tok.float().cpu().numpy()                        # (B, T-1)
        # Window i covers global positions [pos + i*T, pos + i*T + T) and predicts
        # the last T-1 of them.  Flattening the (B, T-1) block and writing it
        # contiguously is WRONG: every window's first token has no prediction, so
        # the rows must be placed with a T-stride gap, not packed.  Getting this
        # wrong silently scrambles token -> document assignment and makes n_doc
        # drift between evaluations.
        for i in range(B):
            start = pos + i * T + 1
            tok_loss[start: start + T - 1] = pt[i]
        pos += B * T

    valid = ~np.isnan(tok_loss)
    lens = np.asarray(doc_lens, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lens)])[:-1]

    sums = np.add.reduceat(np.where(valid, tok_loss, 0.0), starts)
    cnts = np.add.reduceat(valid.astype(np.int64), starts)
    doc_mean = np.divide(sums, np.maximum(cnts, 1))
    ok = cnts > 0

    n_tok = int(valid.sum())
    total = float(sums.sum() / max(n_tok, 1))
    se_doc = float(doc_mean[ok].std(ddof=1) / np.sqrt(ok.sum()))

    model.train()
    return {
        "loss": total,
        "se_doc": se_doc,
        "n_doc": int(ok.sum()),
        "n_tokens": n_tok,
        "doc_mean_std": float(doc_mean[ok].std(ddof=1)),
    }
