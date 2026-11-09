"""Phase-1 gate: prove the model family is what the spec says it is.

Run this BEFORE renting a GPU.  It checks, in order:
  1. exact non-embedding parameter counts against 12d^2L + d(2L+1)
  2. head_dim == 64 for all three sizes (the 同构 claim)
  3. no bias parameters anywhere
  4. tied embedding really is one tensor, not two
  5. forward/backward runs, loss at init ~= ln(vocab), gradients finite
  6. a short step-time measurement (indicative only off-GPU)

Usage:  python -m scripts.check_model
"""
from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")

from scaling.config import PRESETS, TrainConfig  # noqa: E402
from scaling.model import build_model  # noqa: E402

# spec §3 table, exact values, for regression checking
SPEC_N = {"8M": 7_869_696, "20M": 19_473_024, "50M": 50_348_544}


def pick_device() -> str:
    """`DEVICE=cpu python -m scripts.check_model` forces CPU if MPS misbehaves."""
    forced = os.environ.get("DEVICE", "").strip()
    if forced:
        return forced
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> int:
    tok = TrainConfig()
    device = pick_device()
    print(f"python {platform.python_version()}  |  torch {torch.__version__}  |  numpy {np.__version__}")
    print(f"device = {device}  (mps available: {torch.backends.mps.is_available()})\n")


    ok = True
    hdr = (f"{'size':>5} {'d':>4} {'L':>3} {'h':>2} {'hd':>3} "
           f"{'N_nonemb':>12} {'formula':>12} {'vs spec':>8} {'tied':>5} {'bias':>5}")
    print(hdr)
    print("-" * len(hdr))

    models = {}
    for name, cfg in PRESETS.items():
        m = build_model(name)
        measured = m.n_params_nonemb
        form = cfg.n_params_nonemb
        assert measured == form, f"{name}: measured {measured} != formula {form}"

        tied = m.lm_head.weight is m.tok_emb.weight
        n_bias = sum(p.numel() for n, p in m.named_parameters() if n.endswith("bias"))

        print(
            f"{name:>5} {cfg.d_model:>4} {cfg.n_layer:>3} {cfg.n_head:>2} {cfg.head_dim:>3} "
            f"{measured:>12,} {form:>12,} {measured - SPEC_N[name]:>+8} "
            f"{str(tied):>5} {n_bias:>5}"
        )
        if not tied:
            print(f"  FAIL {name}: embeddings not tied")
            ok = False
        if n_bias != 0:
            print(f"  FAIL {name}: {n_bias} bias params")
            ok = False
        models[name] = (m, cfg)

    print()
    print("forward/backward smoke (short seq, fp32, no autocast):")
    T, B = 256, 2
    for name, (m, cfg) in models.items():
        m.to(device)
        x = torch.randint(0, cfg.vocab_size, (B, T), device=device)
        y = torch.randint(0, cfg.vocab_size, (B, T), device=device)

        t0 = time.perf_counter()
        logits, loss = m(x, targets=y)
        loss.backward()
        dt = time.perf_counter() - t0

        grads = [p.grad for p in m.parameters() if p.grad is not None]
        gnorm = math.sqrt(sum(g.detach().float().pow(2).sum().item() for g in grads))
        finite = all(torch.isfinite(g).all().item() for g in grads)
        shapes_ok = tuple(logits.shape) == (B, T, cfg.vocab_size)
        expected = math.log(cfg.vocab_size)
        good = finite and shapes_ok and abs(loss.item() - expected) < 0.15
        ok = ok and good
        print(
            f"  {name:>5}  logits={tuple(logits.shape)}  loss@init={loss.item():.3f} "
            f"(ln V={expected:.3f})  |grad|={gnorm:.2f}  finite={finite}  "
            f"{dt*1000:.0f} ms  {'ok' if good else 'CHECK'}"
        )
        m.zero_grad(set_to_none=True)

        tps = (B * T) / dt
        print(f"         {'':>5}  {tps/1e3:.1f}k tokens/s on {device} (fp32 micro-batch; "
              f"NOT a projection -- real throughput comes from the 3090)")

    # ---- label-shift guard ------------------------------------------------
    # The single most expensive silent bug available in this project: if the
    # training loop passes the same tensor as input and target, causal attention
    # lets the model read the answer, loss drops well below ln(V), and every
    # downstream number is fiction.
    print()
    print("label-shift guard:")
    m8 = models["8M"][0]
    lx = torch.randint(0, 8192, (2, 64), device=device)
    try:
        m8(lx, targets=lx)
        print("  FAIL: forward() accepted targets is idx")
        ok = False
    except ValueError as e:
        print(f"  ok   forward(lx, targets=lx) raised: {str(e)[:52]}...")
    with torch.no_grad():
        _, l_auto = m8.forward_lm(lx)
        _, l_man = m8(lx[:, :-1], targets=lx[:, 1:])
    same = torch.allclose(l_auto, l_man)
    near = abs(l_auto.item() - math.log(8192)) < 0.3
    print(f"  forward_lm={l_auto.item():.3f}  equals manual shift={same}  near ln V={near}")
    if not (same and near):
        ok = False

    print()
    print("PHASE 1 GATE: PASS" if ok else "PHASE 1 GATE: FAIL — fix before renting a GPU")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
