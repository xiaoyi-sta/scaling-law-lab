"""Single training entry point.  Every run is described by flags + config, nothing else.

Deliberate choices worth defending in an interview:

  * `forward_lm` only.  The model's raw forward() would happily accept the input as
    its own target, which leaks the answer through causal attention and produces a
    beautiful, completely fake loss curve.
  * one optimizer step == exactly tokens_per_step tokens == exactly the same number
    of tokens for every run, regardless of micro-batch size or hardware.  The global
    batch is the one invariant the comparison rests on, so it is asserted, not hoped.
  * the data-stream position is checkpointed (TokenBatcher.state), so --resume
    reproduces the original token order instead of quietly starting a new run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch

from scaling.config import PRESETS, TrainConfig
from scaling.data import TokenBatcher
from scaling.eval import autocast_ctx, evaluate
from scaling.model import build_model

# `val_set` was added in v3.1: the ablation is graded on two validation sets,
# because a single raw-web val set is in-distribution for arm A but not for arm B
# (unigram JSD to val: A 0.033, B 0.133).  One row per (breakpoint, val_set).
CSV_COLS = ["run_id", "model_size", "seed", "tokens_seen", "step", "val_set",
            "val_loss", "val_loss_se_doc", "n_doc", "wall_time", "lr"]

VAL_SETS = {"A": ("val.bin", "val_doc_lens.npy"),
            "B": ("valB.bin", "valB_doc_lens.npy")}


def lr_at(step: int, total: int, warmup: int, base: float, min_lr: float) -> float:
    if warmup > 0 and step < warmup:
        return base * (step + 1) / warmup
    p = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
    return min_lr + (base - min_lr) * 0.5 * (1.0 + math.cos(math.pi * p))


def make_optimizer(model, tcfg: TrainConfig):
    decay, nodecay = [], []
    for _, p in model.named_parameters():
        (decay if p.dim() >= 2 else nodecay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": tcfg.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=tcfg.lr, betas=(tcfg.beta1, tcfg.beta2), eps=1e-8,
    )


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def append_row(path: pathlib.Path, row: dict) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if new:
            w.writeheader()
        w.writerow({k: row[k] for k in CSV_COLS})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8M", choices=list(PRESETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="A",
                    help="data tag: A, B, A2, B2 -> data/proc/train<TAG>.bin")
    ap.add_argument("--val-sets", default="A", help="comma separated: A,B")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--steps", type=int, default=None, help="override the step count (smoke tests)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny batch/steps, for local plumbing checks")
    ap.add_argument("--micro-seqs", type=int, default=None,
                    help="sequences per micro-batch (throughput knob; global batch stays exact)")
    ap.add_argument("--data-dir", default="data/proc")
    ap.add_argument("--no-eval", action="store_true", help="benchmark only: skip validation")
    ap.add_argument("--eval-tokens", type=int, default=None,
                    help="cap the validation scan (smoke only)")
    args = ap.parse_args()

    tcfg = TrainConfig(seed=args.seed)
    if args.smoke:
        tcfg.tokens_per_step = 8192
        tcfg.micro_seqs = 2
    if args.micro_seqs:
        tcfg.micro_seqs = args.micro_seqs
    chunk = tcfg.micro_seqs * 1024
    if tcfg.tokens_per_step % chunk:
        raise SystemExit(f"--micro-seqs {tcfg.micro_seqs}: {tcfg.tokens_per_step} "
                         f"is not a multiple of {chunk}")
    mcfg = PRESETS[args.model]
    total_steps = args.steps or tcfg.steps_for(args.model)
    if args.steps:
        bps = sorted({max(1, int(round(f * total_steps))) for f in (0.5, 1.0)})
        bps += [total_steps] if bps[-1] != total_steps else []
    else:
        bps = tcfg.breakpoints(args.model)
    warmup = max(1, int(round(tcfg.warmup_frac * total_steps)))

    run_id = args.run_id or f"{args.model}_seed{args.seed}_{args.arm}"
    proc = pathlib.Path(args.data_dir)
    train_bin = proc / f"train{args.arm}.bin"
    tags = [t.strip().upper() for t in args.val_sets.split(",") if t.strip()]
    for t in tags:
        if t not in VAL_SETS:
            raise SystemExit(f"unknown val set {t!r}, expected one of {list(VAL_SETS)}")
    for p in [train_bin, proc / f"train{args.arm}_doc_lens.npy"] + \
             [proc / VAL_SETS[t][0] for t in tags] + [proc / VAL_SETS[t][1] for t in tags]:
        if not p.exists():
            raise SystemExit(f"missing {p} -- run `make bins` first")

    device = pick_device()
    dtype = tcfg.dtype if device in ("cuda", "cpu") else "fp32"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = build_model(args.model).to(device)
    opt = make_optimizer(model, tcfg)
    batcher = TokenBatcher(str(train_bin), np.load(proc / f"train{args.arm}_doc_lens.npy"),
                           mcfg.seq_len, tcfg.tokens_per_step, tcfg.micro_seqs, args.seed)
    val_lens = {t: np.load(proc / VAL_SETS[t][1]) for t in tags}
    val_bins = {t: str(proc / VAL_SETS[t][0]) for t in tags}

    start_step = 0
    ckpt_dir = pathlib.Path(tcfg.ckpt_dir)
    if args.resume:
        st = torch.load(args.resume, map_location=device, weights_only=False)
        # A checkpoint is only a checkpoint if it resumes the SAME experiment.
        # Silently resuming with a different seed / global batch / model size
        # would splice two different runs together and nobody would notice until
        # the scaling fit came out wrong.
        prev_cfg = st.get("train_cfg", {})
        if st.get("seed") != args.seed:
            raise SystemExit(f"resume seed {st.get('seed')} != --seed {args.seed}")
        if prev_cfg.get("tokens_per_step") != tcfg.tokens_per_step:
            raise SystemExit(
                f"resume tokens_per_step {prev_cfg.get('tokens_per_step')} != "
                f"{tcfg.tokens_per_step}: the global batch is part of the science")
        if (st.get("model_cfg") or {}).get("name") != mcfg.name:
            raise SystemExit(
                f"resume model {st.get('model_cfg', {}).get('name')} != --model {mcfg.name}")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optim"])
        batcher.load_state(st["batcher"])
        start_step = int(st["step"])
        print(f"resumed {args.resume} at step {start_step}")

    print(f"run {run_id}  device={device} dtype={dtype}  params_nonemb={mcfg.n_params_nonemb:,}")
    print(f"steps={total_steps}  tokens/step={tcfg.tokens_per_step:,}  "
          f"accum={batcher.accum_steps}  warmup={warmup}  eval@{bps}")
    print(f"data: {train_bin} ({batcher.epoch_tokens():,} tok)  "
          f"~{total_steps * tcfg.tokens_per_step / batcher.epoch_tokens():.2f} epochs")

    log_path = pathlib.Path(tcfg.log_dir) / f"{run_id}.csv"
    t0 = time.time()
    tokens_at_start = start_step * tcfg.tokens_per_step

    for step in range(start_step, total_steps):
        lr = lr_at(step, total_steps, warmup, tcfg.lr, tcfg.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        seen_here = 0
        for mb in batcher.step_batches():
            x = torch.as_tensor(mb, dtype=torch.long, device=device)
            with autocast_ctx(device, dtype):
                _, loss = model.forward_lm(x)
                loss = loss / batcher.accum_steps
            loss.backward()
            seen_here += x.numel()
        if seen_here != tcfg.tokens_per_step:
            raise RuntimeError(f"step consumed {seen_here} tokens, expected {tcfg.tokens_per_step}")

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()

        global_step = step + 1
        tokens_seen = global_step * tcfg.tokens_per_step
        if global_step == start_step + 1 or global_step % 20 == 0:
            el = time.time() - t0
            rate = (tokens_seen - tokens_at_start) / max(el, 1e-9)
            print(f"  step {global_step}/{total_steps}  loss={loss.item()*batcher.accum_steps:.4f}  "
                  f"lr={lr:.2e}  |g|={gnorm:.2f}  {rate/1e3:.1f}k tok/s  "
                  f"eta {(total_steps-global_step)*tcfg.tokens_per_step/max(rate,1)/60:.1f} min",
                  flush=True)

        if global_step in bps and not args.no_eval:
            wall = time.time() - t0
            parts = []
            for tag in tags:
                m = evaluate(model, val_bins[tag], val_lens[tag], mcfg.seq_len, device,
                             dtype=dtype, max_tokens=args.eval_tokens)
                append_row(log_path, {
                    "run_id": run_id, "model_size": args.model, "seed": args.seed,
                    "tokens_seen": tokens_seen, "step": global_step, "val_set": tag,
                    "val_loss": f"{m['loss']:.6f}", "val_loss_se_doc": f"{m['se_doc']:.6f}",
                    "n_doc": m["n_doc"], "wall_time": f"{wall:.1f}", "lr": f"{lr:.3e}",
                })
                parts.append(f"val{tag}={m['loss']:.5f}(±{m['se_doc']:.5f},n={m['n_doc']})")
            print(f"  EVAL step={global_step} D={tokens_seen:,}  " + "  ".join(parts), flush=True)

            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(), "optim": opt.state_dict(),
                "step": global_step, "tokens_seen": tokens_seen,
                "batcher": batcher.state(), "model_cfg": mcfg.to_dict(),
                "train_cfg": vars(tcfg), "seed": args.seed, "run_id": run_id,
            }, ckpt_dir / f"{run_id}_{tokens_seen}.pt")

    print(f"done {run_id} in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
