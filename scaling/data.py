"""Token streams: memmap .bin -> fixed (B, T) batches, with per-seed order randomisation.

Why this exists instead of "just memmap and slice at a random offset":

    sigma_run (the statistical spine of this project) is supposed to measure
    run-to-run reproducibility.  If three seeds read the same .bin in the same
    order, the only thing that differs between them is weight initialisation, and
    sigma_run comes out systematically too small.  Data order is a real source of
    variance in pretraining, so it has to live inside the replicates.

How order is randomised:

    documents -> ~1M-token blocks aligned to document boundaries -> block order is
    permuted with the run's seed.  Coarse enough that IO stays mostly sequential
    (2 MB reads), fine enough that two seeds see genuinely different curricula.

Why the generator is written as explicit state (order/ptr/buf) instead of `yield`:

    a generator cannot be pickled, so `train.py --resume` could not restore the
    exact position in the data stream.  A resumed run would then see a different
    token order than the one it is supposed to be reproducing, and the checkpoint
    would silently stop being a checkpoint.
"""
from __future__ import annotations

import numpy as np


def build_blocks(doc_lens: np.ndarray, block_tokens: int = 1 << 20) -> list[tuple[int, int]]:
    """Group consecutive documents into ~block_tokens blocks; never split a document."""
    total = int(doc_lens.sum())
    if total == 0:
        raise ValueError("empty shard")
    bounds = [0]
    acc = 0
    for length in doc_lens:
        acc += int(length)
        if acc >= block_tokens:
            bounds.append(bounds[-1] + acc)
            acc = 0
    if acc:
        bounds.append(bounds[-1] + acc)
    assert bounds[-1] == total, f"blocks cover {bounds[-1]} tokens, shard has {total}"
    return list(zip(bounds[:-1], bounds[1:]))


class TokenBatcher:
    """Infinite, resumable stream of (micro_seqs, seq_len) windows for training."""

    def __init__(self, bin_path: str, doc_lens: np.ndarray, seq_len: int,
                 tokens_per_step: int, micro_seqs: int, seed: int,
                 block_tokens: int = 1 << 20):
        self.bin_path = bin_path
        self.mm = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.micro_seqs = micro_seqs
        self.tokens_per_step = tokens_per_step
        self.blocks = build_blocks(np.asarray(doc_lens), block_tokens)
        if tokens_per_step % (micro_seqs * seq_len) != 0:
            raise ValueError(f"{tokens_per_step} not divisible by {micro_seqs}*{seq_len}")
        self.accum_steps = tokens_per_step // (micro_seqs * seq_len)
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(self.blocks))
        self.ptr = 0
        self.epoch = 0
        self.buf = np.empty(0, dtype=np.uint16)

    # ---- stream ----------------------------------------------------------
    def _reshuffle(self) -> None:
        self.epoch += 1
        self.order = self.rng.permutation(len(self.blocks))
        self.ptr = 0

    def next_window(self) -> np.ndarray:
        while self.buf.size < self.seq_len:
            if self.ptr >= len(self.order):
                self._reshuffle()
            s, e = self.blocks[self.order[self.ptr]]
            self.ptr += 1
            self.buf = np.concatenate((self.buf, self.mm[s:e]))
        w = self.buf[: self.seq_len].copy()
        self.buf = self.buf[self.seq_len:].copy()
        return w

    def micro_batch(self) -> np.ndarray:
        return np.stack([self.next_window() for _ in range(self.micro_seqs)])

    def step_batches(self):
        for _ in range(self.accum_steps):
            yield self.micro_batch()

    def epoch_tokens(self) -> int:
        return int(self.mm.size)

    # ---- resumable state -------------------------------------------------
    def state(self) -> dict:
        return {"order": self.order, "ptr": self.ptr, "epoch": self.epoch,
                "buf": self.buf, "rng": self.rng.bit_generator.state}

    def load_state(self, st: dict) -> None:
        self.order = np.asarray(st["order"])
        self.ptr = int(st["ptr"])
        self.epoch = int(st["epoch"])
        self.buf = np.asarray(st["buf"], dtype=np.uint16)
        self.rng.bit_generator.state = st["rng"]


def iter_sequential(bin_path: str, seq_len: int, batch_seqs: int = 8):
    """Non-overlapping contiguous windows, for evaluation only."""
    mm = np.memmap(bin_path, dtype=np.uint16, mode="r")
    n = (mm.size // seq_len) * seq_len
    for start in range(0, n, seq_len * batch_seqs):
        chunk = mm[start: min(start + seq_len * batch_seqs, n)]
        m = (chunk.size // seq_len) * seq_len
        if m == 0:
            continue
        yield np.array(chunk[:m]).reshape(-1, seq_len)   # copy: memmap views are read-only
