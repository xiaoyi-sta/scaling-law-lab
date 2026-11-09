"""Train the 8192-token sentencepiece BPE vocabulary.

This is the one irreversible decision in the project: N (parameter count) and
D (token count) are both defined relative to this vocabulary, and both ablation
arms must share it.  Get it right once, then never touch it again.

Design choices and why (all of them are deliberate, do not "improve" casually):

  byte_fallback=True
      Never emit <unk>.  FineWeb contains arbitrary unicode; a silent unk token
      would corrupt loss in a way that is invisible in the training curve and
      would show up later as an unexplained outlier in the scaling fit.

  normalization_rule_name="identity"
      No NFKC folding.  Keeps token counts stable and makes the tokenizer
      lossless-ish, which matters because we compare fineweb (raw) against
      fineweb-edu (filtered) -- normalisation must not be a hidden variable.

  add_dummy_prefix=False, remove_extra_whitespaces=False
      No implicit leading space, no whitespace collapsing.  Again: fewer hidden
      transforms between the two arms.

  vocab_size=8192
      Small enough that the 8M model's embedding doesn't dominate (21% of total
      params at 8M), large enough that ~4 bytes/token keeps 1e9 tokens within a
      1.5 GB uint16 shard.

Usage:
    export HF_ENDPOINT=https://hf-mirror.com      # if HuggingFace is slow
    python -m data_prep.train_tokenizer --target-mb 300
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import sentencepiece as spm

OUT_DIR = pathlib.Path("tokenizer")
CORPUS = pathlib.Path("data/raw/tokenizer_corpus.txt")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_corpus(target_mb: int, dataset: str, config: str) -> tuple[int, int]:
    """Stream FineWeb and dump raw text until we hit the byte budget.

    Writes to a .tmp and renames only on success, so a dropped connection
    cannot leave a half-corpus that later trains a quietly worse tokenizer.
    """
    from datasets import load_dataset

    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    tmp = CORPUS.with_suffix(".txt.tmp")
    target_bytes = target_mb * 1024 * 1024
    ds = load_dataset(dataset, name=config, split="train", streaming=True)

    written = docs = 0
    first_file = None
    t0 = last = time.time()
    with open(tmp, "w", encoding="utf-8") as f:
        for ex in ds:
            if first_file is None:
                first_file = ex.get("file_path")
            text = ex["text"]
            f.write(text)
            f.write("\n")
            written += len(text.encode("utf-8")) + 1
            docs += 1
            if time.time() - last > 15:
                last = time.time()
                print(f"  {docs:,} docs  {written/1024**2:.0f}/{target_mb} MB  "
                      f"{written/1024**2/max(time.time()-t0, 1e-9):.1f} MB/s", flush=True)
            if written >= target_bytes:
                break

    if written < 0.5 * target_bytes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"stream ended after {written/1024**2:.0f} MB, wanted {target_mb} MB; "
            f"a partial corpus would silently degrade the vocabulary"
        )
    tmp.replace(CORPUS)
    print(f"corpus: {docs:,} docs, {written/1024**2:.1f} MB, first file_path={first_file}")
    return docs, written


def train_spm(vocab_size: int, num_threads: int) -> pathlib.Path:
    model_prefix = str(OUT_DIR / f"spm{vocab_size}")
    spm.SentencePieceTrainer.train(
        input=str(CORPUS),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        byte_fallback=True,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        pad_id=0,
        eos_id=1,
        bos_id=2,
        unk_id=3,
        input_sentence_size=4_000_000,
        shuffle_input_sentence=True,
        num_threads=num_threads,
        minloglevel=1,
    )
    return pathlib.Path(model_prefix + ".model")


def verify(model_path: pathlib.Path, vocab_size: int) -> dict:
    """Report the numbers that actually matter downstream."""
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    assert sp.vocab_size() == vocab_size, f"vocab {sp.vocab_size()} != {vocab_size}"

    samples = [
        "The quick brown fox jumps over the lazy dog.",
        "In 2024, the model achieved 42.7% accuracy on MMLU (5-shot).",
        "def forward(self, x):\n    return x @ self.w",
        "中文测试：这是一个关于缩放定律的项目。",
        "emoji 🤖 and combining a\u0301 accented chars",
        "   leading and   multiple   spaces   ",
    ]
    print("\nround-trip check:")
    ok = True
    for s in samples:
        ids = sp.encode(s)
        back = sp.decode(ids)
        same = back == s
        ok &= same
        print(f"  {'OK ' if same else 'DIFF'} {len(ids):>3} tok | {s[:58]!r}")
        if not same:
            print(f"        decoded -> {back!r}")

    # unknown rate must be exactly 0 thanks to byte_fallback
    with open(CORPUS, encoding="utf-8") as f:
        test_text = f.read(2_000_000)   # read a prefix, not the whole 300 MB
    ids = sp.encode(test_text)
    n_unk = sum(1 for i in ids if i == sp.unk_id())
    n_bytes = len(test_text.encode("utf-8"))
    n_tok = len(ids)
    bpt = n_bytes / n_tok

    # 1e9 tokens at this compression -> shard size on disk
    gib = 1e9 * 2 / 1024**3

    print(f"\n  unk tokens        : {n_unk}   (must be 0)")
    print(f"  bytes/token       : {bpt:.2f}")
    print(f"  1e9 tokens on disk: {gib:.2f} GiB (uint16)")
    print(f"  1e9 tokens of text: {1e9*bpt/1024**3:.2f} GiB raw")
    print(f"  round-trip exact  : {ok}")
    return {
        "vocab_size": vocab_size,
        "unk_count": n_unk,
        "bytes_per_token": round(bpt, 3),
        "roundtrip_exact": ok,
        "corpus_bytes": CORPUS.stat().st_size,
        "corpus_sha256": sha256(CORPUS),
        "model_sha256": sha256(model_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-mb", type=int, default=300,
                    help="raw text budget for tokenizer training (300MB ~= 75M tokens)")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--reuse-corpus", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.reuse_corpus and CORPUS.exists():
        print(f"reusing corpus {CORPUS} ({CORPUS.stat().st_size/1024**2:.0f} MB)")
    else:
        print(f"streaming {args.dataset}/{args.config} -> {CORPUS}")
        build_corpus(args.target_mb, args.dataset, args.config)

    t0 = time.time()
    model_path = train_spm(args.vocab_size, args.threads)
    print(f"\ntrained {model_path} in {time.time()-t0:.0f}s")

    stats = verify(model_path, args.vocab_size)
    meta = {"tokenizer": model_path.name, "corpus": str(CORPUS), **stats}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT_DIR/'meta.json'}")

    hard_fail = stats["unk_count"] != 0 or not stats["roundtrip_exact"]
    print("\nTOKENIZER GATE: " + ("FAIL" if hard_fail else "PASS"))
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
