"""Turn the raw text streams into uint16 .bin shards, with the isolation the spec demands.

Layout produced under data/proc/:

    val.bin              ~8e6 tokens, held-out, NEVER used for training
    val_doc_lens.npy     int32 per-document lengths  -> document-level SE, cluster bootstrap
    val_ids.json         the HF document ids that went into val
    trainA.bin           ~1.1e9 tokens of fineweb      (arm A: raw)
    trainA_doc_lens.npy  document boundaries -> per-seed doc-order shuffling
    trainB.bin           ~1.1e9 tokens of fineweb-edu  (arm B: filtered)
    trainB_doc_lens.npy
    manifest.json        shards used, counts, hashes, disjointness proofs

TRANSPORT: whole parquet files are downloaded once with hf_hub_download and then
read locally with pyarrow.  Streaming parquet through `datasets` issues one HTTP
range request per block, which blows through HuggingFace's resolver quota
(100k requests / 5 min) after a few hundred MB -- it fails at the *first* big arm
and leaves a truncated .tmp behind.  Local files make the HTTP cost O(#files).

Three invariants this script enforces, because each one silently destroys the study:

1.  val comes from a shard that no training arm touches      -> asserted on file names
2.  no val document leaks into the edu arm                    -> asserted on HF `id`
    (fineweb-edu is a *filter of fineweb*, so an edu document can be a val document;
     different filenames prove nothing here)
3.  every .bin is written to .tmp and renamed                  -> a killed job cannot
    leave a truncated shard that trains fine and is quietly wrong

Token counts are measured from what actually landed on disk, never assumed.

Usage:
    export HF_ENDPOINT=https://hf-mirror.com
    python -m data_prep.build_bins --arm all      # ~1 h, ~10 GB download, ~4.1 GB disk
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import sentencepiece as spm
from huggingface_hub import hf_hub_download

PROC = pathlib.Path("data/proc")
CACHE = pathlib.Path("data/raw/parquet")
VAL_TOKENS = 8_000_000
TRAIN_TOKENS = 1_100_000_000
EOS = 1

FINEWEB = "HuggingFaceFW/fineweb"
FINEWEB_EDU = "HuggingFaceFW/fineweb-edu"

# Val lives in the LAST fineweb shard; training arms use the first ones.  Keeping
# the two sets disjoint by shard is the cheapest possible guarantee, and it is
# checked rather than assumed.  Each shard is ~0.67B tokens, so 1.1e9 needs two.
VAL_SHARD = "sample/10BT/014_00000.parquet"
# Second validation set, drawn from fineweb-edu.  The single raw-web val set sits
# in-distribution for arm A and 4x farther (JSD 0.133 vs 0.033) from arm B, so the
# ablation would be graded on A's home turf.  Reporting Delta_L on both sets makes
# that bias visible instead of silent.
VAL_B_SHARD = "sample/100BT/005_00000.parquet"
# Batch-controlled pair: the only two crawl dumps present in BOTH samples.
# Same crawl batches -> the A2/B2 difference is the filter, not the crawl mix.
SHARED_DUMPS = ["CC-MAIN-2013-20", "CC-MAIN-2013-48"]
BATCHCTL_TOKENS = 120_000_000
TRAIN_A_SHARDS = ["sample/10BT/000_00000.parquet", "sample/10BT/001_00000.parquet"]
TRAIN_B_SHARDS = ["sample/100BT/000_00000.parquet", "sample/100BT/000_00001.parquet"]

FLUSH_TOKENS = 20_000_000


def ensure_local(repo: str, files: list[str]) -> list[pathlib.Path]:
    """One HTTP request per file instead of one per 512 KB block.

    hf_hub_download(local_dir=...) mirrors the repo-relative layout underneath
    local_dir, so the destination is local_dir/<filename>, not local_dir/<basename>.
    """
    root = CACHE / repo.split("/")[-1]
    out = []
    for f in files:
        dest = root / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            print(f"    cached  {dest.name}  {dest.stat().st_size/2**30:.2f} GiB", flush=True)
        else:
            print(f"    get     {dest.name} ...", flush=True)
            try:
                hf_hub_download(repo_id=repo, filename=f, repo_type="dataset",
                                local_dir=str(root))
            except Exception as e:
                if "429" in str(e):
                    raise SystemExit(
                        f"\n429 while downloading {f}.\n"
                        f"HuggingFace rate limit (100k resolver requests / 5 min). "
                        f"Wait ~5 min and re-run: finished arms are skipped, so this resumes cheaply.\n"
                    ) from e
                raise
            if not dest.exists():
                raise SystemExit(f"download reported success but {dest} is missing")
        out.append(dest)
    return out


def iter_docs(paths: list[pathlib.Path]):
    """Yield (doc_id, text, dump) from local parquet, batch by batch, no network."""
    for p in paths:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=2048, columns=["id", "text", "dump"]):
            d = batch.to_pydict()
            for doc_id, text, dump in zip(d["id"], d["text"], d["dump"]):
                if text:
                    yield doc_id, text, dump


def build_arm(name: str, repo: str, files: list[str], target: int, sp,
              skip_ids: set[str] | None = None, keep_ids: bool = False,
              force: bool = False, dumps: list[str] | None = None):
    """Stream documents -> flat uint16 shard + per-document lengths."""
    out = PROC / f"{name}.bin"
    lens_path = PROC / f"{name}_doc_lens.npy"
    tmp = PROC / f"{name}.bin.tmp"
    lens_tmp = PROC / f"{name}_doc_lens.npy.tmp"

    if out.exists() and lens_path.exists() and not force:
        n = int(np.load(lens_path, mmap_mode="r").sum())
        print(f"    exists  {out.name}  {n:,} tok  (use --force to rebuild)")
        return {"tokens": n, "docs": int(len(np.load(lens_path))), "skipped_docs": 0,
                "bytes_on_disk": out.stat().st_size, "shards": files, "repo": repo,
                "reused": True}

    tmp.unlink(missing_ok=True)          # drop a truncated shard from any earlier crash
    print(f"  {name} <- {repo}")
    local = ensure_local(repo, files)

    written = docs = skipped = 0
    buf: list[np.ndarray] = []
    buf_len = 0
    lens: list[int] = []
    ids_seen: list[str] = []
    t0 = last = time.time()

    with open(tmp, "wb") as f:
        for doc_id, text, dump in iter_docs(local):
            if skip_ids is not None and doc_id in skip_ids:
                skipped += 1
                continue
            if dumps is not None and dump not in dumps:
                skipped += 1
                continue
            ids = np.asarray(sp.encode(text), dtype=np.uint16)
            if ids.size == 0:
                continue
            ids = np.append(ids, np.uint16(EOS))
            buf.append(ids)
            buf_len += ids.size
            lens.append(int(ids.size))
            if keep_ids:
                ids_seen.append(doc_id)
            docs += 1
            written += ids.size

            if buf_len >= FLUSH_TOKENS:
                f.write(np.concatenate(buf).tobytes())
                buf, buf_len = [], 0
            if time.time() - last > 15:
                last = time.time()
                el = time.time() - t0
                print(f"    {docs:,} docs  {written/1e6:.1f}M tok  "
                      f"{written/el/1e3:.0f}k tok/s  eta {(target-written)/max(written/el,1)/60:.0f} min",
                      flush=True)
            if written >= target:
                break
        if buf:
            f.write(np.concatenate(buf).tobytes())

    if written < 0.9 * target:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"{name}: only {written:,} tokens (<90% of {target:,}); "
                         f"add another shard to the list and re-run")

    # np.save() appends ".npy" when handed a str/Path, which would silently
    # write val_doc_lens.npy.tmp.npy and break the atomic rename below.
    with open(lens_tmp, "wb") as lf:
        np.save(lf, np.asarray(lens, dtype=np.int32))
    tmp.replace(out)
    lens_tmp.replace(lens_path)

    info = {
        "tokens": int(written), "docs": int(docs), "skipped_docs": int(skipped),
        "bytes_on_disk": out.stat().st_size, "shards": files, "repo": repo,
        "add_eos": True, "dtype": "uint16",
    }
    print(f"    -> {out.name}  {written/1e9:.3f}B tok  {docs:,} docs  "
          f"{out.stat().st_size/2**30:.2f} GiB  skipped {skipped:,}")
    if keep_ids:
        (PROC / f"{name}_ids.json").write_text(json.dumps(ids_seen))
        info["ids_file"] = f"{name}_ids.json"
    return info


def verify_shard(name: str, sp) -> dict:
    """Re-read from disk and prove the tokens are where we think they are."""
    path = PROC / f"{name}.bin"
    mm = np.memmap(path, dtype=np.uint16, mode="r")
    lens = np.load(PROC / f"{name}_doc_lens.npy")
    n = mm.size
    assert int(lens.sum()) == n, f"{name}: doc_lens sum {lens.sum()} != tokens on disk {n}"
    text = sp.decode(mm[:40].tolist())
    print(f"  {name}: {n:,} tok, {len(lens):,} docs, first 40 tokens -> {text[:60]!r}")
    return {"tokens": int(n), "docs": int(len(lens)), "first_tokens_decode": text[:60]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="all",
                    choices=["val", "valB", "A", "B", "A2", "B2", "batchctl", "all"])
    ap.add_argument("--tokenizer", default="tokenizer/spm8192.model")
    ap.add_argument("--val-tokens", type=int, default=VAL_TOKENS)
    ap.add_argument("--train-tokens", type=int, default=TRAIN_TOKENS)
    ap.add_argument("--batchctl-tokens", type=int, default=BATCHCTL_TOKENS)
    ap.add_argument("--force", action="store_true", help="rebuild even if the .bin exists")
    args = ap.parse_args()

    PROC.mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    print(f"tokenizer {args.tokenizer}  vocab {sp.vocab_size()}\n")
    manifest = {}

    if args.arm in ("val", "all"):
        assert VAL_SHARD not in TRAIN_A_SHARDS, "val shard must not be a training shard"
        manifest["val"] = build_arm("val", FINEWEB, [VAL_SHARD], args.val_tokens,
                                    sp, keep_ids=True, force=args.force)

    if args.arm in ("valB", "all"):
        assert VAL_B_SHARD not in TRAIN_B_SHARDS, "valB shard must not be a trainB shard"
        manifest["valB"] = build_arm("valB", FINEWEB_EDU, [VAL_B_SHARD], args.val_tokens,
                                     sp, keep_ids=True, force=args.force)

    if args.arm in ("A", "all"):
        manifest["trainA"] = build_arm("trainA", FINEWEB, TRAIN_A_SHARDS,
                                       args.train_tokens, sp, force=args.force)

    if args.arm in ("B", "all"):
        val_ids_path = PROC / "val_ids.json"
        if not val_ids_path.exists():
            raise SystemExit("build the val arm first: arm B needs the val id set to filter against")
        val_ids = set(json.loads(val_ids_path.read_text()))
        print(f"trainB excludes {len(val_ids):,} val ids")
        manifest["trainB"] = build_arm("trainB", FINEWEB_EDU, TRAIN_B_SHARDS,
                                       args.train_tokens, sp, skip_ids=val_ids,
                                       force=args.force)

    if args.arm in ("A2", "B2", "batchctl", "all"):
        # Same crawl dumps on both sides; only the quality filter differs.
        manifest["trainA2"] = build_arm("trainA2", FINEWEB, TRAIN_A_SHARDS,
                                        args.batchctl_tokens, sp, dumps=SHARED_DUMPS,
                                        force=args.force)
        manifest["trainB2"] = build_arm("trainB2", FINEWEB_EDU, TRAIN_B_SHARDS,
                                        args.batchctl_tokens, sp, dumps=SHARED_DUMPS,
                                        force=args.force)

    print("\nisolation checks:")
    problems = []
    if "val" in manifest:
        val_shards = {pathlib.Path(s).name for s in manifest["val"]["shards"]}
        for arm in ("trainA", "trainB"):
            if arm in manifest:
                shared = val_shards & {pathlib.Path(s).name for s in manifest[arm]["shards"]}
                if shared:
                    problems.append(f"{arm} shares a file with val: {shared}")
        print(f"  file-level val disjointness : {'FAIL' if problems else 'ok'}")
        if "trainB" in manifest:
            print(f"  edu arm filtered by val id  : {manifest['trainB']['skipped_docs']:,} docs dropped")

    print("\nverifying shards from disk:")
    for name in [k for k in ("val", "valB", "trainA", "trainB", "trainA2", "trainB2")
                 if k in manifest]:
        manifest[name].update(verify_shard(name, sp))

    manifest["tokenizer"] = {"path": args.tokenizer, "vocab_size": sp.vocab_size()}
    (PROC / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nwrote {PROC/'manifest.json'}")

    if problems:
        print("\nBIN BUILD: FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("\nBIN BUILD: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
