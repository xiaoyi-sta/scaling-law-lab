"""Measure what the ablation actually compares, instead of assuming it.

Two things this answers, and neither is computed by the val-id filter:

1.  |trainA ∩ trainB| -- how many *documents* the two arms share.  The val filter
    only compares arm B against val; it says nothing about A vs B.
2.  dump composition -- Common Crawl batches each arm is drawn from.  If the arms
    come from different batches, then Delta_L bundles the quality filter together
    with crawl-composition differences, and the README must say so.

Output: results/overlap.json + a table on stdout.

Usage: python -m scripts.check_overlap
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import pyarrow.parquet as pq

PARQUET = pathlib.Path("data/raw/parquet")
PROC = pathlib.Path("data/proc")

ARMS = {
    "A": [PARQUET / "fineweb/sample/10BT/000_00000.parquet",
          PARQUET / "fineweb/sample/10BT/001_00000.parquet"],
    "B": [PARQUET / "fineweb-edu/sample/100BT/000_00000.parquet",
          PARQUET / "fineweb-edu/sample/100BT/000_00001.parquet"],
}


def collect(paths, columns):
    out = {c: [] for c in columns}
    for p in paths:
        if not p.exists():
            raise SystemExit(f"missing {p} (run `make bins` first)")
        for b in pq.ParquetFile(p).iter_batches(batch_size=65536, columns=columns):
            for c in columns:
                out[c].extend(b.column(c).to_pylist())
    return out


def main() -> int:
    ids, dumps = {}, {}
    for arm, paths in ARMS.items():
        d = collect(paths, ["id", "dump"])
        ids[arm] = set(d["id"])
        dumps[arm] = collections.Counter(d["dump"])
        print(f"arm {arm}: {len(ids[arm]):,} unique docs, {len(dumps[arm])} dumps")

    shared = ids["A"] & ids["B"]
    denom = len(ids["A"]) * len(ids["B"])
    # expected overlap if both were uniform samples of the same document pool
    expected = denom / 1.33e9 if denom else 0.0
    print(f"\nA n B = {len(shared):,} docs")
    print(f"  = {len(shared)/len(ids['A'])*100:.4f}% of A, {len(shared)/len(ids['B'])*100:.4f}% of B")
    print(f"  under a same-pool independence model we would expect ~{expected:,.0f}"
          f" ({expected/max(len(shared),1):.0f}x more)")

    shared_dumps = set(dumps["A"]) & set(dumps["B"])
    print(f"\ndumps: A={len(dumps['A'])}, B={len(dumps['B'])}, shared={len(shared_dumps)}")
    print(f"  shared: {sorted(shared_dumps)}")
    print(f"  A top: {[d for d, _ in dumps['A'].most_common(3)]}")
    print(f"  B top: {[d for d, _ in dumps['B'].most_common(3)]}")

    val_path = PROC / "val_ids.json"
    val_check = {}
    if val_path.exists():
        V = set(json.loads(val_path.read_text()))
        val_check = {"val_docs": len(V),
                     "val_intersect_A": len(V & ids["A"]),
                     "val_intersect_B": len(V & ids["B"])}
        print(f"\nval: {len(V):,} docs;  val n A = {val_check['val_intersect_A']}, "
              f"val n B = {val_check['val_intersect_B']}")

    out = {
        "docs": {k: len(v) for k, v in ids.items()},
        "shared_docs": len(shared),
        "shared_frac_of_A": round(len(shared) / len(ids["A"]), 6),
        "expected_shared_if_same_pool": round(expected, 1),
        "dumps": {k: dict(v) for k, v in dumps.items()},
        "shared_dumps": sorted(shared_dumps),
        "val": val_check,
        "verdict": (
            "arms drawn from largely different crawl batches; Delta_L confounds the "
            "quality filter with crawl composition -- state this in the README"
            if len(shared_dumps) < min(len(dumps["A"]), len(dumps["B"])) / 2
            else "arms share most dumps; confound is limited"
        ),
    }
    pathlib.Path("results").mkdir(exist_ok=True)
    (pathlib.Path("results") / "overlap.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nverdict: {out['verdict']}")
    print("wrote results/overlap.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
