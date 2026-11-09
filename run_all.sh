#!/usr/bin/env bash
# Full GPU queue, resumable.  Re-run this exact command after any crash or after
# stopping the instance: finished runs are skipped, half-finished ones continue
# from their last checkpoint (which restores weights, optimizer, step AND the
# exact position in the token stream).
#
# Order is deliberate: cheapest runs first, so a budget overrun costs you the
# optional tail, not the scaling-law spine.
#
#   bash run_all.sh            # everything
#   SKIP_OPTIONAL=1 bash run_all.sh   # drop the 20M second seed
#
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export PYTHONUNBUFFERED=1

mkdir -p logs ckpts

# ---- preflight -------------------------------------------------------------
$PY - <<'EOF'
import sys, pathlib, torch
need = ["data/proc/trainA.bin", "data/proc/trainA_doc_lens.npy",
        "data/proc/trainB.bin", "data/proc/trainB_doc_lens.npy",
        "data/proc/val.bin", "data/proc/val_doc_lens.npy",
        "data/proc/valB.bin", "data/proc/valB_doc_lens.npy",
        "data/proc/trainA2.bin", "data/proc/trainB2.bin"]
missing = [p for p in need if not pathlib.Path(p).exists()]
if missing:
    print("MISSING:", *missing, sep="\n  "); sys.exit(1)
print(f"torch {torch.__version__}  cuda {torch.cuda.is_available()}  "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'}")
if not torch.cuda.is_available():
    print("WARNING: no CUDA device -- this queue is meant for a rented GPU")
EOF
[ $? -ne 0 ] && { echo "preflight failed"; exit 1; }

latest_ckpt() {
  ls -1 "ckpts/$1"_*.pt 2>/dev/null \
    | awk -F_ '{n=$NF; sub(/\.pt$/,"",n); print n" "$0}' \
    | sort -n | tail -1 | cut -d' ' -f2-
}

run() {
  local model=$1 arm=$2 seed=$3 rid=$4; shift 4
  local ckpt; ckpt=$(latest_ckpt "$rid")
  local args=(--model "$model" --arm "$arm" --seed "$seed" --run-id "$rid" --val-sets A,B)
  [ -n "${MICRO_SEQS:-}" ] && args+=(--micro-seqs "$MICRO_SEQS")
  if [ -n "$ckpt" ]; then
    echo ">>> RESUME $rid from $(basename "$ckpt")"
    args+=(--resume "$ckpt")
  else
    echo ">>> START  $rid"
  fi
  $PY train.py "${args[@]}" "$@" 2>&1 | tee "logs/${rid}.out"
  local rc=${PIPESTATUS[0]}
  [ $rc -ne 0 ] && { echo "!!! $rid FAILED (rc=$rc). Re-run this script to resume."; return $rc; }
  echo "<<< DONE   $rid"
}

T0=$(date +%s)
echo "================ bench first, then queue ================"
# Measure throughput before committing hours; skip with BENCH=0.
if [ "${BENCH:-1}" = "1" ]; then
  run 50M A 0 bench50M --steps 20 --no-eval
  echo "--- read the 'k tok/s' line above; 50M x 1e9 tokens needs"
  echo "    1e9 / rate / 3600 hours.  Abort now if that exceeds your budget."
fi

echo "================ scaling-law spine ================"
run 8M  A 0 8M_seed0_A
run 8M  A 1 8M_seed1_A
run 8M  A 2 8M_seed2_A
run 20M A 0 20M_seed0_A
run 50M A 0 50M_seed0_A

echo "================ ablation arm B ================"
run 20M B 0 20M_seed0_B

echo "================ optional: second 20M seed ================"
if [ "${SKIP_OPTIONAL:-0}" != "1" ]; then
  run 20M A 1 20M_seed1_A
fi

echo "================ batch-controlled supplement ================"
# Same two crawl dumps on both sides: A2 vs B2 isolates the filter from the
# crawl mix.  120M tokens each -> ~5 min per run, and it is what lets the
# resume line say "filtering" instead of "a different pile of text".
run 20M A2 0 20M_seed0_A2 --steps 915
run 20M B2 0 20M_seed0_B2 --steps 915

echo
echo "================ summary ================"
$PY - <<'EOF'
import csv, glob, collections
rows = collections.defaultdict(list)
for f in glob.glob("logs/*.csv"):
    if "bench" in f:
        continue
    for r in csv.DictReader(open(f)):
        rows[(r["run_id"], r["model_size"])].append(r)
print(f"{'run':<18}{'valA@final':>12}{'valB@final':>12}")
for (rid, m), rs in sorted(rows.items()):
    last = {}
    for r in rs:
        last[r["val_set"]] = r
    a = last.get("A", {}).get("val_loss", "-")
    b = last.get("B", {}).get("val_loss", "-")
    print(f"{rid:<18}{a:>12}{b:>12}")
EOF
echo "total wall clock: $(( ($(date +%s) - T0) / 60 )) min"
