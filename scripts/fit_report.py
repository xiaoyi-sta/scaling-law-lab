"""End-to-end report: from logs/*.csv to the numbers and intervals in the README.

Pipeline (each step's output is written to results/report.json):

  1. load observations, keep the two validation sets separate
  2. estimate sigma_run from replicate seeds          <- the weight that matters
  3. weighted NLS on M1, plus asymptotic / cluster / two-level bootstrap CIs
  4. M1 vs M2 vs M3 by AIC/BIC, nested F test for M1 vs M2
  5. extrapolation: fit {8M all, 20M first 3} -> predict 20M@1B and all of 50M;
     report coverage, Winkler score and MAPE (no "must hit 4/4" gate)
  6. ablation: joint model A/N^a + B_g/D^b + E, gain (B_A/B_B)^(1/beta) + CI
  7. optimal allocation D/N at fixed compute

Usage:
    python -m scripts.fit_report --synthetic    # test the plumbing, no GPU needed
    python -m scripts.fit_report                # real logs
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, ".")

from scaling.config import PRESETS                                    # noqa: E402
from scaling.fit import (aic_bic, ablation_gain, cluster_bootstrap,   # noqa: E402
                         f_test, fit_m2, fit_m3, jackknife_runs,
                         joint_ablation_fit, optimal_allocation,
                         predict, predict_interval,
                         sigma_run_from_replicates, winkler_score, wnls)

RESULTS = pathlib.Path("results")
NAMES = ["A", "B", "alpha", "beta", "E"]


# ------------------------------------------------------------------ loading
def load_logs(include_synthetic: bool = False) -> dict[str, list[dict]]:
    """Read logs/*.csv.  Synthetic fixtures are excluded unless explicitly asked for,
    otherwise a leftover test file silently contaminates the real fit."""
    out: dict[str, list[dict]] = collections.defaultdict(list)
    for f in sorted(glob.glob("logs/*.csv")):
        if "synthetic" in f and not include_synthetic:
            continue
        with open(f) as fh:
            for r in csv.DictReader(fh):
                vs = (r.get("val_set") or "A").strip()
                model = r["model_size"]
                if model not in PRESETS:
                    continue
                out[vs].append({
                    "run_id": r["run_id"], "model": model,
                    "seed": int(r["seed"]), "step": int(r["step"]),
                    "N": PRESETS[model].n_params_nonemb,
                    "D": int(r["tokens_seen"]),
                    "L": float(r["val_loss"]),
                    "se": float(r["val_loss_se_doc"]) if r.get("val_loss_se_doc") else 0.0,
                })
    return out


def arm_of(run_id: str) -> str:
    """Corpus arm from the run id (…_A, …_B, …_A2, …_B2).

    Needed because arm A and arm B are evaluated on the same validation set, so
    their rows land in the same (N, D) cell; pooling them as if they were
    replicate seeds inflates sigma_run from 0.012 to 0.03.
    """
    tail = run_id.rsplit("_", 1)[-1]
    return tail if tail in {"A", "B", "A2", "B2"} else "?"


def arrays(rows: list[dict]):
    runs = {rid: i for i, rid in enumerate(sorted({x["run_id"] for x in rows}))}
    N = np.array([r["N"] for r in rows], float)
    D = np.array([r["D"] for r in rows], float)
    L = np.array([r["L"] for r in rows], float)
    se = np.array([r["se"] for r in rows], float)
    g = np.array([runs[r["run_id"]] for r in rows])
    return N, D, L, se, g


# --------------------------------------------------------------- synthetic
def write_synthetic_logs(sigma_run: float = 0.012, gain: float = 1.28,
                         sigma_obs: float = 0.001, seed: int = 7) -> None:
    """Fabricate logs/*.csv with the real schema so the plumbing can be tested offline.

    Arm B is generated with B_B = B_A / gain**beta, so the report's recovered
    ablation gain should come back to `gain`.
    """
    from scaling.fit import predict
    A, B, alpha, beta, E = 400.0, 400.0, 0.34, 0.28, 1.69
    B_B = B / gain ** beta
    theta = np.array([A, B, alpha, beta, E])
    theta_b = np.array([A, B_B, alpha, beta, E])

    plan = [("8M", 3), ("20M", 1), ("50M", 1)]
    fracs = (.25, .5, .75, 1.0)
    rng = np.random.default_rng(seed)
    pathlib.Path("logs").mkdir(exist_ok=True)

    for vs, th in (("A", theta), ("B", theta_b)):
        rows = []
        for model, n_seeds in plan:
            cfg = PRESETS[model]
            if vs == "B" and model != "20M":
                continue                       # ablation only runs at 20M
            N = cfg.n_params_nonemb
            steps_total = int(1e9 // 131_072) if model != "8M" else int(5e8 // 131_072)
            for s in range(n_seeds):
                run_id = f"{model}_seed{s}_{vs}"
                u = rng.normal(0, sigma_run)
                for f in fracs:
                    step = int(round(f * steps_total))
                    D = step * 131_072
                    loss = predict(th, np.array([N]), np.array([float(D)]))[0]
                    loss += u + rng.normal(0, sigma_obs)
                    rows.append(dict(run_id=run_id, model_size=model, seed=s,
                                     tokens_seen=D, step=step, val_set=vs,
                                     val_loss=f"{loss:.6f}",
                                     val_loss_se_doc=f"{0.0015 + 0.0005*rng.random():.6f}",
                                     n_doc=8882 if vs == "A" else 5443,
                                     wall_time="0.0", lr="1e-4"))
        f = pathlib.Path(f"logs/synthetic_{vs}.csv")
        with open(f, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader(); wr.writerows(rows)
        print(f"  wrote {f} ({len(rows)} rows)")


# ------------------------------------------------------------------ report
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.synthetic:
        write_synthetic_logs(sigma_run=0.012, gain=1.28)

    logs = load_logs(include_synthetic=args.synthetic)
    if not logs:
        print("no logs/*.csv found -- run training first, or use --synthetic")
        return 1

    RESULTS.mkdir(exist_ok=True)
    report: dict = {}
    md: list[str] = ["# Scaling-law report", ""]
    md.append(f"observations: " + ", ".join(f"val{vs}={len(v)}" for vs, v in sorted(logs.items())))
    md.append("")

    # ---- 1. sigma_run ---------------------------------------------------
    # sigma_run must come from replicate SEEDS of one corpus arm, never across arms
    seed_rows = [r for r in logs.get("A", []) if arm_of(r["run_id"]) == "A"]
    N, D, L, se, g = arrays(seed_rows)
    Nc = N; Dc = D
    cells = {}
    for i in range(len(L)):
        cells.setdefault((Nc[i], Dc[i]), []).append(i)
    sr = sigma_run_from_replicates(N, D, L, g) if len(seed_rows) else {"sigma_run": None, "df": 0, "n_cells": 0}
    report["sigma_run"] = sr
    if sr["sigma_run"] is None:
        print("WARNING: no replicate seeds -> sigma_run not identifiable; "
              "weights fall back to 1/se^2 (which is the v2 mistake)")
        sigma_run = 0.0
    else:
        sigma_run = sr["sigma_run"]
        md += [f"## 1. sigma_run",
               f"- sigma_run = **{sr['sigma_run']:.4f}** nats "
               f"(95% chi2 CI [{sr['ci_lo']:.4f}, {sr['ci_hi']:.4f}], df={sr['df']})",
               f"- replicate cells = {sr['n_cells']}", ""]

    w = 1.0 / (sigma_run ** 2 + np.maximum(se, 1e-6) ** 2)

    # ---- 2. main fit (val set A) ---------------------------------------
    Na, Da, La, sea, ga = arrays(logs.get("A", []))
    if Na.size < 6:
        print("not enough observations for a 5-parameter fit yet")
        return 1
    wa = 1.0 / (sigma_run ** 2 + np.maximum(sea, 1e-6) ** 2)
    fit = wnls(Na, Da, La, wa, seed=args.seed)
    boot = cluster_bootstrap(Na, Da, La, ga, wa, B=args.bootstrap, seed=args.seed)
    jk = jackknife_runs(Na, Da, La, ga, wa)

    md += ["## 2. M1 fit (val set A)", "",
           "| param | estimate | asymptotic 95% | two-level bootstrap 95% | LOO-run range |",
           "|---|---|---|---|---|"]
    report["fit"] = fit.as_dict()
    report["bootstrap"] = {
        "n": int(len(boot)),
        "ci": {nm: [float(np.quantile(boot[:, i], .025)),
                    float(np.quantile(boot[:, i], .975))] for i, nm in enumerate(NAMES)},
    } if len(boot) > 10 else {"n": 0}
    for i, nm in enumerate(NAMES):
        est, a_lo, a_hi = fit.theta[i], fit.theta[i] - 1.96 * fit.se[i], fit.theta[i] + 1.96 * fit.se[i]
        if len(boot) > 10:
            b_lo, b_hi = report["bootstrap"]["ci"][nm]
            bcell = f"[{b_lo:.4f}, {b_hi:.4f}]"
        else:
            bcell = "n/a"
        md.append(f"| {nm} | {est:.4f} | [{a_lo:.4f}, {a_hi:.4f}] | {bcell} | "
                  f"[{jk['min'][i]:.4f}, {jk['max'][i]:.4f}] |")
    md += ["", f"n = {fit.n}, runs = {jk['n_runs']}, weighted RSS = {fit.rss:.4e}", ""]

    # cross-check: unweighted vs weighted (the spec promises both)
    fit_u = wnls(Na, Da, La, None, seed=args.seed)
    md.append(f"unweighted fit, alpha = {fit_u.theta[2]:.4f} vs weighted {fit.theta[2]:.4f} "
              f"(difference {abs(fit_u.theta[2]-fit.theta[2]):.2e})")
    md.append("")

    # ---- 3. model comparison -------------------------------------------
    m2 = fit_m2(Na, Da, La, wa, seed=args.seed)
    m3 = fit_m3(Na, Da, La, wa)
    F, p = f_test(fit, m2)
    rows = []
    for nm, m in (("M1 A/N^a+B/D^b+E", fit), ("M2 A/N^a+B/D^b", m2), ("M3 a lnN+b lnD+c", m3)):
        aic, bic = aic_bic(m)
        rows.append((nm, m.k, m.rss, aic, bic))
    md += ["## 3. Function form", "", "| model | k | weighted RSS | AIC | BIC |", "|---|---|---|---|---|"]
    for nm, k, rss, aic, bic in rows:
        md.append(f"| {nm} | {k} | {rss:.4e} | {aic:.2f} | {bic:.2f} |")
    md += ["", f"- nested F test M1 vs M2: F({fit.k-m2.k}, {fit.dof}) = {F:.3f}, p = {p:.4g}",
           f"- M3 coefficients: {', '.join(f'{k}={v:.4f}' for k, v in m3.params.items())}", ""]
    report["model_comparison"] = {
        "F": F, "p": p,
        "models": [{"name": nm, "k": k, "rss": rss, "aic": aic, "bic": bic}
                   for nm, k, rss, aic, bic in rows]}

    # ---- 4. extrapolation ----------------------------------------------
    # fit on {8M all 4 points} + {20M first 3 points}; predict 20M@1B and all 50M
    N8 = PRESETS["8M"].n_params_nonemb
    N20 = PRESETS["20M"].n_params_nonemb
    mask_fit = (Na == N8)
    if np.isin(N20, Na):
        d20max = Da[Na == N20].max()
        mask_fit |= (Na == N20) & (Da < d20max)
    if mask_fit.sum() >= 6:
        fit_x = wnls(Na[mask_fit], Da[mask_fit], La[mask_fit], wa[mask_fit], seed=args.seed)
        boot_x = cluster_bootstrap(Na[mask_fit], Da[mask_fit], La[mask_fit],
                                   ga[mask_fit], wa[mask_fit], B=min(args.bootstrap, 500),
                                   seed=args.seed + 1)
        mask_tst = ~mask_fit
        if mask_tst.sum() > 0:
            pi = predict_interval(fit_x, Na[mask_tst], Da[mask_tst], sigma_run=sigma_run,
                                  sigma_obs=float(np.mean(sea)), boot_thetas=boot_x)
            y = La[mask_tst]
            covered = (pi["lo"] <= y) & (y <= pi["hi"])
            ws = winkler_score(pi["lo"], pi["hi"], y)
            mape = float(np.mean(np.abs(pi["pred"] - y) / y) * 100)
            md += ["## 4. Extrapolation", "",
                   f"fit on {int(mask_fit.sum())} points, predict {int(mask_tst.sum())}", "",
                   "| N | D | actual | predicted | 95% PI | covered |", "|---|---|---|---|---|---|"]
            # pi is indexed by position within the test subset, not by the global row
            test_idx = np.where(mask_tst)[0]
            for k, j in enumerate(test_idx):
                md.append(f"| {Na[j]:,.0f} | {Da[j]:,.0f} | {La[j]:.4f} | {pi['pred'][k]:.4f} | "
                          f"[{pi['lo'][k]:.4f}, {pi['hi'][k]:.4f}] | {'yes' if covered[k] else 'NO'} |")
            md += ["", f"- coverage: {int(covered.sum())}/{int(covered.size)}",
                   f"- mean Winkler score: {float(ws.mean()):.4f} (lower is better; "
                   f"rewards sharpness, not just coverage)",
                   f"- MAPE: {mape:.2f}%"
                   + ("" if mape < 5 else "  <-- above the 5% target, see spec 10.3"), ""]
            report["extrapolation"] = {
                "n_fit": int(mask_fit.sum()), "n_test": int(mask_tst.sum()),
                "coverage": [int(covered.sum()), int(covered.size)],
                "winkler_mean": float(ws.mean()), "mape_pct": mape,
                "points": [{"N": float(Na[j]), "D": float(Da[j]), "actual": float(La[j]),
                            "pred": float(pi["pred"][k]), "lo": float(pi["lo"][k]),
                            "hi": float(pi["hi"][k])}
                           for k, j in enumerate(np.where(mask_tst)[0])]}

    # ---- 5. ablation ----------------------------------------------------
    if logs.get("B"):
        Nb, Db, Lb, seb, gb = arrays(logs["B"])
        # joint fit needs arm A at the same N, so restrict to the size B uses
        # ALL of arm A pins A/alpha/beta/E; the 4 B points only need B_B.  Using
        # just the 20M slice would give 8 observations for 6 parameters.
        keep = np.ones_like(Na, bool)
        Nj = np.concatenate([Na[keep], Nb]); Dj = np.concatenate([Da[keep], Db])
        Lj = np.concatenate([La[keep], Lb]); is_b = np.concatenate(
            [np.zeros(keep.sum(), bool), np.ones(Nb.size, bool)])
        wj = np.concatenate([wa[keep], 1.0 / (sigma_run ** 2 + np.maximum(seb, 1e-6) ** 2)])
        jf = joint_ablation_fit(Nj, Dj, Lj, is_b, wj, seed=args.seed)
        gain = ablation_gain(jf["B_A"], jf["B_B"], jf["beta"])

        # bootstrap the whole joint fit for the gain's CI
        rng = np.random.default_rng(args.seed)
        runs_j = np.unique(np.concatenate([ga[keep], gb + 10_000]))
        gid = np.concatenate([ga[keep], gb + 10_000])
        gains = []
        for _ in range(min(args.bootstrap, 500)):
            pick = rng.choice(runs_j, size=runs_j.size, replace=True)
            idx = np.concatenate([np.where(gid == p)[0] for p in pick])
            try:
                f2 = joint_ablation_fit(Nj[idx], Dj[idx], Lj[idx], is_b[idx], wj[idx],
                                        n_starts=3, seed=int(rng.integers(1 << 30)))
                gg = ablation_gain(f2["B_A"], f2["B_B"], f2["beta"])
                if 0.05 < f2["beta"] < 1.5 and 0.2 < gg < 10:
                    gains.append(gg)
            except Exception:
                continue
        gains = np.asarray(gains)
        lo, hi = (np.quantile(gains, [.025, .975]) if gains.size > 10 else (np.nan, np.nan))

        # per-breakpoint DeltaL, model-free; arm B only exists at one model size
        n_arm = float(Nb[0])
        md += ["## 5. Ablation (arm B minus arm A, same N and D)", "",
               "| D | valA | valB | DeltaL |", "|---|---|---|---|"]
        for d in sorted(set(Db.tolist())):
            ma = (Da == d) & (Na == n_arm); mb = Db == d
            if ma.any() and mb.any():
                md.append(f"| {d:,.0f} | {La[ma].mean():.4f} | {Lb[mb].mean():.4f} | "
                          f"{Lb[mb].mean() - La[ma].mean():+.4f} |")
        md += ["", f"- joint model: B_A = {jf['B_A']:.4f}, B_B = {jf['B_B']:.4f}, "
                   f"beta = {jf['beta']:.4f}",
               f"- token-equivalent gain of corpus B = **{gain:.3f}x** "
               f"(95% cluster bootstrap CI [{lo:.3f}, {hi:.3f}])" if gains.size > 10 else
               f"- token-equivalent gain = {gain:.3f}x (CI unavailable, too few runs)",
               "- interpretation: this is a CORPUS-level effect. A and B share only "
               "0.011% of documents and 2/16 crawl dumps, so it bundles the quality "
               "filter with crawl composition. See spec 4.2.", ""]
        report["ablation"] = {"B_A": jf["B_A"], "B_B": jf["B_B"], "beta": jf["beta"],
                              "gain": gain, "gain_ci": [float(lo), float(hi)],
                              "n_boot": int(gains.size)}

    # ---- 6. optimal allocation -----------------------------------------
    alloc = []
    for label, C in [("8M budget", 6 * PRESETS["8M"].n_params_nonemb * 5e8),
                     ("20M budget", 6 * PRESETS["20M"].n_params_nonemb * 1e9),
                     ("50M budget", 6 * PRESETS["50M"].n_params_nonemb * 1e9)]:
        a = optimal_allocation(fit.theta, C)
        alloc.append({"budget": label, "C": C, **a})
    md += ["## 6. Optimal allocation (D/N)", "",
           "| budget | C (FLOPs) | N | D | D/N |", "|---|---|---|---|---|"]
    for a in alloc:
        md.append(f"| {a['budget']} | {a['C']:.3e} | {a['N']:,.0f} | {a['D']:,.0f} | "
                  f"{a['D_over_N']:.2f} |")
    md += ["", "Note: N only spans 6.4x here, so the exponent and therefore this ratio are "
               "not comparable to Chinchilla's ~20. Report the interval, not the point.", ""]
    report["allocation"] = alloc

    (RESULTS / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (RESULTS / "report.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {RESULTS/'report.md'} and {RESULTS/'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
