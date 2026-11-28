"""Self-check: verify the numbers, not the intentions.

Every check either recomputes a quantity by an independent route and compares, or
tests an algebraic identity that must hold exactly.  Run before publishing numbers.

    python -m scripts.selfcheck
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, ".")

from scaling.config import PRESETS                                          # noqa: E402
from scaling.fit import (ablation_gain, aic_bic, cluster_bootstrap, f_test,  # noqa: E402
                         fit_m2, fit_m3, optimal_allocation, predict,
                         predict_interval, sigma_run_from_replicates, wnls)
from scripts.fit_report import log_paths  # noqa: E402

OK = []
BAD = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (OK if cond else BAD).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


# ------------------------------------------------------------------ 1. logs
def check_logs():
    print("\n1. 训练日志完整性")
    rows = []
    for f in log_paths():
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    if not rows:
        check("logs 非空", False)
        return None

    tps = 131_072
    bad_tok = [r for r in rows if int(r["tokens_seen"]) != int(r["step"]) * tps]
    check("tokens_seen == step x 131072", not bad_tok,
          f"{len(bad_tok)} 行不合规" if bad_tok else f"{len(rows)} 行全部合规")

    keys = [(r["run_id"], r["step"], r["val_set"]) for r in rows]
    check("无重复 (run, step, val_set)", len(keys) == len(set(keys)),
          f"{len(keys)} 行 / {len(set(keys))} 唯一")

    losses = np.array([float(r["val_loss"]) for r in rows])
    ses = np.array([float(r["val_loss_se_doc"]) for r in rows])
    check("val_loss 全为有限值", bool(np.isfinite(losses).all()))
    check("val_loss_se_doc 为正", bool((ses > 0).all()), f"min={ses.min():.2e}")

    # schedule: 8M runs 3814 steps, 20M/50M 7629
    per_run = {}
    for r in rows:
        per_run.setdefault(r["run_id"], set()).add(int(r["step"]))
    bad_plan = []
    for rid, steps in per_run.items():
        model = rid.split("_")[0]
        if "--steps 915" in rid or rid.endswith(("_A2", "_B2")):
            expect_last = 915
        else:
            expect_last = 3814 if model == "8M" else 7629
        if max(steps) != expect_last:
            bad_plan.append((rid, max(steps), expect_last))
    check("每个 run 的最终步数与计划一致", not bad_plan, str(bad_plan[:3]) if bad_plan else "")

    # monotone decreasing loss within a run (training must actually help)
    bad_mono = []
    for rid, _ in per_run.items():
        for vs in ("A", "B"):
            seq = sorted([(int(r["step"]), float(r["val_loss"])) for r in rows
                          if r["run_id"] == rid and r["val_set"] == vs])
            if len(seq) > 1 and any(seq[i+1][1] > seq[i][1] for i in range(len(seq)-1)):
                bad_mono.append((rid, vs))
    check("每条曲线单调下降", not bad_mono, str(bad_mono[:3]) if bad_mono else "")
    return rows


# ------------------------------------------------------- 2. algebra / formulas
def check_formulas():
    print("\n2. 公式与代数恒等式")
    A, B, a, b, E = 412.3, 388.1, 0.342, 0.287, 1.712
    rng = np.random.default_rng(0)

    # noiseless recovery: the optimiser must return the generating parameters
    N, D, L = [], [], []
    for n, dmax, k in [(7_869_696, 5e8, 2), (19_473_024, 1e9, 1), (50_348_544, 1e9, 1)]:
        for _ in range(k):
            for f in (.25, .5, .75, 1.0):
                N.append(n); D.append(dmax * f)
                L.append(A / n**a + B / (dmax*f)**b + E)
    N, D, L = map(np.array, (N, D, L))
    fit = wnls(N, D, L, None, n_starts=8)
    err = [abs(fit.theta[0]-A), abs(fit.theta[1]-B), abs(fit.theta[2]-a),
           abs(fit.theta[3]-b), abs(fit.theta[4]-E)]
    check("无噪声数据能精确恢复参数", max(err) < 1e-3, f"最大误差 {max(err):.2e}")

    # sigma_run vs a hand computation on a tiny known table
    Nt = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
    Dt = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
    Lt = np.array([1.0, 1.2, 0.9, 2.0, 2.4])
    Gt = np.array([0, 1, 2, 0, 1])
    s_cells, df = 0.0, 0
    for key in np.unique(np.c_[Nt, Dt], axis=0):
        v = Lt[(Nt == key[0]) & (Dt == key[1])]
        s_cells += ((v - v.mean())**2).sum(); df += len(v) - 1
    manual = math.sqrt(s_cells / df)
    got = sigma_run_from_replicates(Nt, Dt, Lt, Gt)["sigma_run"]
    check("sigma_run 与手算一致", abs(got - manual) < 1e-12, f"{got:.6f} vs {manual:.6f}")

    # ablation gain: the identity it claims must hold exactly
    BA, BB, beta = 500.0, 400.0, 0.3
    gain = ablation_gain(BA, BB, beta)
    nn, dd = 1e7, 3e8
    # the claim is: arm A needs D*gain tokens to match arm B at D tokens
    lhs = BA / (dd * gain)**beta
    rhs = BB / dd**beta
    check("等效增益恒等式 B_A/D^b == B_B/(D*g)^b", abs(lhs - rhs) < 1e-9,
          f"gain={gain:.4f}")

    # optimal allocation vs brute force
    theta = np.array([A, B, a, b, E])
    C = 6 * 50_348_544 * 1e9
    got = optimal_allocation(theta, C)
    grid = np.linspace(5, 10, 20001)
    brute = min((theta[0]/(10**x)**theta[2] + theta[1]/(C/(6*10**x))**b + E, 10**x)
                for x in grid)
    check("最优配比与暴力网格一致", abs(got["loss"] - brute[0]) < 1e-4 and
          abs(math.log10(got["N"]) - math.log10(brute[1])) < 1e-2,
          f"N={got['N']:.3e} vs {brute[1]:.3e}")

    # F test against a hand computation
    rss_f, k_f, df2 = fit.rss * 1.0, 5, fit.n - 5
    m2 = fit_m2(N, D, L, None)
    F, p = f_test(fit, m2)
    F_manual = ((m2.rss - rss_f) / 1) / (rss_f / df2)
    check("F 统计量由公式可复现", abs(F - F_manual) < 1e-8, f"{F:.6f} vs {F_manual:.6f}")

    # AIC definition
    aic, bic = aic_bic(fit)
    n = fit.n
    ll = -0.5 * n * (math.log(2*math.pi*fit.rss/n) + 1)
    check("AIC = -2logL + 2k", abs(aic - (-2*ll + 2*fit.k)) < 1e-8
          and abs(bic - (-2*ll + fit.k*math.log(n))) < 1e-8)

    # predict_interval: Var = g' Cov g + sigma_run^2 + sigma_obs^2
    N0, D0 = np.array([5e7]), np.array([1e9])
    sr, so = 0.02, 0.001
    pi = predict_interval(fit, N0, D0, sigma_run=sr, sigma_obs=so)
    # rebuild the gradient independently
    A_, B_, a_, b_, E_ = fit.theta
    g = np.array([1/N0[0]**a_, 1/D0[0]**b_,
                  -A_*math.log(N0[0])/N0[0]**a_, -B_*math.log(D0[0])/D0[0]**b_, 1.0])
    var = g @ fit.cov @ g + sr**2 + so**2
    check("预测区间方差与公式一致", abs(pi["sd_total"][0] - math.sqrt(var)) < 1e-10,
          f"{pi['sd_total'][0]:.6f} vs {math.sqrt(var):.6f}")
    return fit


# --------------------------------------------------- 3. resampling correctness
def check_resampling():
    print("\n3. 重采样正确性")
    rng = np.random.default_rng(1)
    N = np.repeat([1e6, 1e7], [8, 8]); D = np.tile(np.linspace(1e8, 1e9, 4), 4)
    G = np.repeat(np.arange(4), 4)
    L = 3.0 - 0.1*np.log(D/1e8) + rng.normal(0, 0.01, N.size)
    bt = cluster_bootstrap(N, D, L, G, None, B=200, seed=0)
    check("bootstrap 产出非空", len(bt) > 150, f"{len(bt)}/200 成功")
    check("bootstrap 参数维度为 5", bt.ndim == 2 and bt.shape[1] == 5)
    check("bootstrap 中心接近点估计", abs(np.median(bt[:, 2]) - wnls(N, D, L, None).theta[2]) < 0.3)


# ------------------------------------------- 4. report matches the raw numbers
def check_report(rows):
    print("\n4. 报告与原始数据一致")
    rp = pathlib.Path("results/report.json")
    if not rp.exists():
        check("results/report.json 存在", False)
        return
    rep = json.loads(rp.read_text())
    A_, B_ = rep["fit"]["A"], rep["fit"]["B"]
    check("report.json 含拟合结果", "alpha" in rep["fit"] and "beta" in rep["fit"],
          f"alpha={rep['fit']['alpha']:.4f}")

    # recompute Delta_L on val A from the CSVs and compare with the report table
    # ablation_block restricts arm A to the model sizes arm B actually has; the
    # check must do the same, or D=250M mixes 8M and 20M runs into one "arm A".
    b_sizes = {r["model_size"] for r in rows
               if r["val_set"] == "A" and r["run_id"].endswith("_B")}

    def losses(arm, vs="A"):
        out = {}
        for r in rows:
            if r["val_set"] != vs or not r["run_id"].endswith("_" + arm):
                continue
            if arm == "A" and r["model_size"] not in b_sizes:
                continue
            out.setdefault(int(r["tokens_seen"]), []).append(float(r["val_loss"]))
        return {k: float(np.mean(v)) for k, v in out.items()}
    la, lb = losses("A"), losses("B")
    abl = rep.get("ablation")
    if abl:
        blk = next((b for b in abl if b["val_set"] == "A" and b["arm_hi"] == "B"), None)
        if blk:
            mism = []
            for d, va, vb in blk["points"]:
                if d in la and d in lb:
                    if abs((vb - va) - (lb[d] - la[d])) > 1e-6:
                        mism.append(d)
            check("报告里的 Delta_L 与原始 CSV 一致", not mism, str(mism[:3]) if mism else "")

    # sign flip: the headline claim must actually be in the data
    lbB = losses("B", "B"); laB = losses("A", "B")
    common = set(laB) & set(lbB)
    if common:
        deltas_val_b = [lbB[d] - laB[d] for d in sorted(common)]
        check("val B 上 Delta_L 为负（符号翻转成立）",
              all(x < 0 for x in deltas_val_b), f"{len(deltas_val_b)} 个断点")
    common_a = set(la) & set(lb)
    if common_a:
        deltas_val_a = [lb[d] - la[d] for d in sorted(common_a)]
        check("val A 上 Delta_L 为正",
              all(x > 0 for x in deltas_val_a), f"{len(deltas_val_a)} 个断点")
        spread = max(deltas_val_a) - min(deltas_val_a)
        check("原始验证集上的 Delta_L 极差小于其均值的 10%",
              spread / abs(np.mean(deltas_val_a)) < 0.10,
              f"极差 {spread:.4f} / 均值 {np.mean(deltas_val_a):.4f}")


def main() -> int:
    rows = check_logs()
    fit = check_formulas()
    check_resampling()
    if rows:
        check_report(rows)
    print(f"\n{'='*52}")
    print(f"PASS {len(OK)}   FAIL {len(BAD)}")
    if BAD:
        print("failed:", *BAD, sep="\n  - ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
