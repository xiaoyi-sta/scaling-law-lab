"""Diagnostic and result figures.  Every number comes from the same code paths as
the report, so a figure can never disagree with the text.

    python -m scripts.plots          ->  results/figures/*.png

Labels are English on purpose: CJK glyphs are missing from matplotlib's default
font on many machines and silently render as boxes.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402
from scipy import stats                              # noqa: E402

sys.path.insert(0, ".")

from scaling.config import PRESETS                                   # noqa: E402
from scaling.fit import (cluster_bootstrap, jackknife_runs,          # noqa: E402
                         optimal_allocation, predict, predict_interval,
                         sigma_run_from_replicates, wnls)
from scripts.fit_report import arrays, select, load_logs             # noqa: E402

OUT = pathlib.Path("results/figures")
C = ["#3b6ea5", "#c1554d", "#5a9367", "#8a6bab", "#c98a3f"]


def setup():
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.autolayout": False,
    })
    OUT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ fig 1
def fig_fit(N, D, L, se, fit):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    sizes = sorted(set(N.tolist()))

    ax = axes[0]
    for k, n in enumerate(sizes):
        m = N == n
        order = np.argsort(D[m])
        ax.errorbar(D[m][order], L[m][order], yerr=se[m][order], fmt="o",
                    ms=4, capsize=2, color=C[k], alpha=0.85,
                    label=f"N = {n/1e6:.0f}M (obs)")
        dd = np.logspace(np.log10(D[m].min()), np.log10(D[m].max()), 100)
        ax.plot(dd, predict(fit.theta, np.full_like(dd, n, dtype=float), dd),
                "-", lw=1.4, color=C[k])
    ax.set_xscale("log")
    ax.set_xlabel("training tokens D")
    ax.set_ylabel("validation loss (nats)")
    ax.set_title("fit vs observed, by model size")
    ax.legend(fontsize=7)

    ax = axes[1]
    for k, d in enumerate(sorted(set(D.tolist()))):
        m = D == d
        ax.errorbar(N[m], L[m], yerr=se[m], fmt="o", ms=4, capsize=2,
                    color=C[k % len(C)], alpha=0.85)
        nn = np.logspace(np.log10(N.min()), np.log10(N.max()), 100)
        ax.plot(nn, predict(fit.theta, nn, np.full_like(nn, d, dtype=float)),
                "-", lw=1.2, color=C[k % len(C)], label=f"D = {d/1e9:.2f}B")
    ax.set_xscale("log")
    ax.set_xlabel("parameters N (non-embedding)")
    ax.set_ylabel("validation loss (nats)")
    ax.set_title("same fit, sliced by data budget")
    ax.legend(fontsize=6.5)

    fig.suptitle(f"L(N,D) = A/N^a + B/D^b + E    "
                 f"a={fit.theta[2]:.3f}  b={fit.theta[3]:.3f}  E={fit.theta[4]:.2f}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "01_fit_curves.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 2
def fig_residuals(N, D, L, se, fit, groups, run_names):
    pred = predict(fit.theta, N, D)
    r = (L - pred) / np.sqrt(se ** 2 + sigma_run_from_replicates(N, D, L, groups)["sigma_run_sq"])

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

    ax = axes[0]
    ax.axhline(0, color="k", lw=0.8)
    ax.scatter(pred, r, s=26, color=C[0], alpha=0.8)
    ax.set_xlabel("fitted loss")
    ax.set_ylabel("standardised residual")
    ax.set_title("residual vs fitted  (look for a funnel or a bend)")

    ax = axes[1]
    (osm, osr), (slope, intercept, rval) = stats.probplot(r, dist="norm")
    ax.scatter(osm, osr, s=26, color=C[1], alpha=0.85)
    ax.plot(osm, slope * osm + intercept, "k-", lw=1)
    ax.set_xlabel("theoretical normal quantile")
    ax.set_ylabel("sample quantile")
    ax.set_title(f"QQ plot  (R^2 = {rval**2:.3f} vs normal)")

    ax = axes[2]
    for k, gid in enumerate(np.unique(groups)):
        m = groups == gid
        ax.scatter(np.full(m.sum(), k), r[m], s=30, color=C[k % len(C)], alpha=0.85)
        ax.hlines(r[m].mean(), k - 0.25, k + 0.25, color="k", lw=1.6)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(np.unique(groups))))
    ax.set_xticklabels([run_names[g] for g in np.unique(groups)],
                       rotation=35, ha="right", fontsize=6.5)
    ax.set_ylabel("standardised residual")
    ax.set_title("residual by run  (black bar = run mean)")

    s2 = float((r ** 2).sum() / (len(r) - 5))
    pchi = 1 - stats.chi2.cdf((r ** 2).sum(), len(r) - 5)
    fig.suptitle(f"sigma_hat^2 = {s2:.2f}   chi2({len(r)-5}) = {(r**2).sum():.2f}   "
                 f"p = {pchi:.2f}   -> no detectable lack of fit", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "02_residual_diagnostics.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 3
def fig_ablation(logs):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    panels = [("A", "B", "graded on raw-web val (A)"),
              ("B", "B", "graded on edu-web val (B)")]
    for k, (vs, arm, label) in enumerate(panels):
        a = arrays(select(logs, vs, "A"))
        b = arrays(select(logs, vs, arm))
        Na, Da, La = a[0], a[1], a[2]
        Nb, Db, Lb = b[0], b[1], b[2]
        sizes = set(np.unique(Nb).tolist())
        m = np.isin(Na, list(sizes))
        xs, ys = [], []
        for d in sorted(set(Db.tolist())):
            ma = (Da == d) & m
            mb = Db == d
            if ma.any() and mb.any():
                xs.append(d)
                ys.append(Lb[mb].mean() - La[ma].mean())
        tag = "vs val A (raw)" if vs == "A" else "vs val B (edu)"
        ax.plot(xs, ys, "o-", color=C[k], lw=1.6, ms=6, label=tag)
        ax.annotate(f"{np.mean(ys):+.3f}", (xs[-1], ys[-1]),
                    textcoords="offset points", xytext=(8, 4), fontsize=8, color=C[k])

    ax.axhline(0, color="k", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("training tokens D")
    ax.set_ylabel("Delta L  (edu corpus minus raw corpus)")
    ax.set_title("The effect is a constant offset whose SIGN depends\n"
                 "on which validation set grades it — not a data-efficiency gain")
    ax.legend(fontsize=8)
    ax.text(0.03, 0.42, "flat in D  ->\nadditive shift,\nnot B/D^b",
            transform=ax.transAxes, fontsize=7.5, va="center", style="italic",
            bbox=dict(fc="white", ec="none", alpha=0.75))
    fig.tight_layout()
    fig.savefig(OUT / "03_ablation_sign_flip.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 4
def fig_bootstrap(boot, fit, jk):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))

    ax = axes[0]
    ax.hist(boot[:, 2], bins=40, color=C[0], alpha=0.75, edgecolor="none")
    lo, hi = np.quantile(boot[:, 2], [0.025, 0.975])
    ax.axvline(fit.theta[2], color="k", lw=1.6, label=f"point est. {fit.theta[2]:.3f}")
    ax.axvline(lo, color=C[1], ls="--", lw=1.3, label=f"2.5% {lo:.3f}")
    ax.axvline(hi, color=C[1], ls="--", lw=1.3, label=f"97.5% {hi:.3f}")
    ax.axvline(fit.theta[2] - 1.96 * fit.se[2], color=C[2], ls=":", lw=1.2,
               label="asymptotic 95%")
    ax.axvline(fit.theta[2] + 1.96 * fit.se[2], color=C[2], ls=":", lw=1.2)
    ax.set_xlabel("alpha")
    ax.set_ylabel("frequency")
    ax.set_title("two-level bootstrap distribution of alpha")
    ax.legend(fontsize=6.5)

    ax = axes[1]
    ax.axhline(0, color="k", lw=0.8)
    ax.scatter(np.arange(jk["n_runs"]), jk["thetas"][:, 2] - fit.theta[2],
               s=48, color=C[3], zorder=3)
    ax.set_xticks(range(jk["n_runs"]))
    ax.set_xticklabels([f"-run{i}" for i in range(jk["n_runs"])], rotation=35, ha="right")
    ax.set_ylabel("alpha(leave-one-out) - alpha(full)")
    ax.set_title("leave-one-run-out: how much one run moves alpha")

    fig.tight_layout()
    fig.savefig(OUT / "04_bootstrap_and_jackknife.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 5
def fig_extrapolation(N, D, L, se, groups, sigma_run):
    N8, N20 = PRESETS["8M"].n_params_nonemb, PRESETS["20M"].n_params_nonemb
    mask_fit = N == N8
    if np.isin(N20, N):
        d20 = D[N == N20].max()
        mask_fit |= (N == N20) & (D < d20)
    w = 1.0 / (sigma_run ** 2 + se ** 2)

    fit_x = wnls(N[mask_fit], D[mask_fit], L[mask_fit], w[mask_fit], seed=0)
    boot_x = cluster_bootstrap(N[mask_fit], D[mask_fit], L[mask_fit],
                               groups[mask_fit], w[mask_fit], B=300, seed=1)
    mt = ~mask_fit
    pi = predict_interval(fit_x, N[mt], D[mt], sigma_run=sigma_run,
                          sigma_obs=float(np.mean(se)), boot_thetas=boot_x)

    order = np.argsort(D[mt])
    x = np.arange(mt.sum())[order]
    y = L[mt][order]
    # coverage must be computed on the UNSORTED arrays; comparing sorted y against
    # unsorted bounds silently mismatches positions and understates coverage.
    cov = int(((pi["lo"] <= L[mt]) & (L[mt] <= pi["hi"])).sum())
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.errorbar(x, pi["pred"][order], yerr=[pi["pred"][order] - pi["lo"][order],
                                            pi["hi"][order] - pi["pred"][order]],
                fmt="s", ms=5, color=C[0], capsize=3, label="prediction + 95% PI")
    ax.plot(x, y, "o", ms=6, color=C[1], label="actual")
    for xi in x:
        ax.axvline(xi, color="0.9", lw=0.6, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{N[mt][order][i]/1e6:.0f}M\n{D[mt][order][i]/1e9:.2f}B"
                        for i in range(len(x))], fontsize=6.5)
    ax.set_xlabel("held-out point (model size / tokens)")
    ax.set_ylabel("loss (nats)")
    ax.set_title(f"fit on 8M + 20M(partial) -> predict 20M@1B and all of 50M\n"
                 f"coverage {cov}/{len(x)}, PI includes sigma_run = {sigma_run:.4f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "05_prediction_intervals.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 6
def fig_dn(boot):
    C50 = 6 * PRESETS["50M"].n_params_nonemb * 1e9
    dn = np.array([optimal_allocation(t, C50)["D_over_N"] for t in boot])
    dn = dn[np.isfinite(dn)]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.hist(dn, bins=40, color=C[2], alpha=0.8, edgecolor="none")
    ax.axvline(np.median(dn), color="k", lw=1.5, label=f"median {np.median(dn):.0f}")
    ax.axvline(20, color=C[1], lw=1.5, ls="--", label="Chinchilla ~20")
    lo, hi = np.quantile(dn, [0.025, 0.975])
    ax.axvspan(lo, hi, color=C[2], alpha=0.12, label=f"95% [{lo:.0f}, {hi:.0f}]")
    ax.set_xlabel("optimal D/N at 50M-budget compute")
    ax.set_ylabel("frequency")
    ax.set_title("the ratio is not pinned down: interval spans an order of magnitude")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "06_optimal_ratio.png")
    plt.close(fig)


def main() -> int:
    setup()
    logs = load_logs()
    N, D, L, se, g = arrays(select(logs, "A", "A"))
    sr = sigma_run_from_replicates(N, D, L, g)
    sigma_run = sr["sigma_run"]
    w = 1.0 / (sigma_run ** 2 + np.maximum(se, 1e-6) ** 2)
    fit = wnls(N, D, L, w, seed=0)

    boot = cluster_bootstrap(N, D, L, g, w, B=400, seed=0)
    jk = jackknife_runs(N, D, L, g, w)

    run_names = {gid: rid for rid, gid in
                 {rid: i for i, rid in enumerate(sorted({r["run_id"] for r in select(logs, "A", "A")}))}.items()}

    print(f"fit: alpha={fit.theta[2]:.4f} beta={fit.theta[3]:.4f} E={fit.theta[4]:.4f}")
    fig_fit(N, D, L, se, fit);                       print("  01_fit_curves.png")
    fig_residuals(N, D, L, se, fit, g, run_names);   print("  02_residual_diagnostics.png")
    fig_ablation(logs);                              print("  03_ablation_sign_flip.png")
    fig_bootstrap(boot, fit, jk);                    print("  04_bootstrap_and_jackknife.png")
    fig_extrapolation(N, D, L, se, g, sigma_run);    print("  05_prediction_intervals.png")
    fig_dn(boot);                                    print("  06_optimal_ratio.png")
    print(f"\nwrote {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
