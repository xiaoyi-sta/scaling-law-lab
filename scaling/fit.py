"""Estimation and inference for L(N, D) = A/N^alpha + B/D^beta + E.

Three separate ideas live here, and conflating them is the classic mistake:

  * point estimate      -- weighted nonlinear least squares
  * parameter uncertainty -- how well we know theta, given these runs
  * predictive uncertainty -- where a NEW run's loss would land (adds sigma_run)

The v2 draft of the project used the token-level standard error of the validation
mean (~3e-4 nats) as the fit weight.  That number is real but irrelevant: reruns
move the loss by ~1e-2 nats, so the weights were 30x too confident and the
"weighted" fit was silently an unweighted one.  Here the weight is
1 / (sigma_run^2 + se_doc^2), with sigma_run estimated from replicate seeds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

PARAM_NAMES = ("A", "B", "alpha", "beta", "E")
LO_BOUNDS = np.array([-10.0, -10.0, 0.01, 0.01, 0.0])
HI_BOUNDS = np.array([20.0, 20.0, 2.0, 2.0, 6.0])


# ---------------------------------------------------------------- forward model
def predict(theta: np.ndarray, N: np.ndarray, D: np.ndarray) -> np.ndarray:
    A, B, alpha, beta, E = theta
    return A / N ** alpha + B / D ** beta + E


def _unpack(z: np.ndarray) -> np.ndarray:
    return np.array([np.exp(z[0]), np.exp(z[1]), z[2], z[3], z[4]])


def _residuals(z: np.ndarray, N, D, L, w) -> np.ndarray:
    return np.sqrt(w) * (predict(_unpack(z), N, D) - L)


def _as_weights(weights, L: np.ndarray) -> np.ndarray:
    """Accept None, a scalar, or a per-observation vector; always return a vector.

    A scalar weight used to survive `wnls` but explode inside the bootstrap as
    `'float' object is not subscriptable`, which is a confusing failure a long way
    from its cause.
    """
    if weights is None:
        return np.ones_like(L, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.ndim == 0:
        return np.full(L.shape, float(w))
    if w.shape != L.shape:
        raise ValueError(f"weights shape {w.shape} != observations {L.shape}")
    return w


@dataclass
class Fit:
    theta: np.ndarray                 # A, B, alpha, beta, E
    se: np.ndarray                    # delta-method SE, same order
    cov: np.ndarray                   # covariance in the same (natural) order
    rss: float                        # weighted residual sum of squares
    n: int
    k: int
    success: bool = True
    message: str = ""

    @property
    def dof(self) -> int:
        return max(self.n - self.k, 1)

    def as_dict(self) -> dict:
        out = {}
        for i, name in enumerate(PARAM_NAMES):
            out[name] = float(self.theta[i])
            out[f"{name}_se"] = float(self.se[i])
            out[f"{name}_lo"] = float(self.theta[i] - 1.96 * self.se[i])
            out[f"{name}_hi"] = float(self.theta[i] + 1.96 * self.se[i])
        out["rss"] = float(self.rss)
        out["n"] = self.n
        out["k"] = self.k
        out["success"] = bool(self.success)
        return out


def _lhs_starts(n_starts: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Latin-hypercube starts over (logA, logB, alpha, beta, E)."""
    dims = 5
    cuts = np.stack([(rng.permutation(n_starts) + rng.random(n_starts)) / n_starts
                     for _ in range(dims)], axis=1)
    lo = np.array([1.0, 1.0, 0.05, 0.05, 1.0])
    hi = np.array([8.0, 8.0, 0.7, 0.7, 3.0])
    return [np.log(np.clip(s[:2], 1e-3, None)).tolist() + s[2:].tolist()
            for s in (lo + cuts * (hi - lo))]


def wnls(N, D, L, weights=None, n_starts: int = 16, seed: int = 0,
         x0=None) -> Fit:
    """Weighted NLS with bounded, log-parametrised multi-start search.

    `method='lm'` (as the v2 spec suggested) cannot take bounds, so A/B/E were free
    to wander negative.  trf with bounds is the only defensible choice here.
    """
    N = np.asarray(N, float); D = np.asarray(D, float); L = np.asarray(L, float)
    w = _as_weights(weights, L)
    rng = np.random.default_rng(seed)

    starts = _lhs_starts(n_starts, rng)
    if x0 is not None:
        # warm start: a bootstrap replicate of the same design should not need a
        # global search, and random starts let replicates land in different local
        # minima, which inflates the spread of the *shape* parameters.
        starts.insert(0, list(np.asarray(x0, float)))
    best = None
    for z0 in starts:
        try:
            r = least_squares(_residuals, z0, bounds=(LO_BOUNDS, HI_BOUNDS),
                              method="trf", args=(N, D, L, w), x_scale="jac",
                              max_nfev=20000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        raise RuntimeError("all starts failed")

    theta = _unpack(best.x)
    n, k = L.size, 5
    dof = max(n - k, 1)
    sigma2 = 2.0 * best.cost / dof

    # delta method from the log-parametrisation to the natural one
    J = np.linalg.pinv(best.jac.T @ best.jac) * sigma2
    se_z = np.sqrt(np.clip(np.diag(J), 0, None))
    scale = np.array([theta[0], theta[1], 1.0, 1.0, 1.0])
    se = se_z * scale
    cov = J * np.outer(scale, scale)

    return Fit(theta=theta, se=se, cov=cov, rss=float(2 * best.cost), n=n, k=k,
               success=bool(best.success), message=str(best.message))


# ------------------------------------------------------- information criteria
def aic_bic(fit: Fit, n_total: int | None = None) -> tuple[float, float]:
    """Gaussian AIC/BIC on the weighted RSS.

    All candidate models share the same weights, so the comparison is fair
    without a separate likelihood scale term.
    """
    n = n_total or fit.n
    rss = max(fit.rss, 1e-300)
    ll = -0.5 * n * (np.log(2 * np.pi * rss / n) + 1)
    return -2 * ll + 2 * fit.k, -2 * ll + fit.k * np.log(n)


def f_test(full: Fit, reduced: Fit) -> tuple[float, float]:
    """Nested F test (full must nest reduced)."""
    df1 = full.k - reduced.k
    df2 = full.dof
    if df1 <= 0:
        raise ValueError("full model must have more parameters than reduced")
    F = ((reduced.rss - full.rss) / df1) / (full.rss / df2)
    from scipy.stats import f as fdist
    return float(F), float(1 - fdist.cdf(F, df1, df2))


# ------------------------------------------------------------- sigma_run
def sigma_run_from_replicates(N, D, L, groups, n_seeds: int | None = None) -> dict:
    """Pooled within-(N, D) variance across replicate seeds, with a chi2 interval.

    groups: integer run id per observation.  Observations are matched on the exact
    (N, D) grid; only cells with >= 2 seeds contribute.
    """
    from scipy.stats import chi2

    N = np.asarray(N, float); D = np.asarray(D, float)
    L = np.asarray(L, float); g = np.asarray(groups)
    cells: dict[tuple, list[float]] = {}
    for n_, d_, l_, g_ in zip(N, D, L, g):
        cells.setdefault((n_, d_), []).append(l_)

    ss, df = 0.0, 0
    n_cells = 0
    for vals in cells.values():
        if len(vals) < 2:
            continue
        v = np.array(vals)
        ss += float(((v - v.mean()) ** 2).sum())
        df += len(v) - 1
        n_cells += 1
    if df == 0:
        return {"sigma_run": None, "df": 0, "n_cells": 0,
                "note": "no replicate cells; sigma_run not identifiable"}

    s2 = ss / df
    lo = df * s2 / chi2.ppf(0.975, df)
    hi = df * s2 / chi2.ppf(0.025, df)
    return {"sigma_run": float(np.sqrt(s2)), "sigma_run_sq": float(s2),
            "df": df, "n_cells": n_cells,
            "ci_lo": float(np.sqrt(max(lo, 0))), "ci_hi": float(np.sqrt(hi))}


# ------------------------------------------------------------ resampling
def cluster_bootstrap(N, D, L, groups, weights=None, B: int = 2000,
                      seed: int = 0, wild: bool = True) -> np.ndarray:
    """Resample whole runs (not observations) and refit.

    Same-run breakpoints share an init and a data order, so treating the 15 rows as
    independent observations is pseudoreplication: the effective sample size is the
    number of runs.  Resampling rows independently would produce confidence
    intervals that are far too narrow.
    """
    N = np.asarray(N, float); D = np.asarray(D, float)
    L = np.asarray(L, float); g = np.asarray(groups)
    w = _as_weights(weights, L)

    runs = np.unique(g)
    rng = np.random.default_rng(seed)

    base = wnls(N, D, L, w, n_starts=8, seed=seed)
    z0 = [np.log(base.theta[0]), np.log(base.theta[1]),
          base.theta[2], base.theta[3], base.theta[4]]
    resid = L - predict(base.theta, N, D)

    out = []
    for b in range(B):
        pick = rng.choice(runs, size=runs.size, replace=True)
        idx = np.concatenate([np.where(g == p)[0] for p in pick])
        Lb = L[idx]
        if wild:
            # Two-level bootstrap.  Whole-run resampling reproduces between-run
            # variation but leaves the within-run breakpoint pattern intact, so
            # beta (identified by that pattern's curvature) comes out far too
            # precise.  Flipping residual signs inside the run restores the
            # within-run noise the resampling did not touch.
            Lb = Lb + rng.choice([-1.0, 1.0], size=idx.size) * resid[idx]
        try:
            fit = wnls(N[idx], D[idx], Lb, w[idx], n_starts=2, seed=b, x0=z0)
        except Exception:
            continue
        if fit.success:
            out.append(fit.theta)
    return np.asarray(out)


def jackknife_runs(N, D, L, groups, weights=None) -> dict:
    """Leave-one-run-out refits: a direct display of the cost of few clusters."""
    N = np.asarray(N, float); D = np.asarray(D, float)
    L = np.asarray(L, float); g = np.asarray(groups)
    w = _as_weights(weights, L)
    runs = np.unique(g)
    thetas = []
    for r in runs:
        m = g != r
        fit = wnls(N[m], D[m], L[m], w[m], n_starts=6, seed=int(r))
        thetas.append(fit.theta)
    T = np.asarray(thetas)
    return {"n_runs": int(runs.size), "min": T.min(0), "max": T.max(0),
            "sd": T.std(0, ddof=1) if len(T) > 1 else np.zeros(T.shape[1]),
            "thetas": T}


# -------------------------------------------------------------- prediction
def predict_interval(fit: Fit, N_new, D_new, sigma_run: float = 0.0,
                     sigma_obs: float = 0.0, boot_thetas: np.ndarray | None = None,
                     level: float = 0.95) -> dict:
    """Point prediction and predictive interval for a NEW observation.

    Var = g' Cov(theta) g + sigma_run^2 + sigma_obs^2.  The sigma_run term is the
    piece the v2 spec omitted, and it is the dominant one: it is what a rerun of
    the same configuration would move by.
    """
    N_new = np.atleast_1d(np.asarray(N_new, float))
    D_new = np.atleast_1d(np.asarray(D_new, float))
    theta = fit.theta
    A, B, alpha, beta, E = theta
    pred = predict(theta, N_new, D_new)

    # gradient of f wrt (A, B, alpha, beta, E)
    G = np.zeros((N_new.size, 5))
    G[:, 0] = 1.0 / N_new ** alpha
    G[:, 1] = 1.0 / D_new ** beta
    G[:, 2] = -A * np.log(N_new) / N_new ** alpha
    G[:, 3] = -B * np.log(D_new) / D_new ** beta
    G[:, 4] = 1.0

    z = 1.959963985 if level == 0.95 else None
    if z is None:
        from scipy.stats import norm
        z = norm.ppf(0.5 + level / 2)

    var_par = np.einsum("ij,jk,ik->i", G, fit.cov, G)
    sd_total = np.sqrt(var_par + sigma_run ** 2 + sigma_obs ** 2)

    out = {"pred": pred, "sd_par": np.sqrt(var_par), "sd_total": sd_total,
           "lo": pred - z * sd_total, "hi": pred + z * sd_total}

    if boot_thetas is not None and len(boot_thetas) > 10:
        bp = np.stack([predict(t, N_new, D_new) for t in boot_thetas])
        lo_b = np.quantile(bp, 0.025, axis=0) - z * np.sqrt(sigma_run ** 2 + sigma_obs ** 2)
        hi_b = np.quantile(bp, 0.975, axis=0) + z * np.sqrt(sigma_run ** 2 + sigma_obs ** 2)
        out["lo_boot"], out["hi_boot"] = lo_b, hi_b
    return out


def winkler_score(lo, hi, y, alpha: float = 0.05) -> np.ndarray:
    """Interval score: rewards coverage AND sharpness (unlike a bare coverage count).

    A 95% interval that always covers because it spans the whole axis scores badly
    here; counting 4/4 covered does not notice.
    """
    lo = np.asarray(lo, float); hi = np.asarray(hi, float); y = np.asarray(y, float)
    width = hi - lo
    score = width.copy()
    below = y < lo
    above = y > hi
    score[below] += (2 / alpha) * (lo[below] - y[below])
    score[above] += (2 / alpha) * (y[above] - hi[above])
    return score


# ------------------------------------------------------------ ablation model
def joint_ablation_fit(N, D, L, is_b, weights=None, n_starts: int = 16,
                       seed: int = 0):
    """L_g(N,D) = A/N^alpha + B_g/D^beta + E, with g in {A, B}.

    v2's plan was to fit arm A alone and invert the curve for B's loss.  Arm A has
    4-5 points and the model has 5 parameters: exactly identified, so the implied
    interval is infinite.  Sharing alpha/beta/A/E across arms makes B_B estimable.
    """
    N = np.asarray(N, float); D = np.asarray(D, float)
    L = np.asarray(L, float); is_b = np.asarray(is_b, bool)
    w = _as_weights(weights, L)

    def res(z):
        A = np.exp(z[0]); BA = np.exp(z[1]); BB = np.exp(z[2])
        a, b, E = z[3], z[4], z[5]
        Bg = np.where(is_b, BB, BA)
        return np.sqrt(w) * (A / N ** a + Bg / D ** b + E - L)

    lo = np.array([-10, -10, -10, 0.01, 0.01, 0.0])
    hi = np.array([20, 20, 20, 2.0, 2.0, 6.0])
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_starts):
        z0 = np.concatenate([
            rng.uniform(1, 6, 3), rng.uniform(0.05, 0.7, 2), rng.uniform(1.0, 3.0, 1)])
        try:
            r = least_squares(res, z0, bounds=(lo, hi), method="trf", x_scale="jac")
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    z = best.x
    return {"A": float(np.exp(z[0])), "B_A": float(np.exp(z[1])),
            "B_B": float(np.exp(z[2])), "alpha": float(z[3]),
            "beta": float(z[4]), "E": float(z[5]),
            "rss": float(2 * best.cost), "n": int(L.size), "k": 6}


def ablation_gain(B_A: float, B_B: float, beta: float) -> float:
    """Token-equivalent (== compute-equivalent, since N is fixed) data gain of B over A.

        A/N^a + B_A/D'^b + E = A/N^a + B_B/D^b + E   =>   D' = D (B_A/B_B)^(1/b)
    """
    return float((B_A / B_B) ** (1.0 / beta))


# ------------------------------------------------------- optimal allocation
def optimal_allocation(theta, compute: float, logN_range=(5.0, 10.0)) -> dict:
    """Default range is N in [1e5, 1e10]; the old (-1, 3) stopped at N=1000 and the
    optimiser simply walked to the boundary, which looks like a result but is not."""
    """argmin_N L(N, C/(6N)) for fixed C = 6ND, by 1-D search over log N."""
    from scipy.optimize import minimize_scalar
    A, B, alpha, beta, E = theta

    def loss(logN):
        N = 10 ** logN
        D = compute / (6 * N)
        return A / N ** alpha + B / D ** beta + E

    r = minimize_scalar(loss, bounds=logN_range, method="bounded")
    N = 10 ** r.x
    D = compute / (6 * N)
    return {"N": float(N), "D": float(D), "D_over_N": float(D / N),
            "loss": float(r.fun)}


# ------------------------------------------------- alternative function forms
@dataclass
class SimpleFit:
    """Holds rss/k/n so aic_bic and f_test can compare forms with different shapes."""
    rss: float
    k: int
    n: int
    params: dict = field(default_factory=dict)

    @property
    def dof(self) -> int:
        return max(self.n - self.k, 1)


def fit_m2(N, D, L, weights=None, n_starts: int = 12, seed: int = 0) -> Fit:
    """M2: A/N^a + B/D^b, i.e. E fixed at 0.  Nests inside M1."""
    N = np.asarray(N, float); D = np.asarray(D, float); L = np.asarray(L, float)
    w = _as_weights(weights, L)
    rng = np.random.default_rng(seed)

    def res(z):
        return np.sqrt(w) * (np.exp(z[0]) / N ** z[2] + np.exp(z[1]) / D ** z[3] - L)

    lo = np.array([-10.0, -10.0, 0.01, 0.01])
    hi = np.array([20.0, 20.0, 2.0, 2.0])
    best = None
    for _ in range(n_starts):
        z0 = np.array([rng.uniform(0, 6), rng.uniform(0, 6),
                       rng.uniform(0.05, 0.7), rng.uniform(0.05, 0.7)])
        try:
            r = least_squares(res, z0, bounds=(lo, hi), method="trf", x_scale="jac")
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r

    theta = np.array([np.exp(best.x[0]), np.exp(best.x[1]), best.x[2], best.x[3], 0.0])
    n, k = L.size, 4
    sigma2 = 2 * best.cost / max(n - k, 1)
    Jz = np.linalg.pinv(best.jac.T @ best.jac) * sigma2
    se_z = np.sqrt(np.clip(np.diag(Jz), 0, None))
    # M2 has 4 parameters; Fit always carries 5 slots so downstream code that
    # indexes by PARAM_NAMES does not need a special case for E.
    scale = np.array([theta[0], theta[1], 1.0, 1.0])
    cov4 = Jz * np.outer(scale, scale)
    cov5 = np.zeros((5, 5)); cov5[:4, :4] = cov4
    se5 = np.zeros(5); se5[:4] = se_z * scale
    return Fit(theta=theta, se=se5, cov=cov5,
               rss=float(2 * best.cost), n=n, k=k, success=bool(best.success))


def fit_m3(N, D, L, weights=None) -> SimpleFit:
    """M3: a*ln N + b*ln D + c.  Not nested in M1, so only AIC/BIC can judge it."""
    N = np.asarray(N, float); D = np.asarray(D, float); L = np.asarray(L, float)
    w = _as_weights(weights, L)
    X = np.column_stack([np.log(N), np.log(D), np.ones_like(N)])
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], L * sw, rcond=None)
    resid = X @ beta - L
    rss = float((w * resid ** 2).sum())
    return SimpleFit(rss=rss, k=3, n=int(L.size),
                     params={"a": float(beta[0]), "b": float(beta[1]),
                             "c": float(beta[2])})
