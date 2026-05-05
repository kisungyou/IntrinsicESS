"""Backbone utilities for intrinsic kernel ESS experiments on S^2.

The experiments are intentionally sphere-only.  The kernel used throughout is the
Schoenberg-valid Abel--Poisson / Legendre generating-function kernel on S^2,

    k_rho(x, y) = (1 - 2 rho <x,y> + rho^2)^(-1/2),  0 < rho < 1.

This is a positive-definite zonal kernel on S^2 because it has the expansion
sum_{l >= 0} rho^l P_l(<x,y>), where P_l are Legendre polynomials.

No SciPy dependency is required.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


Array = np.ndarray


# -----------------------------------------------------------------------------
# Small I/O helpers
# -----------------------------------------------------------------------------


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Mapping, path: Path | str) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path: Path | str) -> Dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv_rows(path: Path | str, rows: Sequence[Mapping], fieldnames: Optional[Sequence[str]] = None) -> None:
    path = Path(path)
    if fieldnames is None:
        if not rows:
            raise ValueError("fieldnames must be supplied when rows is empty")
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_rows(path: Path | str) -> List[Dict[str, str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_latex_table(path: Path | str, rows: Sequence[Mapping], columns: Sequence[str], caption: Optional[str] = None) -> None:
    """Write a minimal LaTeX tabular file from already-formatted values."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        if caption:
            f.write("% " + caption + "\n")
        f.write("\\begin{tabular}{" + "l" * len(columns) + "}\n")
        f.write("\\hline\n")
        f.write(" & ".join(columns) + " \\\\ \n")
        f.write("\\hline\n")
        for row in rows:
            f.write(" & ".join(str(row[c]) for c in columns) + " \\\\ \n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")


# -----------------------------------------------------------------------------
# Geometry and simulation on S^2
# -----------------------------------------------------------------------------


def normalize(x: Array, axis: int = -1, eps: float = 1e-15) -> Array:
    x = np.asarray(x, dtype=float)
    nrm = np.linalg.norm(x, axis=axis, keepdims=True)
    if np.any(nrm < eps):
        raise ValueError("Cannot normalize a vector whose norm is numerically zero.")
    return x / nrm


def sample_uniform_sphere(n: int, rng: np.random.Generator, dim: int = 3) -> Array:
    if dim != 3:
        raise NotImplementedError("The experiments in this repository are written for S^2 embedded in R^3.")
    x = rng.normal(size=(n, dim))
    return normalize(x)


def random_rotation_matrix(rng: np.random.Generator) -> Array:
    """Haar-distributed random rotation matrix in SO(3)."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def orthonormal_frame(mu: Array) -> Tuple[Array, Array, Array]:
    """Return (mu, e1, e2) with e1,e2 spanning the tangent plane at mu."""
    mu = normalize(np.asarray(mu, dtype=float).reshape(1, 3))[0]
    anchor = np.array([1.0, 0.0, 0.0]) if abs(mu[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = anchor - np.dot(anchor, mu) * mu
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(mu, e1)
    return mu, e1, e2


def sample_vmf_s2(mu: Array, kappa: float, n: int, rng: np.random.Generator) -> Array:
    """Exact sampler for vMF(mu, kappa) on S^2.

    The marginal W = mu^T X has density proportional exp(kappa W) on [-1,1].
    The inverse-CDF formula below is stable for the kappa values used in the
    experiments.
    """
    mu = normalize(np.asarray(mu, dtype=float).reshape(1, 3))[0]
    kappa = float(kappa)
    if kappa < 1e-12:
        return sample_uniform_sphere(n, rng)

    u = rng.random(n)
    # W = -1 + log(1 + u * (exp(2 kappa)-1))/kappa, rewritten stably as below.
    w = 1.0 + np.log(u + (1.0 - u) * np.exp(-2.0 * kappa)) / kappa
    phi = 2.0 * np.pi * rng.random(n)
    rad = np.sqrt(np.maximum(0.0, 1.0 - w * w))
    mu, e1, e2 = orthonormal_frame(mu)
    x = w[:, None] * mu[None, :] + rad[:, None] * np.cos(phi)[:, None] * e1[None, :] + rad[:, None] * np.sin(phi)[:, None] * e2[None, :]
    return normalize(x)


def sample_vmf_s2_one(mu: Array, kappa: float, rng: np.random.Generator) -> Array:
    return sample_vmf_s2(mu, kappa, 1, rng)[0]


def log_vmf_normalizer_s2(kappa: float) -> float:
    """Log normalizing constant for vMF on S^2: c(k)=k/(4*pi*sinh(k))."""
    kappa = float(kappa)
    if kappa < 1e-12:
        return -math.log(4.0 * math.pi)
    if kappa > 50.0:
        # log(sinh k) = k - log 2 + log(1 - exp(-2k)); last term is negligible but included.
        log_sinh = kappa - math.log(2.0) + math.log1p(-math.exp(-2.0 * kappa))
    else:
        log_sinh = math.log(math.sinh(kappa))
    return math.log(kappa) - math.log(4.0 * math.pi) - log_sinh


@dataclass
class VMFMixtureS2:
    """Mixture of vMF distributions on S^2 with exact density evaluation."""

    mus: Array
    kappas: Array
    weights: Array

    def __post_init__(self) -> None:
        self.mus = normalize(np.asarray(self.mus, dtype=float))
        self.kappas = np.asarray(self.kappas, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        if self.mus.ndim != 2 or self.mus.shape[1] != 3:
            raise ValueError("mus must have shape (K, 3).")
        if len(self.kappas) != len(self.mus) or len(self.weights) != len(self.mus):
            raise ValueError("mus, kappas, and weights must have the same length.")
        if np.any(self.kappas < 0):
            raise ValueError("kappas must be nonnegative.")
        if np.any(self.weights <= 0):
            raise ValueError("weights must be positive.")
        self.weights = self.weights / self.weights.sum()
        self.log_weights = np.log(self.weights)
        self.log_norms = np.array([log_vmf_normalizer_s2(k) for k in self.kappas])

    @property
    def n_components(self) -> int:
        return int(len(self.weights))

    def log_density(self, x: Array) -> float:
        x = normalize(np.asarray(x, dtype=float).reshape(1, 3))[0]
        vals = self.log_weights + self.log_norms + self.kappas * (self.mus @ x)
        m = float(np.max(vals))
        return float(m + np.log(np.sum(np.exp(vals - m))))

    def sample(self, n: int, rng: np.random.Generator) -> Array:
        comps = rng.choice(self.n_components, size=n, p=self.weights)
        out = np.zeros((n, 3))
        for j in range(self.n_components):
            idx = np.where(comps == j)[0]
            if len(idx) > 0:
                out[idx] = sample_vmf_s2(self.mus[j], float(self.kappas[j]), len(idx), rng)
        return out

    def nearest_component(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, 3)
        return np.argmax(x @ self.mus.T, axis=1)


def tetrahedron_directions() -> Array:
    base = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=float,
    )
    return normalize(base)


# -----------------------------------------------------------------------------
# MCMC kernels on S^2
# -----------------------------------------------------------------------------


def metropolis_local_vmf(
    log_target,
    x0: Array,
    n_steps: int,
    prop_kappa: float,
    rng: np.random.Generator,
) -> Tuple[Array, float]:
    """Random-walk Metropolis with symmetric vMF proposal q(y|x) = vMF(x, prop_kappa)."""
    x = normalize(np.asarray(x0, dtype=float).reshape(1, 3))[0]
    log_px = float(log_target(x))
    chain = np.zeros((n_steps, 3))
    accept = 0
    for t in range(n_steps):
        y = sample_vmf_s2_one(x, prop_kappa, rng)
        log_py = float(log_target(y))
        if np.log(rng.random()) < log_py - log_px:
            x, log_px = y, log_py
            accept += 1
        chain[t] = x
    return chain, accept / n_steps


def log_vmf_mixture_proposal_s2(x: Array, mus: Array, weights: Array, kappa: float) -> float:
    x = normalize(np.asarray(x, dtype=float).reshape(1, 3))[0]
    mus = normalize(np.asarray(mus, dtype=float))
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    log_norm = log_vmf_normalizer_s2(kappa)
    vals = np.log(weights) + log_norm + kappa * (mus @ x)
    m = float(np.max(vals))
    return float(m + np.log(np.sum(np.exp(vals - m))))


def sample_vmf_mixture_proposal_s2(mus: Array, weights: Array, kappa: float, rng: np.random.Generator) -> Array:
    mus = normalize(np.asarray(mus, dtype=float))
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    j = int(rng.choice(len(weights), p=weights))
    return sample_vmf_s2_one(mus[j], kappa, rng)


def metropolis_independence_vmf_mixture(
    target: VMFMixtureS2,
    x0: Array,
    n_steps: int,
    proposal_kappa: float,
    rng: np.random.Generator,
    proposal_weights: Optional[Array] = None,
) -> Tuple[Array, float]:
    """Independence MH with a broad mixture of vMF distributions centered at target modes."""
    if proposal_weights is None:
        proposal_weights = np.ones(target.n_components) / target.n_components
    proposal_weights = np.asarray(proposal_weights, dtype=float)
    proposal_weights = proposal_weights / proposal_weights.sum()

    x = normalize(np.asarray(x0, dtype=float).reshape(1, 3))[0]
    log_px = target.log_density(x)
    log_qx = log_vmf_mixture_proposal_s2(x, target.mus, proposal_weights, proposal_kappa)
    chain = np.zeros((n_steps, 3))
    accept = 0
    for t in range(n_steps):
        y = sample_vmf_mixture_proposal_s2(target.mus, proposal_weights, proposal_kappa, rng)
        log_py = target.log_density(y)
        log_qy = log_vmf_mixture_proposal_s2(y, target.mus, proposal_weights, proposal_kappa)
        log_alpha = (log_py + log_qx) - (log_px + log_qy)
        if np.log(rng.random()) < log_alpha:
            x, log_px, log_qx = y, log_py, log_qy
            accept += 1
        chain[t] = x
    return chain, accept / n_steps


# -----------------------------------------------------------------------------
# S^2 Schoenberg-valid Abel--Poisson kernel and ESS estimators
# -----------------------------------------------------------------------------


def abel_poisson_s2_from_dots(dots: Array, rho: float) -> Array:
    rho = float(rho)
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must lie in (0,1).")
    dots = np.asarray(dots, dtype=float)
    return 1.0 / np.sqrt(np.maximum(1e-15, 1.0 - 2.0 * rho * dots + rho * rho))


def kernel_diag_s2(rho: float) -> float:
    """k_rho(x,x) for the S^2 Abel--Poisson kernel."""
    return 1.0 / (1.0 - float(rho))


def kernel_gram_s2(X: Array, rho: float) -> Array:
    X = normalize(np.asarray(X, dtype=float))
    return abel_poisson_s2_from_dots(X @ X.T, rho)


def centered_gram(K: Array) -> Array:
    K = np.asarray(K, dtype=float)
    row_mean = K.mean(axis=1, keepdims=True)
    col_mean = K.mean(axis=0, keepdims=True)
    grand_mean = float(K.mean())
    return K - row_mean - col_mean + grand_mean


def bartlett_lrv_from_centered_gram(Kc: Array, bandwidth: int) -> Tuple[float, float, List[float]]:
    """Estimate gamma0 and long-run variance from a centered Gram matrix.

    Bartlett weights are 1 - ell/(b+1), ell=1,...,b.
    """
    Kc = np.asarray(Kc, dtype=float)
    n = Kc.shape[0]
    if Kc.shape[1] != n:
        raise ValueError("Kc must be square.")
    b = int(max(1, min(bandwidth, n - 1)))
    gamma0 = float(np.mean(np.diag(Kc)))
    gammas = [gamma0]
    sigma2 = gamma0
    for ell in range(1, b + 1):
        gamma = float(np.mean(np.diag(Kc, k=ell)))
        gammas.append(gamma)
        weight = 1.0 - ell / (b + 1.0)
        sigma2 += 2.0 * weight * gamma
    return gamma0, float(sigma2), gammas


def default_bandwidth(n: int) -> int:
    return max(10, int(np.floor(float(n) ** (1.0 / 3.0))))


def kernel_ess_path_s2(X: Array, rho: float, bandwidth: Optional[int] = None) -> Dict[str, float]:
    X = normalize(np.asarray(X, dtype=float))
    n = int(X.shape[0])
    if bandwidth is None:
        bandwidth = default_bandwidth(n)
    K = kernel_gram_s2(X, rho)
    Kc = centered_gram(K)
    gamma0, sigma2, _ = bartlett_lrv_from_centered_gram(Kc, bandwidth)
    ess = np.inf if sigma2 == 0.0 else n * gamma0 / sigma2
    if sigma2 < 0.0:
        ess = np.nan
    return {
        "ess": float(ess),
        "gamma0": float(gamma0),
        "sigma2": float(sigma2),
        "bandwidth": int(bandwidth),
    }


def scalar_ess(y: Array, bandwidth: Optional[int] = None) -> Dict[str, float]:
    y = np.asarray(y, dtype=float).reshape(-1)
    n = int(y.size)
    if bandwidth is None:
        bandwidth = default_bandwidth(n)
    b = int(max(1, min(bandwidth, n - 1)))
    yc = y - y.mean()
    gamma0 = float(np.dot(yc, yc) / n)
    sigma2 = gamma0
    for ell in range(1, b + 1):
        gamma = float(np.dot(yc[:-ell], yc[ell:]) / (n - ell))
        weight = 1.0 - ell / (b + 1.0)
        sigma2 += 2.0 * weight * gamma
    if gamma0 <= 0.0:
        ess = np.nan
    elif sigma2 == 0.0:
        ess = np.inf
    elif sigma2 < 0.0:
        ess = np.nan
    else:
        ess = n * gamma0 / sigma2
    return {"ess": float(ess), "gamma0": float(gamma0), "sigma2": float(sigma2), "bandwidth": int(b)}


# -----------------------------------------------------------------------------
# Empirical MMD against a large iid reference sample
# -----------------------------------------------------------------------------


def mean_kernel_between_s2(X: Array, Y: Array, rho: float, block_size: int = 512) -> float:
    X = normalize(np.asarray(X, dtype=float))
    Y = normalize(np.asarray(Y, dtype=float))
    n, m = X.shape[0], Y.shape[0]
    total = 0.0
    count = 0
    for i in range(0, n, block_size):
        Xi = X[i : i + block_size]
        for j in range(0, m, block_size):
            Yj = Y[j : j + block_size]
            K = abel_poisson_s2_from_dots(Xi @ Yj.T, rho)
            total += float(K.sum())
            count += K.size
    return total / count


def mean_kernel_within_s2(X: Array, rho: float, block_size: int = 512) -> float:
    X = normalize(np.asarray(X, dtype=float))
    n = X.shape[0]
    total = 0.0
    count = 0
    for i in range(0, n, block_size):
        Xi = X[i : i + block_size]
        for j in range(0, n, block_size):
            Xj = X[j : j + block_size]
            K = abel_poisson_s2_from_dots(Xi @ Xj.T, rho)
            total += float(K.sum())
            count += K.size
    return total / count


def gamma0_from_iid_reference_s2(Y: Array, rho: float, block_size: int = 512) -> float:
    """Estimate gamma0 = E k(Y,Y) - E k(Y,Y') from iid reference draws.

    Uses the off-diagonal U-statistic estimate for E k(Y,Y').
    """
    Y = normalize(np.asarray(Y, dtype=float))
    m = int(Y.shape[0])
    if m < 2:
        raise ValueError("Need at least two reference draws to estimate gamma0.")
    k_diag = kernel_diag_s2(rho)
    mean_all = mean_kernel_within_s2(Y, rho, block_size=block_size)
    offdiag_mean = (m * m * mean_all - m * k_diag) / (m * (m - 1))
    return float(k_diag - offdiag_mean)


def empirical_mmd2_s2(
    X: Array,
    Y: Array,
    rho: float,
    block_size: int = 512,
    Kxx_mean: Optional[float] = None,
    Kyy_mean: Optional[float] = None,
) -> float:
    """V-statistic empirical MMD^2 between empirical measures of X and Y."""
    if Kxx_mean is None:
        Kxx_mean = mean_kernel_within_s2(X, rho, block_size=block_size)
    if Kyy_mean is None:
        Kyy_mean = mean_kernel_within_s2(Y, rho, block_size=block_size)
    Kxy_mean = mean_kernel_between_s2(X, Y, rho, block_size=block_size)
    return float(Kxx_mean + Kyy_mean - 2.0 * Kxy_mean)


def nearest_mode_frequencies(X: Array, target: VMFMixtureS2) -> Array:
    labels = target.nearest_component(X)
    counts = np.bincount(labels, minlength=target.n_components).astype(float)
    return counts / counts.sum()


def total_variation_distance(p: Array, q: Array) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(0.5 * np.sum(np.abs(p - q)))


# -----------------------------------------------------------------------------
# Default experimental target
# -----------------------------------------------------------------------------


def make_default_spherical_mixture() -> VMFMixtureS2:
    """A four-component unequal-weight vMF mixture on S^2.

    All components use the same concentration so that the weights are genuine
    mixture weights even if one works only up to proportionality.
    """
    return VMFMixtureS2(
        mus=tetrahedron_directions(),
        kappas=np.array([28.0, 28.0, 28.0, 28.0]),
        weights=np.array([0.40, 0.30, 0.20, 0.10]),
    )

# -----------------------------------------------------------------------------
# Notebook-friendly experiment runners and summaries
# -----------------------------------------------------------------------------


def run_rotation_experiment(
    outdir: Path | str,
    seed: int = 20260201,
    burn: int = 1000,
    n_keep: int = 3000,
    n_rot: int = 80,
    rho: float = 0.75,
    target_kappa: float = 12.0,
    proposal_kappa: float = 35.0,
    save_chain: bool = True,
) -> Dict[str, float]:
    """Run Experiment 1 and save CSV/JSON outputs.

    The experiment runs one vMF random-walk Metropolis chain on S^2 and then
    applies independent Haar rotations to the retained path.  It records the
    coordinate-wise scalar ESS and the intrinsic kernel ESS for each rotation.
    """
    outdir = ensure_dir(outdir)
    rng = np.random.default_rng(seed)
    target = VMFMixtureS2(
        mus=np.array([[0.0, 0.0, 1.0]]),
        kappas=np.array([target_kappa]),
        weights=np.array([1.0]),
    )
    x0 = sample_uniform_sphere(1, rng)[0]
    chain, accept_rate = metropolis_local_vmf(
        target.log_density,
        x0=x0,
        n_steps=burn + n_keep,
        prop_kappa=proposal_kappa,
        rng=rng,
    )
    X = chain[burn:]
    base_kernel = kernel_ess_path_s2(X, rho=rho)
    rows = []
    for r in range(n_rot):
        Q = random_rotation_matrix(rng)
        XR = X @ Q.T
        coord_stats = [scalar_ess(XR[:, j]) for j in range(3)]
        kernel_stats = kernel_ess_path_s2(XR, rho=rho)
        rows.append(
            {
                "rotation_id": r,
                "coord1_ess": coord_stats[0]["ess"],
                "coord2_ess": coord_stats[1]["ess"],
                "coord3_ess": coord_stats[2]["ess"],
                "kernel_ess": kernel_stats["ess"],
                "kernel_gamma0": kernel_stats["gamma0"],
                "kernel_sigma2": kernel_stats["sigma2"],
                "kernel_bandwidth": kernel_stats["bandwidth"],
            }
        )
    write_csv_rows(outdir / "exp1_rotation_ess.csv", rows)
    if save_chain:
        np.savez_compressed(outdir / "exp1_chain.npz", X=X)
    summary = {
        "experiment": "exp1_rotation_invariance",
        "seed": seed,
        "burn": burn,
        "n_keep": n_keep,
        "n_rot": n_rot,
        "rho": rho,
        "target_kappa": target_kappa,
        "proposal_kappa": proposal_kappa,
        "accept_rate": accept_rate,
        "base_kernel_ess": base_kernel["ess"],
        "base_kernel_gamma0": base_kernel["gamma0"],
        "base_kernel_sigma2": base_kernel["sigma2"],
        "base_kernel_bandwidth": base_kernel["bandwidth"],
    }
    save_json(summary, outdir / "exp1_config_summary.json")
    return summary


def summarize_rotation_experiment(outdir: Path | str):
    """Return summary tables for Experiment 1.

    Returns
    -------
    summary_df: pandas.DataFrame
        Row-level summary used for the paper table.
    long_df: pandas.DataFrame
        Long-format coordinate/kernel ESS data used for plotting.
    config: dict
        Run configuration.
    """
    import pandas as pd

    outdir = Path(outdir)
    df = pd.read_csv(outdir / "exp1_rotation_ess.csv")
    config = load_json(outdir / "exp1_config_summary.json")

    coord_cols = ["coord1_ess", "coord2_ess", "coord3_ess"]
    rows = []
    pooled = df[coord_cols].to_numpy().reshape(-1)
    rows.append(
        {
            "Quantity": "Coordinate ESS, pooled axes",
            "Mean": pooled.mean(),
            "SD": pooled.std(ddof=1),
            "Min": pooled.min(),
            "Max": pooled.max(),
            "Range/mean": (pooled.max() - pooled.min()) / pooled.mean(),
        }
    )
    for j, col in enumerate(coord_cols, start=1):
        x = df[col].to_numpy()
        rows.append(
            {
                "Quantity": f"Coordinate ESS, axis {j}",
                "Mean": x.mean(),
                "SD": x.std(ddof=1),
                "Min": x.min(),
                "Max": x.max(),
                "Range/mean": (x.max() - x.min()) / x.mean(),
            }
        )
    x = df["kernel_ess"].to_numpy()
    rows.append(
        {
            "Quantity": "Intrinsic kernel ESS",
            "Mean": x.mean(),
            "SD": x.std(ddof=1) if len(x) > 1 else 0.0,
            "Min": x.min(),
            "Max": x.max(),
            "Range/mean": (x.max() - x.min()) / x.mean(),
        }
    )
    summary_df = pd.DataFrame(rows)

    long_rows = []
    for _, row in df.iterrows():
        for j, col in enumerate(coord_cols, start=1):
            long_rows.append({"rotation_id": row["rotation_id"], "type": f"axis {j}", "ess": row[col]})
        long_rows.append({"rotation_id": row["rotation_id"], "type": "kernel", "ess": row["kernel_ess"]})
    long_df = pd.DataFrame(long_rows)
    return summary_df, long_df, config


def plot_rotation_experiment(outdir: Path | str, figdir: Path | str, formats: Sequence[str] = ("png", "pdf")) -> None:
    """Generate the Experiment 1 figure used in the paper."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    outdir = Path(outdir)
    figdir = ensure_dir(figdir)
    summary_df, long_df, config = summarize_rotation_experiment(outdir)

    coord_df = long_df[long_df["type"].str.startswith("axis")]
    kernel_df = long_df[long_df["type"] == "kernel"]
    base_kernel = float(config["base_kernel_ess"])

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))

    # Panel (a): boxplots of coordinate axes and kernel ESS.
    labels = ["axis 1", "axis 2", "axis 3", "kernel"]
    data = [long_df.loc[long_df["type"] == lab, "ess"].to_numpy() for lab in labels]
    axes[0].boxplot(
        data,
        labels=["axis 1", "axis 2", "axis 3", "kernel"],
        showmeans=True,
        meanprops={"marker": "^", "markerfacecolor": "tab:green", "markeredgecolor": "tab:green", "markersize": 6},
        medianprops={"color": "tab:orange", "linewidth": 1.8},
    )
    axes[0].axhline(base_kernel, linestyle="--", linewidth=1.0, color="0.3")
    axes[0].set_ylabel("Estimated ESS")
    axes[0].set_title("(a) ESS after rotations")
    handles = [
        Line2D([0], [0], color="tab:orange", linewidth=1.8, label="median"),
        Line2D([0], [0], marker="^", linestyle="None", color="tab:green", markerfacecolor="tab:green", label="mean"),
    ]
    axes[0].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False)

    # Panel (b): relative variation from unrotated value.
    pooled_coord = coord_df["ess"].to_numpy()
    coord_rel = (pooled_coord.max() - pooled_coord.min()) / pooled_coord.mean()
    kern = kernel_df["ess"].to_numpy()
    kern_rel = (kern.max() - kern.min()) / kern.mean()
    axes[1].bar([0, 1], [coord_rel, kern_rel], tick_label=["coordinate\nESS", "kernel\nESS"])
    axes[1].set_ylabel("Range / mean")
    axes[1].set_title("(b) Relative variation")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    for ext in formats:
        fig.savefig(figdir / f"fig1.{ext}", bbox_inches="tight")
    plt.close(fig)


def run_spherical_mixture_experiment(
    outdir: Path | str,
    seed: int = 20260202,
    burn: int = 1000,
    n_keep: int = 2500,
    n_rep: int = 20,
    ref_n: int = 8000,
    rhos: Sequence[float] = (0.35, 0.60, 0.85),
    local_proposal_kappa: float = 90.0,
    independence_proposal_kappa: float = 12.0,
    save_reference: bool = True,
) -> Dict:
    """Run Experiment 2 and save CSV/JSON outputs."""
    outdir = ensure_dir(outdir)
    rng = np.random.default_rng(seed)
    target = make_default_spherical_mixture()
    rhos = [float(r) for r in rhos]
    bandwidth = default_bandwidth(n_keep)

    Y_ref = target.sample(ref_n, rng)
    if save_reference:
        np.savez_compressed(outdir / "exp2_reference_sample.npz", Y_ref=Y_ref)

    ref_mode_probs = nearest_mode_frequencies(Y_ref, target)
    ref_mode_rows = [
        {"component": j, "reference_nearest_mode_probability": ref_mode_probs[j], "nominal_mixture_weight": target.weights[j]}
        for j in range(target.n_components)
    ]
    write_csv_rows(outdir / "exp2_reference_mode_probs.csv", ref_mode_rows)

    ref_terms = {}
    for rho in rhos:
        Kyy_mean = mean_kernel_within_s2(Y_ref, rho)
        gamma0_ref = gamma0_from_iid_reference_s2(Y_ref, rho)
        ref_terms[rho] = {"Kyy_mean": Kyy_mean, "gamma0_ref": gamma0_ref}

    rows = []
    mode_rows = []
    samplers = ["local_vmf", "independence_mixture"]
    for rep in range(n_rep):
        x0 = sample_uniform_sphere(1, rng)[0]
        chain_local, acc_local = metropolis_local_vmf(
            target.log_density,
            x0=x0,
            n_steps=burn + n_keep,
            prop_kappa=local_proposal_kappa,
            rng=rng,
        )
        chain_ind, acc_ind = metropolis_independence_vmf_mixture(
            target,
            x0=x0,
            n_steps=burn + n_keep,
            proposal_kappa=independence_proposal_kappa,
            rng=rng,
        )
        paths = {
            "local_vmf": (chain_local[burn:], acc_local),
            "independence_mixture": (chain_ind[burn:], acc_ind),
        }
        for sampler_name in samplers:
            X, acc = paths[sampler_name]
            freq = nearest_mode_frequencies(X, target)
            mode_tv = total_variation_distance(freq, ref_mode_probs)
            for j in range(target.n_components):
                mode_rows.append(
                    {
                        "rep": rep,
                        "sampler": sampler_name,
                        "component": j,
                        "chain_nearest_mode_frequency": freq[j],
                        "reference_nearest_mode_probability": ref_mode_probs[j],
                    }
                )
            for rho in rhos:
                ess_stats = kernel_ess_path_s2(X, rho=rho, bandwidth=bandwidth)
                Kxx_mean = mean_kernel_within_s2(X, rho)
                mmd2_raw = empirical_mmd2_s2(
                    X,
                    Y_ref,
                    rho,
                    Kxx_mean=Kxx_mean,
                    Kyy_mean=ref_terms[rho]["Kyy_mean"],
                )
                gamma0_ref = ref_terms[rho]["gamma0_ref"]
                mmd2_reference_corrected = mmd2_raw - gamma0_ref / ref_n
                rows.append(
                    {
                        "rep": rep,
                        "sampler": sampler_name,
                        "rho": rho,
                        "n_keep": n_keep,
                        "accept_rate": acc,
                        "kernel_ess": ess_stats["ess"],
                        "kernel_gamma0": ess_stats["gamma0"],
                        "kernel_sigma2": ess_stats["sigma2"],
                        "kernel_bandwidth": ess_stats["bandwidth"],
                        "mmd2_raw_reference": mmd2_raw,
                        "mmd2_reference_corrected": mmd2_reference_corrected,
                        "n_times_mmd2_reference_corrected": n_keep * mmd2_reference_corrected,
                        "calibration_ratio": (n_keep * mmd2_reference_corrected) / ess_stats["sigma2"],
                        "gamma0_ref": gamma0_ref,
                        "reference_size": ref_n,
                        "mode_tv_error": mode_tv,
                    }
                )
    write_csv_rows(outdir / "exp2_results.csv", rows)
    write_csv_rows(outdir / "exp2_mode_frequencies.csv", mode_rows)
    config = {
        "experiment": "exp2_spherical_mixture",
        "seed": seed,
        "burn": burn,
        "n_keep": n_keep,
        "n_rep": n_rep,
        "ref_n": ref_n,
        "rhos": rhos,
        "bandwidth": bandwidth,
        "local_proposal_kappa": local_proposal_kappa,
        "independence_proposal_kappa": independence_proposal_kappa,
        "target_mus": target.mus.tolist(),
        "target_kappas": target.kappas.tolist(),
        "target_weights": target.weights.tolist(),
        "reference_nearest_mode_probs": ref_mode_probs.tolist(),
        "reference_terms": {str(k): v for k, v in ref_terms.items()},
    }
    save_json(config, outdir / "exp2_config_summary.json")
    return config


def summarize_spherical_mixture_experiment(outdir: Path | str):
    """Return summary tables for Experiment 2."""
    import pandas as pd

    outdir = Path(outdir)
    df = pd.read_csv(outdir / "exp2_results.csv")
    config = load_json(outdir / "exp2_config_summary.json")
    group_cols = ["sampler", "rho"]
    summary = (
        df.groupby(group_cols)
        .agg(
            kernel_ess_mean=("kernel_ess", "mean"),
            kernel_ess_sd=("kernel_ess", "std"),
            sigma2_mean=("kernel_sigma2", "mean"),
            sigma2_sd=("kernel_sigma2", "std"),
            n_mmd_mean=("n_times_mmd2_reference_corrected", "mean"),
            n_mmd_sd=("n_times_mmd2_reference_corrected", "std"),
            ratio_mean=("calibration_ratio", "mean"),
            ratio_sd=("calibration_ratio", "std"),
            mode_tv_mean=("mode_tv_error", "mean"),
            mode_tv_sd=("mode_tv_error", "std"),
            accept_mean=("accept_rate", "mean"),
            accept_sd=("accept_rate", "std"),
        )
        .reset_index()
    )
    return df, summary, config


def plot_spherical_mixture_experiment(outdir: Path | str, figdir: Path | str, formats: Sequence[str] = ("png", "pdf")) -> None:
    """Generate Figures 2 and 3 for Experiment 2."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    outdir = Path(outdir)
    figdir = ensure_dir(figdir)
    df, summary, config = summarize_spherical_mixture_experiment(outdir)
    samplers = ["local_vmf", "independence_mixture"]
    sampler_labels = {"local_vmf": "local\nvMF", "independence_mixture": "independence\nmixture"}
    rhos = sorted(df["rho"].unique())

    # Figure 2: kernel ESS by rho and mode TV error.
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    offsets = {"local_vmf": -0.015, "independence_mixture": 0.015}
    for sampler in samplers:
        sub = summary[summary["sampler"] == sampler].sort_values("rho")
        x = sub["rho"].to_numpy() + offsets[sampler]
        axes[0].errorbar(x, sub["kernel_ess_mean"], yerr=sub["kernel_ess_sd"], marker="o", capsize=3, label=sampler_labels[sampler].replace("\n", " "))
    axes[0].set_xlabel(r"Kernel scale $\rho$")
    axes[0].set_ylabel("Estimated kernel ESS")
    axes[0].set_title("(a) Intrinsic kernel ESS")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)

    data = [df[df["sampler"] == s].groupby("rep")["mode_tv_error"].first().to_numpy() for s in samplers]
    axes[1].boxplot(
        data,
        labels=[sampler_labels[s] for s in samplers],
        showmeans=True,
        meanprops={"marker": "^", "markerfacecolor": "tab:green", "markeredgecolor": "tab:green", "markersize": 6},
        medianprops={"color": "tab:orange", "linewidth": 1.8},
    )
    axes[1].set_ylabel("Mode-frequency TV error")
    axes[1].set_title("(b) Mode mass error")
    handles = [
        Line2D([0], [0], color="tab:orange", linewidth=1.8, label="median"),
        Line2D([0], [0], marker="^", linestyle="None", color="tab:green", markerfacecolor="tab:green", label="mean"),
    ]
    axes[1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.31), ncol=2, frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.31)
    for ext in formats:
        fig.savefig(figdir / f"fig2.{ext}", bbox_inches="tight")
    plt.close(fig)

    # Figure 3: calibration scatter and ratio boxplot.
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    markers = {"local_vmf": "o", "independence_mixture": "s"}
    for sampler in samplers:
        sub = df[df["sampler"] == sampler]
        axes[0].scatter(sub["kernel_sigma2"], sub["n_times_mmd2_reference_corrected"], alpha=0.75, s=24, marker=markers[sampler], label=sampler_labels[sampler].replace("\n", " "))
    minv = min(df["kernel_sigma2"].min(), df["n_times_mmd2_reference_corrected"].min())
    maxv = max(df["kernel_sigma2"].max(), df["n_times_mmd2_reference_corrected"].max())
    axes[0].plot([minv, maxv], [minv, maxv], linestyle="--", color="0.3", linewidth=1.0)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Estimated $\widehat\sigma_k^2$")
    axes[0].set_ylabel(r"Corrected $n\widehat{\mathrm{MMD}}_k^2$")
    axes[0].set_title("(a) Risk calibration")
    axes[0].legend(frameon=False)

    data = []
    labels = []
    for sampler in samplers:
        for rho in rhos:
            data.append(df[(df["sampler"] == sampler) & (df["rho"] == rho)]["calibration_ratio"].to_numpy())
            labels.append(f"{sampler_labels[sampler]}\n$\\rho={rho:.2f}$")
    axes[1].boxplot(
        data,
        labels=labels,
        showmeans=True,
        meanprops={"marker": "^", "markerfacecolor": "tab:green", "markeredgecolor": "tab:green", "markersize": 6},
        medianprops={"color": "tab:orange", "linewidth": 1.8},
    )
    axes[1].axhline(1.0, linestyle="--", color="0.3", linewidth=1.0)
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"Calibration ratio")
    axes[1].set_title("(b) Ratio by sampler and scale")
    axes[1].tick_params(axis="x", labelrotation=0)
    axes[1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2, frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.37)
    for ext in formats:
        fig.savefig(figdir / f"fig3.{ext}", bbox_inches="tight")
    plt.close(fig)
