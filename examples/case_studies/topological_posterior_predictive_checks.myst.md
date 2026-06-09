---
kernelspec:
  display_name: Python 3
  language: python
  name: python3
language_info:
  name: python
  pygments_lexer: ipython3
myst_substitutions:
  conda_dependencies: ripser
  extra_dependencies: ripser
  extra_install_notes: '`ripser` is used to compute Vietoris--Rips persistent homology
    for the topological posterior predictive checks.'
  pip_dependencies: ripser
jupytext:
  formats: ipynb,myst
  text_representation:
    extension: .myst.md
    format_name: myst
    format_version: '0.13'
    jupytext_version: 1.16.7
---

(topological_posterior_predictive_checks)=
# Topological posterior predictive checks with PyMC

:::{post} May 21, 2026
:tags: posterior predictive checks, model criticism, topology, time series
:category: intermediate, how-to
:author: Guillaume Sidoine
:::

Posterior predictive checks are a central part of Bayesian workflow: after fitting a model, we simulate replicated data from the posterior predictive distribution and compare those replicas to the observed data {cite:p}`gelman2013bayesian`.

This notebook demonstrates a complementary diagnostic: **topological posterior predictive checks**. The idea is to use persistent-homology summaries as posterior predictive discrepancy statistics. These summaries can detect global shape features, such as connected components and loops, that ordinary low-order summaries can miss. Persistent homology summarizes the birth and death of topological features across scales {cite:p}`edelsbrunner2010computational,ghrist2008barcodes`.

We use a realistic synthetic example: a noisy seasonal time series. In the raw time domain the data is one-dimensional, but its delay-coordinate embedding has a recurrent loop. We fit two PyMC models:

1. A misspecified Gaussian model on the delay embedding. It can match means and covariance but cannot reproduce the recurrent loop.
2. A revised Fourier seasonal model on the time series. Its posterior predictive delay embeddings recover the loop.

The example illustrates a Bayesian workflow pattern:

> ordinary posterior predictive checks pass → topological posterior predictive checks detect missing recurrent structure → model is revised → topological posterior predictive checks improve.

The topology is used as an external diagnostic on posterior predictive samples. Related work has used topological summaries in likelihood-free and approximate Bayesian computation {cite:p}`thorne2022topological`; here the focus is posterior predictive model criticism in a standard PyMC workflow.

## Dependencies

This notebook requires PyMC, ArviZ, NumPy, Pandas, Matplotlib, and `ripser` for Vietoris--Rips persistent homology.

:::{include} ../extra_installs.md
:::

The notebook intentionally keeps the example small enough to run quickly, but the Markov chain Monte Carlo cells may still take a minute or two depending on hardware.

```{code-cell} ipython3
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

try:
    from ripser import ripser
except ImportError as err:
    raise ImportError(
        "This notebook requires the `ripser` package. Install it with `pip install ripser`."
    ) from err

warnings.filterwarnings("ignore", category=FutureWarning)
```

```{code-cell} ipython3
RANDOM_SEED = 20260520
rng = np.random.default_rng(RANDOM_SEED)

az.style.use("arviz-darkgrid")
print(f"PyMC version: {pm.__version__}")
```

## A short intuition for homology

Homology is a way to summarize the qualitative shape of an object. In low dimensions, the most useful summaries are intuitive:

- **H0** tracks connected components. A point cloud with two well-separated clusters has two prominent H0 features.
- **H1** tracks loops or holes. A noisy circle has one prominent H1 feature.
- **H2** tracks enclosed voids in three-dimensional objects, such as the hollow center of a sphere.

For point-cloud data, there is no single fixed shape at the start. Persistent homology therefore builds a family of shapes indexed by a distance scale. At a small scale, each point is almost isolated. As the scale grows, nearby points connect, loops can appear, and eventually loops are filled in. A feature's **persistence** is the gap between the scale where it appears and the scale where it disappears.

This is useful for posterior predictive checking because some model failures are global rather than marginal. A model can reproduce means, variances, and correlations while still failing to reproduce a connected component, a loop, or another large-scale geometric structure.

```{code-cell} ipython3
# Visual intuition: H0 sees connected components; H1 sees loops.
theta = np.linspace(0.0, 2.0 * np.pi, 160, endpoint=False)
noisy_circle = np.column_stack([np.cos(theta), np.sin(theta)])
noisy_circle += rng.normal(scale=0.06, size=noisy_circle.shape)

left_cluster = rng.normal(loc=(-1.0, 0.0), scale=0.15, size=(80, 2))
right_cluster = rng.normal(loc=(1.0, 0.0), scale=0.15, size=(80, 2))
two_clusters = np.vstack([left_cluster, right_cluster])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(two_clusters[:, 0], two_clusters[:, 1], s=18, alpha=0.8)
axes[0].set_title("H0 intuition: two connected components")
axes[0].set_aspect("equal", adjustable="box")

axes[1].scatter(noisy_circle[:, 0], noisy_circle[:, 1], s=18, alpha=0.8)
axes[1].set_title("H1 intuition: one persistent loop")
axes[1].set_aspect("equal", adjustable="box")

for ax in axes:
    ax.set_xlabel("x")
    ax.set_ylabel("y")

plt.tight_layout()
```

## Generate a seasonal time series

We generate a synthetic seasonal signal with noise and a weak second harmonic. The observed data is not constructed as a literal circle. The loop appears only after delay-coordinate embedding, which is a standard way to reveal recurrence in a time series.

```{code-cell} ipython3
def make_seasonal_timeseries(
    n_time: int = 300,
    period: int = 50,
    amplitude: float = 1.0,
    noise: float = 0.15,
    trend_strength: float = 0.0,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Generate a noisy seasonal time series with a weak second harmonic.

    The signal is deliberately simple: it is periodic, mildly non-sinusoidal,
    and noisy. This makes it useful for demonstrating how a delay embedding can
    reveal recurrent structure that is not explicit in the one-dimensional time
    series plot.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_time, dtype=float)
    seasonal = amplitude * np.sin(2.0 * np.pi * t / period)
    harmonic = 0.25 * amplitude * np.sin(4.0 * np.pi * t / period + 0.7)
    trend = trend_strength * (t - t.mean()) / max(t.max(), 1.0)
    y = seasonal + harmonic + trend + rng.normal(scale=noise, size=n_time)
    return y.astype(float)


def delay_embedding(y: np.ndarray, lag: int = 12, dim: int = 2) -> np.ndarray:
    """Return a delay-coordinate embedding of a univariate time series.

    The output has columns ``(y_t, y_{t-lag}, ...)``. For periodic or recurrent
    signals, this embedding can turn temporal recurrence into visible geometric
    structure, such as a loop. That geometry is what the topological posterior
    predictive check compares between observed and replicated data.
    """
    y = np.asarray(y, dtype=float)
    if lag <= 0:
        raise ValueError("lag must be positive")
    if dim < 2:
        raise ValueError("dim must be at least 2")
    n_points = len(y) - (dim - 1) * lag
    if n_points <= 5:
        raise ValueError("time series is too short for the requested delay embedding")
    cols = [y[(dim - 1 - j) * lag : (dim - 1 - j) * lag + n_points] for j in range(dim)]
    return np.column_stack(cols)


N_TIME = 300
PERIOD = 50
LAG = 12
DELAY_DIM = 2

# The lag is approximately one quarter of the period, which makes the recurrent loop visible.
observed_series = make_seasonal_timeseries(
    n_time=N_TIME, period=PERIOD, noise=0.15, seed=RANDOM_SEED
)
observed_embedding = delay_embedding(observed_series, lag=LAG, dim=DELAY_DIM)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(observed_series, lw=1.5)
axes[0].set(title="Observed seasonal time series", xlabel="time", ylabel="y")
axes[1].scatter(observed_embedding[:, 0], observed_embedding[:, 1], s=16, alpha=0.8)
axes[1].set(title="Delay-coordinate embedding", xlabel="$y_t$", ylabel=f"$y_{{t-{LAG}}}$")
axes[1].set_aspect("equal", adjustable="box")
plt.tight_layout()
```

The delay embedding has an annular or loop-like shape. A model that treats the embedded points as a single Gaussian cloud can match the embedding's mean and covariance, but it has no reason to preserve the hole.

## Topological posterior predictive check helper functions

For each posterior predictive replicate, we compute Vietoris--Rips persistent homology with `ripser`. We summarize the diagrams using a few scalar summaries:

- `h1_max_persistence`: the longest-lived loop;
- `h1_persistence_gap`: the gap between the longest and second-longest loop lifetimes;
- `h1_total_persistence_p2`: squared total H1 persistence, which emphasizes large features;
- `h1_n_features_ge_threshold`: number of loops with lifetime above a threshold.

For this example, the main diagnostic is `h1_max_persistence`.

```{code-cell} ipython3
Diagram = np.ndarray
Diagrams = list[Diagram]


@dataclass
class TopoPPCResult:
    """Container for ordinary and topological posterior predictive check results."""

    name: str
    observed_embedding: np.ndarray
    replicated_embeddings: np.ndarray
    obs_diagrams: Diagrams
    rep_diagrams: list[Diagrams]
    ordinary_table: pd.DataFrame
    topo_table: pd.DataFrame


def compute_persistence_diagrams(X: np.ndarray, maxdim: int = 1) -> Diagrams:
    """Compute Vietoris--Rips persistence diagrams for a point cloud.

    ``ripser`` returns one diagram per homological dimension. In this notebook,
    dimension 0 tracks connected components and dimension 1 tracks loops.
    """
    return ripser(X, maxdim=maxdim)["dgms"]


def finite_lifetimes(diagram: Diagram) -> np.ndarray:
    """Return finite persistence lifetimes from a persistence diagram."""
    if diagram is None or diagram.size == 0:
        return np.array([], dtype=float)
    dgm = np.asarray(diagram, dtype=float)
    dgm = dgm[np.isfinite(dgm[:, 1])]
    if dgm.size == 0:
        return np.array([], dtype=float)
    return np.maximum(dgm[:, 1] - dgm[:, 0], 0.0)


def max_persistence(diagram: Diagram) -> float:
    """Return the lifetime of the longest-lived topological feature."""
    lifetimes = finite_lifetimes(diagram)
    return float(np.max(lifetimes)) if lifetimes.size else 0.0


def second_max_persistence(diagram: Diagram) -> float:
    """Return the second-largest finite lifetime, or zero if it is absent."""
    lifetimes = finite_lifetimes(diagram)
    if lifetimes.size < 2:
        return 0.0
    return float(np.sort(lifetimes)[-2])


def persistence_gap(diagram: Diagram) -> float:
    """Return the gap between the two longest-lived topological features."""
    return float(max_persistence(diagram) - second_max_persistence(diagram))


def total_persistence(diagram: Diagram, power: float = 1.0) -> float:
    """Return total persistence, optionally emphasizing long-lived features."""
    lifetimes = finite_lifetimes(diagram)
    return float(np.sum(lifetimes**power)) if lifetimes.size else 0.0


def n_persistent_features(diagram: Diagram, threshold: float) -> int:
    """Count features whose lifetimes exceed a chosen persistence threshold."""
    lifetimes = finite_lifetimes(diagram)
    return int(np.sum(lifetimes >= threshold))


def summarize_diagram(diagram: Diagram, dim: int, threshold: float = 0.5) -> dict[str, float]:
    """Convert one persistence diagram into scalar discrepancy statistics."""
    prefix = f"h{dim}"
    return {
        f"{prefix}_n_finite_features": float(finite_lifetimes(diagram).size),
        f"{prefix}_max_persistence": max_persistence(diagram),
        f"{prefix}_second_max_persistence": second_max_persistence(diagram),
        f"{prefix}_persistence_gap": persistence_gap(diagram),
        f"{prefix}_total_persistence_p1": total_persistence(diagram, power=1.0),
        f"{prefix}_total_persistence_p2": total_persistence(diagram, power=2.0),
        f"{prefix}_n_features_ge_threshold": float(n_persistent_features(diagram, threshold)),
    }


def smoothed_ppc_p_value(
    obs_value: float, rep_values: Iterable[float], direction: str = "greater"
) -> float:
    """Compute a conservative Monte Carlo posterior predictive p-value.

    The +1 smoothing avoids returning exactly zero when the observed statistic
    is more extreme than all finite posterior predictive draws.
    """
    values = np.asarray(list(rep_values), dtype=float)
    p_greater = (1.0 + np.sum(values >= obs_value)) / (values.size + 1.0)
    p_less = (1.0 + np.sum(values <= obs_value)) / (values.size + 1.0)
    if direction == "greater":
        return float(p_greater)
    if direction == "less":
        return float(p_less)
    if direction == "two-sided":
        return float(min(1.0, 2.0 * min(p_greater, p_less)))
    raise ValueError(f"unknown direction: {direction}")


def ordinary_embedding_table(
    observed_embedding: np.ndarray, replicated_embeddings: np.ndarray
) -> pd.DataFrame:
    """Compare mean and covariance summaries of observed and replicated embeddings."""

    def summarize(X: np.ndarray) -> dict[str, float]:
        mean = X.mean(axis=0)
        cov = np.cov(X.T)
        eigvals = np.linalg.eigvalsh(cov)
        return {
            "mean_x": float(mean[0]),
            "mean_y": float(mean[1]),
            "cov_xx": float(cov[0, 0]),
            "cov_yy": float(cov[1, 1]),
            "cov_xy": float(cov[0, 1]),
            "cov_eig_min": float(eigvals[0]),
            "cov_eig_max": float(eigvals[-1]),
        }

    obs_summary = summarize(observed_embedding)
    rep_summaries = pd.DataFrame([summarize(X) for X in replicated_embeddings])
    rows = []
    for metric, observed in obs_summary.items():
        rep_values = rep_summaries[metric].to_numpy()
        q05, q50, q95 = np.quantile(rep_values, [0.05, 0.50, 0.95])
        rows.append(
            {
                "metric": metric,
                "observed": observed,
                "rep_q05": q05,
                "rep_q50": q50,
                "rep_q95": q95,
                "inside_90_interval": bool(q05 <= observed <= q95),
            }
        )
    return pd.DataFrame(rows)


def topological_table(
    obs_diagrams: Diagrams, rep_diagrams: list[Diagrams], h1_threshold: float = 0.5
) -> pd.DataFrame:
    """Compare observed topological summaries to posterior predictive summaries."""
    obs_summary = {}
    for dim in [0, 1]:
        obs_summary.update(summarize_diagram(obs_diagrams[dim], dim=dim, threshold=h1_threshold))

    rep_rows = []
    for diagrams in rep_diagrams:
        row = {}
        for dim in [0, 1]:
            row.update(summarize_diagram(diagrams[dim], dim=dim, threshold=h1_threshold))
        rep_rows.append(row)
    rep_summary = pd.DataFrame(rep_rows)

    rows = []
    for metric, observed in obs_summary.items():
        rep_values = rep_summary[metric].to_numpy(dtype=float)
        q05, q50, q95 = np.quantile(rep_values, [0.05, 0.50, 0.95])
        rows.append(
            {
                "metric": metric,
                "observed": observed,
                "rep_q05": q05,
                "rep_q50": q50,
                "rep_q95": q95,
                "p_high": smoothed_ppc_p_value(observed, rep_values, "greater"),
                "p_low": smoothed_ppc_p_value(observed, rep_values, "less"),
                "p_two_sided": smoothed_ppc_p_value(observed, rep_values, "two-sided"),
            }
        )
    return pd.DataFrame(rows)


def run_topoppc(
    name: str,
    observed_embedding: np.ndarray,
    replicated_embeddings: np.ndarray,
    h1_threshold: float = 0.5,
) -> TopoPPCResult:
    """Run ordinary and topological posterior predictive checks for embeddings."""
    obs_diagrams = compute_persistence_diagrams(observed_embedding, maxdim=1)
    rep_diagrams = [compute_persistence_diagrams(X, maxdim=1) for X in replicated_embeddings]
    return TopoPPCResult(
        name=name,
        observed_embedding=observed_embedding,
        replicated_embeddings=replicated_embeddings,
        obs_diagrams=obs_diagrams,
        rep_diagrams=rep_diagrams,
        ordinary_table=ordinary_embedding_table(observed_embedding, replicated_embeddings),
        topo_table=topological_table(obs_diagrams, rep_diagrams, h1_threshold=h1_threshold),
    )
```

```{code-cell} ipython3
def plot_replicates(result: TopoPPCResult, rep_ids: tuple[int, ...] = (0, 1, 2)) -> None:
    """Plot the observed embedding next to a few posterior predictive embeddings."""
    fig, axes = plt.subplots(1, 1 + len(rep_ids), figsize=(14, 3.6), sharex=True, sharey=True)
    axes[0].scatter(
        result.observed_embedding[:, 0], result.observed_embedding[:, 1], s=13, alpha=0.8
    )
    axes[0].set_title("Observed")
    axes[0].set_aspect("equal", adjustable="box")
    for ax, rep_id in zip(axes[1:], rep_ids):
        X = result.replicated_embeddings[rep_id]
        ax.scatter(X[:, 0], X[:, 1], s=13, alpha=0.8)
        ax.set_title(f"Replicate {rep_id}")
        ax.set_aspect("equal", adjustable="box")
    for ax in axes:
        ax.set_xlabel("$y_t$")
        ax.set_ylabel(f"$y_{{t-{LAG}}}$")
    fig.suptitle(result.name, y=1.05)
    plt.tight_layout()


def plot_metric_histogram(result: TopoPPCResult, metric: str) -> None:
    """Plot the posterior predictive distribution of one topological statistic."""
    row = result.topo_table.query("metric == @metric").iloc[0]
    obs = float(row["observed"])

    # Recompute replicated scalar values from diagrams for plotting.
    dim = int(metric[1])
    values = []
    for diagrams in result.rep_diagrams:
        summary = summarize_diagram(diagrams[dim], dim=dim)
        values.append(summary[metric])
    values = np.asarray(values)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(values, bins=25, alpha=0.75, label="posterior predictive replicas")
    ax.axvline(
        obs,
        lw=2.5,
        label=f"observed; p_high={row['p_high']:.3f}, p_low={row['p_low']:.3f}",
    )
    ax.set(title=f"{result.name}: PPC for {metric}", xlabel=metric, ylabel="count")
    ax.legend()
    plt.tight_layout()


def plot_h1_diagram(result: TopoPPCResult, rep_id: int = 0) -> None:
    """Plot finite H1 birth/death pairs for the observed data and one replicate."""
    obs = result.obs_diagrams[1]
    rep = result.rep_diagrams[rep_id][1]
    obs = obs[np.isfinite(obs[:, 1])] if obs.size else np.empty((0, 2))
    rep = rep[np.isfinite(rep[:, 1])] if rep.size else np.empty((0, 2))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    if rep.size:
        ax.scatter(rep[:, 0], rep[:, 1], s=25, alpha=0.55, label=f"replicate {rep_id}")
    if obs.size:
        ax.scatter(obs[:, 0], obs[:, 1], s=35, alpha=0.85, label="observed")
    if obs.size or rep.size:
        stacked = np.vstack([x for x in [obs, rep] if x.size])
        hi = float(np.max(stacked[:, 1])) * 1.05
    else:
        hi = 1.0
    ax.plot([0, hi], [0, hi], ls="--", lw=1.5, label="death = birth")
    ax.set(xlabel="birth", ylabel="death", title=f"{result.name}: finite H1 persistence diagram")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.legend()
    plt.tight_layout()


def display_selected_tables(result: TopoPPCResult) -> None:
    """Display compact ordinary and topological posterior predictive check tables."""
    selected_metrics = [
        "h1_max_persistence",
        "h1_persistence_gap",
        "h1_total_persistence_p2",
        "h1_n_features_ge_threshold",
        "h0_max_persistence",
    ]
    print("Ordinary PPC table")
    display(result.ordinary_table)
    print("Selected topological PPC table")
    display(result.topo_table[result.topo_table["metric"].isin(selected_metrics)])


def replace_inf_deaths(diagram: Diagram, max_eps: float) -> Diagram:
    """Replace infinite deaths by ``max_eps`` for bounded Betti-curve plotting."""
    if diagram is None or diagram.size == 0:
        return np.empty((0, 2), dtype=float)
    dgm = np.asarray(diagram, dtype=float).copy()
    dgm[np.isinf(dgm[:, 1]), 1] = max_eps
    return dgm


def betti_curve(diagram: Diagram, eps_grid: np.ndarray, max_eps: float) -> np.ndarray:
    """Compute a Betti curve from a persistence diagram over a grid of scales."""
    dgm = replace_inf_deaths(diagram, max_eps=max_eps)
    if dgm.size == 0:
        return np.zeros_like(eps_grid, dtype=float)

    births = dgm[:, 0]
    deaths = dgm[:, 1]
    curve = np.zeros_like(eps_grid, dtype=float)
    for i, eps in enumerate(eps_grid):
        curve[i] = np.sum((births <= eps) & (eps < deaths))
    return curve


def betti_curves_for_result(
    result: TopoPPCResult,
    dim: int = 1,
    n_grid: int = 200,
    eps_max_quantile: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return observed and replicated Betti curves for one homological dimension."""
    finite_deaths = []
    for dgm in [result.obs_diagrams[dim]] + [dgms[dim] for dgms in result.rep_diagrams]:
        if dgm.size:
            finite_deaths.extend(dgm[np.isfinite(dgm[:, 1]), 1].tolist())

    max_eps = float(np.quantile(finite_deaths, eps_max_quantile)) if finite_deaths else 1.0
    max_eps = max(max_eps, 1e-8)
    eps_grid = np.linspace(0.0, max_eps, n_grid)

    obs_curve = betti_curve(result.obs_diagrams[dim], eps_grid, max_eps=max_eps)
    rep_curves = np.vstack(
        [betti_curve(dgms[dim], eps_grid, max_eps=max_eps) for dgms in result.rep_diagrams]
    )
    return eps_grid, obs_curve, rep_curves


def plot_betti_envelope(result: TopoPPCResult, dim: int = 1) -> None:
    """Plot an observed Betti curve against its posterior predictive envelope."""
    eps_grid, obs_curve, rep_curves = betti_curves_for_result(result, dim=dim)

    q05 = np.quantile(rep_curves, 0.05, axis=0)
    q25 = np.quantile(rep_curves, 0.25, axis=0)
    q50 = np.quantile(rep_curves, 0.50, axis=0)
    q75 = np.quantile(rep_curves, 0.75, axis=0)
    q95 = np.quantile(rep_curves, 0.95, axis=0)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.fill_between(eps_grid, q05, q95, alpha=0.20, label="posterior predictive 90% envelope")
    ax.fill_between(eps_grid, q25, q75, alpha=0.35, label="posterior predictive 50% envelope")
    ax.plot(eps_grid, q50, ls="--", label="posterior predictive median")
    ax.plot(eps_grid, obs_curve, lw=2.5, label="observed")
    ax.set(
        xlabel="filtration scale epsilon",
        ylabel=f"Betti-{dim}",
        title=f"{result.name}: Betti-{dim} posterior predictive envelope",
    )
    ax.legend()
    plt.tight_layout()
```

## Model 1: a Gaussian model on the delay embedding

The first model treats the delay-embedded points as independent draws from a single bivariate normal distribution. This is deliberately misspecified: a Gaussian cloud can reproduce mean and covariance, but it cannot express the recurrent hole in the embedding.

```{code-cell} ipython3
coords_bad = {
    "obs_id": np.arange(observed_embedding.shape[0]),
    "coord": ["y_t", f"y_t_minus_{LAG}"],
}

with pm.Model(coords=coords_bad) as gaussian_embedding_model:
    mu = pm.Normal("mu", mu=0.0, sigma=2.0, dims="coord")
    chol, corr, stds = pm.LKJCholeskyCov(
        "chol_cov",
        n=2,
        eta=2.0,
        sd_dist=pm.Exponential.dist(1.0),
        compute_corr=True,
    )
    x = pm.MvNormal("x", mu=mu, chol=chol, observed=observed_embedding, dims=("obs_id", "coord"))

    idata_bad = pm.sample(
        draws=400,
        tune=400,
        chains=2,
        cores=1,
        target_accept=0.9,
        random_seed=RANDOM_SEED,
        progressbar=False,
    )
    ppc_bad = pm.sample_posterior_predictive(
        idata_bad,
        var_names=["x"],
        random_seed=RANDOM_SEED + 1,
        progressbar=False,
    )
```

```{code-cell} ipython3
az.summary(idata_bad, var_names=["mu", "chol_cov_stds", "chol_cov_corr"], round_to=2)

rhat_bad = az.rhat(idata_bad, var_names=["mu", "chol_cov_stds", "chol_cov_corr"]).to_array()
assert float(rhat_bad.max()) < 1.03
```

```{code-cell} ipython3
def posterior_predictive_embedding_array(
    ppc,
    var_name: str = "x",
    n_rep: int = 100,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Extract a manageable subset of posterior predictive embedding replicas."""
    arr = ppc.posterior_predictive[var_name]
    arr = arr.stack(sample=("chain", "draw")).transpose("sample", "obs_id", "coord").values
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.shape[0], size=min(n_rep, arr.shape[0]), replace=False)
    return arr[idx]


replicated_embeddings_bad = posterior_predictive_embedding_array(ppc_bad, var_name="x", n_rep=100)
bad_result = run_topoppc(
    "Gaussian embedding model", observed_embedding, replicated_embeddings_bad, h1_threshold=0.5
)

display_selected_tables(bad_result)
plot_replicates(bad_result)
plot_metric_histogram(bad_result, "h1_max_persistence")
plot_h1_diagram(bad_result, rep_id=0)
plot_betti_envelope(bad_result, dim=1)
```

The scalar diagnostic below is the primary decision statistic, but the Betti-1 envelope is also useful as a visual posterior predictive check. It shows how the number of active one-dimensional holes evolves over the filtration scale, while the maximum-persistence statistic focuses on the single most persistent loop.

The Gaussian model usually passes the ordinary mean/covariance checks. But the topological check based on `h1_max_persistence` should fail: the observed delay embedding contains one persistent loop, while the Gaussian posterior predictive replicas contain only short-lived noisy cycles.

## Model 2: a revised Fourier seasonal model

The topological failure suggests that the model is missing recurrence or periodic structure. We therefore fit a simple Fourier seasonal regression to the original time series, generate posterior predictive time series, and then delay-embed those posterior predictive samples.

This is the key workflow step: the topology is not forced into the sampler. Instead, the topological posterior predictive check diagnoses a structural failure and motivates a better generative model.

```{code-cell} ipython3
def fourier_design(t: np.ndarray, period: int, harmonics: int = 2) -> np.ndarray:
    """Build an intercept plus sine/cosine Fourier design matrix."""
    t = np.asarray(t, dtype=float)
    cols = [np.ones_like(t)]
    for h in range(1, harmonics + 1):
        cols.append(np.sin(2.0 * np.pi * h * t / period))
        cols.append(np.cos(2.0 * np.pi * h * t / period))
    return np.column_stack(cols)


HARMONICS = 2
time = np.arange(N_TIME)
fourier_design_matrix = fourier_design(time, period=PERIOD, harmonics=HARMONICS)

coords_good = {
    "time": time,
    "coef": ["intercept"]
    + [f"h{h}_{kind}" for h in range(1, HARMONICS + 1) for kind in ["sin", "cos"]],
}

with pm.Model(coords=coords_good) as fourier_model:
    beta = pm.Normal("beta", mu=0.0, sigma=1.5, dims="coef")
    sigma = pm.HalfNormal("sigma", sigma=0.5)
    mu = pt.dot(fourier_design_matrix, beta)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=observed_series, dims="time")

    idata_good = pm.sample(
        draws=400,
        tune=400,
        chains=2,
        cores=1,
        target_accept=0.9,
        random_seed=RANDOM_SEED + 2,
        progressbar=False,
    )
    ppc_good = pm.sample_posterior_predictive(
        idata_good,
        var_names=["y"],
        random_seed=RANDOM_SEED + 3,
        progressbar=False,
    )
```

```{code-cell} ipython3
az.summary(idata_good, var_names=["beta", "sigma"], round_to=2)

rhat_good = az.rhat(idata_good, var_names=["beta", "sigma"]).to_array()
assert float(rhat_good.max()) < 1.03
```

```{code-cell} ipython3
def posterior_predictive_delay_embeddings(
    ppc,
    var_name: str = "y",
    lag: int = LAG,
    dim: int = DELAY_DIM,
    n_rep: int = 100,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Delay-embed posterior predictive time series replicas."""
    arr = ppc.posterior_predictive[var_name]
    arr = arr.stack(sample=("chain", "draw")).transpose("sample", "time").values
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.shape[0], size=min(n_rep, arr.shape[0]), replace=False)
    return np.stack([delay_embedding(arr[i], lag=lag, dim=dim) for i in idx], axis=0)


replicated_embeddings_good = posterior_predictive_delay_embeddings(
    ppc_good, var_name="y", n_rep=100
)
good_result = run_topoppc(
    "Fourier seasonal model", observed_embedding, replicated_embeddings_good, h1_threshold=0.5
)

display_selected_tables(good_result)
plot_replicates(good_result)
plot_metric_histogram(good_result, "h1_max_persistence")
plot_h1_diagram(good_result, rep_id=0)
plot_betti_envelope(good_result, dim=1)
```

After adding seasonal structure, the posterior predictive delay embeddings should now reproduce the observed loop. The `h1_max_persistence` value for the observed data should be typical under the revised model's posterior predictive distribution.

## Compare the two models through the topological posterior predictive check

The table below compares the main topological discrepancy, `h1_max_persistence`, between the misspecified Gaussian embedding model and the revised Fourier seasonal model.

```{code-cell} ipython3
def extract_metric(result: TopoPPCResult, metric: str) -> pd.Series:
    """Extract one discrepancy row and attach the model name."""
    row = result.topo_table.query("metric == @metric").iloc[0].copy()
    row["model"] = result.name
    return row


comparison = pd.DataFrame(
    [
        extract_metric(bad_result, "h1_max_persistence"),
        extract_metric(good_result, "h1_max_persistence"),
    ]
)[["model", "observed", "rep_q05", "rep_q50", "rep_q95", "p_high", "p_low", "p_two_sided"]]
comparison
```

The intended interpretation is:

- The Gaussian embedding model can pass ordinary moment-based posterior predictive checks but fails the topological posterior predictive check because it cannot reproduce the recurrent loop.
- The Fourier seasonal model is structurally closer to the data-generating process and produces posterior predictive delay embeddings with similar H1 persistence to the observed embedding.

This demonstrates how persistent homology can be used as a model criticism tool: it detects a missing global shape feature and suggests a direction for model revision.

## Discussion and limitations

Topological posterior predictive checks should not replace ordinary posterior predictive checks. They answer a different question: whether the posterior predictive distribution reproduces the global shape of the observed data in a chosen representation. In this notebook, the representation is a delay-coordinate embedding of a time series.

Several choices matter:

- the data representation used before computing topology;
- the filtration and persistent homology dimension;
- the scalar topological summary used as a discrepancy;
- the number of posterior predictive replicates;
- whether the topological feature is scientifically meaningful rather than an artifact of preprocessing.

The method is most useful when the scientific question involves recurrence, loops, disconnected components, branching structure, voids, or other global geometric features.

A conservative way to phrase the contribution is:

> Persistent homology summaries can be used as posterior predictive discrepancy statistics in Bayesian workflow, allowing model criticism for global shape features that ordinary scalar summaries may miss.

## Appendix: why homology works for point clouds

This section gives a compact mathematical view of the construction used above. It is not needed to use the diagnostic, but it explains why persistent homology is a natural source of discrepancy statistics.

A **simplicial complex** is a shape built from vertices, edges, triangles, tetrahedra, and their higher-dimensional analogues. For a point cloud, the Vietoris--Rips construction builds such a complex at a distance scale $\epsilon$: points become vertices, nearby pairs become edges, triples whose pairwise distances are small become filled triangles, and so on.

For each dimension $k$, a **$k$-chain** is a formal sum of $k$-simplices. The **boundary operator** $\partial_k$ maps each $k$-simplex to its oriented boundary. For example, the boundary of an edge is its two endpoints, and the boundary of a triangle is its three edges. A central identity is

$$
\partial_k \circ \partial_{k+1} = 0,
$$

which says that the boundary of a boundary is empty.

Homology compares two kinds of objects:

- **cycles**, which have zero boundary and therefore look closed;
- **boundaries**, which are cycles that bound a higher-dimensional filled object.

The $k$-th homology group is

$$
H_k = \ker(\partial_k) / \operatorname{im}(\partial_{k+1}).
$$

Its rank is the **Betti number** $\beta_k$. In the examples above, $\beta_0$ counts connected components and $\beta_1$ counts loops that have not yet been filled by triangles.

Persistent homology repeats this calculation over a sequence of scales. A loop may be born when enough edges connect around a hole, and it may die when enough triangles fill the hole. Long persistence means that the feature is stable across scales, which is why maximum H1 persistence is a useful statistic for detecting the recurrent loop in the delay embedding.

In practice, this notebook delegates the matrix-reduction algorithm to `ripser`. The Bayesian workflow only needs the resulting diagrams and their scalar summaries, which can be compared to posterior predictive replicas like any other discrepancy statistic.

## Authors

* Authored by Guillaume Sidoine in May, 2026 ([pymc-examples#881](https://github.com/pymc-devs/pymc-examples/pull/881))

## References

:::{bibliography}
:filter: docname in docnames
:::

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,xarray
```

:::{include} ../page_footer.md
:::

