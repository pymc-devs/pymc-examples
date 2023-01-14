---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# DEMetropolis(Z): Population vs. History efficiency comparison
The idea behind `DEMetropolis` is quite simple: Over time, a population of MCMC chains converges to the posterior, therefore the population can be used to inform joint proposals.
But just like the most recent positions of an entire population converges, so does the history of each individual chain.

In [ter Braak & Vrugt, 2008](https://doi.org/10.1007/s11222-008-9104-9) this history of posterior samples is used in the "DE-MCMC-Z" variant to make proposals.

The implementation in PyMC3 is based on `DE-MCMC-Z`, but a few details are different. Namely, each `DEMetropolisZ` chain only looks into its own history. Also we use a different tuning scheme.

In this notebook, a D-dimenstional multivariate normal target densities are sampled with `DEMetropolis` and `DEMetropolisZ` at different $N_{chains}$ settings.

```{code-cell} ipython3
import pathlib
import time

import arviz as az
import fastprogress
import ipywidgets
import numpy as np
import pandas as pd
import pymc3 as pm

from matplotlib import cm
from matplotlib import pyplot as plt

print(f"Running on PyMC3 v{pm.__version__}")
```

## Benchmarking with a D-dimensional MVNormal model
The function below constructs a fresh model for a given dimensionality and runs either `DEMetropolis` or `DEMetropolisZ` with the given settings. The resulting trace is saved with ArviZ.

If the saved trace is already found, it is loaded from disk.

Note that all traces are sampled with `cores=1`. This is because parallelization of `DEMetropolis` chains is slow at $O(N_{chains})$ and the comparison would be different depending on the number of available CPUs.

```{code-cell} ipython3
def get_mvnormal_model(D: int) -> pm.Model:
    true_mu = np.zeros(D)
    true_cov = np.eye(D)
    true_cov[:5, :5] = np.array(
        [
            [1, 0.5, 0, 0, 0],
            [0.5, 2, 2, 0, 0],
            [0, 2, 3, 0, 0],
            [0, 0, 0, 4, 4],
            [0, 0, 0, 4, 5],
        ]
    )

    with pm.Model() as pmodel:
        x = pm.MvNormal("x", mu=true_mu, cov=true_cov, shape=(D,))

    true_samples = x.random(size=1000)
    truth_id = az.data.convert_to_inference_data(true_samples[np.newaxis, :], group="random")
    return pmodel, truth_id


def run_setting(D, N_tune, N_draws, N_chains, algorithm):
    savename = f"{algorithm}_{D}_{N_tune}_{N_draws}_{N_chains}.nc"
    print(f"Scenario filename: {savename}")
    if not pathlib.Path(savename).exists():
        pmodel, truth_id = get_mvnormal_model(D)
        with pmodel:
            if algorithm == "DE-MCMC":
                step = pm.DEMetropolis()
            elif algorithm == "DE-MCMC-Z":
                step = pm.DEMetropolisZ()
            idata = pm.sample(
                cores=1,
                tune=N_tune,
                draws=N_draws,
                chains=N_chains,
                step=step,
                start={"x": [0] * D},
                discard_tuned_samples=False,
                return_inferencedata=True,
            )
        idata.to_netcdf(savename)
    else:
        idata = az.from_netcdf(savename)
    return idata
```

## Running the Benchmark Scenarios
Here a variety of different scenarios is computed and the results are aggregated in a multi-indexed DataFrame.

```{code-cell} ipython3
df_results = pd.DataFrame(columns="algorithm,D,N_tune,N_draws,N_chains,t,idata".split(","))
df_results = df_results.set_index("algorithm,D,N_tune,N_draws,N_chains".split(","))

for algorithm in {"DE-MCMC", "DE-MCMC-Z"}:
    for D in (10, 20, 40):
        N_tune = 10000
        N_draws = 10000
        for N_chains in (5, 10, 20, 30, 40, 80):
            idata = run_setting(D, N_tune, N_draws, N_chains, algorithm)
            t = idata.posterior.sampling_time
            df_results.loc[(algorithm, D, N_tune, N_draws, N_chains)] = (t, idata)
```

```{code-cell} ipython3
df_results[["t"]]
```

## Analyzing the traces
From the traces, we need to compute the absolute and relative $N_{eff}$ and the $\hat{R}$ to see if we can trust the posteriors.

```{code-cell} ipython3
df_temp = df_results.reset_index(["N_tune", "N_draws"])
df_temp["N_samples"] = [row.N_draws * row.Index[2] for row in df_temp.itertuples()]
df_temp["ess"] = [
    float(az.ess(idata.posterior).x.mean()) for idata in fastprogress.progress_bar(df_temp.idata)
]
df_temp["rel_ess"] = [row.ess / (row.N_samples) for row in df_temp.itertuples()]
df_temp["r_hat"] = [
    float(az.rhat(idata.posterior).x.mean()) for idata in fastprogress.progress_bar(df_temp.idata)
]
df_temp = df_temp.sort_index(level=["algorithm", "D", "N_chains"])
```

```{code-cell} ipython3
df_temp
```

## Visualizing Effective Sample Size
In this diagram, we'll plot the relative effective sample size against the number of chains.

Because our computation above ran everything with $N_{cores}=1$, we can't make a realistic comparison of effective sampling rates.

```{code-cell} ipython3
fig, right = plt.subplots(dpi=140, ncols=1, sharey="row", figsize=(12, 6))

for algorithm, linestyle in zip(["DE-MCMC", "DE-MCMC-Z"], ["-", "--"]):
    dimensionalities = list(sorted(set(df_temp.reset_index().D)))[::-1]
    N_dimensionalities = len(dimensionalities)
    for d, dim in enumerate(dimensionalities):
        color = cm.autumn(d / N_dimensionalities)
        df = df_temp.loc[(algorithm, dim)].reset_index()
        right.plot(
            df.N_chains,
            df.rel_ess * 100,
            linestyle=linestyle,
            color=color,
            label=f"{algorithm}, {dim} dimensions",
        )

right.legend()
right.set_ylabel("$S_{eff}$   [%]")
right.set_xlabel("$N_{chains}$   [-]")
right.set_ylim(0)
right.set_xlim(0)
plt.show()
```

## Visualizing Computation Time
As all traces were sampled with `cores=1`, we expect the computation time to grow linearly with the number of samples.

```{code-cell} ipython3
fig, ax = plt.subplots(dpi=140)

for alg in ["DE-MCMC", "DE-MCMC-Z"]:
    df = df_temp.sort_values("N_samples").loc[alg]
    ax.scatter(df.N_samples / 1000, df.t, label=alg)
ax.legend()
ax.set_xlabel("$N_{samples} / 1000$   [-]")
ax.set_ylabel("$t_{sampling}$   [s]")
fig.tight_layout()
plt.show()
```

## Visualizing the Traces
By comparing DE-MCMC and DE-MCMC-Z for a setting such as D=10, $N_{chains}$=5, you can see how DE-MCMC-Z has a clear advantage over a DE-MCMC that is run with too few chains.

```{code-cell} ipython3
def plot_trace(algorithm, D, N_chains):
    n_plot = min(10, N_chains)
    fig, axs = plt.subplots(nrows=n_plot, figsize=(12, 2 * n_plot))
    idata = df_results.loc[(algorithm, D, 10000, 10000, N_chains), "idata"]
    for c in range(n_plot):
        samples = idata.posterior.x[c, :, 0]
        axs[c].plot(samples, linewidth=0.5)
    plt.show()
    return


ipywidgets.interact_manual(
    plot_trace,
    algorithm=["DE-MCMC", "DE-MCMC-Z"],
    D=sorted(set(df_results.reset_index().D)),
    N_chains=sorted(set(df_results.reset_index().N_chains)),
);
```

## Inspecting the Sampler Stats
With the following widget, you can explore the sampler stats to better understand the tuning phase.

The `tune=None` default setting of `DEMetropolisZ` is the most robust tuning strategy. However, setting `tune='lambda'` can improves the initial convergence by doing a swing-in that makes it diverge much faster than it would with a constant `lambda`. The downside of tuning `lambda` is that if the tuning is stopped too early, it can get stuck with a very inefficient `lambda`.

Therefore, you should always inspect the `lambda` and rolling mean of `accepted` sampler stats when picking $N_{tune}$.

```{code-cell} ipython3
def plot_stat(*, sname: str = "accepted", rolling=True, algorithm, D, N_chains):
    fig, ax = plt.subplots(ncols=1, figsize=(12, 7), sharey="row")
    row = df_results.loc[(algorithm, D, 10000, 10000, N_chains)]
    for c in df_results.idata[0].posterior.chain:
        S = np.hstack(
            [
                # idata.warmup_sample_stats[sname].sel(chain=c),
                idata.sample_stats[sname].sel(chain=c)
            ]
        )
        y = pd.Series(S).rolling(window=500).mean().iloc[500 - 1 :].values if rolling else S
        ax.plot(y, linewidth=0.5)
    ax.set_xlabel("iteration")
    ax.set_ylabel(sname)
    plt.show()
    return


ipywidgets.interact_manual(
    plot_stat,
    sname=set(df_results.idata[0].sample_stats.keys()),
    rolling=True,
    algorithm=["DE-MCMC-Z", "DE-MCMC"],
    D=sorted(set(df_results.reset_index().D)),
    N_chains=sorted(set(df_results.reset_index().N_chains)),
);
```

## Conclusion
When used with the recommended settings, `DEMetropolis` is on par with `DEMetropolisZ`. On high-dimensional problems however, `DEMetropolisZ` can achieve the same effective sample sizes with less chains.

On problems where not enough CPUs are available to run $N_{chains}=2\cdot D$ `DEMetropolis` chains, the `DEMetropolisZ` should have much better scaling.

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w
```
