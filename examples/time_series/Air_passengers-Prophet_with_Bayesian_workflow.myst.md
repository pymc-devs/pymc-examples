---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(Air_passengers-Prophet_with_Bayesian_workflow)=
# Air passengers - Prophet-like model

:::{post} April, 2022
:tags: time series, prophet 
:category: intermediate
:author: Marco Gorelli, Danh Phan
:::

+++

We're going to look at the "air passengers" dataset, which tracks the monthly totals of a US airline passengers from 1949 to 1960. We could fit this using the [Prophet](https://facebook.github.io/prophet/) model {cite:p}`taylor2018forecasting` (indeed, this dataset is one of the examples they provide in their documentation), but instead we'll make our own Prophet-like model in PyMC3. This will make it a lot easier to inspect the model's components and to do prior predictive checks (an integral component of the [Bayesian workflow](https://arxiv.org/abs/2011.01808) {cite:p}`gelman2020bayesian`).

```{code-cell} ipython3
import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
```

```{code-cell} ipython3
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
%config InlineBackend.figure_format = 'retina'
```

```{code-cell} ipython3
# get data
try:
    df = pd.read_csv("../data/AirPassengers.csv", parse_dates=["Month"])
except FileNotFoundError:
    df = pd.read_csv(pm.get_data("AirPassengers.csv"), parse_dates=["Month"])
```

## Before we begin: visualise the data

```{code-cell} ipython3
df.plot.scatter(x="Month", y="#Passengers", color="k");
```

There's an increasing trend, with multiplicative seasonality. We'll fit a linear trend, and "borrow" the multiplicative seasonality part of it from Prophet.

+++

## Part 0: scale the data

+++

First, we'll scale time to be between 0 and 1:

```{code-cell} ipython3
t = (df["Month"] - pd.Timestamp("1900-01-01")).dt.days.to_numpy()
t_min = np.min(t)
t_max = np.max(t)
t = (t - t_min) / (t_max - t_min)
```

Next, for the target variable, we divide by the maximum. We do this, rather than standardising, so that the sign of the observations in unchanged - this will be necessary for the seasonality component to work properly later on.

```{code-cell} ipython3
y = df["#Passengers"].to_numpy()
y_max = np.max(y)
y = y / y_max
```

## Part 1: linear trend

The model we'll fit, for now, will just be

$$\text{Passengers} \sim \alpha + \beta\ \text{time}$$

First, let's try using the default priors set by prophet, and we'll do a prior predictive check:

```{code-cell} ipython3
with pm.Model(check_bounds=False) as linear:
    α = pm.Normal("α", mu=0, sigma=5)
    β = pm.Normal("β", mu=0, sigma=5)
    σ = pm.HalfNormal("σ", sigma=5)
    trend = pm.Deterministic("trend", α + β * t)
    pm.Normal("likelihood", mu=trend, sigma=σ, observed=y)

    linear_prior = pm.sample_prior_predictive()

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
ax[0].plot(
    df["Month"],
    az.extract_dataset(linear_prior, group="prior_predictive", num_samples=100)["likelihood"]
    * y_max,
    color="blue",
    alpha=0.05,
)
df.plot.scatter(x="Month", y="#Passengers", color="k", ax=ax[0])
ax[0].set_title("Prior predictive")
ax[1].plot(
    df["Month"],
    az.extract_dataset(linear_prior, group="prior", num_samples=100)["trend"] * y_max,
    color="blue",
    alpha=0.05,
)
df.plot.scatter(x="Month", y="#Passengers", color="k", ax=ax[1])
ax[1].set_title("Prior trend lines");
```

We can do better than this. These priors are evidently too wide, as we end up with implausibly many passengers. Let's try setting tighter priors.

```{code-cell} ipython3
with pm.Model(check_bounds=False) as linear:
    α = pm.Normal("α", mu=0, sigma=0.5)
    β = pm.Normal("β", mu=0, sigma=0.5)
    σ = pm.HalfNormal("σ", sigma=0.1)
    trend = pm.Deterministic("trend", α + β * t)
    pm.Normal("likelihood", mu=trend, sigma=σ, observed=y)

    linear_prior = pm.sample_prior_predictive(samples=100)

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
ax[0].plot(
    df["Month"],
    az.extract_dataset(linear_prior, group="prior_predictive", num_samples=100)["likelihood"]
    * y_max,
    color="blue",
    alpha=0.05,
)
df.plot.scatter(x="Month", y="#Passengers", color="k", ax=ax[0])
ax[0].set_title("Prior predictive")
ax[1].plot(
    df["Month"],
    az.extract_dataset(linear_prior, group="prior", num_samples=100)["trend"] * y_max,
    color="blue",
    alpha=0.05,
)
df.plot.scatter(x="Month", y="#Passengers", color="k", ax=ax[1])
ax[1].set_title("Prior trend lines");
```

Cool. Before going on to anything more complicated, let's try conditioning on the data and doing a posterior predictive check:

```{code-cell} ipython3
with linear:
    linear_trace = pm.sample(return_inferencedata=True)
    linear_prior = pm.sample_posterior_predictive(trace=linear_trace)

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
ax[0].plot(
    df["Month"],
    az.extract_dataset(linear_prior, group="posterior_predictive", num_samples=100)["likelihood"]
    * y_max,
    color="blue",
    alpha=0.01,
)
df.plot.scatter(x="Month", y="#Passengers", color="k", ax=ax[0])
ax[0].set_title("Posterior predictive")
ax[1].plot(
    df["Month"],
    az.extract_dataset(linear_trace, group="posterior", num_samples=100)["trend"] * y_max,
    color="blue",
    alpha=0.01,
)
df.plot.scatter(x="Month", y="#Passengers", color="k", ax=ax[1])
ax[1].set_title("Posterior trend lines");
```

## Part 2: enter seasonality

To model seasonality, we'll "borrow" the approach taken by Prophet - see [the Prophet paper](https://peerj.com/preprints/3190/) {cite:p}`taylor2018forecasting` for details, but the idea is to make a matrix of Fourier features which get multiplied by a vector of coefficients. As we'll be using multiplicative seasonality, the final model will be

$$\text{Passengers} \sim (\alpha + \beta\ \text{time}) (1 + \text{seasonality})$$

```{code-cell} ipython3
n_order = 10
periods = (df["Month"] - pd.Timestamp("1900-01-01")).dt.days / 365.25

fourier_features = pd.DataFrame(
    {
        f"{func}_order_{order}": getattr(np, func)(2 * np.pi * periods * order)
        for order in range(1, n_order + 1)
        for func in ("sin", "cos")
    }
)
fourier_features
```

Again, let's use the default Prophet priors, just to see what happens.

```{code-cell} ipython3
coords = {"fourier_features": np.arange(2 * n_order)}
with pm.Model(check_bounds=False, coords=coords) as linear_with_seasonality:
    α = pm.Normal("α", mu=0, sigma=0.5)
    β = pm.Normal("β", mu=0, sigma=0.5)
    σ = pm.HalfNormal("σ", sigma=0.1)
    β_fourier = pm.Normal("β_fourier", mu=0, sigma=10, dims="fourier_features")
    seasonality = pm.Deterministic(
        "seasonality", pm.math.dot(β_fourier, fourier_features.to_numpy().T)
    )
    trend = pm.Deterministic("trend", α + β * t)
    μ = trend * (1 + seasonality)
    pm.Normal("likelihood", mu=μ, sigma=σ, observed=y)

    linear_seasonality_prior = pm.sample_prior_predictive()

fig, ax = plt.subplots(nrows=3, ncols=1, sharex=False, figsize=(8, 6))
ax[0].plot(
    df["Month"],
    az.extract_dataset(linear_seasonality_prior, group="prior_predictive", num_samples=100)[
        "likelihood"
    ]
    * y_max,
    color="blue",
    alpha=0.05,
)
df.plot.scatter(x="Month", y="#Passengers", color="k", ax=ax[0])
ax[0].set_title("Prior predictive")
ax[1].plot(
    df["Month"],
    az.extract_dataset(linear_seasonality_prior, group="prior", num_samples=100)["trend"] * y_max,
    color="blue",
    alpha=0.05,
)
df.plot.scatter(x="Month", y="#Passengers", color="k", ax=ax[1])
ax[1].set_title("Prior trend lines")
ax[2].plot(
    df["Month"].iloc[:12],
    az.extract_dataset(linear_seasonality_prior, group="prior", num_samples=100)["seasonality"][:12]
    * 100,
    color="blue",
    alpha=0.05,
)
ax[2].set_title("Prior seasonality")
ax[2].set_ylabel("Percent change")
formatter = mdates.DateFormatter("%b")
ax[2].xaxis.set_major_formatter(formatter);
```

Again, this seems implausible. The default priors are too wide for our use-case, and it doesn't make sense to use them when we can do prior predictive checks to set more sensible ones. Let's try with some narrower ones:

```{code-cell} ipython3
coords = {"fourier_features": np.arange(2 * n_order)}
with pm.Model(check_bounds=False, coords=coords) as linear_with_seasonality:
    α = pm.Normal("α", mu=0, sigma=0.5)
    β = pm.Normal("β", mu=0, sigma=0.5)
    trend = pm.Deterministic("trend", α + β * t)

    β_fourier = pm.Normal("β_fourier", mu=0, sigma=0.1, dims="fourier_features")
    seasonality = pm.Deterministic(
        "seasonality", pm.math.dot(β_fourier, fourier_features.to_numpy().T)
    )

    μ = trend * (1 + seasonality)
    σ = pm.HalfNormal("σ", sigma=0.1)
    pm.Normal("likelihood", mu=μ, sigma=σ, observed=y)

    linear_seasonality_prior = pm.sample_prior_predictive()

fig, ax = plt.subplots(nrows=3, ncols=1, sharex=False, figsize=(8, 6))
ax[0].plot(
    df["Month"],
    az.extract_dataset(linear_seasonality_prior, group="prior_predictive", num_samples=100)[
        "likelihood"
    ]
    * y_max,
    color="blue",
    alpha=0.05,
)
df.plot.scatter(x="Month", y="#Passengers", color="k", ax=ax[0])
ax[0].set_title("Prior predictive")
ax[1].plot(
    df["Month"],
    az.extract_dataset(linear_seasonality_prior, group="prior", num_samples=100)["trend"] * y_max,
    color="blue",
    alpha=0.05,
)
df.plot.scatter(x="Month", y="#Passengers", color="k", ax=ax[1])
ax[1].set_title("Prior trend lines")
ax[2].plot(
    df["Month"].iloc[:12],
    az.extract_dataset(linear_seasonality_prior, group="prior", num_samples=100)["seasonality"][:12]
    * 100,
    color="blue",
    alpha=0.05,
)
ax[2].set_title("Prior seasonality")
ax[2].set_ylabel("Percent change")
formatter = mdates.DateFormatter("%b")
ax[2].xaxis.set_major_formatter(formatter);
```

Seems a lot more believable. Time for a posterior predictive check:

```{code-cell} ipython3
with linear_with_seasonality:
    linear_seasonality_trace = pm.sample(return_inferencedata=True)
    linear_seasonality_posterior = pm.sample_posterior_predictive(trace=linear_seasonality_trace)

fig, ax = plt.subplots(nrows=3, ncols=1, sharex=False, figsize=(8, 6))
ax[0].plot(
    df["Month"],
    az.extract_dataset(linear_seasonality_posterior, group="posterior_predictive", num_samples=100)[
        "likelihood"
    ]
    * y_max,
    color="blue",
    alpha=0.05,
)
df.plot.scatter(x="Month", y="#Passengers", color="k", ax=ax[0])
ax[0].set_title("Posterior predictive")
ax[1].plot(
    df["Month"],
    az.extract_dataset(linear_trace, group="posterior", num_samples=100)["trend"] * y_max,
    color="blue",
    alpha=0.05,
)
df.plot.scatter(x="Month", y="#Passengers", color="k", ax=ax[1])
ax[1].set_title("Posterior trend lines")
ax[2].plot(
    df["Month"].iloc[:12],
    az.extract_dataset(linear_seasonality_trace, group="posterior", num_samples=100)["seasonality"][
        :12
    ]
    * 100,
    color="blue",
    alpha=0.05,
)
ax[2].set_title("Posterior seasonality")
ax[2].set_ylabel("Percent change")
formatter = mdates.DateFormatter("%b")
ax[2].xaxis.set_major_formatter(formatter);
```

Neat!

+++

## Conclusion

We saw how we could implement a Prophet-like model ourselves and fit it to the air passengers dataset. Prophet is an awesome library and a net-positive to the community, but by implementing it ourselves, however, we can take whichever components of it we think are relevant to our problem, customise them, and carry out the Bayesian workflow {cite:p}`gelman2020bayesian`). Next time you have a time series problem, I hope you will try implementing your own probabilistic model rather than using Prophet as a "black-box" whose arguments are tuneable hyperparameters.

For reference, you might also want to check out:
- [TimeSeeers](https://github.com/MBrouns/timeseers), a hierarchical Bayesian Time Series model based on Facebooks Prophet, written in PyMC3
- [PM-Prophet](https://github.com/luke14free/pm-prophet), a Pymc3-based universal time series prediction and decomposition library inspired by Facebook Prophet

+++

## Authors
* Authored by [Marco Gorelli](https://github.com/MarcoGorelli) in June, 2021 ([pymc-examples#183](https://github.com/pymc-devs/pymc-examples/pull/183))
* Updated by Danh Phan in May, 2022 ([pymc-examples#320](https://github.com/pymc-devs/pymc-examples/pull/320))

+++

## References
:::{bibliography}
:filter: docname in docnames
:::

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,aeppl,xarray
```

:::{include} ../page_footer.md
:::
