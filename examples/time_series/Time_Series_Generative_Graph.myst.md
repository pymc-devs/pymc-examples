---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: pymc_env
  language: python
  name: python3
---

(time_series_generative_graph)=
# Time Series Models Derived From a Generative Graph

:::{post} March, 2024
:tags: time-series, 
:category: intermediate, reference
:author: Juan Orduz and Ricardo Vieira
:::

+++

In This notebook, we show to model and fit a time series model starting from a generative graph. In particular, we explain how to use {func}`~pytensor.scan.basic.scan` to loop efficiently inside a PyMC model.

For this example, we consider an autoregressive model AR(2). Recall that an AR(2) model is defined as:

$$
\begin{align*}
y_t &= \rho_1 y_{t-1} + \rho_2 y_{t-2} + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2)
\end{align*}
$$

That is, we have a recursive linear model in term of the first two lags of the time series.

```{code-cell} ipython3
:tags: [hide-input]

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from pymc.pytensorf import collect_default_updates

az.style.use("arviz-darkgrid")
plt.rcParams["figure.figsize"] = [12, 7]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.facecolor"] = "white"

%config InlineBackend.figure_format = "retina"

rng = np.random.default_rng(42)
```

## Define AR(2) Process

We start by encoding the generative graph of the AR(2) model as a function `ar_dist`. The strategy is to pass this function as a custom distribution via {class}`~pymc.CustomDist` inside a PyMC model. 

We need to specify the initial state (`ar_init`), the autoregressive coefficients (`rho`), and the standard deviation of the noise (`sigma`). Given such parameters, we can define the generative graph of the AR(2) model using the  {func}`~pytensor.scan.basic.scan` operation.

```{code-cell} ipython3
lags = 2  # Number of lags
trials = 100  # Time series length


def ar_dist(ar_init, rho, sigma, size):
    def ar_step(x_tm2, x_tm1, rho, sigma):
        mu = x_tm1 * rho[0] + x_tm2 * rho[1]
        x = mu + pm.Normal.dist(sigma=sigma)
        return x, collect_default_updates([x])

    ar_innov, _ = pytensor.scan(
        fn=ar_step,
        outputs_info=[{"initial": ar_init, "taps": range(-lags, 0)}],
        non_sequences=[rho, sigma],
        n_steps=trials - lags,
        strict=True,
    )

    return ar_innov
```

## Generate AR(2) Graph

Now that we have implemented the AR(2) step, we can assign priors to the parameters `rho` and `sigma`. We also specify the initial state `ar_init` as a zero vector.

```{code-cell} ipython3
coords = {
    "lags": range(-lags, 0),
    "steps": range(trials - lags),
    "trials": range(trials),
}
with pm.Model(coords=coords, check_bounds=False) as model:
    rho = pm.Normal(name="rho", mu=0, sigma=0.2, dims=("lags",))
    sigma = pm.HalfNormal(name="sigma", sigma=0.2)

    ar_init_obs = pm.MutableData(name="ar_init_obs", value=np.zeros(lags), dims=("lags",))
    ar_init = pm.Normal(name="ar_init", observed=ar_init_obs, dims=("lags",))

    ar_innov_obs = pm.MutableData("ar_innov_obs", np.zeros(trials - lags), dims=("steps",))
    ar_innov = pm.CustomDist(
        "ar_dist",
        ar_init,
        rho,
        sigma,
        dist=ar_dist,
        observed=ar_innov_obs,
        dims=("steps",),
    )

    ar = pm.Deterministic(
        name="ar", var=pt.concatenate([ar_init, ar_innov], axis=-1), dims=("trials",)
    )


pm.model_to_graphviz(model)
```

## Prior

Let's sample from the prior distribution to see how the AR(2) model behaves.

```{code-cell} ipython3
with model:
    prior = pm.sample_prior_predictive(samples=100, random_seed=rng)
```

```{code-cell} ipython3
_, ax = plt.subplots()
for i, hdi_prob in enumerate((0.94, 0.64), 1):
    hdi = az.hdi(prior.prior["ar"], hdi_prob=hdi_prob)["ar"]
    lower = hdi.sel(hdi="lower")
    upper = hdi.sel(hdi="higher")
    ax.fill_between(
        x=np.arange(trials),
        y1=lower,
        y2=upper,
        alpha=(i - 0.2) * 0.2,
        color="C0",
        label=f"{hdi_prob:.0%} HDI",
    )
ax.plot(prior.prior["ar"].mean(("chain", "draw")), color="C0", label="Mean")
ax.legend(loc="upper right")
ax.set_xlabel("time")
ax.set_title("AR(2) Prior Samples", fontsize=18, fontweight="bold")
```

It is not surprising that the prior distribution is a stationary process around zero given that the prior of the  `rho` parameter is far from one.

Let's look into individual samples to get a feeling of the heterogeneity of the generated series:

```{code-cell} ipython3
fig, ax = plt.subplots(
    nrows=5, ncols=1, figsize=(12, 12), sharex=True, sharey=True, layout="constrained"
)
chosen_draw = 2
for i, axi in enumerate(ax, start=chosen_draw):
    axi.plot(
        prior.prior["ar"].isel(draw=i, chain=0),
        color="C0" if i == chosen_draw else "black",
    )
    axi.set_title(f"Sample {i}", fontsize=18, fontweight="bold")
ax[-1].set_xlabel("time")
```

## Posterior

Next, we want to condition the AR(2) model on some observed data so that we can do a parameter recovery analysis.

```{code-cell} ipython3
# Pick a random draw from the prior (i.e. a time series)
prior_draw = prior.prior.isel(chain=0, draw=chosen_draw)

# Set the observed values
ar_init_obs.set_value(prior_draw["ar"].values[:lags])
ar_innov_obs.set_value(prior_draw["ar"].values[lags:])
ar_obs = prior_draw["ar"].to_numpy()
rho_true = prior_draw["rho"].to_numpy()
sigma_true = prior_draw["sigma"].to_numpy()

# Output the true values
print(f"rho_true={np.round(rho_true, 3)}, {sigma_true=:.3f}")
```

We now run the MCMC algorithm to sample from the posterior distribution.

```{code-cell} ipython3
with model:
    trace = pm.sample(random_seed=rng)
```

Let's plot the trace and the posterior distribution of the parameters.

```{code-cell} ipython3
axes = az.plot_trace(
    data=trace,
    var_names=["rho", "sigma"],
    compact=True,
    lines=[
        ("rho", {}, rho_true),
        ("sigma", {}, sigma_true),
    ],
    backend_kwargs={"figsize": (12, 5), "layout": "constrained"},
)
plt.gcf().suptitle("AR(2) Model Trace", fontsize=18, fontweight="bold")
```

```{code-cell} ipython3
axes = az.plot_posterior(
    trace, var_names=["rho", "sigma"], ref_val=[*rho_true, sigma_true], figsize=(15, 5)
)
plt.gcf().suptitle("AR(2) Model Parameters Posterior", fontsize=18, fontweight="bold")
```

We see we have successfully recovered the true parameters of the model.

+++

## Posterior Predictive

Finally, we can use the posterior samples to generate new data from the AR(2) model. We can then compare the generated data with the observed data to check the goodness of fit of the model.

```{code-cell} ipython3
with model:
    post_pred = pm.sample_posterior_predictive(trace, random_seed=rng)
```

```{code-cell} ipython3
post_pred_ar = post_pred.posterior_predictive["ar"]

_, ax = plt.subplots()
for i, hdi_prob in enumerate((0.94, 0.64), 1):
    hdi = az.hdi(post_pred_ar, hdi_prob=hdi_prob)["ar"]
    lower = hdi.sel(hdi="lower")
    upper = hdi.sel(hdi="higher")
    ax.fill_between(
        x=np.arange(trials),
        y1=lower,
        y2=upper,
        alpha=(i - 0.2) * 0.2,
        color="C0",
        label=f"{hdi_prob:.0%} HDI",
    )
ax.plot(prior.prior["ar"].mean(("chain", "draw")), color="C0", label="Mean")
ax.plot(ar_obs, color="black", label="Observed")
ax.legend(loc="upper right")
ax.set_xlabel("time")
ax.set_title("AR(2) Posterior Predictive Samples", fontsize=18, fontweight="bold")
```

Overall, we the model is capturing the global dynamics of the time series. In order to have a abetter insight of the model, we can plot a subset of the posterior samples and compare them with the observed data.

```{code-cell} ipython3
fig, ax = plt.subplots(
    nrows=5, ncols=1, figsize=(12, 12), sharex=True, sharey=True, layout="constrained"
)
for i, axi in enumerate(ax):
    axi.plot(post_pred.posterior_predictive["ar"].isel(draw=i, chain=0), color="C0")
    axi.plot(ar_obs, color="black", label="Observed")
    axi.legend(loc="upper right")
    axi.set_title(f"Sample {i}")

ax[-1].set_xlabel("time")

fig.suptitle("AR(2) Posterior Predictive Samples", fontsize=18, fontweight="bold", y=1.05)
```

## Authors
- Authored by [Juan Orduz](https://juanitorduz.github.io/) and [Ricardo Vieira](https://github.com/ricardoV94) in March 2024

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor
```

:::{include} ../page_footer.md
:::
