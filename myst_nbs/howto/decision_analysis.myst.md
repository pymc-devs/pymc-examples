---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Decision analysis with a utility function

:::{post} Dec 15, 2022
:tags: Bayesian workflow, decision analysis
:category: beginner, how-to
:author: Oriol Abril Pla
:::

```{code-cell} ipython3
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from scipy import stats
from xarray_einstats.stats import XrContinuousRV, XrDiscreteRV
```

```{code-cell} ipython3
RANDOM_SEED = 4235
rng = np.random.default_rng(RANDOM_SEED)
```

## Overview

In this example we will show how to perform Bayesian decison analysis over a set of discrete decisions. We will use as an example, we will take the role of a shopper, who has done their weekly grocery shopping at a different supermarket every week for the last year, and has logged:

* The supermarket they went to that week
* The cumulative time it took from exiting home to getting back, so both travel time and unintuitive supermarket layout are penalized (in hours)
* The money spent (in euros)
* The number of rubbish bins that were filled that week

+++

## Example data

```{code-cell} ipython3
obs_supermarket = rng.choice(["aldi", "caprabo", "condis", "consum", "keisy"], size=52)
super_id, supermarkets = pd.factorize(obs_supermarket, sort=True)
time_means = np.array([7, 5, 4.7, 6, 5.2])
cost_means = np.array([90, 125, 130, 120, 150])
rubbish_means = np.array([5, 3, 4, 4, 3])
obs_time = rng.normal(time_means[super_id], 0.2)
obs_cost = rng.normal(cost_means[super_id], 1)
obs_rubbish = rng.poisson(rubbish_means[super_id])
```

## Model definition

```{code-cell} ipython3
coords = {
    "supermarket": supermarkets,
    "measure": ["time", "cost"],
}
with pm.Model(coords=coords) as model:
    model.add_coord("week", length=len(super_id), mutable=True)
    supermarket_id = pm.MutableData("supermarket_id", super_id, dims="week")

    mu = pm.Normal("mu", 0, 1, dims=("measure", "supermarket"))
    sigma = pm.HalfNormal("sigma", 0.5, dims=("measure", "supermarket"))
    lam = pm.Exponential("lam", 0.3, dims="supermarket")

    pm.Normal(
        "time",
        mu=mu[0, supermarket_id],
        sigma=sigma[0, supermarket_id],
        dims="week",
        observed=obs_time,
    )
    pm.Normal(
        "cost",
        mu=mu[1, supermarket_id],
        sigma=sigma[1, supermarket_id],
        dims="week",
        observed=obs_cost,
    )
    pm.Poisson("rubbish", mu=lam[supermarket_id], dims="week", observed=obs_rubbish)
```

## Inference

```{code-cell} ipython3
with model:
    idata = pm.sample()
```

```{code-cell} ipython3
idata
```

## Posterior predictive sampling

```{code-cell} ipython3
with model:
    model.set_dim("week", len(supermarkets))
    model.set_data("supermarket_id", np.arange(len(supermarkets)))

    pm.sample_posterior_predictive(idata, extend_inferencedata=True)

# we can't rename dimensions in pymc, so we do it now
post_pred = idata.posterior_predictive
post_pred = post_pred.rename(week="supermarket").assign_coords(
    supermarket=idata.posterior["supermarket"]
)
```

## Decision analysis

```{code-cell} ipython3
def utility(time, cost, rubbish):
    return -(cost + 20 * time + 2 * (rubbish + 1) ** 2)
```

```{code-cell} ipython3
util = utility(post_pred["time"], post_pred["cost"], post_pred["rubbish"])
util.mean(dim=("chain", "draw"))
```

```{code-cell} ipython3
def utility_no_rubbish(time, cost, rubbish):
    return -(cost + 30 * time)


util2 = utility_no_rubbish(post_pred["time"], post_pred["cost"], post_pred["rubbish"])
```

```{code-cell} ipython3
import matplotlib.pyplot as plt

_, ax = plt.subplots(1, 2, figsize=(12, 6))
az.plot_density(
    [util.sel(supermarket=s) for s in supermarkets],
    data_labels=supermarkets,
    labeller=az.labels.NoVarLabeller(),
    ax=ax[0],
)
ax[0].set_title("Utility function **with** rubbish term")
az.plot_density(
    [util2.sel(supermarket=s) for s in supermarkets],
    data_labels=supermarkets,
    labeller=az.labels.NoVarLabeller(),
    ax=ax[1],
)
ax[1].set_title("Utility function **without** rubbish term")
```

## Authors
* Authored by Oriol Abril Pla in Dec 2022

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

```{code-cell} ipython3

```
