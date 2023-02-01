---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: pymc_env
  language: python
  name: pymc_env
---

(gaussian_mixture_model)=
# Gaussian Mixture Model

:::{post} April, 2022
:tags: mixture model, classification 
:category: beginner
:author: Abe Flaxman
:::

A [mixture model](https://en.wikipedia.org/wiki/Mixture_model) allows us to make inferences about the component contributors to a distribution of data. More specifically, a Gaussian Mixture Model allows us to make inferences about the means and standard deviations of a specified number of underlying component Gaussian distributions.

This could be useful in a number of ways. For example, we may be interested in simply describing a complex distribution parametrically (i.e. a [mixture distribution](https://en.wikipedia.org/wiki/Mixture_distribution)). Alternatively, we may be interested in [classification](https://en.wikipedia.org/wiki/Classification) where we seek to probabilistically classify which of a number of classes a particular observation is from.

```{code-cell} ipython3
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from scipy.stats import norm
from xarray_einstats.stats import XrContinuousRV
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
```

First we generate some simulated observations.

```{code-cell} ipython3
:tags: [hide-input]

k = 3
ndata = 500
centers = np.array([-5, 0, 5])
sds = np.array([0.5, 2.0, 0.75])
idx = rng.integers(0, k, ndata)
x = rng.normal(loc=centers[idx], scale=sds[idx], size=ndata)
plt.hist(x, 40);
```

In the PyMC model, we will estimate one $\mu$ and one $\sigma$ for each of the 3 clusters. Writing a Gaussian Mixture Model is very easy with the `pm.NormalMixture` distribution.

```{code-cell} ipython3
with pm.Model(coords={"cluster": range(k)}) as model:
    μ = pm.Normal(
        "μ",
        mu=0,
        sigma=5,
        transform=pm.distributions.transforms.univariate_ordered,
        initval=[-4, 0, 4],
        dims="cluster",
    )
    σ = pm.HalfNormal("σ", sigma=1, dims="cluster")
    weights = pm.Dirichlet("w", np.ones(k), dims="cluster")
    pm.NormalMixture("x", w=weights, mu=μ, sigma=σ, observed=x)

pm.model_to_graphviz(model)
```

```{code-cell} ipython3
with model:
    idata = pm.sample()
```

We can also plot the trace to check the nature of the MCMC chains, and compare to the ground truth values.

```{code-cell} ipython3
az.plot_trace(idata, var_names=["μ", "σ"], lines=[("μ", {}, [centers]), ("σ", {}, [sds])]);
```

And if we wanted, we could calculate the probability density function and examine the estimated group membership probabilities, based on the posterior mean estimates.

```{code-cell} ipython3
xi = np.linspace(-7, 7, 500)
post = idata.posterior
pdf_components = XrContinuousRV(norm, post["μ"], post["σ"]).pdf(xi) * post["w"]
pdf = pdf_components.sum("cluster")

fig, ax = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
# empirical histogram
ax[0].hist(x, 50)
ax[0].set(title="Data", xlabel="x", ylabel="Frequency")
# pdf
pdf_components.mean(dim=["chain", "draw"]).sum("cluster").plot.line(ax=ax[1])
ax[1].set(title="PDF", xlabel="x", ylabel="Probability\ndensity")
# plot group membership probabilities
(pdf_components / pdf).mean(dim=["chain", "draw"]).plot.line(hue="cluster", ax=ax[2])
ax[2].set(title="Group membership", xlabel="x", ylabel="Probability");
```

## Authors
- Authored by Abe Flaxman.
- Updated by Thomas Wiecki.
- Updated by [Benjamin T. Vincent](https://github.com/drbenvincent) in April 2022 ([#310](https://github.com/pymc-devs/pymc-examples/pull/310)) to use `pm.NormalMixture`.
- Updated by Benjamin T. Vincent in February 2023 to run on PyMC v5.

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,aeppl,xarray,xarray_einstats
```

:::{include} ../page_footer.md
:::
