---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: default
  language: python
  name: python3
---

(pathfinder)=

# Pathfinder Variational Inference

:::{post} Feb 5, 2023 
:tags: variational inference, JAX
:category: advanced, how-to
:author: Thomas Wiecki
:::

+++

Pathfinder {cite:p}`zhang2021pathfinder` is a variational inference algorithm that produces samples from the posterior of a Bayesian model. It compares favorably to the widely used ADVI algorithm. On large problems, it should scale better than most MCMC algorithms, including dynamic HMC (i.e. NUTS), at the cost of a more biased estimate of the posterior. For details on the algorithm, see the [arxiv preprint](https://arxiv.org/abs/2108.03782).

PyMC's implementation of Pathfinder is now natively integrated using PyTensor. The Pathfinder implementation can be accessed through [pymc-extras](https://github.com/pymc-devs/pymc-extras/), which can be installed via:

`pip install git+https://github.com/pymc-devs/pymc-extras`

```{code-cell} ipython3
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pymc_extras as pmx

print(f"Running on PyMC v{pm.__version__}")
```

First, define your PyMC model. Here, we use the 8-schools model.

```{code-cell} ipython3
# Data of the Eight Schools Model
J = 8
y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

with pm.Model() as model:
    mu = pm.Normal("mu", mu=0.0, sigma=10.0)
    tau = pm.HalfCauchy("tau", 5.0)

    z = pm.Normal("z", mu=0, sigma=1, shape=J)
    theta = pm.Deterministic("theta", mu + tau * z)
    obs = pm.Normal("obs", mu=theta, sigma=sigma, shape=J, observed=y)
```

Next, we call `pmx.fit()` and pass in the algorithm we want it to use.

```{code-cell} ipython3
rng = np.random.default_rng(123)
with model:
    idata_ref = pm.sample(target_accept=0.9, random_seed=rng)
    idata_path = pmx.fit(
        method="pathfinder",
        jitter=12,
        num_draws=1000,
        random_seed=123,
    )
```

Just like `pymc.sample()`, this returns an idata with samples from the posterior. Note that because these samples do not come from an MCMC chain, convergence can not be assessed in the regular way.

```{code-cell} ipython3
az.plot_forest(
    [idata_ref, idata_path],
    var_names=["~z"],
    model_names=["ref", "path"],
    combined=True,
);
```

```{code-cell} ipython3
az.plot_trace(idata_path)
plt.tight_layout();
```

## References

:::{bibliography}
:filter: docname in docnames
:::

+++

## Authors

* Authored by Thomas Wiecki on Oct 11 2022 ([pymc-examples#429](https://github.com/pymc-devs/pymc-examples/pull/429))
* Re-execute notebook by Reshama Shaikh on Feb 5, 2023
* Bug fix by Chris Fonnesbeck on Jul 17, 2024
* Updated to PyMC implementation by Michael Cao on Feb 13, 2025
* Updated text by Chris Fonnesbeck on Feb 19, 2025

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p xarray
```

:::{include} ../page_footer.md
:::
