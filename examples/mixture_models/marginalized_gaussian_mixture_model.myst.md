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

+++ {"papermill": {"duration": 0.012112, "end_time": "2020-12-20T20:45:32.375345", "exception": false, "start_time": "2020-12-20T20:45:32.363233", "status": "completed"}}

# Marginalized Gaussian Mixture Model

:::{post} Sept 18, 2021
:tags: mixture model, 
:category: intermediate
:::

```{code-cell} ipython3
---
papermill:
  duration: 5.906876
  end_time: '2020-12-20T20:45:38.293074'
  exception: false
  start_time: '2020-12-20T20:45:32.386198'
  status: completed
---
import arviz as az
import numpy as np
import pymc3 as pm
import seaborn as sns

from matplotlib import pyplot as plt

print(f"Running on PyMC3 v{pm.__version__}")
```

```{code-cell} ipython3
---
papermill:
  duration: 0.034525
  end_time: '2020-12-20T20:45:38.340340'
  exception: false
  start_time: '2020-12-20T20:45:38.305815'
  status: completed
---
%config InlineBackend.figure_format = 'retina'
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
```

+++ {"papermill": {"duration": 0.011094, "end_time": "2020-12-20T20:45:38.362640", "exception": false, "start_time": "2020-12-20T20:45:38.351546", "status": "completed"}}

Gaussian mixtures are a flexible class of models for data that exhibits subpopulation heterogeneity.  A toy example of such a data set is shown below.

```{code-cell} ipython3
---
papermill:
  duration: 0.019101
  end_time: '2020-12-20T20:45:38.392974'
  exception: false
  start_time: '2020-12-20T20:45:38.373873'
  status: completed
---
N = 1000

W = np.array([0.35, 0.4, 0.25])

MU = np.array([0.0, 2.0, 5.0])
SIGMA = np.array([0.5, 0.5, 1.0])
```

```{code-cell} ipython3
---
papermill:
  duration: 0.018854
  end_time: '2020-12-20T20:45:38.422840'
  exception: false
  start_time: '2020-12-20T20:45:38.403986'
  status: completed
---
component = rng.choice(MU.size, size=N, p=W)
x = rng.normal(MU[component], SIGMA[component], size=N)
```

```{code-cell} ipython3
---
papermill:
  duration: 0.422847
  end_time: '2020-12-20T20:45:38.856513'
  exception: false
  start_time: '2020-12-20T20:45:38.433666'
  status: completed
---
fig, ax = plt.subplots(figsize=(8, 6))

ax.hist(x, bins=30, density=True, lw=0);
```

+++ {"papermill": {"duration": 0.012072, "end_time": "2020-12-20T20:45:38.881581", "exception": false, "start_time": "2020-12-20T20:45:38.869509", "status": "completed"}}

A natural parameterization of the Gaussian mixture model is as the [latent variable model](https://en.wikipedia.org/wiki/Latent_variable_model)

$$
\begin{align*}
\mu_1, \ldots, \mu_K
    & \sim N(0, \sigma^2) \\
\tau_1, \ldots, \tau_K
    & \sim \textrm{Gamma}(a, b) \\
\boldsymbol{w}
    & \sim \textrm{Dir}(\boldsymbol{\alpha}) \\
z\ |\ \boldsymbol{w}
    & \sim \textrm{Cat}(\boldsymbol{w}) \\
x\ |\ z
    & \sim N(\mu_z, \tau^{-1}_z).
\end{align*}
$$

An implementation of this parameterization in PyMC3 is available at {doc}`gaussian_mixture_model`.  A drawback of this parameterization is that is posterior relies on sampling the discrete latent variable $z$.  This reliance can cause slow mixing and ineffective exploration of the tails of the distribution.

An alternative, equivalent parameterization that addresses these problems is to marginalize over $z$.  The marginalized model is

$$
\begin{align*}
\mu_1, \ldots, \mu_K
    & \sim N(0, \sigma^2) \\
\tau_1, \ldots, \tau_K
    & \sim \textrm{Gamma}(a, b) \\
\boldsymbol{w}
    & \sim \textrm{Dir}(\boldsymbol{\alpha}) \\
f(x\ |\ \boldsymbol{w})
    & = \sum_{i = 1}^K w_i\ N(x\ |\ \mu_i, \tau^{-1}_i),
\end{align*}
$$

where

$$N(x\ |\ \mu, \sigma^2) = \frac{1}{\sqrt{2 \pi} \sigma} \exp\left(-\frac{1}{2 \sigma^2} (x - \mu)^2\right)$$

is the probability density function of the normal distribution.

Marginalizing $z$ out of the model generally leads to faster mixing and better exploration of the tails of the posterior distribution.  Marginalization over discrete parameters is a common trick in the [Stan](http://mc-stan.org/) community, since Stan does not support sampling from discrete distributions.  For further details on marginalization and several worked examples, see the [_Stan User's Guide and Reference Manual_](http://www.uvm.edu/~bbeckage/Teaching/DataAnalysis/Manuals/stan-reference-2.8.0.pdf).

PyMC3 supports marginalized Gaussian mixture models through its `NormalMixture` class.  (It also supports marginalized general mixture models through its `Mixture` class)  Below we specify and fit a marginalized Gaussian mixture model to this data in PyMC3.

```{code-cell} ipython3
---
papermill:
  duration: 71.268227
  end_time: '2020-12-20T20:46:50.162293'
  exception: false
  start_time: '2020-12-20T20:45:38.894066'
  status: completed
---
with pm.Model(coords={"cluster": np.arange(len(W)), "obs_id": np.arange(N)}) as model:
    w = pm.Dirichlet("w", np.ones_like(W))

    mu = pm.Normal(
        "mu",
        np.zeros_like(W),
        1.0,
        dims="cluster",
        transform=pm.transforms.ordered,
        testval=[1, 2, 3],
    )
    tau = pm.Gamma("tau", 1.0, 1.0, dims="cluster")

    x_obs = pm.NormalMixture("x_obs", w, mu, tau=tau, observed=x, dims="obs_id")
```

```{code-cell} ipython3
---
papermill:
  duration: 587.834129
  end_time: '2020-12-20T20:56:38.008902'
  exception: false
  start_time: '2020-12-20T20:46:50.174773'
  status: completed
---
with model:
    trace = pm.sample(5000, n_init=10000, tune=1000, return_inferencedata=True)

    # sample posterior predictive samples
    ppc_trace = pm.sample_posterior_predictive(trace, var_names=["x_obs"], keep_size=True)

trace.add_groups(posterior_predictive=ppc_trace)
```

+++ {"papermill": {"duration": 0.013524, "end_time": "2020-12-20T20:56:38.036405", "exception": false, "start_time": "2020-12-20T20:56:38.022881", "status": "completed"}}

We see in the following plot that the posterior distribution on the weights and the component means has captured the true value quite well.

```{code-cell} ipython3
az.plot_trace(trace, var_names=["w", "mu"], compact=False);
```

```{code-cell} ipython3
az.plot_posterior(trace, var_names=["w", "mu"]);
```

+++ {"papermill": {"duration": 0.035988, "end_time": "2020-12-20T20:56:44.871074", "exception": false, "start_time": "2020-12-20T20:56:44.835086", "status": "completed"}}

We see that the posterior predictive samples have a distribution quite close to that of the observed data.

```{code-cell} ipython3
az.plot_ppc(trace);
```

Author: [Austin Rochford](http://austinrochford.com)

+++

## Watermark

```{code-cell} ipython3
---
papermill:
  duration: 0.108022
  end_time: '2020-12-20T20:58:55.011403'
  exception: false
  start_time: '2020-12-20T20:58:54.903381'
  status: completed
---
%load_ext watermark
%watermark -n -u -v -iv -w -p theano,xarray
```
