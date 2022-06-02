---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: pymc
  language: python
  name: pymc
---

# GLM: Mini-batch ADVI on hierarchical regression model

:::{post} Sept 23, 2021
:tags: generalized linear model, hierarchical model, variational inference
:category: intermediate
:::

+++

Unlike Gaussian mixture models, (hierarchical) regression models have independent variables. These variables affect the likelihood function, but are not random variables. When using mini-batch, we should take care of that.

```{code-cell} ipython3
%env AESARA_FLAGS=device=cpu, floatX=float32, warn_float64=ignore

import os

import aesara
import aesara.tensor as at
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

from scipy import stats

print(f"Running on PyMC v{pm.__version__}")
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
pm.set_at_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
```

```{code-cell} ipython3
try:
    data = pd.read_csv(os.path.join("..", "data", "radon.csv"))
except FileNotFoundError:
    data = pd.read_csv(pm.get_data("radon.csv"))

data
```

```{code-cell} ipython3
county_idx = data["county_code"].values
floor_idx = data["floor"].values
log_radon_idx = data["log_radon"].values

coords = {"counties": data.county.unique()}
```

Here, `log_radon_idx_t` is a dependent variable, while `floor_idx_t` and `county_idx_t` determine the independent variable.

```{code-cell} ipython3
log_radon_idx_t = pm.Minibatch(log_radon_idx, 100)
floor_idx_t = pm.Minibatch(floor_idx, 100)
county_idx_t = pm.Minibatch(county_idx, 100)
```

```{code-cell} ipython3
with pm.Model(coords=coords) as hierarchical_model:
    # Hyperpriors for group nodes
    mu_a = pm.Normal("mu_alpha", mu=0.0, sigma=100**2)
    sigma_a = pm.Uniform("sigma_alpha", lower=0, upper=100)
    mu_b = pm.Normal("mu_beta", mu=0.0, sigma=100**2)
    sigma_b = pm.Uniform("sigma_beta", lower=0, upper=100)
```

Intercept for each county, distributed around group mean `mu_a`. Above we just set `mu` and `sd` to a fixed value while here we plug in a common group distribution for all `a` and `b` (which are vectors with the same length as the number of unique counties in our example).

```{code-cell} ipython3
with hierarchical_model:

    a = pm.Normal("alpha", mu=mu_a, sigma=sigma_a, dims="counties")
    # Intercept for each county, distributed around group mean mu_a
    b = pm.Normal("beta", mu=mu_b, sigma=sigma_b, dims="counties")
```

Model prediction of radon level `a[county_idx]` translates to `a[0, 0, 0, 1, 1, ...]`, we thus link multiple household measures of a county to its coefficients.

```{code-cell} ipython3
with hierarchical_model:

    radon_est = a[county_idx_t] + b[county_idx_t] * floor_idx_t
```

Finally, we specify the likelihood:

```{code-cell} ipython3
with hierarchical_model:

    # Model error
    eps = pm.Uniform("eps", lower=0, upper=100)

    # Data likelihood
    radon_like = pm.Normal(
        "radon_like", mu=radon_est, sigma=eps, observed=log_radon_idx_t, total_size=len(data)
    )
```

Random variables `radon_like`, associated with `log_radon_idx_t`, should be given to the function for ADVI to denote that as observations in the likelihood term.

+++

On the other hand, `minibatches` should include the three variables above.

+++

Then, run ADVI with mini-batch.

```{code-cell} ipython3
with hierarchical_model:
    approx = pm.fit(100000, callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)])
    idata_advi = approx.sample(500)
```

Check the trace of ELBO and compare the result with MCMC.

```{code-cell} ipython3
plt.plot(approx.hist);
```

```{code-cell} ipython3
# Inference button (TM)!
with pm.Model(coords=coords):

    mu_a = pm.Normal("mu_alpha", mu=0.0, sigma=100**2)
    sigma_a = pm.Uniform("sigma_alpha", lower=0, upper=100)
    mu_b = pm.Normal("mu_beta", mu=0.0, sigma=100**2)
    sigma_b = pm.Uniform("sigma_beta", lower=0, upper=100)

    a = pm.Normal("alpha", mu=mu_a, sigma=sigma_a, dims="counties")
    b = pm.Normal("beta", mu=mu_b, sigma=sigma_b, dims="counties")

    # Model error
    eps = pm.Uniform("eps", lower=0, upper=100)

    radon_est = a[county_idx] + b[county_idx] * floor_idx

    radon_like = pm.Normal("radon_like", mu=radon_est, sigma=eps, observed=log_radon_idx)

    # essentially, this is what init='advi' does
    step = pm.NUTS(scaling=approx.cov.eval(), is_cov=True)
    hierarchical_trace = pm.sample(
        2000,
        step,
        # sampling different initial values from the trace
        initvals=list(approx.sample(return_inferencedata=False, size=4)[i] for i in range(4)),
        progressbar=True,
        return_inferencedata=True,
    )
```

```{code-cell} ipython3
az.plot_density(
    [idata_advi, hierarchical_trace], var_names=["~alpha", "~beta"], data_labels=["ADVI", "NUTS"]
);
```

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p xarray
```
