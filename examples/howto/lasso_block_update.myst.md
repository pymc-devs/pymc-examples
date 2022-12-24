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

(lasso_block_update)=
# Lasso regression with block updating

:::{post} Feb 10, 2022
:tags: regression 
:category: beginner
:author: Chris Fonnesbeck, Raul Maldonado, Michael Osthege, Thomas Wiecki, Lorenzo Toniazzi
:::

```{code-cell} ipython3
:tags: []

%matplotlib inline
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

print(f"Running on PyMC v{pm.__version__}")
```

```{code-cell} ipython3
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
```

Sometimes, it is very useful to update a set of parameters together. For example, variables that are highly correlated are often good to update together. In PyMC block updating is simple. This will be demonstrated using the parameter `step` of {class}`pymc.sample`.

Here we have a [LASSO regression model](https://en.wikipedia.org/wiki/Lasso_(statistics)#Bayesian_interpretation) where the two coefficients are strongly correlated. Normally, we would define the coefficient parameters as a single random variable, but here we define them separately to show how to do block updates.

First we generate some fake data.

```{code-cell} ipython3
x = rng.standard_normal(size=(3, 30))
x1 = x[0] + 4
x2 = x[1] + 4
noise = x[2]
y_obs = x1 * 0.2 + x2 * 0.3 + noise
```

Then define the random variables.

```{code-cell} ipython3
:tags: []

lam = 3000

with pm.Model() as model:
    sigma = pm.Exponential("sigma", 1)
    tau = pm.Uniform("tau", 0, 1)
    b = lam * tau
    beta1 = pm.Laplace("beta1", 0, b)
    beta2 = pm.Laplace("beta2", 0, b)

    mu = x1 * beta1 + x2 * beta2

    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)
```

For most samplers, including {class}`pymc.Metropolis` and {class}`pymc.HamiltonianMC`, simply pass a list of variables to sample as a block. This works with both scalar and array parameters.

```{code-cell} ipython3
with model:
    step1 = pm.Metropolis([beta1, beta2])

    step2 = pm.Slice([sigma, tau])

    idata = pm.sample(draws=10000, step=[step1, step2])
```

We conclude by plotting the sampled marginals and the joint distribution of `beta1` and `beta2`.

```{code-cell} ipython3
:tags: []

az.plot_trace(idata);
```

```{code-cell} ipython3
az.plot_pair(
    idata,
    var_names=["beta1", "beta2"],
    kind="hexbin",
    marginals=True,
    figsize=(10, 10),
    gridsize=50,
)
```

## Authors

* Authored by [Chris Fonnesbeck](https://github.com/fonnesbeck) in Dec, 2020
* Updated by [Raul Maldonado](https://github.com/CloudChaoszero) in Jan, 2021
* Updated by Raul Maldonado in Mar, 2021
* Reexecuted by [Thomas Wiecki](https://github.com/twiecki) and [Michael Osthege](https://github.com/michaelosthege) with PyMC v4 in Jan, 2022 ([pymc-examples#264](https://github.com/pymc-devs/pymc-examples/pull/264))
* Updated by [Lorenzo Toniazzi](https://github.com/ltoniazzi) in Feb, 2022 ([pymc-examples#279](https://github.com/pymc-devs/pymc-examples/pull/279))

+++

## Watermark

```{code-cell} ipython3
:tags: []

%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,aeppl,xarray
```

:::{include} ../page_footer.md
:::
