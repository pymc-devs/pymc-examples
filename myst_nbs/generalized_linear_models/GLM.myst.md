---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# (Generalized) Linear and Hierarchical Linear Models in PyMC3

```{code-cell} ipython3
import os

import arviz as az
import bambi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import theano
import xarray as xr

from numpy.random import default_rng

print(f"Running on PyMC3 v{pm.__version__}")
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'
# Initialize random number generator
RANDOM_SEED = 8927
rng = default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
```

## Linear Regression

Lets generate some data with known slope and intercept and fit a simple linear GLM.

```{code-cell} ipython3
size = 50
true_intercept = 1
true_slope = 2
x = np.linspace(0, 1, size)
y = true_intercept + x * true_slope + rng.normal(scale=0.5, size=size)
data = pd.DataFrame({"x": x, "y": y})
```

> Bambi is a high-level Bayesian model-building interface written in Python. It's built on top of the PyMC3  probabilistic programming framework, and is designed to make it extremely easy to fit mixed-effects models common 
in social sciences settings using a Bayesian approach.

We construct a model using Bambi(it uses formula based input), If no priors are given explicitly by the user, then Bambi chooses smart default priors for all parameters of the model based on plausible implied partial correlations between the outcome and the predictors ( [Documentation](https://bambinos.github.io/bambi/master), [Reference](https://arxiv.org/abs/2012.10754) )

```{code-cell} ipython3
model = bambi.Model("y ~ x", data)
fitted = model.fit(draws=1000)
```

```{code-cell} ipython3
x_axis = xr.DataArray(np.linspace(0, 1, num=100), dims=["x_plot"])
mu_pred = fitted.posterior["Intercept"] + fitted.posterior["x"] * x_axis
mu_mean = mu_pred.mean(dim=("chain", "draw"))
mu_plot = mu_pred.stack(sample=("chain", "draw"))
random_subset = rng.permutation(np.arange(len(mu_plot.sample)))[:200]
plt.scatter(x, y)
plt.plot(x_axis, mu_plot.isel(sample=random_subset), color="black", alpha=0.025)
plt.plot(x_axis, mu_mean, color="C1");
```

## Robust GLM

Lets try the same model but with a few outliers in the data.

```{code-cell} ipython3
x_out = np.append(x, [0.1, 0.15, 0.2])
y_out = np.append(y, [8, 6, 9])
data_outlier = pd.DataFrame({"x": x_out, "y": y_out})
```

```{code-cell} ipython3
model = bambi.Model("y ~ x", data_outlier)
fitted = model.fit(draws=1000)
fitted
```

Here we generate a graphviz model of the Bambi/PyMC3 model

```{code-cell} ipython3
model.graph()
```

```{code-cell} ipython3
x_axis = xr.DataArray(np.linspace(0, 1, num=100), dims=["x_plot"])
mu_pred = fitted.posterior["Intercept"] + fitted.posterior["x"] * x_axis
mu_mean = mu_pred.mean(dim=("chain", "draw"))
mu_plot = mu_pred.stack(sample=("chain", "draw"))
random_subset = rng.permutation(np.arange(len(mu_plot.sample)))[:200]
plt.scatter(x_out, y_out)
plt.plot(x_axis, mu_plot.isel(sample=random_subset), color="black", alpha=0.025)
plt.plot(x_axis, mu_mean, color="C1");
```

Because the Normal distribution does not have a lot of mass in the tails, an outlier will affect the fit strongly. Instead, we can replace the Normal likelihood with a Student T distribution which has heavier tails and is more robust towards outliers. Bambi does not support passing family="StudentT" yet to indicate a StudentT likelihood, but we can make our own custom [Family object](https://bambinos.github.io/bambi/master/notebooks/getting_started.html#Families)

```{code-cell} ipython3
family = bambi.Family(
    "t", prior=bambi.Prior("StudentT", lam=1, nu=1.5), link="identity", parent="mu"
)
```

`link = "identity"` implies that no transformation is applied to the linear predictor. </br>

`parent = "mu"` means the linear predictor is modeling the `mu` parameter of the Student T distribution.

Here `common` key in priors dict defines the prior for all common effects at the same time. As explained in a section below, Bambi supports two types of effects, common and group effects.

```{code-cell} ipython3
model = bambi.Model(
    "y ~ x",
    data_outlier,
    priors={"common": bambi.Prior("HalfNormal", sigma=10)},
    family=family,
)
```

```{code-cell} ipython3
fitted = model.fit(target_accept=0.9)
x_axis = xr.DataArray(np.linspace(0, 1, num=100), dims=["x_plot"])
mu_pred = fitted.posterior["Intercept"] + fitted.posterior["x"] * x_axis
mu_mean = mu_pred.mean(dim=("chain", "draw"))
mu_plot = mu_pred.stack(sample=("chain", "draw"))
random_subset = rng.permutation(np.arange(len(mu_plot.sample)))[:200]
plt.scatter(x_out, y_out)
plt.plot(x_axis, mu_plot.isel(sample=random_subset), color="black", alpha=0.025)
plt.plot(x_axis, mu_mean, color="C1");
```

## Hierarchical GLM

+++

Learn more about uses and caveats of Hierarchical models from Thomas Wiecki [here](https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/)

```{code-cell} ipython3
try:
    sat_data = pd.read_csv(os.path.join("..", "data", "Guber1999data.txt"))
except:
    sat_data = pd.read_csv(pm.get_data("Guber1999data.txt"))
```

Bambi uses [formulae based syntax](https://bambinos.github.io/bambi/master/notebooks/getting_started.html#Formula-based-specification), so fixed and random effects, or common and group specific effects are specified using different syntax in Bambi, just as you would do with [lme4](https://cran.r-project.org/web/packages/lme4/vignettes/lmer.pdf). Group-specific effects are specified with the (variable|group) syntax ,and (1|group) if you want group-specific intercepts, which is what we did for spend, stu_tea_rat, salary and prcnt_take.

Group-specific effects are specified with the `(variable|group)` syntax ,and `(1|group)` if you want group-specific intercepts, which is what we did for `spend`, `stu_tea_rat`, `salary` and `prcnt_take`.

```{code-cell} ipython3
grp_mean = bambi.Prior("Normal", mu=0, sigma=10)
grp_sd = bambi.Prior("HalfCauchy", beta=5)

priors = {
    "Intercept": bambi.Prior("Normal", mu=sat_data.sat_t.mean(), sigma=sat_data.sat_t.std()),
    "1|spend": bambi.Prior("Normal", mu=grp_mean, sigma=grp_sd),
    "1|stu_tea_rat": bambi.Prior("Normal", mu=grp_mean, sigma=grp_sd),
    "1|salary": bambi.Prior("Normal", mu=grp_mean, sigma=grp_sd),
    "1|prcnt_take": bambi.Prior("Normal", mu=grp_mean, sigma=grp_sd),
}
```

```{code-cell} ipython3
model = bambi.Model(
    "sat_t ~ (1|spend) + (1|stu_tea_rat) + (1|salary) + (1|prcnt_take)", sat_data, priors=priors
)
results = model.fit()
```

```{code-cell} ipython3
az.plot_pair(results, marginals=True);
```

```{code-cell} ipython3
grp_mean = bambi.Prior("Normal", mu=0, sigma=10)
grp_prec = bambi.Prior("Gamma", alpha=1, beta=0.1, testval=1.0)

priors = {
    "Intercept": bambi.Prior("Normal", mu=sat_data.sat_t.mean(), sigma=sat_data.sat_t.std()),
    "slope": bambi.Prior("StudentT", mu=grp_mean, lam=grp_prec, nu=1),
}
model = bambi.Model("sat_t ~ spend + stu_tea_rat + salary + prcnt_take", sat_data, priors=priors)
results = model.fit()
```

```{code-cell} ipython3
az.plot_pair(results);
```

## Logistic Regression

```{code-cell} ipython3
try:
    htwt_data = pd.read_csv(os.path.join("..", "data", "HtWt.csv"))
except FileNotFoundError:
    htwt_data = pd.read_csv(pm.get_data("HtWt.csv"))

htwt_data.head()
```

```{code-cell} ipython3
model = bambi.Model("male ~ height + weight", htwt_data, family="bernoulli")
results = model.fit()

az.summary(results)
```

```{code-cell} ipython3
az.plot_posterior(results, ref_val=0);
```

```{code-cell} ipython3
az.plot_pair(results);
```

## Bayesian Logistic Lasso

```{code-cell} ipython3
lp = pm.Laplace.dist(mu=0, b=0.05)
x_eval = np.linspace(-0.5, 0.5, 300)
plt.plot(x_eval, theano.tensor.exp(lp.logp(x_eval)).eval())
plt.xlabel("x")
plt.ylabel("Probability")
plt.title("Laplace distribution");
```

```{code-cell} ipython3
priors = {
    "Intercept": bambi.Prior("Normal", mu=0, sigma=50),
    "Regressor": bambi.Prior("Laplace", mu=0, b=0.05),
}

model = bambi.Model("male ~ height + weight", htwt_data, priors=priors, family="bernoulli")
results = model.fit()

az.summary(results)
```

```{code-cell} ipython3
az.plot_pair(results);
```

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p graphviz
```
