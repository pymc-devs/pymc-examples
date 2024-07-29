---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: pymc
  language: python
  name: python3
---

(weibull_aft)=

# Reparameterizing the Weibull Accelerated Failure Time Model

:::{post} January 17, 2023
:tags: censored, survival analysis, weibull
:category: intermediate, how-to
:author: Junpeng Lao, George Ho, Chris Fonnesbeck
:::

```{code-cell} ipython3
import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import statsmodels.api as sm

print(f"Running on PyMC v{pm.__version__}")
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")
```

## Dataset

The {ref}`previous example notebook on Bayesian parametric survival analysis <bayes_param_survival_pymc3>` introduced two different accelerated failure time (AFT) models: Weibull and log-linear. In this notebook, we present three different parameterizations of the Weibull AFT model.

The data set we'll use is the `flchain` R data set, which comes from a medical study investigating the effect of serum free light chain (FLC) on lifespan. Read the full documentation of the data by running:

`print(sm.datasets.get_rdataset(package='survival', dataname='flchain').__doc__)`.

```{code-cell} ipython3
# Fetch and clean data
data = (
    sm.datasets.get_rdataset(package="survival", dataname="flchain")
    .data.sample(500)  # Limit ourselves to 500 observations
    .reset_index(drop=True)
)
```

```{code-cell} ipython3
y = data.futime.values
censored = ~data["death"].values.astype(bool)
```

```{code-cell} ipython3
y[:5]
```

```{code-cell} ipython3
censored[:5]
```

## Using `pm.Potential`

We have an unique problem when modelling censored data. Strictly speaking, we don't have any _data_ for censored values: we only know the _number_ of values that were censored. How can we include this information in our model?

One way do this is by making use of `pm.Potential`. The [PyMC2 docs](https://pymc-devs.github.io/pymc/modelbuilding.html#the-potential-class) explain its usage very well. Essentially, declaring `pm.Potential('x', logp)` will add `logp` to the log-likelihood of the model.

+++

## Parameterization 1

This parameterization is an intuitive, straightforward parameterization of the Weibull survival function. This is probably the first parameterization to come to one's mind.

```{code-cell} ipython3
def weibull_lccdf(x, alpha, beta):
    """Log complementary cdf of Weibull distribution."""
    return -((x / beta) ** alpha)
```

```{code-cell} ipython3
with pm.Model() as model_1:
    alpha_sd = 10.0

    mu = pm.Normal("mu", mu=0, sigma=100)
    alpha_raw = pm.Normal("a0", mu=0, sigma=0.1)
    alpha = pm.Deterministic("alpha", pt.exp(alpha_sd * alpha_raw))
    beta = pm.Deterministic("beta", pt.exp(mu / alpha))

    y_obs = pm.Weibull("y_obs", alpha=alpha, beta=beta, observed=y[~censored])
    y_cens = pm.Potential("y_cens", weibull_lccdf(y[censored], alpha, beta))
```

```{code-cell} ipython3
with model_1:
    # Change init to avoid divergences
    data_1 = pm.sample(target_accept=0.9, init="adapt_diag")
```

```{code-cell} ipython3
az.plot_trace(data_1, var_names=["alpha", "beta"])
```

```{code-cell} ipython3
az.summary(data_1, var_names=["alpha", "beta"], round_to=2)
```

## Parameterization 2

Note that, confusingly, `alpha` is now called `r`, and `alpha` denotes a prior; we maintain this notation to stay faithful to the original implementation in Stan. In this parameterization, we still model the same parameters `alpha` (now `r`) and `beta`.

For more information, see [this Stan example model](https://github.com/stan-dev/example-models/blob/5e9c5055dcea78ad756a6fb9b3ff9a77a0a4c22b/bugs_examples/vol1/kidney/kidney.stan) and [the corresponding documentation](https://www.mrc-bsu.cam.ac.uk/wp-content/uploads/WinBUGS_Vol1.pdf).

```{code-cell} ipython3
with pm.Model() as model_2:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    r = pm.Gamma("r", alpha=1, beta=0.001, testval=0.25)
    beta = pm.Deterministic("beta", pt.exp(-alpha / r))

    y_obs = pm.Weibull("y_obs", alpha=r, beta=beta, observed=y[~censored])
    y_cens = pm.Potential("y_cens", weibull_lccdf(y[censored], r, beta))
```

```{code-cell} ipython3
with model_2:
    # Increase target_accept to avoid divergences
    data_2 = pm.sample(target_accept=0.9)
```

```{code-cell} ipython3
az.plot_trace(data_2, var_names=["r", "beta"])
```

```{code-cell} ipython3
az.summary(data_2, var_names=["r", "beta"], round_to=2)
```

## Parameterization 3

In this parameterization, we model the log-linear error distribution with a Gumbel distribution instead of modelling the survival function directly. For more information, see [this blog post](http://austinrochford.com/posts/2017-10-02-bayes-param-survival.html).

```{code-cell} ipython3
logtime = np.log(y)


def gumbel_sf(y, mu, sigma):
    """Gumbel survival function."""
    return 1.0 - pt.exp(-pt.exp(-(y - mu) / sigma))
```

```{code-cell} ipython3
with pm.Model() as model_3:
    s = pm.HalfNormal("s", tau=5.0)
    gamma = pm.Normal("gamma", mu=0, sigma=5)

    y_obs = pm.Gumbel("y_obs", mu=gamma, beta=s, observed=logtime[~censored])
    y_cens = pm.Potential("y_cens", gumbel_sf(y=logtime[censored], mu=gamma, sigma=s))
```

```{code-cell} ipython3
with model_3:
    # Change init to avoid divergences
    data_3 = pm.sample(init="adapt_diag")
```

```{code-cell} ipython3
az.plot_trace(data_3)
```

```{code-cell} ipython3
az.summary(data_3, round_to=2)
```

## Authors

- Originally collated by [Junpeng Lao](https://junpenglao.xyz/) on Apr 21, 2018. See original code [here](https://github.com/junpenglao/Planet_Sakaar_Data_Science/blob/65447fdb431c78b15fbeaef51b8c059f46c9e8d6/PyMC3QnA/discourse_1107.ipynb).
- Authored and ported to Jupyter notebook by [George Ho](https://eigenfoo.xyz/) on Jul 15, 2018.
- Updated for compatibility with PyMC v5 by Chris Fonnesbeck on Jan 16, 2023.

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w
```

:::{include} ../page_footer.md
:::
