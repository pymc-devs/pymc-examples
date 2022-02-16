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

(censored_data)=
# Censored Data Models

```{code-cell} ipython3
from copy import copy

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import seaborn as sns

from numpy.random import default_rng
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'
rng = default_rng(1234)
az.style.use("arviz-darkgrid")
```

[This example notebook on Bayesian survival
analysis](http://docs.pymc.io/notebooks/survival_analysis.html) touches on the
point of censored data. _Censoring_ is a form of missing-data problem, in which
observations greater than a certain threshold are clipped down to that
threshold, or observations less than a certain threshold are clipped up to that
threshold, or both. These are called right, left and interval censoring,
respectively. In this example notebook we consider interval censoring.

Censored data arises in many modelling problems. Two common examples are:

1. _Survival analysis:_ when studying the effect of a certain medical treatment
   on survival times, it is impossible to prolong the study until all subjects
   have died. At the end of the study, the only data collected for many patients
   is that they were still alive for a time period $T$ after the treatment was
   administered: in reality, their true survival times are greater than $T$.

2. _Sensor saturation:_ a sensor might have a limited range and the upper and
   lower limits would simply be the highest and lowest values a sensor can
   report. For instance, many mercury thermometers only report a very narrow
   range of temperatures.

This example notebook presents two different ways of dealing with censored data
in PyMC3:

1. An imputed censored model, which represents censored data as parameters and
   makes up plausible values for all censored values. As a result of this
   imputation, this model is capable of generating plausible sets of made-up
   values that would have been censored. Each censored element introduces a
   random variable.

2. An unimputed censored model, where the censored data are integrated out and
   accounted for only through the log-likelihood. This method deals more
   adequately with large amounts of censored data and converges more quickly.

To establish a baseline we compare to an uncensored model of the uncensored
data.

```{code-cell} ipython3
# Produce normally distributed samples
size = 500
true_mu = 13.0
true_sigma = 5.0
samples = rng.normal(true_mu, true_sigma, size)

# Set censoring limits
low = 3.0
high = 16.0


def censor(x, low, high):
    x = copy(x)
    x[x <= low] = low
    x[x >= high] = high
    return x


# Censor samples
censored = censor(samples, low, high)
```

```{code-cell} ipython3
# Visualize uncensored and censored data
_, ax = plt.subplots(figsize=(10, 3))
edges = np.linspace(-5, 35, 30)
ax.hist(samples, bins=edges, density=True, histtype="stepfilled", alpha=0.2, label="Uncensored")
ax.hist(censored, bins=edges, density=True, histtype="stepfilled", alpha=0.2, label="Censored")
[ax.axvline(x=x, c="k", ls="--") for x in [low, high]]
ax.legend();
```

## Uncensored Model

```{code-cell} ipython3
def uncensored_model(data):
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=((high - low) / 2) + low, sigma=(high - low))
        sigma = pm.HalfNormal("sigma", sigma=(high - low) / 2.0)
        observed = pm.Normal("observed", mu=mu, sigma=sigma, observed=data)
    return model
```

We should predict that running the uncensored model on uncensored data, we will get reasonable estimates of the mean and variance.

```{code-cell} ipython3
uncensored_model_1 = uncensored_model(samples)
with uncensored_model_1:
    trace = pm.sample(tune=1000, return_inferencedata=True)
    az.plot_posterior(trace, ref_val=[true_mu, true_sigma], round_to=3);
```

And that is exactly what we find. 

The problem however, is that in censored data contexts, we do not have access to the true values. If we were to use the same uncensored model on the censored data, we would anticipate that our parameter estimates will be biased. If we calculate point estimates for the mean and std, then we can see that we are likely to underestimate the mean and std for this particular dataset and censor bounds.

```{code-cell} ipython3
np.mean(censored), np.std(censored)
```

```{code-cell} ipython3
uncensored_model_2 = uncensored_model(censored)
with uncensored_model_2:
    trace = pm.sample(tune=1000, return_inferencedata=True)
    az.plot_posterior(trace, ref_val=[true_mu, true_sigma], round_to=3);
```

The figure above confirms this.

## Censored data models

The models below show 2 approaches to dealing with censored data. First, we need to do a bit of data pre-processing to count the number of observations that are left or right censored. We also also need to extract just the non-censored data that we observe.

```{code-cell} ipython3
n_right_censored = sum(censored >= high)
n_left_censored = sum(censored <= low)
n_observed = len(censored) - n_right_censored - n_left_censored
uncensored = censored[(censored > low) & (censored < high)]
assert len(uncensored) == n_observed
```

(censored_data/model1)=
### Model 1 - Imputed Censored Model of Censored Data

In this model, we impute the censored values from the same distribution as the uncensored data. Sampling from the posterior generates possible uncensored data sets.

This model makes use of [PyMC3's bounded variables](https://docs.pymc.io/api/bounds.html).

```{code-cell} ipython3
with pm.Model() as imputed_censored_model:
    mu = pm.Normal("mu", mu=((high - low) / 2) + low, sigma=(high - low))
    sigma = pm.HalfNormal("sigma", sigma=(high - low) / 2.0)
    right_censored = pm.Bound(pm.Normal, lower=high)(
        "right_censored", mu=mu, sigma=sigma, shape=n_right_censored
    )
    left_censored = pm.Bound(pm.Normal, upper=low)(
        "left_censored", mu=mu, sigma=sigma, shape=n_left_censored
    )
    observed = pm.Normal("observed", mu=mu, sigma=sigma, observed=uncensored, shape=n_observed)
```

```{code-cell} ipython3
with imputed_censored_model:
    trace = pm.sample(return_inferencedata=True)
    az.plot_posterior(trace, var_names=["mu", "sigma"], ref_val=[true_mu, true_sigma], round_to=3);
```

We can see that the bias in the estimates of the mean and variance (present in the uncensored model) have been largely removed.

+++

### Model 2 - Unimputed Censored Model of Censored Data

In this model, we do not impute censored data, but instead integrate them out through the likelihood.

The implementations of the likelihoods are non-trivial. See the [Stan manual](https://github.com/stan-dev/stan/releases/download/v2.14.0/stan-reference-2.14.0.pdf) (section 11.3 on censored data) and the [original PyMC3 issue on GitHub](https://github.com/pymc-devs/pymc3/issues/1833) for more information.

This model makes use of [PyMC3's `Potential`](https://docs.pymc.io/api/model.html#pymc3.model.Potential).

```{code-cell} ipython3
# Import the log cdf and log complementary cdf of the normal Distribution from PyMC3
from pymc3.distributions.dist_math import normal_lccdf, normal_lcdf


# Helper functions for unimputed censored model
def left_censored_likelihood(mu, sigma, n_left_censored, lower_bound):
    """Likelihood of left-censored data."""
    return n_left_censored * normal_lcdf(mu, sigma, lower_bound)


def right_censored_likelihood(mu, sigma, n_right_censored, upper_bound):
    """Likelihood of right-censored data."""
    return n_right_censored * normal_lccdf(mu, sigma, upper_bound)
```

```{code-cell} ipython3
with pm.Model() as unimputed_censored_model:
    mu = pm.Normal("mu", mu=0.0, sigma=(high - low) / 2.0)
    sigma = pm.HalfNormal("sigma", sigma=(high - low) / 2.0)
    observed = pm.Normal(
        "observed",
        mu=mu,
        sigma=sigma,
        observed=uncensored,
    )
    left_censored = pm.Potential(
        "left_censored", left_censored_likelihood(mu, sigma, n_left_censored, low)
    )
    right_censored = pm.Potential(
        "right_censored", right_censored_likelihood(mu, sigma, n_right_censored, high)
    )
```

Sampling

```{code-cell} ipython3
with unimputed_censored_model:
    trace = pm.sample(tune=1000, return_inferencedata=True)
    az.plot_posterior(trace, var_names=["mu", "sigma"], ref_val=[true_mu, true_sigma], round_to=3);
```

Again, the bias in the estimates of the mean and variance (present in the uncensored model) have been largely removed.

+++

## Discussion

As we can see, both censored models appear to capture the mean and variance of the underlying distribution as well as the uncensored model! In addition, the imputed censored model is capable of generating data sets of censored values (sample from the posteriors of `left_censored` and `right_censored` to generate them), while the unimputed censored model scales much better with more censored data, and converges faster.

+++

## Authors

- Originally authored by [Luis Mario Domenzain](https://github.com/domenzain) on Mar 7, 2017.
- Updated by [George Ho](https://github.com/eigenfoo) on Jul 14, 2018.
- Updated by [Benjamin Vincent](https://github.com/drbenvincent) in May 2021.

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p theano,xarray
```
