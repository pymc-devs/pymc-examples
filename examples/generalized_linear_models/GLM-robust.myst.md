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

(GLM-robust)=
# GLM: Robust Linear Regression

:::{post} January 10, 2023
:tags: regression, linear model, robust
:category: beginner
:author: Thomas Wiecki, Chris Fonnesbeck, Abhipsha Das, Conor Hassan, Igor Kuvychko, Reshama Shaikh, Oriol Abril Pla
:::

+++

# GLM: Robust Linear Regression

The tutorial is the second of a three-part series on Bayesian *generalized linear models (GLMs)*, that first appeared on [Thomas Wiecki's blog](https://twiecki.io/):

  1. {ref}`Linear Regression <pymc:GLM_linear>`
  2. {ref}`Robust Linear Regression <GLM-robust>`
  3. {ref}`Hierarchical Linear Regression <GLM-hierarchical>`
  
In this blog post I will write about:

 - How a few outliers can largely affect the fit of linear regression models.
 - How replacing the normal likelihood with Student T distribution produces robust regression.

In the {ref}`linear regression tutorial <pymc:GLM_linear>` I described how minimizing the squared distance of the regression line is the same as maximizing the likelihood of a Normal distribution with the mean coming from the regression line. This latter probabilistic expression allows us to easily formulate a Bayesian linear regression model.

This worked splendidly on simulated data. The problem with simulated data though is that it's, well, simulated. In the real world things tend to get more messy and assumptions like normality are easily violated by a few outliers. 

Lets see what happens if we add some outliers to our simulated data from the last post.

+++

First, let's import our modules.

```{code-cell} ipython3
%matplotlib inline

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr
```

```{code-cell} ipython3
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
%config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")
```

Create some toy data but also add some outliers.

```{code-cell} ipython3
size = 100
true_intercept = 1
true_slope = 2

x = np.linspace(0, 1, size)
# y = a + b*x
true_regression_line = true_intercept + true_slope * x
# add noise
y = true_regression_line + rng.normal(scale=0.5, size=size)

# Add outliers
x_out = np.append(x, [0.1, 0.15, 0.2])
y_out = np.append(y, [8, 6, 9])

data = pd.DataFrame(dict(x=x_out, y=y_out))
```

Plot the data together with the true regression line (the three points in the upper left corner are the outliers we added).

```{code-cell} ipython3
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, xlabel="x", ylabel="y", title="Generated data and underlying model")
ax.plot(x_out, y_out, "x", label="sampled data")
ax.plot(x, true_regression_line, label="true regression line", lw=2.0)
plt.legend(loc=0);
```

## Robust Regression

### Normal Likelihood

Lets see what happens if we estimate our Bayesian linear regression model by fitting a regression model with a normal likelihood.
Note that the bambi library provides an easy to use such that an equivalent model can be built using one line of code.
A version of this same notebook using Bambi is available at {doc}`bambi's docs <bambi:notebooks/t_regression>`

```{code-cell} ipython3
with pm.Model() as model:
    xdata = pm.ConstantData("x", x_out, dims="obs_id")

    # define priors
    intercept = pm.Normal("intercept", mu=0, sigma=1)
    slope = pm.Normal("slope", mu=0, sigma=1)
    sigma = pm.HalfCauchy("sigma", beta=10)

    mu = pm.Deterministic("mu", intercept + slope * xdata, dims="obs_id")

    # define likelihood
    likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=y_out, dims="obs_id")

    # inference
    trace = pm.sample(tune=2000)
```

To evaluate the fit, the code below calculates the posterior predictive regression lines by taking regression parameters from the posterior distribution and plots a regression line for 20 of them.

```{code-cell} ipython3
post = az.extract(trace, num_samples=20)
x_plot = xr.DataArray(np.linspace(x_out.min(), x_out.max(), 100), dims="plot_id")
lines = post["intercept"] + post["slope"] * x_plot

plt.scatter(x_out, y_out, label="data")
plt.plot(x_plot, lines.transpose(), alpha=0.4, color="C1")
plt.plot(x, true_regression_line, label="True regression line", lw=3.0, c="C2")
plt.legend(loc=0)
plt.title("Posterior predictive for normal likelihood");
```

As you can see, the fit is quite skewed and we have a fair amount of uncertainty in our estimate as indicated by the wide range of different posterior predictive regression lines. Why is this? The reason is that the normal distribution does not have a lot of mass in the tails and consequently, an outlier will affect the fit strongly.

A Frequentist would estimate a [Robust Regression](http://en.wikipedia.org/wiki/Robust_regression) and use a non-quadratic distance measure to evaluate the fit.

But what's a Bayesian to do? Since the problem is the light tails of the Normal distribution we can instead assume that our data is not normally distributed but instead distributed according to the [Student T distribution](http://en.wikipedia.org/wiki/Student%27s_t-distribution) which has heavier tails as shown next {cite:p}`gelman2013bayesian,kruschke2014doing`.

Lets look at those two distributions to get a feel for them.

```{code-cell} ipython3
normal_dist = pm.Normal.dist(mu=0, sigma=1)
t_dist = pm.StudentT.dist(mu=0, lam=1, nu=1)
x_eval = np.linspace(-8, 8, 300)
plt.plot(x_eval, pm.math.exp(pm.logp(normal_dist, x_eval)).eval(), label="Normal", lw=2.0)
plt.plot(x_eval, pm.math.exp(pm.logp(t_dist, x_eval)).eval(), label="Student T", lw=2.0)
plt.xlabel("x")
plt.ylabel("Probability density")
plt.legend();
```

As you can see, the probability of values far away from the mean (0 in this case) are much more likely under the `T` distribution than under the Normal distribution.

Below is a PyMC model, with the `likelihood` term following a `StudentT` distribution with $\nu=3$ degrees of freedom, opposed to the `Normal` distribution.

```{code-cell} ipython3
with pm.Model() as robust_model:
    xdata = pm.ConstantData("x", x_out, dims="obs_id")

    # define priors
    intercept = pm.Normal("intercept", mu=0, sigma=1)
    slope = pm.Normal("slope", mu=0, sigma=1)
    sigma = pm.HalfCauchy("sigma", beta=10)

    mu = pm.Deterministic("mu", intercept + slope * xdata, dims="obs_id")

    # define likelihood
    likelihood = pm.StudentT("y", mu=mu, sigma=sigma, nu=3, observed=y_out, dims="obs_id")

    # inference
    robust_trace = pm.sample(tune=4000)
```

```{code-cell} ipython3
robust_post = az.extract(robust_trace, num_samples=20)
x_plot = xr.DataArray(np.linspace(x_out.min(), x_out.max(), 100), dims="plot_id")
robust_lines = robust_post["intercept"] + robust_post["slope"] * x_plot

plt.scatter(x_out, y_out, label="data")
plt.plot(x_plot, robust_lines.transpose(), alpha=0.4, color="C1")
plt.plot(x, true_regression_line, label="True regression line", lw=3.0, c="C2")
plt.legend(loc=0)
plt.title("Posterior predictive for Student-T likelihood")
```

There, much better! The outliers are barely influencing our estimation at all because our likelihood function assumes that outliers are much more probable than under the Normal distribution.

+++

## Summary

 - By changing the likelihood from a Normal distribution to a Student T distribution -- which has more mass in the tails -- we can perform *Robust Regression*.

*Extensions*: 

 - The Student-T distribution has, besides the mean and variance, a third parameter called *degrees of freedom* that describes how much mass should be put into the tails. Here it is set to 1 which gives maximum mass to the tails (setting this to infinity results in a Normal distribution!). One could easily place a prior on this rather than fixing it which I leave as an exercise for the reader ;).
 - T distributions can be used as priors as well. See {ref}`GLM-hierarchical`
 - How do we test if our data is normal or violates that assumption in an important way? Check out this great blog post, [Probably Overthinking It](http://allendowney.blogspot.com/2013/08/are-my-data-normal.html), by Allen Downey.

+++

## Authors 

* Adapted from [Thomas Wiecki's](https://twitter.com/twiecki) blog
* Updated by @fonnesbeck in September 2016 (pymc#1378)
* Updated by @chiral-carbon in August 2021 (pymc-examples#205)
* Updated by Conor Hassan, Igor Kuvychko, Reshama Shaikh and [Oriol Abril Pla](https://oriolabrilpla.cat/en/) in 2022
* Rerun using PyMC v5, by Reshama Shaikh, January 2023

+++

## References

:::{bibliography}
:filter: docname in docnames
:::
            
## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p xarray
```

:::{include} ../page_footer.md
:::

```{code-cell} ipython3

```
