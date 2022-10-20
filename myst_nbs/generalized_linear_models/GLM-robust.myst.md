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

(GLM-robust)=
# GLM: Robust Linear Regression

:::{post} August, 2013
:tags: regression, linear model, robust
:category: beginner
:author: Thomas Wiecki
:::

+++

# GLM: Robust Linear Regression

This tutorial first appeard as a post in small series on Bayesian GLMs on:

  1. [The Inference Button: Bayesian GLMs made easy with PyMC3](https://twiecki.io/blog/2013/08/12/bayesian-glms-1/)
  2. [This world is far from Normal(ly distributed): Robust Regression in PyMC3](https://twiecki.io/blog/2013/08/27/bayesian-glms-2/)
  3. [The Best Of Both Worlds: Hierarchical Linear Regression in PyMC3](https://twiecki.io/blog/2014/03/17/bayesian-glms-3/)
  
In this blog post I will write about:

 - How a few outliers can largely affect the fit of linear regression models.
 - How replacing the normal likelihood with Student T distribution produces robust regression.
 - How this can easily be done with the `Bambi` library by passing a `family` object or passing the family name as an argument.
 
This is the second part of a series on Bayesian GLMs (click [here for part I about linear regression](http://twiecki.github.io/blog/2013/08/12/bayesian-glms-1/)). In this prior post I described how minimizing the squared distance of the regression line is the same as maximizing the likelihood of a Normal distribution with the mean coming from the regression line. This latter probabilistic expression allows us to easily formulate a Bayesian linear regression model.

This worked splendidly on simulated data. The problem with simulated data though is that it's, well, simulated. In the real world things tend to get more messy and assumptions like normality are easily violated by a few outliers. 

Lets see what happens if we add some outliers to our simulated data from the last post.

+++

First, let's import our modules.

```{code-cell} ipython3
%matplotlib inline

import aesara
import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

print(f"Running on pymc3 v{pm.__version__}")
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

```{code-cell} ipython3
data.head()
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


Lets see what happens if we estimate our Bayesian linear regression model using the `bambi`. This function takes a [`formulae`](https://bambinos.github.io/formulae/api_reference.html) string to describe the linear model and adds a Normal likelihood for Intercept and Slope by default.

```{code-cell} ipython3
model = bmb.Model("y ~ x", data)
trace = model.fit(draws=2000, cores=2)
```

```{code-cell} ipython3
model.graph()
```

```{code-cell} ipython3
az.summary(trace)
```

To evaluate the fit, the code below calculates the posterior predictive regression lines by taking regression parameters from the posterior distribution and plots a regression line for every 10th of them.

```{code-cell} ipython3
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, xlabel="x", ylabel="y", title="Generated data and underlying model")
ax.plot(x_out, y_out, "x", label="sampled data")

# calculate posterior regression lines (for every 10th point from posterior)
for chain in range(2):
    for i in range(0, 2000, 10):
        regression_line = (
            trace.posterior.Intercept[chain, i].values + trace.posterior.x[chain, i].values * x
        )
        ax.plot(x, regression_line, lw=0.5, alpha=0.25, color="black", label="posterior regression")

ax.plot(x, true_regression_line, label="true regression line", lw=2.0)

# remove duplicate legend labels for posterior regression lines
handles, labels = plt.gca().get_legend_handles_labels()
newLabels, newHandles = [], []
for handle, label in zip(handles, labels):
    if label not in newLabels:
        newLabels.append(label)
        newHandles.append(handle)
_ = ax.legend(newHandles, newLabels, loc=0)
```

As you can see, the fit is quite skewed and we have a fair amount of uncertainty in our estimate as indicated by the wide range of different posterior predictive regression lines. Why is this? The reason is that the normal distribution does not have a lot of mass in the tails and consequently, an outlier will affect the fit strongly.

A Frequentist would estimate a [Robust Regression](http://en.wikipedia.org/wiki/Robust_regression) and use a non-quadratic distance measure to evaluate the fit.

But what's a Bayesian to do? Since the problem is the light tails of the Normal distribution we can instead assume that our data is not normally distributed but instead distributed according to the [Student T distribution](http://en.wikipedia.org/wiki/Student%27s_t-distribution) which has heavier tails as shown next (I read about this trick in ["The Kruschke"](https://www.elsevier.com/books/doing-bayesian-data-analysis/kruschke/978-0-12-405888-0), aka the puppy-book; but I think [Gelman](http://www.stat.columbia.edu/~gelman/book/) was the first to formulate this).

Lets look at those two distributions to get a feel for them.

```{code-cell} ipython3
normal_dist = pm.Normal.dist(mu=0, sigma=1)
t_dist = pm.StudentT.dist(mu=0, lam=1, nu=1)
x_eval = np.linspace(-8, 8, 300)
plt.plot(x_eval, aesara.tensor.exp(pm.logp(normal_dist, x_eval)).eval(), label="Normal", lw=2.0)
plt.plot(x_eval, aesara.tensor.exp(pm.logp(t_dist, x_eval)).eval(), label="Student T", lw=2.0)
plt.xlabel("x")
plt.ylabel("Probability density")
plt.legend();
```

As you can see, the probability of values far away from the mean (0 in this case) are much more likely under the `T` distribution than under the Normal distribution.

To define the usage of a T distribution in `Bambi` we can pass the distribution name to the `family` argument  -- `t` -- that specifies that our data is Student T-distributed. Note that this is the same syntax as `R` and `statsmodels` use.

```{code-cell} ipython3
model_robust = bmb.Model("y ~ x", data, family="t")
model_robust.set_priors({"nu": bmb.Prior("Gamma", alpha=3, beta=1)})
trace_robust = model_robust.fit(draws=2000, cores=2)
```

```{code-cell} ipython3
model_robust.graph()
```

```{code-cell} ipython3
az.summary(trace_robust)
```

```{code-cell} ipython3
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, xlabel="x", ylabel="y", title="Generated data and underlying model")
ax.plot(x_out, y_out, "x", label="sampled data")

# calculate posterior regression lines (for every 10th point from posterior)
for chain in range(2):
    for i in range(0, 2000, 10):
        regression_line = (
            trace_robust.posterior.Intercept[chain, i].values
            + trace_robust.posterior.x[chain, i].values * x
        )
        ax.plot(x, regression_line, lw=0.5, alpha=0.25, color="black", label="posterior regression")

ax.plot(x, true_regression_line, label="true regression line", lw=2.0)

# remove duplicate legend labels for posterior regression lines
handles, labels = plt.gca().get_legend_handles_labels()
newLabels, newHandles = [], []
for handle, label in zip(handles, labels):
    if label not in newLabels:
        newLabels.append(label)
        newHandles.append(handle)
_ = ax.legend(newHandles, newLabels, loc=0)
```

There, much better! The outliers are barely influencing our estimation at all because our likelihood function assumes that outliers are much more probable than under the Normal distribution.

+++

## Summary

- `Bambi` allows you to pass in a `family` argument that contains information about the likelihood.
 - By changing the likelihood from a Normal distribution to a Student-T distribution -- which has more mass in the tails -- we can perform *Robust Regression*.

The next post will be about logistic regression in PyMC3 and what the posterior and oatmeal have in common.

*Extensions*: 

 - The Student-T distribution has, besides the mean and variance, a third parameter called *degrees of freedom* that describes how much mass should be put into the tails. Here it is set to 1 which gives maximum mass to the tails (setting this to infinity results in a Normal distribution!). One could easily place a prior on this rather than fixing it which I leave as an exercise for the reader ;).
 - T distributions can be used as priors as well. I will show this in a future post on hierarchical GLMs.
 - How do we test if our data is normal or violates that assumption in an important way? Check out this [great blog post](http://allendowney.blogspot.com/2013/08/are-my-data-normal.html) by Allen Downey.

+++

## Authors

* Authored by Thomas Wiecki in August, 2013
* Updated by Igor Kuvychko in October, 2022

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p xarray
```
