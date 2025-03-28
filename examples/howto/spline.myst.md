---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: pymc-examples
  language: python
  name: pymc-examples
---

(spline)=
# Splines

:::{post} June 4, 2022 
:tags: patsy, regression, spline 
:category: beginner
:author: Joshua Cook
:::

+++

## Introduction

Often, the model we want to fit is not a perfect line between some $x$ and $y$.
Instead, the parameters of the model are expected to vary over $x$.
There are multiple ways to handle this situation, one of which is to fit a *spline*.
Spline fit is effectively a sum of multiple individual curves (piecewise polynomials), each fit to a different section of $x$, that are tied together at their boundaries, often called *knots*.

The spline is effectively multiple individual lines, each fit to a different section of $x$, that are tied together at their boundaries, often called *knots*.

Below is a full working example of how to fit a spline using PyMC. The data and model are taken from [*Statistical Rethinking* 2e](https://xcelab.net/rm/statistical-rethinking/) by [Richard McElreath's](https://xcelab.net/rm/) {cite:p}`mcelreath2018statistical`.

For more information on this method of non-linear modeling, I suggesting beginning with [chapter 5 of Bayesian Modeling and Computation in Python](https://bayesiancomputationbook.com/markdown/chp_05.html) {cite:p}`martin2021bayesian`.

```{code-cell} ipython3
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from patsy import build_design_matrices, dmatrix
```

```{code-cell} ipython3
%matplotlib inline
%config InlineBackend.figure_format = "retina"

seed = sum(map(ord, "splines"))
rng = np.random.default_rng(seed)
az.style.use("arviz-darkgrid")
```

## Cherry blossom data

The data for this example is the number of days (`doy` for "days of year") that the cherry trees were in bloom in each year (`year`). 
For convenience, years missing a `doy` were dropped (which is a bad idea to deal with missing data in general!).

```{code-cell} ipython3
try:
    blossom_data = pd.read_csv(Path("..", "data", "cherry_blossoms.csv"), sep=";")
except FileNotFoundError:
    blossom_data = pd.read_csv(pm.get_data("cherry_blossoms.csv"), sep=";")


blossom_data.dropna().describe()
```

```{code-cell} ipython3
blossom_data = blossom_data.dropna(subset=["doy"]).reset_index(drop=True)
blossom_data.head(n=10)
```

After dropping rows with missing data, there are 827 years with the numbers of days in which the trees were in bloom.

```{code-cell} ipython3
blossom_data.shape
```

If we visualize the data, it is clear that there a lot of annual variation, but some evidence for a non-linear trend in bloom days over time.

```{code-cell} ipython3
blossom_data.plot.scatter(
    "year",
    "doy",
    color="cornflowerblue",
    s=10,
    title="Cherry Blossom Data",
    ylabel="Days in bloom",
);
```

## The model

We will fit the following model.

$D \sim \mathcal{N}(\mu, \sigma)$  
$\quad \mu = a + Bw$  
$\qquad a \sim \mathcal{N}(100, 10)$  
$\qquad w \sim \mathcal{N}(0, 10)$  
$\quad \sigma \sim \text{Exp}(1)$

The number of days of bloom $D$ will be modeled as a normal distribution with mean $\mu$ and standard deviation $\sigma$. In turn, the mean will be a linear model composed of a y-intercept $a$ and spline defined by the basis $B$ multiplied by the model parameter $w$ with a variable for each region of the basis. Both have relatively weak normal priors.

### Prepare the spline

The spline will have 15 *knots*, splitting the year into 16 sections (including the regions covering the years before and after those in which we have data). The knots are the boundaries of the spline, the name owing to how the individual lines will be tied together at these boundaries to make a continuous and smooth curve.  The knots will be unevenly spaced over the years such that each region will have the same proportion of data.

```{code-cell} ipython3
num_knots = 15
knot_list = np.percentile(blossom_data.year, np.linspace(0, 100, num_knots + 2))[1:-1]
knot_list
```

Below is a plot of the locations of the knots over the data.

```{code-cell} ipython3
blossom_data.plot.scatter(
    "year",
    "doy",
    color="cornflowerblue",
    s=10,
    title="Cherry Blossom Data",
    ylabel="Day of Year",
)
for knot in knot_list:
    plt.gca().axvline(knot, color="grey", alpha=0.4)
```

We can use `patsy` to create the matrix $B$ that will be the b-spline basis for the regression.
The degree is set to 3 to create a cubic b-spline.

```{code-cell} ipython3
:tags: [hide-output]

B = dmatrix(
    "bs(year, knots=knots, degree=3, include_intercept=True) - 1",
    {"year": blossom_data.year.values, "knots": knot_list},
)
B
```

The b-spline basis is plotted below, showing the *domain* of each piece of the spline. The height of each curve indicates how influential the corresponding model covariate (one per spline region) will be on model's inference of that region. The overlapping regions represent the knots, showing how the smooth transition from one region to the next is formed.

```{code-cell} ipython3
spline_df = (
    pd.DataFrame(B)
    .assign(year=blossom_data.year.values)
    .melt("year", var_name="spline_i", value_name="value")
)

color = plt.cm.magma(np.linspace(0, 0.80, len(spline_df.spline_i.unique())))

fig = plt.figure()
for i, c in enumerate(color):
    subset = spline_df.query(f"spline_i == {i}")
    subset.plot("year", "value", c=c, ax=plt.gca(), label=i)
plt.legend(title="Spline Index", loc="upper center", fontsize=8, ncol=6);
```

### Fit the model

Finally, the model can be built using PyMC. A graphical diagram shows the organization of the model parameters (note that this requires the installation of {ref}`python-graphviz`, which I recommend doing in a `conda` virtual environment).

```{code-cell} ipython3
COORDS = {"splines": np.arange(B.shape[1])}
with pm.Model(coords=COORDS) as spline_model:
    a = pm.Normal("a", 100, 5)
    w = pm.Normal("w", mu=0, sigma=3, size=B.shape[1], dims="splines")

    mu = pm.Deterministic(
        "mu",
        a + pm.math.dot(np.asarray(B, order="F"), w.T),
    )
    sigma = pm.Exponential("sigma", 1)

    D = pm.Normal("D", mu=mu, sigma=sigma, observed=blossom_data.doy)
```

```{code-cell} ipython3
pm.model_to_graphviz(spline_model)
```

```{code-cell} ipython3
with spline_model:
    idata = pm.sample_prior_predictive()
    idata.extend(
        pm.sample(
            nuts_sampler="nutpie",
            draws=1000,
            tune=1000,
            random_seed=rng,
            chains=4,
        )
    )
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)
```

## Analysis

Now we can analyze the draws from the posterior of the model.

### Parameter Estimates

Below is a table summarizing the posterior distributions of the model parameters.
The posteriors of $a$ and $\sigma$ are quite narrow while those for $w$ are wider.
This is likely because all of the data points are used to estimate $a$ and $\sigma$ whereas only a subset are used for each value of $w$.
(It could be interesting to model these hierarchically allowing for the sharing of information and adding regularization across the spline.) 
The effective sample size and $\widehat{R}$ values all look good, indicating that the model has converged and sampled well from the posterior distribution.

```{code-cell} ipython3
az.summary(idata, var_names=["a", "w", "sigma"])
```

The trace plots of the model parameters look good (homogeneous and no sign of trend), further indicating that the chains converged and mixed.

```{code-cell} ipython3
az.plot_trace(idata, var_names=["a", "w", "sigma"]);
```

```{code-cell} ipython3
az.plot_forest(idata, var_names=["w"], combined=False, r_hat=True);
```

Another visualization of the fit spline values is to plot them multiplied against the basis matrix.
The knot boundaries are shown as vertical lines again, but now the spline basis is multiplied against the values of $w$ (represented as the rainbow-colored curves). The dot product of $B$ and $w$ – the actual computation in the linear model – is shown in black.

```{code-cell} ipython3
wp = idata.posterior["w"].mean(("chain", "draw")).values

spline_df = (
    pd.DataFrame(B * wp.T)
    .assign(year=blossom_data.year.values)
    .melt("year", var_name="spline_i", value_name="value")
)

spline_df_merged = (
    pd.DataFrame(np.dot(B, wp.T))
    .assign(year=blossom_data.year.values)
    .melt("year", var_name="spline_i", value_name="value")
)


color = plt.cm.rainbow(np.linspace(0, 1, len(spline_df.spline_i.unique())))
fig = plt.figure()
for i, c in enumerate(color):
    subset = spline_df.query(f"spline_i == {i}")
    subset.plot("year", "value", c=c, ax=plt.gca(), label=i)
spline_df_merged.plot("year", "value", c="black", lw=2, ax=plt.gca())
plt.legend(title="Spline Index", loc="lower center", fontsize=8, ncol=6)

for knot in knot_list:
    plt.gca().axvline(knot, color="grey", alpha=0.4)
```

### Model predictions

Lastly, we can visualize the predictions of the model using the posterior predictive check.

```{code-cell} ipython3
post_pred = az.summary(idata, var_names=["mu"]).reset_index(drop=True)
blossom_data_post = blossom_data.copy().reset_index(drop=True)
blossom_data_post["pred_mean"] = post_pred["mean"]
blossom_data_post["pred_hdi_lower"] = post_pred["hdi_3%"]
blossom_data_post["pred_hdi_upper"] = post_pred["hdi_97%"]
```

```{code-cell} ipython3
blossom_data.plot.scatter(
    "year",
    "doy",
    color="cornflowerblue",
    s=10,
    title="Cherry blossom data with posterior predictions",
    ylabel="Days in bloom",
)
for knot in knot_list:
    plt.gca().axvline(knot, color="grey", alpha=0.4)

blossom_data_post.plot("year", "pred_mean", ax=plt.gca(), lw=3, color="firebrick")
plt.fill_between(
    blossom_data_post.year,
    blossom_data_post.pred_hdi_lower,
    blossom_data_post.pred_hdi_upper,
    color="firebrick",
    alpha=0.4,
);
```

## Predicting on new data

Now imagine we got a new data set, with the same range of years as the original data set, and we want to get predictions for this new data set with our fitted model. We can do this with the classic PyMC workflow of `Data` containers and `set_data` method.

Before we get there though, let's note that we didn't say the new data set contains *new* years, i.e out-of-sample years. And that's on purpose, because splines can't extrapolate beyond the range of the data set used to fit the model -- hence their limitation for time series analysis. On data ranges previously seen though, that's no problem. 

That precision out of the way, let's redefine our model, this time adding `Data` containers.

```{code-cell} ipython3
COORDS = {"obs": blossom_data.index}
```

```{code-cell} ipython3
with pm.Model(coords=COORDS) as spline_model:
    year_data = pm.Data("year", blossom_data.year)
    doy = pm.Data("doy", blossom_data.doy)

    # intercept
    a = pm.Normal("a", 100, 5)

    # Create spline bases & coefficients
    ## Store knots & design matrix for prediction
    spline_model.knots = np.percentile(year_data.eval(), np.linspace(0, 100, num_knots + 2))[1:-1]
    spline_model.dm = dmatrix(
        "bs(x, knots=spline_model.knots, degree=3, include_intercept=False) - 1",
        {"x": year_data.eval()},
    )
    spline_model.add_coords({"spline": np.arange(spline_model.dm.shape[1])})
    splines_basis = pm.Data("splines_basis", np.asarray(spline_model.dm), dims=("obs", "spline"))
    w = pm.Normal("w", mu=0, sigma=3, dims="spline")

    mu = pm.Deterministic(
        "mu",
        a + pm.math.dot(splines_basis, w),
    )
    sigma = pm.Exponential("sigma", 1)

    D = pm.Normal("D", mu=mu, sigma=sigma, observed=doy)
```

```{code-cell} ipython3
pm.model_to_graphviz(spline_model)
```

```{code-cell} ipython3
with spline_model:
    idata = pm.sample(
        nuts_sampler="nutpie",
        random_seed=rng,
    )
    idata.extend(pm.sample_posterior_predictive(idata, random_seed=rng))
```

Now we can swap out the data and update the design matrix with the new data:

```{code-cell} ipython3
new_blossom_data = (
    blossom_data.sample(50, random_state=rng).sort_values("year").reset_index(drop=True)
)

# update design matrix with new data
year_data_new = new_blossom_data.year.to_numpy()
dm_new = build_design_matrices([spline_model.dm.design_info], {"x": year_data_new})[0]
```

Use `set_data` to update the data in the model:

```{code-cell} ipython3
with spline_model:
    pm.set_data(
        new_data={
            "year": year_data_new,
            "doy": new_blossom_data.doy.to_numpy(),
            "splines_basis": np.asarray(dm_new),
        },
        coords={
            "obs": new_blossom_data.index,
        },
    )
```

And all that's left is to sample from the posterior predictive distribution:

```{code-cell} ipython3
with spline_model:
    preds = pm.sample_posterior_predictive(idata, var_names=["mu"])
```

Plot the predictions, to check if everything went well:

```{code-cell} ipython3
_, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True, sharey=True)

blossom_data.plot.scatter(
    "year",
    "doy",
    color="cornflowerblue",
    s=10,
    title="Posterior predictions",
    ylabel="Days in bloom",
    ax=axes[0],
)
axes[0].vlines(
    spline_model.knots,
    blossom_data.doy.min(),
    blossom_data.doy.max(),
    color="grey",
    alpha=0.4,
)
axes[0].plot(
    blossom_data.year,
    idata.posterior["mu"].mean(("chain", "draw")),
    color="firebrick",
)
az.plot_hdi(blossom_data.year, idata.posterior["mu"], color="firebrick", ax=axes[0])

new_blossom_data.plot.scatter(
    "year",
    "doy",
    color="cornflowerblue",
    s=10,
    title="Predictions on new data",
    ylabel="Days in bloom",
    ax=axes[1],
)
axes[1].vlines(
    spline_model.knots,
    blossom_data.doy.min(),
    blossom_data.doy.max(),
    color="grey",
    alpha=0.4,
)
axes[1].plot(
    new_blossom_data.year,
    preds.posterior_predictive.mu.mean(("chain", "draw")),
    color="firebrick",
)
az.plot_hdi(new_blossom_data.year, preds.posterior_predictive.mu, color="firebrick", ax=axes[1]);
```

And... voilà! Granted, this example is not the most realistic one, but we trust you to adapt it to your wildest dreams ;)

+++

## References

:::{bibliography}
:filter: docname in docnames
:::

+++

## Authors

- Created by Joshua Cook
- Updated by Tyler James Burch
- Updated by Chris Fonnesbeck
- Predictions on new data added by Alex Andorra

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,xarray,patsy
```

:::{include} ../page_footer.md
:::
