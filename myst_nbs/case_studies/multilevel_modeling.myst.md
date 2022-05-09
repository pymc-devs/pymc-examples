---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3.9.10 ('pymc-dev-py39')
  language: python
  name: python3
---

(multilevel_modeling)=
# A Primer on Bayesian Methods for Multilevel Modeling

:::{post} 27 February, 2022
:tags: hierarchical model, case study
:category: intermediate
:author: Chris Fonnesbeck, Colin Carroll, Alex Andorra, Oriol Abril, Farhan Reynaldo
:::

+++

Hierarchical or multilevel modeling is a generalization of regression modeling. *Multilevel models* are regression models in which the constituent model parameters are given **probability models**. This implies that model parameters are allowed to **vary by group**. Observational units are often naturally **clustered**. Clustering induces dependence between observations, despite random sampling of clusters and random sampling within clusters.

A *hierarchical model* is a particular multilevel model where parameters are nested within one another. Some multilevel structures are not hierarchical -- e.g. "country" and "year" are not nested, but may represent separate, but overlapping, clusters of parameters. We will motivate this topic using an environmental epidemiology example.

+++

## Example: Radon contamination {cite:t}`gelman2006data`

Radon is a radioactive gas that enters homes through contact points with the ground. It is a carcinogen that is the primary cause of lung cancer in non-smokers. Radon levels vary greatly from household to household.

![radon](https://www.cgenarchive.org/uploads/2/5/2/6/25269392/7758459_orig.jpg)

The EPA did a study of radon levels in 80,000 houses. There are two important predictors:

* measurement in basement or first floor (radon higher in basements)
* county uranium level (positive correlation with radon levels)

We will focus on modeling radon levels in Minnesota.

The hierarchy in this example is households within county.

+++

## Data organization

+++

First, we import the data from a local file, and extract Minnesota's data.

```{code-cell} ipython3
import os

import aesara.tensor as at
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

print(f"Running on PyMC v{pm.__version__}")
```

```{code-cell} ipython3
RANDOM_SEED = 8924
az.style.use("arviz-darkgrid")
```

```{code-cell} ipython3
# Import radon data
try:
    srrs2 = pd.read_csv(os.path.join("..", "data", "srrs2.dat"))
except FileNotFoundError:
    srrs2 = pd.read_csv(pm.get_data("srrs2.dat"))

srrs2.columns = srrs2.columns.map(str.strip)
srrs_mn = srrs2[srrs2.state == "MN"].copy()
```

Next, obtain the county-level predictor, uranium, by combining two variables.

```{code-cell} ipython3
srrs_mn["fips"] = srrs_mn.stfips * 1000 + srrs_mn.cntyfips
cty = pd.read_csv(pm.get_data("cty.dat"))
cty_mn = cty[cty.st == "MN"].copy()
cty_mn["fips"] = 1000 * cty_mn.stfips + cty_mn.ctfips
```

Use the `merge` method to combine home- and county-level information in a single DataFrame.

```{code-cell} ipython3
srrs_mn = srrs_mn.merge(cty_mn[["fips", "Uppm"]], on="fips")
srrs_mn = srrs_mn.drop_duplicates(subset="idnum")
u = np.log(srrs_mn.Uppm).unique()

n = len(srrs_mn)
```

```{code-cell} ipython3
srrs_mn.head()
```

We also need a lookup table (`dict`) for each unique county, for indexing.

```{code-cell} ipython3
srrs_mn.county = srrs_mn.county.map(str.strip)
mn_counties = srrs_mn.county.unique()
counties = len(mn_counties)
county_lookup = dict(zip(mn_counties, range(counties)))
```

Finally, create local copies of variables.

```{code-cell} ipython3
county = srrs_mn["county_code"] = srrs_mn.county.replace(county_lookup).values
radon = srrs_mn.activity
srrs_mn["log_radon"] = log_radon = np.log(radon + 0.1).values
floor = srrs_mn.floor.values
```

Distribution of radon levels in MN (log scale):

```{code-cell} ipython3
srrs_mn.log_radon.hist(bins=25);
```

## Conventional approaches

The two conventional alternatives to modeling radon exposure represent the two extremes of the bias-variance tradeoff:

***Complete pooling***: 

Treat all counties the same, and estimate a single radon level.

$$y_i = \alpha + \beta x_i + \epsilon_i$$

***No pooling***:

Model radon in each county independently.

$$y_i = \alpha_{j[i]} + \beta x_i + \epsilon_i$$

where $j = 1,\ldots,85$

The errors $\epsilon_i$ may represent measurement error, temporal within-house variation, or variation among houses.

+++

We'll start by estimating the slope and intercept for the complete pooling model. You'll notice that we used an *index* variable instead of an *indicator* variable in the linear model below. There are two main reasons. One, this generalizes well to more-than-two-category cases. Two, this approach correctly considers that neither category has more prior uncertainty than the other. On the contrary, the indicator variable approach necessarily assumes that one of the categories has more uncertainty than the other: here, the cases when `floor=1` would take into account 2 priors ($\alpha + \beta$), whereas cases when `floor=0` would have only one prior ($\alpha$). But *a priori* we aren't more unsure about floor measurements than about basement measurements, so it makes sense to give them the same prior uncertainty.

Now for the model:

```{code-cell} ipython3
coords = {"Level": ["Basement", "Floor"]}
with pm.Model(coords=coords, rng_seeder=RANDOM_SEED) as pooled_model:
    floor_idx = pm.Data("floor_idx", floor, dims="obs_id", mutable=True)
    a = pm.Normal("a", 0.0, sigma=10.0, dims="Level")

    theta = a[floor_idx]
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")

pm.model_to_graphviz(pooled_model)
```

You may be wondering why we are using the `pm.Data` container above even though the variable `floor_idx` is not an observed variable nor a parameter of the model. As you'll see, this will make our lives much easier when we'll plot and diagnose our model. In short, this will tell {doc}`Arviz <arviz:index>` that `floor_idx` is information used by the model to index variables. ArviZ will thus include `floor_idx` as a variable in the `constant_data` group of the resulting {ref}`InferenceData <xarray_for_arviz>` object. Moreover, including `floor_idx` in the `InferenceData` object makes sharing and reproducing analysis much easier, all the data needed to analyze or rerun the model is stored there.

+++

Before running the model let's do some prior predictive checks. Indeed, having sensible priors is not only a way to incorporate scientific knowledge into the model, it can also help and make the MCMC machinery faster -- here we are dealing with a simple linear regression, so no link function comes and distorts the outcome space; but one day this will happen to you and you'll need to think hard about your priors to help your MCMC sampler. So, better to train ourselves when it's quite easy than having to learn when it's very hard... There is a really neat function to do that in PyMC:

```{code-cell} ipython3
with pooled_model:
    prior_checks = pm.sample_prior_predictive()

_, ax = plt.subplots()
prior_checks.prior.plot.scatter(x="Level", y="a", color="k", alpha=0.2, ax=ax)
ax.set_ylabel("Mean log radon level");
```

ArviZ `InferenceData` uses `xarray.Dataset`s under the hood, which give access to several common plotting functions with `.plot`. In this case, we want scatter plot of the mean log radon level (which is stored in variable `a`) for each of the two levels we are considering. If our desired plot is supported by xarray plotting capabilities, we can take advantage of xarray to automatically generate both plot and labels for us. Notice how everything is directly plotted and annotated, the only change we need to do is renaming the y axis label from `a` to `Mean log radon level`.

+++

I'm no expert in radon levels, but, before seing the data, these priors seem to allow for quite a wide range of the mean log radon level. But don't worry, we can always change these priors if sampling gives us hints that they might not be appropriate -- after all, priors are assumptions, not oaths; and as most assumptions, they can be tested.

However, we can already think of an improvement. Do you see it? Remember what we said at the beginning: radon levels tend to be higher in basements, so we could incorporate this prior scientific knowledge into our model by giving $a_{basement}$ a higher mean than $a_{floor}$. Here, there are so much data that the prior should be washed out anyway, but we should keep this fact in mind -- for future cases or if sampling proves more difficult than expected...

Speaking of sampling, let's fire up the Bayesian machinery!

```{code-cell} ipython3
with pooled_model:
    pooled_trace = pm.sample()

pooled_trace.extend(prior_checks)
az.summary(pooled_trace, round_to=2)
```

No divergences and a sampling that only took seconds -- this is the Flash of samplers! Here the chains look very good (good R hat, good effective sample size, small sd), but remember to check your chains after sampling -- `az.plot_trace` is usually a good start.

Let's see what it means on the outcome space: did the model pick-up the negative relationship between floor measurements and log radon levels? What's the uncertainty around its estimates? To estimate the uncertainty around the household radon levels (not the average level, but measurements that would be likely in households), we need to sample the likelihood `y` from the model. In another words, we need to do posterior predictive checks:

```{code-cell} ipython3
with pooled_model:
    ppc = pm.sample_posterior_predictive(pooled_trace)
```

+++ {"raw_mimetype": "text/html"}

We have now converted our trace and posterior predictive samples into an `arviz.InferenceData` object. `InferenceData` is specifically designed to centralize all the relevant quantities of a Bayesian inference workflow into a single object. This will also make the rendering of plots and diagnostics faster -- otherwise ArviZ will need to convert your trace to InferenceData each time you call it.

```{code-cell} ipython3
pooled_trace.extend(ppc)
```

+++ {"raw_mimetype": "text/html"}

We now want to calculate the highest density interval given by the posterior predictive on Radon levels. However, we are not interested in the HDI of each observation but in the HDI of each level. We can groupby xarray objects using variable or coordinate names or using other xarray objects with the same dimensions (`obs_id` in this case). `az.hdi` works with both xarray Datasets and groupby objects. Moreover, all calculations keep track and update the dimensions and coordinate values.

When calling aggregation functions, xarray by default reduces all the dimensions in each variable whereas ArviZ by default reduces only `chain` and `draw` dimensions. This generally means that we have to be explicit when calling `Dataset.mean(dim=("chain", "draw")` if using xarray functions but not with ArviZ ones -- compare for example calls to `.mean` with calls to `az.hdi`. Notice how using labeled dimensions helps in understanding what exactly is being reduced with a quick glance at the code.

Now that we have some context on reducing dims in ArviZ and xarray, let's move to the case at hand. In this particular case, we want ArviZ to reduce all dimensions in each groupby group. Here, each groupby will have the same 3 dimensions as the original input `(chain, draw, obs_id)` what will change is the length of the `obs_id` dimension, in the first group it will be the number of basement level observations and in the second the number of floor level observations. In `az.hdi`, the dimensions to be reduce can be specified with the `input_core_dims` argument.

```{code-cell} ipython3
hdi_helper = lambda ds: az.hdi(ds, input_core_dims=[["chain", "draw", "obs_id"]])
hdi_ppc = (
    pooled_trace.posterior_predictive.y.groupby(pooled_trace.constant_data.floor_idx)
    .apply(hdi_helper)
    .y
)
hdi_ppc
```

In addition, ArviZ has also included the `hdi_prob` as an attribute of the `hdi` coordinate, click on its file icon to see it!

+++

We will now add one extra coordinate to the `observed_data` group: the `Level` labels (not indices). This will allow xarray to automatically generate the correct xlabel and xticklabels so we don't have to worry about labeling too much. In this particular case we will only do one plot, which makes the adding of a coordinate a bit of an overkill. In many cases however, we will have several plots and using this approach will automate labeling for _all_ plots. Eventually, we will sort by Level coordinate to make sure `Basement` is the first value and goes at the left of the plot.

```{code-cell} ipython3
level_labels = pooled_trace.posterior.Level[pooled_trace.constant_data.floor_idx]
pooled_trace.observed_data = pooled_trace.observed_data.assign_coords(Level=level_labels).sortby(
    "Level"
)
```

We can then use these samples in our plot:

```{code-cell} ipython3
pooled_means = pooled_trace.posterior.mean(dim=("chain", "draw"))

_, ax = plt.subplots()
pooled_trace.observed_data.plot.scatter(x="Level", y="y", label="Observations", alpha=0.4, ax=ax)

az.plot_hdi(
    [0, 1],
    hdi_data=hdi_ppc,
    fill_kwargs={"alpha": 0.2, "label": "Exp. distrib. of Radon levels"},
    ax=ax,
)

az.plot_hdi(
    [0, 1], pooled_trace.posterior.a, fill_kwargs={"alpha": 0.5, "label": "Exp. mean HPD"}, ax=ax
)
ax.plot([0, 1], pooled_means.a, label="Exp. mean")

ax.set_ylabel("Log radon level")
ax.legend(ncol=2, fontsize=9, frameon=True);
```

The 94% interval of the expected value is very narrow, and even narrower for basement measurements, meaning that the model is slightly more confident about these observations. The sampling distribution of individual radon levels is much wider. We can infer that floor level does account for some of the variation in radon levels. We can see however that the model underestimates the dispersion in radon levels across households -- lots of them lie outside the light orange prediction envelope. So this model is a good start but we can't stop there.

Let's compare it to the unpooled model, where we estimate the radon level for each county:

```{code-cell} ipython3
coords["County"] = mn_counties
with pm.Model(coords=coords, rng_seeder=RANDOM_SEED) as unpooled_model:
    floor_idx = pm.Data("floor_idx", floor, dims="obs_id", mutable=True)
    county_idx = pm.Data("county_idx", county, dims="obs_id", mutable=True)
    a = pm.Normal("a", 0.0, sigma=10.0, dims=("County", "Level"))

    theta = a[county_idx, floor_idx]
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")
pm.model_to_graphviz(unpooled_model)
```

```{code-cell} ipython3
with unpooled_model:
    unpooled_trace = pm.sample()
```

Sampling went fine again. Let's look at the expected values for both basement (dimension 0) and floor (dimension 1) in each county:

```{code-cell} ipython3
az.plot_forest(
    unpooled_trace, var_names="a", figsize=(6, 32), r_hat=True, combined=True, textsize=8
);
```

Sampling was good for all counties, but you can see that some are more uncertain than others, and all of these uncertain estimates are for floor measurements. This probably comes from the fact that some counties just have a handful of floor measurements, so the model is pretty uncertain about them.

To identify counties with high radon levels, we can plot the ordered mean estimates, as well as their 94% HPD:

```{code-cell} ipython3
unpooled_means = unpooled_trace.posterior.mean(dim=("chain", "draw"))
unpooled_hdi = az.hdi(unpooled_trace)
```

We will now take advantage of label based indexing for Datasets with the `sel` method and of automagical sorting capabilities. We first sort using the values of a specific 1D variable `a`. Then, thanks to `unpooled_means` and `unpooled_hdi` both having the `County` dimension, we can pass a 1D DataArray to sort the second dataset too.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
xticks = np.arange(0, 86, 6)
fontdict = {"horizontalalignment": "right", "fontsize": 10}
for ax, level in zip(axes, ["Basement", "Floor"]):
    unpooled_means_iter = unpooled_means.sel(Level=level).sortby("a")
    unpooled_hdi_iter = unpooled_hdi.sel(Level=level).sortby(unpooled_means_iter.a)
    unpooled_means_iter.plot.scatter(x="County", y="a", ax=ax, alpha=0.8)
    ax.vlines(
        np.arange(counties),
        unpooled_hdi_iter.a.sel(hdi="lower"),
        unpooled_hdi_iter.a.sel(hdi="higher"),
        color="orange",
        alpha=0.6,
    )
    ax.set(title=f"{level.title()} estimates", ylabel="Radon estimate", ylim=(-2, 4.5))
    ax.set_xticks(xticks)
    ax.set_xticklabels(unpooled_means_iter.County.values[xticks], fontdict=fontdict)
    ax.tick_params(rotation=30)
```

There seems to be more dispersion in radon levels for floor measurements than for basement ones. Moreover, as we saw in the forest plot, floor estimates are globally more uncertain, especially in some counties. We speculated that this is due to smaller sample sizes in the data, but let's verify it!

```{code-cell} ipython3
n_floor_meas = srrs_mn.groupby("county").sum().floor
uncertainty = unpooled_hdi.a.sel(hdi="higher", Level="Floor") - unpooled_hdi.a.sel(
    hdi="lower", Level="Floor"
)

plt.plot(n_floor_meas, uncertainty, "o", alpha=0.4)
plt.xlabel("Nbr floor measurements in county")
plt.ylabel("Estimates' uncertainty");
```

Bingo! This makes sense: it's very hard to estimate floor radon levels in counties where there are no floor measurements, and the model is telling us that by being very uncertain in its estimates for those counties. This is a classic issue with no-pooling models: when you estimate clusters independently from each other, what do you with small-sample-size counties?

Another way to see this phenomenon is to visually compare the pooled and unpooled estimates for a subset of counties representing a range of sample sizes:

+++

In cases where label based indexing is not powerful enough (for example when repeated labels are present) we can still index xarray objects with boolean masks or positional indices. Here we create a mask with the `isin` method and index with `where`. Note that xarray objects are generally high dimensional and condition based indexing is bound to generate ragged arrays. Thus, `xarray.where` by default replaces the _unselected_ values with NaNs. In our case, the variable we are indexing is 1D and we can therefore use `drop=True` to remove the values instead of replacing by NaN.

Like we did above, we add a couple of extra coordinates to help in data processing and plotting.

```{code-cell} ipython3
SAMPLE_COUNTIES = (
    "LAC QUI PARLE",
    "AITKIN",
    "KOOCHICHING",
    "DOUGLAS",
    "CLAY",
    "STEARNS",
    "RAMSEY",
    "ST LOUIS",
)

unpooled_trace.observed_data = unpooled_trace.observed_data.assign_coords(
    {
        "County": ("obs_id", mn_counties[unpooled_trace.constant_data.county_idx]),
        "Level": (
            "obs_id",
            np.array(["Basement", "Floor"])[unpooled_trace.constant_data.floor_idx],
        ),
    }
)

fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey=True, sharex=True)
for ax, c in zip(axes.ravel(), SAMPLE_COUNTIES):
    sample_county_mask = unpooled_trace.observed_data.County.isin([c])

    # plot obs:
    unpooled_trace.observed_data.where(sample_county_mask, drop=True).sortby("Level").plot.scatter(
        x="Level", y="y", ax=ax, alpha=0.4
    )

    # plot both models:
    ax.plot([0, 1], unpooled_means.a.sel(County=c), "b")
    ax.plot([0, 1], pooled_means.a, "r--")

    ax.set_title(c)
    ax.set_xlabel("")
    ax.set_ylabel("Log radon level")
```

Neither of these models are satisfactory:

* If we are trying to identify high-radon counties, pooling is useless -- because, by definition, the pooled model estimates radon at the state-level. In other words, pooling leads to maximal *underfitting*: the variation across counties is not taken into account and only the overall population is estimated.
* We do not trust extreme unpooled estimates produced by models using few observations. This leads to maximal *overfitting*: only the within-county variations are taken into account and the overall population (i.e the state-level, which tells us about similarites across counties) is not estimated. 

This issue is acute for small sample sizes, as seen above: in counties where we have few floor measurements, if radon levels are higher for those data points than for basement ones (Aitkin, Koochiching, Ramsey), the model will estimate that radon levels are higher in floors than basements for these counties. But we shouldn't trust this conclusion, because both scientific knowledge and the situation in other counties tell us that it is usually the reverse (basement radon > floor radon). So unless we have a lot of observations telling us otherwise for a given county, we should be skeptical and shrink our county-estimates to the state-estimates -- in other words, we should balance between cluster-level and population-level information, and the amount of shrinkage will depend on how extreme and how numerous the data in each cluster are. 

But how do we do that? Well, ladies and gentlemen, let me introduce you to... hierarchical models!

+++

## Multilevel and hierarchical models

When we pool our data, we imply that they are sampled from the same model. This ignores any variation among sampling units (other than sampling variance) -- we assume that counties are all the same:

![pooled](pooled_model.png)

When we analyze data unpooled, we imply that they are sampled independently from separate models. At the opposite extreme from the pooled case, this approach claims that differences between sampling units are too large to combine them -- we assume that counties have no similarity whatsoever:

![unpooled](unpooled_model.png)

In a hierarchical model, parameters are viewed as a sample from a population distribution of parameters. Thus, we view them as being neither entirely different or exactly the same. This is ***partial pooling***:

![hierarchical](partial_pooled_model.png)

We can use PyMC to easily specify multilevel models, and fit them using Markov chain Monte Carlo.

+++

## Partial pooling model

The simplest partial pooling model for the household radon dataset is one which simply estimates radon levels, without any predictors at any level. A partial pooling model represents a compromise between the pooled and unpooled extremes, approximately a weighted average (based on sample size) of the unpooled county estimates and the pooled estimates.

$$\hat{\alpha} \approx \frac{(n_j/\sigma_y^2)\bar{y}_j + (1/\sigma_{\alpha}^2)\bar{y}}{(n_j/\sigma_y^2) + (1/\sigma_{\alpha}^2)}$$

Estimates for counties with smaller sample sizes will shrink towards the state-wide average.

Estimates for counties with larger sample sizes will be closer to the unpooled county estimates and will influence the the state-wide average.

```{code-cell} ipython3
with pm.Model(coords=coords, rng_seeder=RANDOM_SEED) as partial_pooling:
    county_idx = pm.Data("county_idx", county, dims="obs_id", mutable=True)
    # Hyperpriors:
    a = pm.Normal("a", mu=0.0, sigma=10.0)
    sigma_a = pm.Exponential("sigma_a", 1.0)

    # Varying intercepts:
    a_county = pm.Normal("a_county", mu=a, sigma=sigma_a, dims="County")

    # Expected value per county:
    theta = a_county[county_idx]
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")
pm.model_to_graphviz(partial_pooling)
```

```{code-cell} ipython3
with partial_pooling:
    partial_pooling_trace = pm.sample(tune=2000)
```

To compare partial-pooling and no-pooling estimates, let's run the unpooled model without the `floor` predictor:

```{code-cell} ipython3
with pm.Model(coords=coords, rng_seeder=RANDOM_SEED) as unpooled_bis:
    county_idx = pm.Data("county_idx", county, dims="obs_id", mutable=True)
    a_county = pm.Normal("a_county", 0.0, sigma=10.0, dims="County")

    theta = a_county[county_idx]
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")

    unpooled_trace_bis = pm.sample(tune=2000)
```

Now let's compare both models' estimates for all 85 counties. We'll plot the estimates against each county's sample size, to let you see more clearly what hierarchical models bring to the table:

```{code-cell} ipython3
N_county = srrs_mn.groupby("county")["idnum"].count().values

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
for ax, trace, level in zip(
    axes,
    (unpooled_trace_bis, partial_pooling_trace),
    ("no pooling", "partial pooling"),
):

    # add variable with x values to xarray dataset
    trace.posterior = trace.posterior.assign_coords({"N_county": ("County", N_county)})
    # plot means
    trace.posterior.mean(dim=("chain", "draw")).plot.scatter(
        x="N_county", y="a_county", ax=ax, alpha=0.9
    )
    ax.hlines(
        partial_pooling_trace.posterior.a.mean(),
        0.9,
        max(N_county) + 1,
        alpha=0.4,
        ls="--",
        label="Est. population mean",
    )

    # plot hdi
    hdi = az.hdi(trace).a_county
    ax.vlines(N_county, hdi.sel(hdi="lower"), hdi.sel(hdi="higher"), color="orange", alpha=0.5)

    ax.set(
        title=f"{level.title()} Estimates",
        xlabel="Nbr obs in county (log scale)",
        xscale="log",
        ylabel="Log radon",
    )
    ax.legend(fontsize=10)
```

Notice the difference between the unpooled and partially-pooled estimates, particularly at smaller sample sizes: As expected, the former are both more extreme and more imprecise. Indeed, in the partially-pooled model, estimates in small-sample-size counties are informed by the population parameters -- hence more precise estimates. Moreover, the smaller the sample size, the more regression towards the overall mean (the dashed gray line) -- hence less extreme estimates. In other words, the model is skeptical of extreme deviations from the population mean in counties where data is sparse. 

Now let's try to integrate the `floor` predictor! To show you an example with a slope we're gonna take the indicator variable road, but we could stay with the index variable approach that we used for the no-pooling model. Then we would have one intercept for each category -- basement and floor.

+++

## Varying intercept model

As above, this model allows intercepts to vary across county, according to a random effect. We just add a fixed slope for the predictor (i.e all counties will have the same slope):

$$y_i = \alpha_{j[i]} + \beta x_{i} + \epsilon_i$$

where

$$\epsilon_i \sim N(0, \sigma_y^2)$$

and the intercept random effect:

$$\alpha_{j[i]} \sim N(\mu_{\alpha}, \sigma_{\alpha}^2)$$

As with the the no-pooling model, we set a separate intercept for each county, but rather than fitting separate regression models for each county, multilevel modeling **shares strength** among counties, allowing for more reasonable inference in counties with little data. Here is what that looks in code:

```{code-cell} ipython3
with pm.Model(coords=coords, rng_seeder=RANDOM_SEED) as varying_intercept:
    floor_idx = pm.Data("floor_idx", floor, dims="obs_id", mutable=True)
    county_idx = pm.Data("county_idx", county, dims="obs_id", mutable=True)
    # Hyperpriors:
    a = pm.Normal("a", mu=0.0, sigma=10.0)
    sigma_a = pm.Exponential("sigma_a", 1.0)

    # Varying intercepts:
    a_county = pm.Normal("a_county", mu=a, sigma=sigma_a, dims="County")
    # Common slope:
    b = pm.Normal("b", mu=0.0, sigma=10.0)

    # Expected value per county:
    theta = a_county[county_idx] + b * floor_idx
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")
pm.model_to_graphviz(varying_intercept)
```

Let's fit this bad boy with MCMC:

```{code-cell} ipython3
with varying_intercept:
    varying_intercept_trace = pm.sample(tune=2000, init="adapt_diag")
```

```{code-cell} ipython3
az.plot_forest(
    varying_intercept_trace, var_names=["a", "a_county"], r_hat=True, combined=True, textsize=9
);
```

```{code-cell} ipython3
az.plot_trace(varying_intercept_trace, var_names=["a", "sigma_a", "b", "sigma"]);
```

```{code-cell} ipython3
az.summary(varying_intercept_trace, var_names=["a", "sigma_a", "b", "sigma"], round_to=2)
```

As we suspected, the estimate for the `floor` coefficient is reliably negative and centered around -0.66. This can be interpreted as houses without basements having about half ($\exp(-0.66) = 0.52$) the radon levels of those with basements, after accounting for county. Note that this is only the *relative* effect of floor on radon levels: conditional on being in a given county, radon is expected to be half lower in houses without basements than in houses with. To see how much difference a basement makes on the *absolute* level of radon, we'd have to push the parameters through the model, as we do with posterior predictive checks and as we'll do just now.

+++

To do so, we will take advantage of automatic broadcasting with xarray. We want to create a 2D array with dimensions `(County, Level)`, our variable `a_county` already has the `County` dimension. `b` however is a scalar. We will multiply `b` with an `xvals` `DataArray` to introduce the `Level` dimension into the mix. xarray will handle everything from there, no loops nor reshapings required.

```{code-cell} ipython3
xvals = xr.DataArray([0, 1], dims="Level", coords={"Level": ["Basement", "Floor"]})
post = varying_intercept_trace.posterior  # alias for readability
theta = (
    (post.a_county + post.b * xvals).mean(dim=("chain", "draw")).to_dataset(name="Mean log radon")
)

_, ax = plt.subplots()
theta.plot.scatter(x="Level", y="Mean log radon", alpha=0.2, color="k", ax=ax)  # scatter
ax.plot(xvals, theta["Mean log radon"].T, "k-", alpha=0.2)
# add lines too
ax.set_title("MEAN LOG RADON BY COUNTY");
```

The graph above shows, for each county, the expected log radon level and the average effect of having no basement -- these are the absolute effects we were talking about. Two caveats though:
1. This graph doesn't show the uncertainty for each county -- how confident are we that the average estimates are where the graph shows? For that we'd need to combine the uncertainty in `a_county` and `b`, and this would of course vary by county. I didn't show it here because the graph would get cluttered, but go ahead and do it for a subset of counties.
2. These are only *average* estimates at the *county-level* (`theta` in the model): they don't take into account the variation by household. To add this other layer of uncertainty we'd need to take stock of the effect of `sigma` and generate samples from the `y` variable to see the effect on given households (that's exactly the role of posterior predictive checks).

That being said, it is easy to show that the partial pooling model provides more objectively reasonable estimates than either the pooled or unpooled models, at least for counties with small sample sizes:

```{code-cell} ipython3
fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey=True, sharex=True)
for ax, c in zip(axes.ravel(), SAMPLE_COUNTIES):
    sample_county_mask = unpooled_trace.observed_data.County.isin([c])

    # plot obs:
    unpooled_trace.observed_data.where(sample_county_mask, drop=True).sortby("Level").plot.scatter(
        x="Level", y="y", ax=ax, alpha=0.4
    )

    # plot models:
    ax.plot([0, 1], unpooled_means.a.sel(County=c), "k:", alpha=0.5, label="No pooling")
    ax.plot([0, 1], pooled_means.a, "r--", label="Complete pooling")

    ax.plot([0, 1], theta["Mean log radon"].sel(County=c), "b", label="Partial pooling")

    ax.set_title(c)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=10)

axes[0, 0].set_ylabel("Log radon level")
axes[1, 0].set_ylabel("Log radon level")
axes[0, 0].legend(fontsize=8, frameon=True), axes[1, 0].legend(fontsize=8, frameon=True);
```

Here we clearly see the notion that partial-pooling is a compromise between no pooling and complete pooling, as its mean estimates are usually between the other models' estimates. And interestingly, the bigger (smaller) the sample size in a given county, the closer the partial-pooling estimates are to the no-pooling (complete-pooling) estimates.

We see however that counties vary by more than just their baseline rates: the effect of floor seems to be different from one county to another. It would be great if our model could take that into account, wouldn't it? Well to do that, we need to allow the slope to vary by county -- not only the intercept -- and here is how you can do it with PyMC.

+++

## Varying intercept and slope model

The most general model allows both the intercept and slope to vary by county:

$$y_i = \alpha_{j[i]} + \beta_{j[i]} x_{i} + \epsilon_i$$

```{code-cell} ipython3
with pm.Model(coords=coords, rng_seeder=RANDOM_SEED) as varying_intercept_slope:
    floor_idx = pm.Data("floor_idx", floor, dims="obs_id", mutable=True)
    county_idx = pm.Data("county_idx", county, dims="obs_id", mutable=True)

    # Hyperpriors:
    a = pm.Normal("a", mu=0.0, sigma=5.0)
    sigma_a = pm.Exponential("sigma_a", 1.0)
    b = pm.Normal("b", mu=0.0, sigma=1.0)
    sigma_b = pm.Exponential("sigma_b", 0.5)

    # Varying intercepts:
    a_county = pm.Normal("a_county", mu=a, sigma=sigma_a, dims="County")
    # Varying slopes:
    b_county = pm.Normal("b_county", mu=b, sigma=sigma_b, dims="County")

    # Expected value per county:
    theta = a_county[county_idx] + b_county[county_idx] * floor_idx
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")
pm.model_to_graphviz(varying_intercept_slope)
```

Now, if you run this model, you'll get divergences (some or a lot, depending on your random seed). We don't want that -- divergences are your Voldemort to your models. In these situations it's usually wise to reparametrize your model using the "non-centered parametrization" (I know, it's really not a great term, but please indulge me). We're not gonna explain it here, but there are [great resources out there](https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/). In a nutshell, it's an algebraic trick that helps computation but leaves the model unchanged -- the model is statistically equivalent to the "centered" version. In that case, here is what it would look like:

```{code-cell} ipython3
with pm.Model(coords=coords, rng_seeder=RANDOM_SEED) as varying_intercept_slope:
    floor_idx = pm.Data("floor_idx", floor, dims="obs_id", mutable=True)
    county_idx = pm.Data("county_idx", county, dims="obs_id", mutable=True)

    # Hyperpriors:
    a = pm.Normal("a", mu=0.0, sigma=5.0)
    sigma_a = pm.Exponential("sigma_a", 1.0)
    b = pm.Normal("b", mu=0.0, sigma=1.0)
    sigma_b = pm.Exponential("sigma_b", 0.5)

    # Varying intercepts:
    za_county = pm.Normal("za_county", mu=0.0, sigma=1.0, dims="County")
    # Varying slopes:
    zb_county = pm.Normal("zb_county", mu=0.0, sigma=1.0, dims="County")

    # Expected value per county:
    theta = (a + za_county[county_idx] * sigma_a) + (b + zb_county[county_idx] * sigma_b) * floor
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")

    varying_intercept_slope_trace = pm.sample(1000, tune=2000, target_accept=0.99)
```

True, the code is uglier (for you, not for the computer), but:
1. The interpretation stays pretty much the same: `a` and `b` are still the mean state-wide intercept and slope. `sigma_a` and `sigma_b` still estimate the dispersion across counties of the intercepts and slopes (the more alike the counties, the smaller the corresponding sigma). The big change is that now the counties estimates (`za_county` and `zb_county`) are z-scores. But the strategy to see what this means for mean radon levels per county is the same: push all these parameters through the model to get samples from `theta`.
2. We don't have any divergence: the model sampled more efficiently and converged more quickly than in the centered form.

Notice however that we had to increase the number of tuning steps. Looking at the trace helps us understand why:

```{code-cell} ipython3
az.plot_trace(varying_intercept_slope_trace, compact=True, chain_prop={"ls": "-"});
```

All chains look good and we get a negative `b` coefficient, illustrating the mean downward effect of no-basement on radon levels at the state level. But notice that `sigma_b` often gets very near zero -- which would indicate that counties don't vary that much in their answer to the `floor` "treatment". That's probably what bugged MCMC when using the centered parametrization: these situations usually yield a weird geometry for the sampler, causing the divergences. In other words, the non-centered form often perfoms better when one of the sigmas gets close to zero. But here, even with the non-centered model the sampler is not that comfortable with `sigma_b`: in fact if you look at the estimates with `az.summary` you'll probably see that the number of effective samples is quite low for `sigma_b`.

Also note that `sigma_a` is not that big either -- i.e counties do differ in their baseline radon levels, but not by a lot. However we don't have that much of a problem to sample from this distribution because it's much narrower than `sigma_b` and doesn't get dangerously close to 0.

To wrap up this model, let's plot the relationship between radon and floor for each county:

```{code-cell} ipython3
xvals = xr.DataArray([0, 1], dims="Level", coords={"Level": ["Basement", "Floor"]})
post = varying_intercept_slope_trace.posterior  # alias for readability
avg_a_county = (post.a + post.za_county * post.sigma_a).mean(dim=("chain", "draw"))
avg_b_county = (post.b + post.zb_county * post.sigma_b).mean(dim=("chain", "draw"))
theta = (avg_a_county + avg_b_county * xvals).to_dataset(name="Mean log radon")

_, ax = plt.subplots()
theta.plot.scatter(x="Level", y="Mean log radon", alpha=0.2, color="k", ax=ax)  # scatter
ax.plot(xvals, theta["Mean log radon"].T, "k-", alpha=0.2)
# add lines too
ax.set_title("MEAN LOG RADON BY COUNTY");
```

With the same caveats as earlier, we can see that now *both* the intercept and the slope vary by county -- and isn't that a marvel of statistics? But wait, there is more! We can (and maybe should) take into account the covariation between intercepts and slopes: when baseline radon is low in a given county, maybe that means the difference between floor and basement measurements will decrease -- because there isn't that much radon anyway. That would translate into a positive correlation between `a_county` and `b_county`, and adding that into our model would make even more efficient use the available data. 

Or maybe the correlation is negative? In any case, we can't know that unless we model it. To do that, we'll use a multivariate Normal distribution instead of two different Normals for `a_county` and `b_county`. This simply means that each county's parameters come from a common distribution with mean `a` for intercepts and `b` for slopes, and slopes and intercepts co-vary according to the covariance matrix `S`. In mathematical form:

$$y \sim Normal(\theta, \sigma)$$

$$\theta = \alpha_{COUNTY} + \beta_{COUNTY} \times floor$$

$$\begin{bmatrix} \alpha_{COUNTY} \\ \beta_{COUNTY} \end{bmatrix} \sim MvNormal(\begin{bmatrix} \alpha \\ \beta \end{bmatrix}, \Sigma)$$

$$\Sigma = \begin{pmatrix} \sigma_{\alpha} & 0 \\ 0 & \sigma_{\beta} \end{pmatrix}
     P
     \begin{pmatrix} \sigma_{\alpha} & 0 \\ 0 & \sigma_{\beta} \end{pmatrix}$$
     
where $\alpha$ and $\beta$ are the mean intercept and slope respectively, $\sigma_{\alpha}$ and $\sigma_{\beta}$ represent the variation in intercepts and slopes respectively, and $P$ is the correlation matrix of intercepts and slopes. In this case, as their is only one slope, $P$ contains only one relevant figure: the correlation between $\alpha$ and $\beta$.

This translates quite easily in PyMC:

```{code-cell} ipython3
coords["param"] = ["a", "b"]
coords["param_bis"] = ["a", "b"]
with pm.Model(coords=coords, rng_seeder=RANDOM_SEED) as covariation_intercept_slope:
    floor_idx = pm.Data("floor_idx", floor, dims="obs_id", mutable=True)
    county_idx = pm.Data("county_idx", county, dims="obs_id", mutable=True)

    # prior stddev in intercepts & slopes (variation across counties):
    sd_dist = pm.Exponential.dist(0.5, shape=(2,))

    # get back standard deviations and rho:
    chol, corr, stds = pm.LKJCholeskyCov("chol", n=2, eta=2.0, sd_dist=sd_dist)

    # prior for average intercept:
    a = pm.Normal("a", mu=0.0, sigma=5.0)
    # prior for average slope:
    b = pm.Normal("b", mu=0.0, sigma=1.0)
    # population of varying effects:
    ab_county = pm.MvNormal("ab_county", mu=at.stack([a, b]), chol=chol, dims=("County", "param"))

    # Expected value per county:
    theta = ab_county[county_idx, 0] + ab_county[county_idx, 1] * floor_idx
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")
```

This is by far the most complex model we've done so far, so it's normal if you're confused. Just take some time to let it sink in. The centered version mirrors the mathematical notions very closely, so you should be able to get the gist of it. Of course, you guessed it, we're gonna need the non-centered version. There is actually just one change:

```{code-cell} ipython3
with pm.Model(coords=coords, rng_seeder=RANDOM_SEED) as covariation_intercept_slope:
    floor_idx = pm.Data("floor_idx", floor, dims="obs_id", mutable=True)
    county_idx = pm.Data("county_idx", county, dims="obs_id", mutable=True)

    # prior stddev in intercepts & slopes (variation across counties):
    sd_dist = pm.Exponential.dist(0.5, shape=(2,))

    # get back standard deviations and rho:
    chol, corr, stds = pm.LKJCholeskyCov("chol", n=2, eta=2, sd_dist=sd_dist)

    # prior for average intercept:
    a = pm.Normal("a", mu=0.0, sigma=5.0)
    # prior for average slope:
    b = pm.Normal("b", mu=0.0, sigma=1.0)
    # population of varying effects:
    z = pm.Normal("z", 0.0, 1.0, dims=("param", "County"))
    ab_county = pm.Deterministic("ab_county", at.dot(chol, z).T, dims=("County", "param"))

    # Expected value per county:
    theta = a + ab_county[county_idx, 0] + (b + ab_county[county_idx, 1]) * floor_idx
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")

    covariation_intercept_slope_trace = pm.sample(
        1000,
        tune=3000,
        target_accept=0.95,
        idata_kwargs={"dims": {"chol_stds": ["param"], "chol_corr": ["param", "param_bis"]}},
    )
```

```{code-cell} ipython3
az.plot_trace(
    covariation_intercept_slope_trace,
    var_names=["~z", "~chol", "~chol_corr"],
    compact=True,
    chain_prop={"ls": "-"},
);
```

```{code-cell} ipython3
az.plot_trace(
    covariation_intercept_slope_trace,
    var_names="chol_corr",
    lines=[("chol_corr", {}, 0.0)],
    compact=True,
    chain_prop={"ls": "-"},
    coords={
        "param": xr.DataArray(["a"], dims=["pointwise_sel"]),
        "param_bis": xr.DataArray(["b"], dims=["pointwise_sel"]),
    },
);
```

So the correlation between slopes and intercepts seems to be negative: when `a_county` increases, `b_county` tends to decrease. In other words, when basement radon in a county gets bigger, the difference with floor radon tends to get bigger too (because floor readings get smaller while basement readings get bigger). But again, the uncertainty is wide on `Rho` so it's possible the correlation goes the other way around or is simply close to zero. 

And how much variation is there across counties? It's not easy to read `sigma_ab` above, so let's do a forest plot and compare the estimates with the model that doesn't include the covariation between slopes and intercepts:

```{code-cell} ipython3
az.plot_forest(
    [varying_intercept_slope_trace, covariation_intercept_slope_trace],
    model_names=["No covariation", "With covariation"],
    var_names=["a", "b", "sigma_a", "sigma_b", "chol_stds", "chol_corr"],
    combined=True,
    figsize=(8, 6),
);
```

The estimates are very close to each other, both for the means and the standard deviations. But remember, the information given by `Rho` is only seen at the county level: in theory it uses even more information from the data to get an even more informed pooling of information for all county parameters. So let's visually compare estimates of both models at the county level:

```{code-cell} ipython3
# posterior means of covariation model:
a_county_cov = (
    covariation_intercept_slope_trace.posterior["a"]
    + covariation_intercept_slope_trace.posterior["ab_county"].sel(param="a")
).mean(dim=("chain", "draw"))
b_county_cov = (
    covariation_intercept_slope_trace.posterior["b"]
    + covariation_intercept_slope_trace.posterior["ab_county"].sel(param="b")
).mean(dim=("chain", "draw"))

# plot both and connect with lines
plt.scatter(avg_a_county, avg_b_county, label="No cov estimates", alpha=0.6)
plt.scatter(
    a_county_cov,
    b_county_cov,
    facecolors="none",
    edgecolors="k",
    lw=1,
    label="With cov estimates",
    alpha=0.8,
)
plt.plot([avg_a_county, a_county_cov], [avg_b_county, b_county_cov], "k-", alpha=0.5)
plt.xlabel("Intercept")
plt.ylabel("Slope")
plt.legend();
```

The negative correlation is somewhat clear here: when the intercept increases, the slope decreases. So we understand why the model put most of the posterior weight into negative territory for `Rho`. Nevertheless, the negativity isn't *that* obvious, which is why the model gives a non-trivial posterior probability to the possibility that `Rho` could in fact be zero or positive.

Interestingly, the differences between both models occur at extreme slope and intercept values. This is because the second model used the slightly negative correlation between intercepts and slopes to adjust their estimates: when intercepts are *larger* (smaller) than average, the model pushes *down* (up) the associated slopes.

Globally, there is a lot of agreement here: modeling the correlation didn’t change inference that much. We already saw that radon levels tended to be lower in floors than basements, and when we checked the posterior distributions of the average effects (`a` and `b`) and standard deviations, we noticed that they were almost identical. But on average the model with covariation will be more accurate -- because it squeezes additional information from the data, to shrink estimates in both dimensions.

+++

## Adding group-level predictors

A primary strength of multilevel models is the ability to handle predictors on multiple levels simultaneously. If we consider the varying-intercepts model above:

$$y_i = \alpha_{j[i]} + \beta x_{i} + \epsilon_i$$

we may, instead of a simple random effect to describe variation in the expected radon value, specify another regression model with a county-level covariate. Here, we use the county uranium reading $u_j$, which is thought to be related to radon levels:

$$\alpha_j = \gamma_0 + \gamma_1 u_j + \zeta_j$$

$$\zeta_j \sim N(0, \sigma_{\alpha}^2)$$

Thus, we are now incorporating a house-level predictor (floor or basement) as well as a county-level predictor (uranium).

Note that the model has both indicator variables for each county, plus a county-level covariate. In classical regression, this would result in collinearity. In a multilevel model, the partial pooling of the intercepts towards the expected value of the group-level linear model avoids this.

Group-level predictors also serve to reduce group-level variation, $\sigma_{\alpha}$ (here it would be the variation across counties, `sigma_a`). An important implication of this is that the group-level estimate induces stronger pooling -- by definition, a smaller $\sigma_{\alpha}$ means a stronger shrinkage of counties parameters towards the overall state mean. 

This is fairly straightforward to implement in PyMC -- we just add another level:

```{code-cell} ipython3
with pm.Model(coords=coords, rng_seeder=RANDOM_SEED) as hierarchical_intercept:
    floor_idx = pm.Data("floor_idx", floor, dims="obs_id", mutable=True)
    county_idx = pm.Data("county_idx", county, dims="obs_id", mutable=True)
    uranium = pm.Data("uranium", u, dims="County", mutable=True)

    # Hyperpriors:
    g = pm.Normal("g", mu=0.0, sigma=10.0, shape=2)
    sigma_a = pm.Exponential("sigma_a", 1.0)

    # Varying intercepts uranium model:
    a = g[0] + g[1] * uranium
    a_county = pm.Normal("a_county", mu=a, sigma=sigma_a, dims="County")
    # Common slope:
    b = pm.Normal("b", mu=0.0, sigma=1.0)

    # Expected value per county:
    theta = a_county[county_idx] + b * floor_idx
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")
pm.model_to_graphviz(hierarchical_intercept)
```

Do you see the new level, with `sigma_a` and `g`, which is two-dimensional because it contains the linear model for `a_county`? Now, if we run this model we're gonna get... divergences, you guessed it! So we're gonna switch to the non-centered form again:

```{code-cell} ipython3
with pm.Model(coords=coords, rng_seeder=RANDOM_SEED) as hierarchical_intercept:
    floor_idx = pm.Data("floor_idx", floor, dims="obs_id", mutable=True)
    county_idx = pm.Data("county_idx", county, dims="obs_id", mutable=True)
    uranium = pm.Data("uranium", u, dims="County", mutable=True)

    # Hyperpriors:
    g = pm.Normal("g", mu=0.0, sigma=10.0, shape=2)
    sigma_a = pm.Exponential("sigma_a", 1.0)

    # Varying intercepts uranium model:
    a = pm.Deterministic("a", g[0] + g[1] * uranium, dims="County")
    za_county = pm.Normal("za_county", mu=0.0, sigma=1.0, dims="County")
    a_county = pm.Deterministic("a_county", a + za_county * sigma_a, dims="County")
    # Common slope:
    b = pm.Normal("b", mu=0.0, sigma=1.0)

    # Expected value per county:
    theta = a_county[county_idx] + b * floor_idx
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")

    hierarchical_intercept_trace = pm.sample(1000, tune=2000, target_accept=0.99)
```

```{code-cell} ipython3
uranium = hierarchical_intercept_trace.constant_data.uranium
post = hierarchical_intercept_trace.posterior.assign_coords(uranium=uranium)
avg_a = post["a"].mean(dim=("chain", "draw")).sortby("uranium")
avg_a_county = post["a_county"].mean(dim=("chain", "draw"))
avg_a_county_hdi = az.hdi(post, var_names="a_county")["a_county"]

_, ax = plt.subplots()
ax.plot(avg_a.uranium, avg_a, "k--", alpha=0.6, label="Mean intercept")
az.plot_hdi(
    uranium,
    post["a"],
    fill_kwargs={"alpha": 0.1, "color": "k", "label": "Mean intercept HPD"},
    ax=ax,
)
ax.scatter(uranium, avg_a_county, alpha=0.8, label="Mean county-intercept")
ax.vlines(
    uranium,
    avg_a_county_hdi.sel(hdi="lower"),
    avg_a_county_hdi.sel(hdi="higher"),
    alpha=0.5,
    color="orange",
)
plt.xlabel("County-level uranium")
plt.ylabel("Intercept estimate")
plt.legend(fontsize=9);
```

Uranium is indeed much associated with baseline radon levels in each county. The graph above shows the average relationship and its uncertainty: the baseline radon level in an average county as a function of uranium, as well as the 94% HPD of this radon level (grey line and envelope). The blue points and orange bars represent the relationship between baseline radon and uranium, but now for each county. As you see, the uncertainty is bigger now, because it adds on top of the average uncertainty -- each county has its idyosyncracies after all.

If we compare the county-intercepts for this model with those of the partial-pooling model without a county-level covariate:

```{code-cell} ipython3
az.plot_forest(
    [varying_intercept_trace, hierarchical_intercept_trace],
    model_names=["W/t. county pred.", "With county pred."],
    var_names=["a_county"],
    combined=True,
    figsize=(6, 40),
    textsize=9,
);
```

We see that the compatibility intervals are narrower for the model including the county-level covariate. This is expected, as the effect of a covariate is to reduce the variation in the outcome variable -- provided the covariate is of predictive value. More importantly, with this model we were able to squeeze even more information out of the data.

+++

## Correlations among levels

In some instances, having predictors at multiple levels can reveal correlation between individual-level variables and group residuals. We can account for this by including the average of the individual predictors as a covariate in the model for the group intercept:

$$\alpha_j = \gamma_0 + \gamma_1 u_j + \gamma_2 \bar{x} + \zeta_j$$

These are broadly referred to as ***contextual effects***.

To add these effects to our model, let's create a new variable containing the mean of `floor` in each county and add that to our previous model:

```{code-cell} ipython3
avg_floor_data = srrs_mn.groupby("county")["floor"].mean().rename(county_lookup).values

with pm.Model(coords=coords, rng_seeder=RANDOM_SEED) as contextual_effect:
    floor_idx = pm.Data("floor_idx", floor, mutable=True)
    county_idx = pm.Data("county_idx", county, mutable=True)
    uranium = pm.Data("uranium", u, dims="County", mutable=True)
    avg_floor = pm.Data("avg_floor", avg_floor_data, dims="County", mutable=True)

    # Hyperpriors:
    g = pm.Normal("g", mu=0.0, sigma=10.0, shape=3)
    sigma_a = pm.Exponential("sigma_a", 1.0)

    # Varying intercepts uranium model:
    a = pm.Deterministic("a", g[0] + g[1] * u + g[2] * avg_floor, dims="County")
    za_county = pm.Normal("za_county", mu=0.0, sigma=1.0, dims="County")
    a_county = pm.Deterministic("a_county", a + za_county * sigma_a, dims="County")
    # Common slope:
    b = pm.Normal("b", mu=0.0, sigma=1.0)

    # Expected value per county:
    theta = a_county[county_idx] + b * floor_idx
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon)

    contextual_effect_trace = pm.sample(1000, tune=2000, target_accept=0.99)
```

```{code-cell} ipython3
az.summary(contextual_effect_trace, var_names=["g"], round_to=2)
```

So we might infer from this that counties with higher proportions of houses without basements tend to have higher baseline levels of radon. This seems to be new, as up to this point we saw that `floor` was *negatively* associated with radon levels. But remember this was at the household-level: radon tends to be higher in houses with basements. But at the county-level it seems that the less basements on average in the county, the more radon. So it's not that contradictory. What's more, the estimate for $\gamma_2$ is quite uncertain and overlaps with zero, so it's possible that the relationship is not that strong. And finally, let's note that $\gamma_2$ estimates something else than uranium's effect, as this is already taken into account by $\gamma_1$ -- it answers the question "once we know uranium level in the county, is there any value in learning about the proportion of houses without basements?".

All of this is to say that we shouldn't interpret this causally: there is no credible mecanism by which a basement (or absence thereof) *causes* radon emissions. More probably, our causal graph is missing something: a confounding variable, one that influences both basement construction and radon levels, is lurking somewhere in the dark... Perhaps is it the type of soil, which might influence what type of structures are built *and* the level of radon? Maybe adding this to our model would help with causal inference.

+++

## Prediction

Gelman (2006) used cross-validation tests to check the prediction error of the unpooled, pooled, and partially-pooled models

**root mean squared cross-validation prediction errors**:

* unpooled = 0.86
* pooled = 0.84
* multilevel = 0.79

There are two types of prediction that can be made in a multilevel model:

1. a new individual within an *existing* group
2. a new individual within a *new* group

The first type is the easiest one, as we've generally already sampled from the existing group. For this model, the first type of posterior prediction is the only one making sense, as counties are not added or deleted every day. So, if we wanted to make a prediction for, say, a new house with no basement in St. Louis and Kanabec counties, we just need to sample from the radon model with the appropriate intercept:

```{code-cell} ipython3
county_lookup["ST LOUIS"], county_lookup["KANABEC"]
```

That is, 

$$\tilde{y}_i \sim N(\alpha_{69} + \beta (x_i=1), \sigma_y^2)$$

Because we judiciously set the county index and floor values as shared variables earlier, we can modify them directly to the desired values (69 and 1 respectively) and sample corresponding posterior predictions, without having to redefine and recompile our model. Using the model just above:

```{code-cell} ipython3
prediction_coords = {"obs_id": ["ST LOUIS", "KANABEC"]}
with contextual_effect:
    pm.set_data({"county_idx": np.array([69, 31]), "floor_idx": np.array([1, 1])})
    contextual_effect_trace = pm.sample_posterior_predictive(
        contextual_effect_trace, predictions=True, extend_inferencedata=True
    );
```

```{code-cell} ipython3
az.plot_posterior(contextual_effect_trace, group="predictions");
```

## Benefits of Multilevel Models

- Accounting for natural hierarchical structure of observational data.

- Estimation of coefficients for (under-represented) groups.

- Incorporating individual- and group-level information when estimating group-level coefficients.

- Allowing for variation among individual-level coefficients across groups.

## References

:::{bibliography}
:filter: docname in docnames

gelman2006multilevel
mcelreath2018statistical
:::

+++

## Authors

* Authored by Chris Fonnesbeck in May, 2017 ([pymc#2124](https://github.com/pymc-devs/pymc/pull/2124))
* Updated by Colin Carroll in June, 2018 ([pymc#3049](https://github.com/pymc-devs/pymc/pull/3049))
* Updated by Alex Andorra in January, 2020 ([pymc#3765](https://github.com/pymc-devs/pymc/pull/3765))
* Updated by Oriol Abril in June, 2020 ([pymc#3963](https://github.com/pymc-devs/pymc/pull/3963))
* Updated by Farhan Reynaldo in November 2021 ([pymc-examples#246](https://github.com/pymc-devs/pymc-examples/pull/246))
* Updated by Chris Fonnesbeck in Februry 2022 ([pymc-examples#285](https://github.com/pymc-devs/pymc-examples/pull/285)

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w
```

:::{include} ../page_footer.md
:::
