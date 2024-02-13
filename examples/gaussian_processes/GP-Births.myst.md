---
jupytext:
  notebook_metadata_filter: substitutions
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
myst:
  substitutions:
    extra_dependencies: numpyro preliz scikit-learn seaborn tensorflow-probability
---

(gp-birthdays)=
# Baby Births Modelling with HSGP

:::{post} January, 2024
:tags: gaussian processes, hilbert space approximation,
:category: intermediate, how-to
:author: [Bill Engels](https://github.com/bwengals) and [Juan Orduz](https://juanitorduz.github.io/)
:::

+++

In this notebook we want to reproduce a simpler version of the classical example of using Gaussian processes to model time series data: The birthdays data set (I first encountered this example in the seminal book {cite:p}`gelman2013bayesian`  [Chapter 21] when learning about the subject). The objective of this example is to illustrate how to use the Hilbert Space Gaussian Process (HSGP) approximation method introduced in {cite:p}`solin2020Hilbert` to speed up models with Gaussian processes components.
The main idea of this method relies on the Laplacian's spectral decomposition to approximate kernels' spectral measures as a function of basis functions. The key observation is that the basis functions in the reduced-rank approximation do not depend on the hyperparameters of the covariance function for the Gaussian process. This allows us to speed up the computations tremendously.
We do not go into the mathematical details here as the original article is very well written and easy to follow (see also the great paper {cite:p}`riutort2022PracticalHilbertSpaceApproximate`). Instead, we reproduce a simplified version presented in various sources:

- {cite:p}`vehtari2022Birthdays` by [Aki Vehtari](https://users.aalto.fi/~ave/)
- {cite:p}`numpyroBirthdays`, from [NumPyro](https://num.pyro.ai/en/stable/) documentation, which is a great resource to learn about the method internals (so it is also strongly recommended!).

```{tip}
For a complete treatment of this example please refer to the **amazing guide**: {cite:p}`vehtari2022Birthdays` by [Aki Vehtari](https://users.aalto.fi/~ave/). This is a step-by-step to develop this model in Stan. All the code can be found on [this repository](https://github.com/avehtari/casestudies/tree/master/Birthdays).
```

```{note}
This notebook is based on the blog post {cite:p}`orduz2024Birthdays` where Juan presents a more complete model, similar to the one described in the references above.
```

+++

## Prepare Notebook

+++

:::{include} ../extra_installs.md
:::

```{code-cell} ipython3
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import preliz as pz
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns
import xarray as xr

from matplotlib.ticker import MaxNLocator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

warnings.filterwarnings("ignore")

az.style.use("arviz-darkgrid")
plt.rcParams["figure.figsize"] = [12, 7]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.facecolor"] = "white"

%load_ext autoreload
%autoreload 2
%config InlineBackend.figure_format = "retina"
```

```{code-cell} ipython3
seed: int = sum(map(ord, "birthdays"))
rng: np.random.Generator = np.random.default_rng(seed=seed)
```

## Read Data

We read the data from the repository [Bayesian workflow book - Birthdays](https://avehtari.github.io/casestudies/Birthdays/birthdays.html) by [Aki Vehtari](https://users.aalto.fi/~ave/).

```{code-cell} ipython3
raw_df = pd.read_csv(
    "https://raw.githubusercontent.com/avehtari/casestudies/master/Birthdays/data/births_usa_1969.csv",
)

raw_df.info()
```

The data set contains the number of births per day in USA in the period 1969-1988. All the columns are self-explanatory except for `day_of_year2` which is the day of the year (from 1 to 366) with leap day being 60 and 1st March 61 also on non-leap-years.

```{code-cell} ipython3
raw_df.head()
```

## EDA and Feature Engineering

+++

First, we look into the `births` distribution:

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.histplot(data=raw_df, x="births", kde=True, ax=ax)
ax.set_title(
    label="Number of Births in the USA in 1969 - 1988",
    fontsize=18,
    fontweight="bold",
)
```

We create a couple of features:
-  A `date`stamp.
-  `births_relative100`: the number of births relative to $100$.
-  `obs`: data index.

```{code-cell} ipython3
data_df = raw_df.copy().assign(
    date=lambda x: pd.to_datetime(x[["year", "month", "day"]]),
    births_relative100=lambda x: x["births"] / x["births"].mean() * 100,
    obs=lambda x: x.index,
)
```

Now, let's look into the development over time of the relative births, which is the target variable we will model.

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.scatterplot(data=data_df, x="date", y="births_relative100", c="C0", s=8, ax=ax)
ax.axhline(100, color="black", linestyle="--", label="mean level")
ax.legend()
ax.set(xlabel="date", ylabel="relative number of births")
ax.set_title(label="Relative Births in the USA in 1969 - 1988", fontsize=18, fontweight="bold")
```

We see a clear long term trend component and a clear yearly seasonality. The plot above has many many data points and we want to make sure we understand seasonal patters at different levels (which might be hidden in the plot above). Hence, we systematically check seasonality at various levels.

Let's continue looking by averaging over the day of the year:

```{code-cell} ipython3
fig, ax = plt.subplots()
(
    data_df.groupby(["day_of_year2"], as_index=False)
    .agg(meanbirths=("births_relative100", "mean"))
    .pipe((sns.scatterplot, "data"), x="day_of_year2", y="meanbirths", c="C0", ax=ax)
)
ax.axhline(100, color="black", linestyle="--", label="mean level")
ax.legend()
ax.set(xlabel="day of year", ylabel="relative number of births per day of year")
ax.set_title(
    label="Relative Births in the USA in 1969 - 1988\nMean over Day of Year",
    fontsize=18,
    fontweight="bold",
)
```

Overall, we see a relatively smooth behavior with the exception of certain holidays (memorial day, thanks giving and labor day) and the new year's day.

+++

Next, we split by month and year to see if there if we spot any changes in the pattern over time.

```{code-cell} ipython3
fig, ax = plt.subplots()
(
    data_df.groupby(["year", "month"], as_index=False)
    .agg(meanbirths=("births_relative100", "mean"))
    .assign(month=lambda x: pd.Categorical(x["month"]))
    .pipe(
        (sns.lineplot, "data"),
        x="year",
        y="meanbirths",
        marker="o",
        markersize=7,
        hue="month",
        palette="tab20",
    )
)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.legend(title="month", loc="center left", bbox_to_anchor=(1, 0.5))
ax.set(xlabel="year", ylabel="relative number of births")
ax.set_title(
    label="Relative Births in the USA in 1969 - 1988\nMean over Month and Year",
    fontsize=18,
    fontweight="bold",
)
```

Besides the global trend, we do not see any clear differences between months.

We continue looking into the weekly seasonality.

```{code-cell} ipython3
fig, ax = plt.subplots()
(
    sns.lineplot(
        data=data_df,
        x="day_of_week",
        y="births_relative100",
        marker="o",
        c="C0",
        markersize=10,
        ax=ax,
    )
)
ax.axhline(100, color="black", linestyle="--", label="mean level")
ax.legend()
ax.set(xlabel="day of week", ylabel="relative number of births per day of week")
ax.set_title(
    label="Relative Births in the USA in 1969 - 1988\nMean over Day of Week",
    fontsize=18,
    fontweight="bold",
)
```

It seems that there are on average less births during the weekend.

We can also plot the time development over the years.

```{code-cell} ipython3
fig, ax = plt.subplots()
(
    data_df.assign(day_name=lambda x: x["date"].dt.day_name())
    .groupby(["year", "day_name"], as_index=False)
    .agg(meanbirths=("births_relative100", "mean"))
    .pipe(
        (sns.lineplot, "data"),
        x="year",
        y="meanbirths",
        marker="o",
        markersize=7,
        hue="day_name",
    )
)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.legend(title="day of week", loc="center left", bbox_to_anchor=(1, 0.5))
ax.set(xlabel="year", ylabel="relative number of births per day of week")
ax.set_title(
    label="Relative Births in the USA in 1969 - 1988\nMean over Day of Week and Year",
    fontsize=18,
    fontweight="bold",
)
```

We see that the trends behave differently over the years for weekdays and weekends.

+++

```{tip}
Let's summarize the main findings of the EDA:
-  There is a clear non-linear long term trend.
-  There is a clear smooth yearly seasonality up to some special holidays and the end of the year drop.
-  There is a clear weekly seasonality.
-  There are differences in the trends over the years for weekdays and weekends.
```

+++

## Data Pre-Processing

After having a better understanding of the data and the patters we want to capture with the model, we can proceed to pre-process the data.

+++

- Extract relevant features

```{code-cell} ipython3
n = data_df.shape[0]
obs = data_df["obs"].to_numpy()
date = data_df["date"].to_numpy()
year = data_df["year"].to_numpy()
day_of_week_idx, day_of_week = data_df["day_of_week"].factorize(sort=True)
day_of_week_no_monday = day_of_week[day_of_week != 1]
day_of_year2_idx, day_of_year2 = data_df["day_of_year2"].factorize(sort=True)
births_relative100 = data_df["births_relative100"].to_numpy()
```

```{code-cell} ipython3
data_df.head(10)
```

We want to work on the normalized log scale of the relative births. The reason for this is to work on a scale where is easier to set up priors (scaled space) and so that the heteroscedasticity is reduced (log transform).

```{code-cell} ipython3
# we want to use the scale of the data size to set up the priors.
# we are mainly interested in the standard deviation.
obs_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
obs_pipeline.fit(obs.reshape(-1, 1))
normalized_obs = obs_pipeline.transform(obs.reshape(-1, 1)).flatten()
obs_std = obs_pipeline["scaler"].scale_.item()

# we first take a log transform and then normalize the data.
births_relative100_pipeline = Pipeline(
    steps=[
        ("log", FunctionTransformer(func=np.log, inverse_func=np.exp)),
        ("scaler", StandardScaler()),
    ]
)
births_relative100_pipeline.fit(births_relative100.reshape(-1, 1))
normalized_log_births_relative100 = births_relative100_pipeline.transform(
    births_relative100.reshape(-1, 1)
).flatten()
normalized_log_births_relative100_std = births_relative100_pipeline["scaler"].scale_.item()
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(normalized_obs, normalized_log_births_relative100, "o", c="C0", markersize=2)
ax.set(xlabel="normalized obs", ylabel="relative number of births - Transformed")
ax.set_title(
    label="Relative Births in the USA in 1969 - 1988\nTransformed Data",
    fontsize=18,
    fontweight="bold",
)
```

## Model Specification

### Model Components

Let's describe the model components. All of these building blocks should not come as a surprise after looking into the EDA section.

1. **Global trend.** We use a Gaussian process with an exponential quadratic kernel.
2. **Periodicity over years**: We use a Gaussian process with a periodic kernel. Observe that, since we are working on the normalized scale, the period should be `period=365.25 / obs_std` (and not `period=365.25` !).
3. **Weekly seasonality**: We use a normal distribution on the day of the week one-hot-encoded values. As the data is standardized, in particular centered around zero, we do not need to add an intercept term. In addition, we set the coefficient of Monday to zero to avoid identifiability issues.
4. **Likelihood**: We use a Gaussian distribution.

For all of the Gaussian processes components we use the Hilbert Space Gaussian Process (HSGP) approximation.

+++

```{note}
This model corresponds to `Model 3: Slow trend + yearly seasonal trend + day of week` in {cite:p}`vehtari2022Birthdays`.
```

+++

### Prior Specifications

Most of the priors are not very informative. The only tricky part here is to think that we are working on the normalized log scale of the relative births data. For example, for the global trend we use a Gaussian process with an exponential quadratic kernel. We use the following priors for the length scale:

```{code-cell} ipython3
fig, ax = plt.subplots()
pz.LogNormal(mu=np.log(700 / obs_std), sigma=1).plot_pdf(ax=ax)
ax.set(xlim=(None, 4))
ax.set_title(
    label="Prior distribution for the global trend Gaussian process",
    fontsize=18,
    fontweight="bold",
)
```

The motivation is that we have around $7.3$K data points and whe want to consider the in between data points distance in the normalized scale. That is why we consider the ratio `7_000 / obs_str`. Note that we want to capture the long term trend, so we want to consider a length scale that is larger than the data points distance. We increase the order of magnitude by dividing by $10$. We then take a log transform as we are using a log-normal prior.

+++

### Model Implementation

We now specify the model in PyMC.

```{code-cell} ipython3
coords = {
    "obs": obs,
    "day_of_week_no_monday": day_of_week_no_monday,
    "day_of_week": day_of_week,
    "day_of_year2": day_of_year2,
}

with pm.Model(coords=coords) as model:
    # --- Data Containers ---

    normalized_obs_data = pm.Data(
        name="normalized_obs_data", value=normalized_obs, mutable=False, dims="obs"
    )

    day_of_week_idx_data = pm.Data(
        name="day_of_week_idx_data", value=day_of_week_idx, mutable=False, dims="obs"
    )
    normalized_log_births_relative100_data = pm.Data(
        name="log_births_relative100",
        value=normalized_log_births_relative100,
        mutable=False,
        dims="obs",
    )

    # --- Priors ---

    # global trend
    amplitude_trend = pm.HalfNormal(name="amplitude_trend", sigma=1.0)
    ls_trend = pm.LogNormal(name="ls_trend", mu=np.log(700 / obs_std), sigma=1)
    cov_trend = amplitude_trend * pm.gp.cov.ExpQuad(input_dim=1, ls=ls_trend)
    gp_trend = pm.gp.HSGP(m=[20], c=1.5, cov_func=cov_trend)
    f_trend = gp_trend.prior(name="f_trend", X=normalized_obs_data[:, None], dims="obs")

    ## year periodic
    amplitude_year_periodic = pm.HalfNormal(name="amplitude_year_periodic", sigma=1)
    ls_year_periodic = pm.LogNormal(name="ls_year_periodic", mu=np.log(7_000 / obs_std), sigma=1)
    gp_year_periodic = pm.gp.HSGPPeriodic(
        m=20,
        scale=amplitude_year_periodic,
        cov_func=pm.gp.cov.Periodic(input_dim=1, period=365.25 / obs_std, ls=ls_year_periodic),
    )
    f_year_periodic = gp_year_periodic.prior(
        name="f_year_periodic", X=normalized_obs_data[:, None], dims="obs"
    )

    ## day of week
    b_day_of_week_no_monday = pm.Normal(
        name="b_day_of_week_no_monday", sigma=1, dims="day_of_week_no_monday"
    )

    b_day_of_week = pm.Deterministic(
        name="b_day_of_week",
        var=pt.concatenate(([0], b_day_of_week_no_monday)),
        dims="day_of_week",
    )

    # global noise
    sigma = pm.HalfNormal(name="sigma", sigma=0.5)

    # --- Parametrization ---
    mu = pm.Deterministic(
        name="mu",
        var=f_trend
        + f_year_periodic
        + b_day_of_week[day_of_week_idx_data] * (day_of_week_idx_data > 0),
        dims="obs",
    )

    # --- Likelihood ---
    pm.Normal(
        name="likelihood",
        mu=mu,
        sigma=sigma,
        observed=normalized_log_births_relative100_data,
        dims="obs",
    )

pm.model_to_graphviz(model=model)
```

```{attention}
The first two basis vectors for the (periodic) {class}`~pymc.gp.HSGP` sometimes come out to be either all ones or all zeros. This is a problem because it brings an extra intercept in the model and this can hurt sampling. To avoid this, you can use the `drop_first` argument in the {class}`~pymc.gp.HSGP` class.
```

+++

## Prior Predictive Checks

We run the model with the prior predictive checks to see if the model is able to generate data in a similar scale as the data.

```{code-cell} ipython3
with model:
    prior_predictive = pm.sample_prior_predictive(samples=2_000, random_seed=rng)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
az.plot_ppc(data=prior_predictive, group="prior", kind="kde", ax=ax)
ax.set_title(label="Prior Predictive", fontsize=18, fontweight="bold")
```

It looks very reasonable as the prior samples are withing a reasonable range of the observed data.

+++

## Model Fitting and Diagnostics

We now proceed to fit the model using the `Numpyro` sampler. It takes around $5$ minutes to run the model locally (Intel MacBook Pro, $4$ cores, $16$ GB RAM).

```{code-cell} ipython3
with model:
    idata = pm.sample(
        target_accept=0.9,
        draws=2_000,
        chains=4,
        nuts_sampler="numpyro",
        random_seed=rng,
    )
    posterior_predictive = pm.sample_posterior_predictive(trace=idata, random_seed=rng)
```

## Diagnostics

We do not see any divergences or very high r-hat values:

```{code-cell} ipython3
idata["sample_stats"]["diverging"].sum().item()
```

```{code-cell} ipython3
var_names = [
    "amplitude_trend",
    "ls_trend",
    "amplitude_year_periodic",
    "ls_year_periodic",
    "b_day_of_week_no_monday",
    "sigma",
]

az.summary(data=idata, var_names=var_names, round_to=3)
```

We can also look into the trace plots.

```{code-cell} ipython3
axes = az.plot_trace(
    data=idata,
    var_names=var_names,
    compact=True,
    backend_kwargs={"figsize": (15, 12), "layout": "constrained"},
)
plt.gcf().suptitle("Trace", fontsize=16)
```

```{note}
Observe we get the same results as in `Model 3: Slow trend + yearly seasonal trend + day of week` in blog post {cite:p}`vehtari2022Birthdays`.
```

+++

## Posterior Distribution Analysis

Now we want to do a deep dive into the posterior distribution of the model and its components. We want to do this in the original scale. Therefore the first step is to transform the posterior samples back to the original scale.

+++

- Model Components

```{code-cell} ipython3
pp_vars_original_scale = {
    var_name: xr.apply_ufunc(
        births_relative100_pipeline.inverse_transform,
        idata["posterior"][var_name].expand_dims(dim={"_": 1}, axis=-1),
        input_core_dims=[["obs", "_"]],
        output_core_dims=[["obs", "_"]],
        vectorize=True,
    ).squeeze(dim="_")
    for var_name in ["f_trend", "f_year_periodic"]
}
```

- Likelihood

```{code-cell} ipython3
pp_likelihood_original_scale = xr.apply_ufunc(
    births_relative100_pipeline.inverse_transform,
    posterior_predictive["posterior_predictive"]["likelihood"].expand_dims(dim={"_": 1}, axis=-1),
    input_core_dims=[["obs", "_"]],
    output_core_dims=[["obs", "_"]],
    vectorize=True,
).squeeze(dim="_")
```

We start by plotting the likelihood.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(15, 9))
sns.scatterplot(data=data_df, x="date", y="births_relative100", c="C0", s=8, label="data", ax=ax)
ax.axhline(100, color="black", linestyle="--", label="mean level")
az.plot_hdi(
    x=date,
    y=pp_likelihood_original_scale,
    hdi_prob=0.94,
    color="C1",
    fill_kwargs={"alpha": 0.2, "label": r"likelihood $94\%$ HDI"},
    smooth=False,
    ax=ax,
)
az.plot_hdi(
    x=date,
    y=pp_likelihood_original_scale,
    hdi_prob=0.5,
    color="C1",
    fill_kwargs={"alpha": 0.6, "label": r"likelihood $50\%$ HDI"},
    smooth=False,
    ax=ax,
)

ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=4)
ax.set(xlabel="date", ylabel="relative number of births")
ax.set_title(
    label="""Relative Births in the USA in 1969 - 1988
    Posterior Predictive (Likelihood)""",
    fontsize=18,
    fontweight="bold",
)
```

It looks that we are capturing the global variation. Let’s look into the posterior distribution plot to get a better understanding of the model.

```{code-cell} ipython3
fig, ax = plt.subplots()
az.plot_ppc(
    data=posterior_predictive,
    num_pp_samples=1_000,
    observed_rug=True,
    random_seed=seed,
    ax=ax,
)
ax.set_title(label="Posterior Predictive", fontsize=18, fontweight="bold")
```

This does not seem very good as there is a pretty big discrepancy between black line and shaded blue in the bulk of posterior, tails look good. This suggests we might be missing some covariates. We explore this in a latter more complex model.

+++

To get a better understanding of the model fit, we need to look into the individual components.

+++

## Model Components

+++

- Global Trend

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(15, 9))
sns.scatterplot(data=data_df, x="date", y="births_relative100", c="C0", s=8, label="data", ax=ax)
ax.axhline(100, color="black", linestyle="--", label="mean level")
az.plot_hdi(
    x=date,
    y=pp_vars_original_scale["f_trend"],
    hdi_prob=0.94,
    color="C3",
    fill_kwargs={"alpha": 0.2, "label": r"$f_\text{trend}$ $94\%$ HDI"},
    smooth=False,
    ax=ax,
)
az.plot_hdi(
    x=date,
    y=pp_vars_original_scale["f_trend"],
    hdi_prob=0.5,
    color="C3",
    fill_kwargs={"alpha": 0.6, "label": r"$f_\text{trend}$ $50\%$ HDI"},
    smooth=False,
    ax=ax,
)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=4)
ax.set(xlabel="date", ylabel="relative number of births")
ax.set_title(
    label="""Relative Births in the USA in 1969-1988
    Posterior Predictive (Global Trend)""",
    fontsize=18,
    fontweight="bold",
)
```

- Yearly Periodicity

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(15, 9))
sns.scatterplot(data=data_df, x="date", y="births_relative100", c="C0", s=8, label="data", ax=ax)
ax.axhline(100, color="black", linestyle="--", label="mean level")
az.plot_hdi(
    x=date,
    y=pp_vars_original_scale["f_year_periodic"],
    hdi_prob=0.94,
    color="C4",
    fill_kwargs={"alpha": 0.2, "label": r"$f_\text{yearly periodic}$ $94\%$ HDI"},
    smooth=False,
    ax=ax,
)
az.plot_hdi(
    x=date,
    y=pp_vars_original_scale["f_year_periodic"],
    hdi_prob=0.5,
    color="C4",
    fill_kwargs={"alpha": 0.6, "label": r"$f_\text{yearly periodic}$ $50\%$ HDI"},
    smooth=False,
    ax=ax,
)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=4)
ax.set(xlabel="date", ylabel="relative number of births")
ax.set_title(
    label="Relative Births in the USA in 1969\nPosterior Predictive (Periodic Yearly)",
    fontsize=18,
    fontweight="bold",
)
```

- Global Trend plus Yearly Periodicity

+++

If we want to combine the global trend and the yearly periodicity, we can't simply sum the components in the original scale as we would be adding the mean term twice. Instead we need to first sum the posterior samples and then take the inverse transform (these operation do not commute!).

```{code-cell} ipython3
pp_vars_original_scale["f_trend_periodic"] = xr.apply_ufunc(
    births_relative100_pipeline.inverse_transform,
    (idata["posterior"]["f_trend"] + idata["posterior"]["f_year_periodic"]).expand_dims(
        dim={"_": 1}, axis=-1
    ),
    input_core_dims=[["obs", "_"]],
    output_core_dims=[["obs", "_"]],
    vectorize=True,
).squeeze(dim="_")
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(15, 9))
sns.scatterplot(data=data_df, x="date", y="births_relative100", c="C0", s=8, label="data", ax=ax)
ax.axhline(100, color="black", linestyle="--", label="mean level")
az.plot_hdi(
    x=date,
    y=pp_vars_original_scale["f_trend_periodic"],
    hdi_prob=0.94,
    color="C3",
    fill_kwargs={"alpha": 0.2, "label": r"$f_\text{trend + periodic}$ $94\%$ HDI"},
    smooth=False,
    ax=ax,
)
az.plot_hdi(
    x=date,
    y=pp_vars_original_scale["f_trend_periodic"],
    hdi_prob=0.5,
    color="C3",
    fill_kwargs={"alpha": 0.6, "label": r"$f_\text{trend  + periodic}$ $50\%$ HDI"},
    smooth=False,
    ax=ax,
)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=4)
ax.set(xlabel="date", ylabel="relative number of births")
ax.set_title(
    label="""Relative Births in the USA in 1969-1988
    Posterior Predictive (Global Trend + Periodic Yearly)""",
    fontsize=18,
    fontweight="bold",
)
```

## Authors
- Authored by [Juan Orduz](https://juanitorduz.github.io/) in January 2024 

+++

## References
:::{bibliography}
:filter: docname in docnames
:::

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor
```

:::{include} ../page_footer.md
:::
