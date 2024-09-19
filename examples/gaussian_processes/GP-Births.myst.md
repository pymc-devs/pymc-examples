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
# Baby Births Modelling with HSGPs

:::{post} January, 2024
:tags: gaussian processes, hilbert space approximation,
:category: intermediate, how-to
:author: [Juan Orduz](https://juanitorduz.github.io/)
:::

+++

This notebook provides an example of using the Hilbert Space Gaussian Process (HSGP) technique, introduced in {cite:p}`solin2020Hilbert`, in the context of time series modeling. This technique has proven successful in speeding up models with Gaussian process components.

To illustrate the main concepts, we use the classic *birthdays* example dataset (see {cite:p}`gelman2013bayesian`  [Chapter 21] and [here](https://statmodeling.stat.columbia.edu/2020/10/25/birthday-data/) for a comment on the data source) and reproduce one of the models presented in the excellent case study {cite:p}`vehtari2022Birthdays` by [Aki Vehtari](https://users.aalto.fi/~ave/) (you can find the Stan code on [this repository](https://github.com/avehtari/casestudies/tree/master/Birthdays)). In his exposition, the author presents an extensive iterative approach to analyze the relative number of births per day in the USA from 1969-1988 using HSGPs for various components:  the long-term trend, seasonal, weekly, day the year, and special floating day variation. As this resource is very detailed and gives many relevant explanations, we do not reproduce the whole process but instead focus on reproducing one of the intermediate models. Namely, the model with a slow trend, yearly seasonal trend, and day-of-week components (Model 3 in the original case study). The reason for reproducing a simpler model than the final one is to make this an introductory notebook for users willing to learn about this technique. We will provide a subsequent example where we implement the final model with all components.

In this notebook, we do not want to deep-dive into the mathematical details but rather focus on the implementation and how to use PyMC's {class}`~pymc.gp.HSGP` and {class}`~pymc.gp.HSGPPeriodic` API. This class provides a convenient way of using HSGPs in PyMC models. The user needs to input certain parameters to control the number of terms in the approximation and the domain of definition. Of course, understanding what these parameters do is important, so let's briefly touch upon the main idea of the approximation and the most relevant parameters:

## The main idea of the approximation

Recall that a *kernel* (associated with a covariance function) is the main ingredient of a Gaussian process as it encodes a measure of similarity (and smoothness) between points (see {ref}`GP-MeansAndCovs`). The Hilbert space approximation idea is to decompose such kernel as a linear combination of an orthonormal basis so that when replacing the kernel by this expansion, we can fit a linear model in terms of these basis functions. Sampling from a truncated expansion will be much faster than the vanilla Gaussian process formulation. The key observation is that the basis functions in the approximation do not depend on the hyperparameters of the covariance function for the Gaussian process, allowing the computations to speed up tremendously.

Where does the Hilbert Space come from? It turns out that the orthonormal basis comes from the spectral decomposition of the Laplace operator on a compact set (think about the Fourier decomposition on the circle, for example). In other words, the basis functions are eigenvectors of the Laplace operator on the square-integrable-functions space $L^{2}([-L, L])$, which is a Hilbert Space. Returning to the class {class}`~pymc.gp.HSGP`, the two most important parameters are:

- **$m$:** The number of basis vectors to use in the approximation.
- **$L$**: The boundary of the space of definition. Choose L such that the domain $[-L, L]$ contains all points in the domain. (Note that the compact set is the closed interval $[-L, L]$ 😉)

One can also use a *proportion extension factor* $c > 0$ used to construct $L$ from the domain of definition of the Gaussian process $X$. Concretely, $L$ can be specified as the product $cS$, where $S = \max|X|$.

We recommend the paper {cite:p}`riutort2022PracticalHilbertSpaceApproximate` for a practical discussion of this technique.

```{note}
You can find a similar example in [`Numpyro`](https://num.pyro.ai/en/stable/)'s documentation: {cite:p}`numpyroBirthdays`. This example is a great resource to learn about the method internals.
```

```{note}
This notebook is based on the blog post {cite:p}`orduz2024Birthdays`.
```

---

+++

:::{include} ../extra_installs.md
:::

```{code-cell} ipython3
:tags: [hide-input]

import warnings

from collections.abc import Callable

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
:tags: [hide-input]

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
);
```

We create a couple of features:
-  A `date`stamp.
-  `births_relative100`: the number of births relative to $100$.
-  `time`: data index.

```{code-cell} ipython3
data_df = raw_df.copy().assign(
    date=lambda x: pd.to_datetime(x[["year", "month", "day"]]),
    births_relative100=lambda x: x["births"] / x["births"].mean() * 100,
    time=lambda x: x.index,
)
```

```{note}
We scale the data to be as close as possible to Aki's case study. We do not need to scale the data for the HSGP model to work.
```

+++

Now, let's look into the development over time of the relative births, which is the target variable we will model.

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.scatterplot(data=data_df, x="date", y="births_relative100", c="C0", s=8, ax=ax)
ax.axhline(100, color="black", linestyle="--", label="mean level")
ax.legend()
ax.set(xlabel="date", ylabel="relative number of births")
ax.set_title(label="Relative Births in the USA in 1969 - 1988", fontsize=18, fontweight="bold");
```

We see a clear long term trend component and a clear yearly seasonality. We also see how the variance grows with time, this is known as [heteroscedasticity](https://en.wikipedia.org/wiki/Homoscedasticity_and_heteroscedasticity).

The plot above has many many data points and we want to make sure we understand seasonal patterns at different levels (which might be hidden in the plot above). Hence, we systematically check seasonality at various levels.

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
);
```

Overall, we see a relatively smooth behavior with the exception of certain holidays (Memorial Day, Thanksgiving and Labor Day) and the new year's day.

+++

Next, we split by month and year to see if we spot any changes in the pattern over time.

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
);
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
);
```

It seems that there are on average less births during the weekend.

+++

`````{admonition} EDA Summary
:class: tip
Let's summarize the main findings of the EDA:
-  There is a clear non-linear long term trend.
-  There is a clear smooth yearly seasonality up to some special holidays and the end of the year drop.
-  There is a clear weekly seasonality.
``````

+++

## Data Pre-Processing

After having a better understanding of the data and the patterns we want to capture with the model, we can proceed to pre-process the data.

+++

- Extract relevant features

```{code-cell} ipython3
n = data_df.shape[0]
time = data_df["time"].to_numpy()
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

We want to work on the normalized log scale of the relative births. The reason for this is to work on a scale where it is easier to set up priors (scaled space) and so that the heteroscedasticity is reduced (log transform).

```{code-cell} ipython3
# we want to use the scale of the data size to set up the priors.
# we are mainly interested in the standard deviation.
time_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
time_pipeline.fit(time.reshape(-1, 1))
normalized_time = time_pipeline.transform(time.reshape(-1, 1)).flatten()
time_std = time_pipeline["scaler"].scale_.item()

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
ax.plot(normalized_time, normalized_log_births_relative100, "o", c="C0", markersize=2)
ax.set(xlabel="normalized time", ylabel="relative number of births - Transformed")
ax.set_title(
    label="Relative Births in the USA in 1969 - 1988\nTransformed Data",
    fontsize=18,
    fontweight="bold",
);
```

## Model Specification

### Model Components

In this example notebook, we implement the `Model 3: Slow trend + yearly seasonal trend + day of the week` described in {cite:p}`vehtari2022Birthdays`. The EDA above should help us understand the motivation behind each of the following components of the model:

1. **Global trend.** We use a Gaussian process with an exponential quadratic kernel.
2. **Periodicity over years**: We use a Gaussian process with a periodic kernel. Observe that, since we are working on the normalized scale, the period should be `period=365.25 / time_std` (and not `period=365.25` !).
3. **Weekly seasonality**: We use a normal distribution on the day of the week one-hot-encoded values. As the data is standardized, in particular centered around zero, we do not need to add an intercept term. In addition, we set the coefficient of Monday to zero to avoid identifiability issues.
4. **Likelihood**: We use a Gaussian distribution.

For all of the Gaussian processes components we use the Hilbert Space Gaussian Process (HSGP) approximation.

+++

### Prior Specifications

Most of the priors are not very informative. The only tricky part here is to think that we are working on the normalized log scale of the relative births data. For example, for the global trend we use a Gaussian process with an exponential quadratic kernel. We use the following priors for the length scale:

```{code-cell} ipython3
fig, ax = plt.subplots()
pz.LogNormal(mu=np.log(700 / time_std), sigma=1).plot_pdf(ax=ax)
ax.set(xlim=(None, 4))
ax.set_title(
    label="Prior distribution for the global trend Gaussian process",
    fontsize=18,
    fontweight="bold",
);
```

The motivation is that we have around $7.3$K data points and we want to consider the in between data points distance in the normalized scale. That is why we consider the ratio `7_000 / time_str`. Note that we want to capture the long term trend, so we want to consider a length scale that is larger than the data points distance. We increase the order of magnitude by dividing by $10$. Finally, since a {class}`~pymc.distributions.continuous.LogNormal` distribution has positive support and a common choice for length scales, we take a log-transform on the resulting quantity `700 / time_str` so ensure the mean of the prior is close to this value.

+++

### Model Implementation

We now specify the model in PyMC.

```{code-cell} ipython3
coords = {
    "time": time,
    "day_of_week_no_monday": day_of_week_no_monday,
    "day_of_week": day_of_week,
    "day_of_year2": day_of_year2,
}

with pm.Model(coords=coords) as model:
    # --- Data Containers ---

    normalized_time_data = pm.Data(
        name="normalized_time_data", value=normalized_time, mutable=False, dims="time"
    )

    day_of_week_idx_data = pm.Data(
        name="day_of_week_idx_data", value=day_of_week_idx, mutable=False, dims="time"
    )
    normalized_log_births_relative100_data = pm.Data(
        name="log_births_relative100",
        value=normalized_log_births_relative100,
        mutable=False,
        dims="time",
    )

    # --- Priors ---

    # global trend
    amplitude_trend = pm.HalfNormal(name="amplitude_trend", sigma=1.0)
    ls_trend = pm.LogNormal(name="ls_trend", mu=np.log(700 / time_std), sigma=1)
    cov_trend = amplitude_trend * pm.gp.cov.ExpQuad(input_dim=1, ls=ls_trend)
    gp_trend = pm.gp.HSGP(m=[20], c=1.5, cov_func=cov_trend)
    f_trend = gp_trend.prior(name="f_trend", X=normalized_time_data[:, None], dims="time")

    ## year periodic
    amplitude_year_periodic = pm.HalfNormal(name="amplitude_year_periodic", sigma=1)
    ls_year_periodic = pm.LogNormal(name="ls_year_periodic", mu=np.log(7_000 / time_std), sigma=1)
    gp_year_periodic = pm.gp.HSGPPeriodic(
        m=20,
        scale=amplitude_year_periodic,
        cov_func=pm.gp.cov.Periodic(input_dim=1, period=365.25 / time_std, ls=ls_year_periodic),
    )
    f_year_periodic = gp_year_periodic.prior(
        name="f_year_periodic", X=normalized_time_data[:, None], dims="time"
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
        dims="time",
    )

    # --- Likelihood ---
    pm.Normal(
        name="likelihood",
        mu=mu,
        sigma=sigma,
        observed=normalized_log_births_relative100_data,
        dims="time",
    )

pm.model_to_graphviz(model=model)
```

```{tip}
There is an alternative parametrization of the day of week as described in {cite:p}`orduz2024Birthdays`. We can use a {class}`~pymc.ZeroSumNormal` distribution to parametrize via relative difference across weekdays. We would simply replace the prior `b_day_of_week` as:

```python
b_day_of_week = pm.ZeroSumNormal(name="b_day_of_week", sigma=1, dims="day_of_week")
```

+++

```{attention}
The first two basis vectors for the (periodic) {class}`~pymc.gp.HSGP` sometimes come out to be either all ones or all zeros. In general, when there is an intercept term in the model (in this example is not the case), this is a problem because it brings an extra intercept in the model and this can hurt sampling. To avoid this, you can use the `drop_first` argument in the {class}`~pymc.gp.HSGP` class.
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
ax.set_title(label="Prior Predictive", fontsize=18, fontweight="bold");
```

It looks very reasonable as the prior samples are within a reasonable range of the observed data.

+++

## Model Fitting and Diagnostics

We now proceed to fit the model using the `NumPyro` sampler. It takes around $5$ minutes to run the model locally (Intel MacBook Pro, $4$ cores, $16$ GB RAM).

```{code-cell} ipython3
with model:
    idata = pm.sample(
        target_accept=0.9,
        draws=2_000,
        chains=4,
        nuts_sampler="numpyro",
        random_seed=rng,
    )
    idata.extend(pm.sample_posterior_predictive(trace=idata, random_seed=rng))
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
plt.gcf().suptitle("Model Trace", fontsize=18, fontweight="bold");
```

```{note}
Observe we get the same results as in `Model 3: Slow trend + yearly seasonal trend + day of week` in blog post {cite:p}`vehtari2022Birthdays`.
```

+++

## Posterior Distribution Analysis

Now we want to do a deep dive into the posterior distribution of the model and its components. We want to do this in the original scale. Therefore the first step is to transform the posterior samples back to the original scale. For that purpose we use the following utility function (the code is not important).

```{code-cell} ipython3
:tags: [hide-input]

def apply_fn_along_dims(fn: Callable, a: xr.DataArray, dim: str) -> xr.DataArray:
    """Apply a function along a specific dimension.

    We need to expand the dimensions of the input array to make it compatible with the
    function which we assume acts on a matrix.
    """
    return xr.apply_ufunc(
        fn,
        a.expand_dims(
            dim={"_": 1}, axis=-1
        ),  # The auxiliary dimension `_` is used to broadcast the function.
        input_core_dims=[[dim, "_"]],
        output_core_dims=[[dim, "_"]],
        vectorize=True,
    ).squeeze(dim="_")
```

- Model Components

```{code-cell} ipython3
pp_vars_original_scale = {
    var_name: apply_fn_along_dims(
        fn=births_relative100_pipeline.inverse_transform,
        a=idata["posterior"][var_name],
        dim="time",
    )
    for var_name in ["f_trend", "f_year_periodic"]
}
```

- Likelihood

```{code-cell} ipython3
pp_likelihood_original_scale = apply_fn_along_dims(
    fn=births_relative100_pipeline.inverse_transform,
    a=idata["posterior_predictive"]["likelihood"],
    dim="time",
)
```

We start by plotting the likelihood.

```{code-cell} ipython3
:tags: [hide-input]

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
);
```

It looks that we are capturing the global variation. Let’s look into the posterior distribution plot to get a better understanding of the model.

```{code-cell} ipython3
fig, ax = plt.subplots()
az.plot_ppc(
    data=idata,
    num_pp_samples=1_000,
    observed_rug=True,
    random_seed=seed,
    ax=ax,
)
ax.set_title(label="Posterior Predictive", fontsize=18, fontweight="bold");
```

This does not seem very good as there is a pretty big discrepancy between black line and shaded blue in the bulk of posterior, tails look good. This suggests we might be missing some covariates. We explore this in a latter more complex model.

+++

To get a better understanding of the model fit, we need to look into the individual components.

+++

## Model Components

Next, we visualize each of the main components of the model. We write a utility function to do this.

```{code-cell} ipython3
:tags: [hide-input]

def plot_component(
    component_name: str, color: str, component_label: str
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(15, 9))
    sns.scatterplot(
        data=data_df, x="date", y="births_relative100", c="C0", s=8, label="data", ax=ax
    )
    ax.axhline(100, color="black", linestyle="--", label="mean level")
    az.plot_hdi(
        x=date,
        y=pp_vars_original_scale[component_name],
        hdi_prob=0.94,
        color=color,
        fill_kwargs={"alpha": 0.2, "label": rf"{component_label} $94\%$ HDI"},
        smooth=False,
        ax=ax,
    )
    az.plot_hdi(
        x=date,
        y=pp_vars_original_scale[component_name],
        hdi_prob=0.5,
        color=color,
        fill_kwargs={"alpha": 0.6, "label": rf"{component_label} $50\%$ HDI"},
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
    return fig, ax
```

### Global Trend

```{code-cell} ipython3
fig, ax = plot_component(component_name="f_trend", color="C3", component_label="$f_{trend}$")
```

### Yearly Periodicity

```{code-cell} ipython3
fig, ax = plot_component(
    component_name="f_year_periodic",
    color="C4",
    component_label=r"$f_{year \: periodic}$",
)
```

### Global Trend plus Yearly Periodicity

+++

If we want to combine the global trend and the yearly periodicity, we can't simply sum the components in the original scale as we would be adding the mean term twice. Instead we need to first sum the posterior samples and then take the inverse transform (these operation do not commute!).

```{code-cell} ipython3
pp_vars_original_scale["f_trend_periodic"] = apply_fn_along_dims(
    fn=births_relative100_pipeline.inverse_transform,
    a=idata["posterior"]["f_trend"] + idata["posterior"]["f_year_periodic"],
    dim="time",
)

fig, ax = plot_component(
    component_name="f_trend_periodic",
    color="C3",
    component_label=r"$f_{trend \: + \: periodic}$",
)
```

---

## Conclusion

We hope you better understand HSGPs and how to use them in practice with the very convenient PyMC's API. It's great to be able to strategically fold GPs into larger models. It's "possible" with GPs, but HSGPs make that actually possible. The reason is that the complexity of each GP component the is reduced by the approximation from $\mathcal{O}(n^3)$ to $\mathcal{O}(nm + m)$, where $m$ is the number of basis functions used in the approximation. This is a huge speedup!

``````{admonition} HSGP Limitations
Keep in mind that HSGPs are not a silver bullet. 
- They only apply to stationary covariances (in practice, {class}`~pymc.gp.cov.ExpQuad`, {class}`~pymc.gp.cov.Matern52`, {class}`~pymc.gp.cov.Matern32`).
- They don't scale well with input dimension. For dimensions $1$ and $2$ they are fine.

In practice this not a huge limitation as most of the time we work with stationary covariances and low input dimensions.
``````

In a future notebook, we will present a more complete model to compare with Vehtari's results. Stay tuned!

+++

## Authors
- Authored by [Juan Orduz](https://juanitorduz.github.io/) in January 2024 

+++

## Acknowledgements

I would like to thank [Alex Andorra](https://github.com/AlexAndorra) and [Bill Engels](https://github.com/bwengals) for their valuable feedback and suggestions during the writing of this notebook.

+++

## References
:::{bibliography}
:filter: docname in docnames
:::

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p numpyro,pytensor
```

:::{include} ../page_footer.md
:::
