---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: pymc_env
  language: python
  name: pymc_env
---

(excess_deaths)=
# Counterfactual inference: calculating excess deaths due to Covid-19

:::{post} July, 2022
:tags: counterfactuals, causal inference, time series, case study, Bayesian workflow
:category: intermediate
:author: Benjamin T. Vincent
:::

Counterfactual inference and causal reasoning is a broad and deep topic. This notebook provides a concise introduction to using PyMC for Bayesian counterfactual inference. To do this we will examine the notion of 'excess deaths' which has unfortunately become prominent in our minds as the effects of Covid-19 have unfolded over the world. Excess deaths are defined as:

$$
\text{Excess deaths} = 
  \underbrace{\text{Reported Deaths}}_{\text{noisy measure of actual deaths}} 
  - \underbrace{\text{Expected Deaths}}_{\text{unmeasurable counterfactual}}
$$

This concept of excess deaths is particularly relevant to Bayesians. Firstly, in an ideal world we would compare the actual number of deaths against the expected deaths. But the actual number of deaths is not directly observable - depending on the country we live in and the reporting infrastructure there may be noise or a time lag in terms of recording actual deaths. Secondly, the expected number of deaths is not measurable - but not because there is observation noise, this is a purely hypothetical quantity of the number of deaths we could expect _if_ Covid-19 had not occurred. So the expected number of deaths is a counterfactual and so making statements about the excess number of deaths requires counterfactual reasoning.

+++

## Overall strategy
The strategy we will take is as follows:
1. Import data on reported number of deaths, as well as average monthly temperature data which we use as a predictor variable.
2. Split into `pre` and `post` covid datasets.
3. Estimate model parameters (deaths per month and linear trend) on the `pre` dataset. From this we can calculate the number of reported deaths expected by the model in the pre period. This is not a counterfactual.
4. Counterfactual inference - we will calculate a posterior predictive distribution which describes the expected number of deaths we would expect based upon the pre-Covid-19 death data. We do this by providing the model with data (time, and temperature) and asking it to predict the expected number of deaths for this new data, based on the posterior distributions over parameters that it had estimated from the pre Covid-19 data.
5. Calculate the excess deaths by comparing the reported deaths with our counterfactual (expected number of deaths).

+++

## Modelling strategy
As ever, there we could take many different approaches to the modelling. Because we are dealing with time series data, then it would be very sensible to use a time series modelling approach. However, because the focus of this case study is on the counterfactual reasoning, I chose the simpler approach of using linear regression to model the time series. Interested readers can find out more about this in {cite:t}`martin2021bayesian`.

```{code-cell} ipython3
import calendar
import os

import aesara.tensor as at
import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import xarray as xr
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
month_strings = calendar.month_name[1:]
```

```{code-cell} ipython3
def format_x_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y %b"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(which="major", linestyle="-", axis="x")
    for label in ax.get_xticklabels(which="both"):
        label.set(rotation=70, horizontalalignment="right")
```

## Import data
For our purposes we will obtain number of deaths (per month) reported in England and Wales. This data is available from the Office of National Statistics dataset [Deaths registered monthly in England and Wales](https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/datasets/monthlyfiguresondeathsregisteredbyareaofusualresidence). I manually downloaded this data for the years 2006-2022 and aggregated it into a single `.csv` file. Below we import this and create columns for the year, the month and the observation number.

```{code-cell} ipython3
try:
    df = pd.read_csv(os.path.join("..", "data", "total_deaths.csv"))
except FileNotFoundError:
    df = pd.read_csv(pm.get_data("total_deaths.csv"))

df = df.assign(
    date=pd.to_datetime(df["date"], format="%m-%Y"),
    year=lambda x: x["date"].dt.year,
    month=lambda x: x["date"].dt.month,
    t=df.index,
).set_index("date")
df["pre"] = df.index < "2020"
display(df)
```

We are also going to use temperature data as a predictor. So below we import [average UK temperature from the Met Office](https://www.metoffice.gov.uk/research/climate/maps-and-data/uk-and-regional-series) by month and do some processing to get it into the right format.

```{code-cell} ipython3
try:
    w = pd.read_csv(os.path.join("..", "data", "weather.csv"))
except FileNotFoundError:
    w = pd.read_csv(pm.get_data("weather.csv"))

w = pd.melt(
    w,
    id_vars="year",
    value_vars=[
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ],
    var_name="month",
    value_name="temp",
)
w["date"] = w["year"].map(str) + "-" + w["month"].map(str)
w["date"] = pd.to_datetime(w["date"])
w = w.drop(["month", "year"], axis=1)
w = w.sort_values("date")
w = w.set_index("date")
w = w.dropna()
display(w)
```

We merge these two data sources into a single dataframe.

```{code-cell} ipython3
# merge the dataframes
df = w.merge(df, on="date", how="inner")
df = df.rename(columns={"temp_x": "temp"})
display(df)
```

Finally, we split into `pre` and `post` Covid-19 onset dataframes. It is documented that the first Covid-19 cases appeared in the UK in January 2020, so we will take this time point as the cutoff for the pre vs. post Covid-19 era.

```{code-cell} ipython3
pre = df[df.index < "2020"]
post = df[df.index >= "2020"]
```

## Visualise data

**TODO: Why doesn't this changing of default figure size work?**

```{code-cell} ipython3
# set default figure sizes
sns.set(rc={"figure.figsize": (10, 6)})
plt.rcParams["figure.figsize"] = [10, 6]
```

### Reported deaths over time
Plotting the time series shows that there is clear seasonality in the number of deaths, and we can also take a guess that there may be an increase in the average number of deaths per year.

```{code-cell} ipython3
ax = sns.lineplot(data=df, x="date", y="deaths", hue="pre")
format_x_axis(ax)
```

### Seasonality

Let's take a closer look at the seasonal pattern (just of the pre-covid data) by plotting deaths as a function of month, and we will color code the year. This confirms our suspicion of a yearly trend in numbers of deaths with there being more deaths in the winter season than the summer. We can also see a large number of deaths in January, followed by a slight dip in February which bounces back in March. This could be due to a combination of:
- `push-back` of deaths that actually occurred in December being registered in January
- or `pull-forward` where money of the vulnerable people who would have died in February ended up dying in January, potentially due to the cold conditions.

The colour coding supports our suspicion that there is a positive main effect of year - that the baseline number of deaths per year is increasing.

```{code-cell} ipython3
ax = sns.lineplot(data=pre, x="month", y="deaths", hue="year", lw=3)
ax.set(title="Pre Covid-19 data");
```

### Linear trend

Let's look at that more closely by plotting the total deaths over time, pre Covid-19. While there is some variability here, it seems like adding a linear trend as a predictor will capture some of the variance in reported deaths, and therefore make for a better model of reported deaths.

```{code-cell} ipython3
annual_deaths = pd.DataFrame(pre.groupby("year")["deaths"].sum()).reset_index()
sns.regplot(x="year", y="deaths", data=annual_deaths);
```

### Effects of temperature on deaths

Looking at the `pre` data alone, there is a clear negative relationship between monthly average temperature and the number of deaths. This relationship could plausibly be quadratic, but for our purposes a linear relationship seems like a reasonable place to start.

```{code-cell} ipython3
fig, ax = plt.subplots(1, 2)
sns.regplot(x="temp", y="deaths", data=pre, scatter_kws={"s": 40}, order=1, ax=ax[0])
ax[0].set(title="Linear fit (pre Covid-19 data)")
sns.regplot(x="temp", y="deaths", data=pre, scatter_kws={"s": 40}, order=2, ax=ax[1])
ax[1].set(title="Quadratic fit (pre Covid-19 data)");
```

Let's examine the slope of this relationship, which will be useful in defining a prior for a temperature coefficient in our model.

```{code-cell} ipython3
# NOTE: results are returned from higher to lower polynomial powers
slope, intercept = np.polyfit(pre["temp"], pre["deaths"], 1)
print(f"{slope:.0f} deaths/degree")
```

Based on this, if we focus only on the relationship between temperature and deaths, we expect there to be 764 _fewer_ deaths for every $1^\circ C$ increase in average monthly temperature. So we can use this figure to guide our prior, but we will include a very high sigma on this prior.

+++

## Modelling
Here we are going to estimate month average deaths and a linear trend coefficient. And this will just be based upon the pre Covid-19 data.

**TODO: write down maths of model here**

```{code-cell} ipython3
# immutable coords
COORDS = {"month": month_strings}
```

```{code-cell} ipython3
with pm.Model(coords=COORDS) as model:

    # observed data
    month = pm.MutableData("month", pre["month"].to_numpy(), dims="t")
    time = pm.MutableData("time", pre["t"].to_numpy(), dims="t")
    temp = pm.MutableData("temp", pre["temp"].to_numpy(), dims="t")

    # observed outcome
    deaths = pm.MutableData("deaths", pre["deaths"].to_numpy(), dims="t")

    # priors
    intercept = pm.Normal("intercept", 40_000, 10_000)
    _month_mu = pm.Normal("_month_mu", 0, 3000, dims="month")
    # remove a degree of freedom by subtracting mean
    month_mu = pm.Deterministic(
        "month mu",
        _month_mu - at.mean(_month_mu),
        dims="month",
    )
    linear_trend = pm.TruncatedNormal("linear trend", 0, 50, lower=0)
    temp_coeff = pm.Normal("temp coeff", 0, 200)

    # model
    mu = pm.Deterministic(
        "mu",
        intercept + (linear_trend * time) + month_mu[month - 1] + (temp_coeff * temp),
        dims="t",
    )
    sigma = pm.HalfNormal("sigma", 2_000)
    # likelihood
    pm.TruncatedNormal("obs", mu=mu, sigma=sigma, lower=0, observed=deaths, dims="t")
```

```{code-cell} ipython3
pm.model_to_graphviz(model)
```

## Prior predictive check

```{code-cell} ipython3
with model:
    idata = pm.sample_prior_predictive()
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(pre.shape[0])

# prior predictive
az.plot_hdi(x, idata.prior_predictive["obs"], hdi_prob=0.50, smooth=False)
az.plot_hdi(x, idata.prior_predictive["obs"], hdi_prob=0.95, smooth=False)

ax.plot(x, pre["deaths"], label="observed")
ax.set(title="Prior predictive distribution in the pre Covid-19 era")
plt.legend();
```

**TODO: Better formatting of x-axis. `az.plot_hdi` can't deal with time series inputs on the x-axis**

+++

This seems reasonable:
- The _a priori_ number of deaths looks centred on the observed numbers.
- Given the priors, the predicted range of deaths is quite broad, and so is unlikely to over-constrain the model.
- The model does not predict negative numbers of deaths per month.

We can look at this in more detail with the Arviz prior predictive check (ppc) plot. Again we see that the distribution of the observations is centered on the actual observations but has more spread. This is useful as we know the priors are not too restrictive and are unlikely to systematically influence our posterior predictions upwards or downwards.

```{code-cell} ipython3
az.plot_ppc(idata, group="prior");
```

## Inference
Draw samples for the posterior and the posterior predictive distributions.

```{code-cell} ipython3
with model:
    idata.extend(pm.sample(tune=2000, target_accept=0.85))
```

```{code-cell} ipython3
az.plot_trace(idata, var_names=["~mu", "~_month_mu"]);
```

## Posterior predictive check

```{code-cell} ipython3
with model:
    idata.extend(pm.sample_posterior_predictive(idata))
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(15, 6))
x = np.arange(pre.shape[0])
az.plot_hdi(x, idata.posterior_predictive["obs"], hdi_prob=0.5, smooth=False)
az.plot_hdi(x, idata.posterior_predictive["obs"], hdi_prob=0.95, smooth=False)
ax.plot(x, pre["deaths"], label="observed")
ax.set(title="Posterior predictive distribution in the pre Covid-19 era")
plt.legend();
```

**TODO: Better formatting of x-axis. `az.plot_hdi` can't deal with time series inputs on the x-axis**

```{code-cell} ipython3
az.plot_forest(idata.posterior, var_names="month mu");
```

```{code-cell} ipython3
temp = idata.posterior["mu"].mean(dim=["chain", "draw"]).to_dataframe()
pre["deaths_predicted"] = temp["mu"].values

fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
sns.lineplot(data=pre, x="month", y="deaths", hue="year", ax=ax[0], lw=3)
ax[0].set(title="Observed")
sns.lineplot(data=pre, x="month", y="deaths_predicted", hue="year", ax=ax[1], lw=3)
ax[1].set(title="Model predicted mean");
```

The model is doing a pretty good job of capturing the properties of the data. On the right, we can clearly see the main effect of `month` and of `year`. However, we can see that there is something interesting happening in the data (left) in January which the model is not capturing. This might be able to be captured in the model by adding an interaction between `month` and `year`, but this is left as an exercise for the reader.

+++

## Excess deaths: Pre-Covid

```{code-cell} ipython3
# convert deaths into an XArray object with a labelled dimension to help in the next step
deaths = xr.DataArray(pre["deaths"].to_numpy(), dims=["t"])

# do the calculation by taking the difference
excess_deaths = deaths - idata.posterior_predictive["obs"]

fig, ax = plt.subplots(figsize=(15, 5))
x = np.arange(pre.shape[0])
# the transpose is to keep arviz happy, ordering the dimensions as (chain, draw, t)
az.plot_hdi(x, excess_deaths.transpose(..., "t"), hdi_prob=0.5, smooth=False)
az.plot_hdi(x, excess_deaths.transpose(..., "t"), hdi_prob=0.95, smooth=False)
ax.axhline(y=0, color="k")
ax.set(title="Excess deaths, pre Covid-19");
```

**TODO: Better formatting of x-axis. `az.plot_hdi` can't deal with time series inputs on the x-axis**

+++

We can see that we have a few spikes here where the number of excess deaths is plausibly greater than zero. Such occasions are above and beyond what we could expect from: a) seasonal effects, b) the linearly increasing trend, b) the effect of cold winters. 

If we were interested, then we could start to generate hypotheses about what additional predictors may account for this. Some ideas could include: 
- monthly minimum temperatures which may not be reflected in the monthly mean temperature
- prevalence of the common cold. 

But we are so close to our objective of calculating excess deaths during the Covid-19 period, so we will move on.

+++

## Counterfactual inference
Now we will use our model to predict the reported deaths in the 'what if?' scenario of business as usual.

So we update the model with the `month` and time (`t`) data from the `post` dataframe and run posterior predictive sampling to predict the number of reported deaths we would observe in this counterfactual scenario.

```{code-cell} ipython3
with model:
    pm.set_data(
        {
            "month": post["month"].to_numpy(),
            "time": post["t"].to_numpy(),
            "temp": post["temp"].to_numpy(),
        }
    )
    counterfactual = pm.sample_posterior_predictive(idata, var_names=["obs"])
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(15, 6))
x = np.arange(post.shape[0])
az.plot_hdi(x, counterfactual.posterior_predictive["obs"], hdi_prob=0.5, smooth=False)
az.plot_hdi(x, counterfactual.posterior_predictive["obs"], hdi_prob=0.95, smooth=False)
ax.plot(x, post["deaths"], label="observed")
ax.set(title="Posterior predictive distribution since Covid-19 onset in the UK")
plt.legend();
```

**TODO: Better formatting of x-axis. `az.plot_hdi` can't deal with time series inputs on the x-axis**

+++

## Excess deaths: since Covid onset

+++

No we'll use the predicted number of reported deaths under the counterfactual scenario of assuming that nothing novel happened from 2020 onwards.

```{code-cell} ipython3
# convert deaths into an XArray object with a labelled dimension to help in the next step
deaths = xr.DataArray(post["deaths"].to_numpy(), dims=["t"])

# do the calculation by taking the difference
excess_deaths = deaths - counterfactual.posterior_predictive["obs"]
```

And we can easily compute the cumulative excess deaths

```{code-cell} ipython3
# calculate the cumulative excess deaths
cumsum = excess_deaths.cumsum(dim="t")
```

```{code-cell} ipython3
fig, ax = plt.subplots(2, 1, figsize=(15, 7))
x = np.arange(post.shape[0])

# Plot the excess deaths
# the transpose is to keep arviz happy, ordering the dimensions as (chain, draw, t)
az.plot_hdi(x, excess_deaths.transpose(..., "t"), hdi_prob=0.5, smooth=False, ax=ax[0])
az.plot_hdi(x, excess_deaths.transpose(..., "t"), hdi_prob=0.95, smooth=False, ax=ax[0])
ax[0].axhline(y=0, color="k")
ax[0].set(title="Excess deaths, since Covid-19 onset")

# Plot the cumulative excess deaths
az.plot_hdi(x, cumsum.transpose(..., "t"), hdi_prob=0.5, smooth=False, ax=ax[1])
az.plot_hdi(x, cumsum.transpose(..., "t"), hdi_prob=0.95, smooth=False, ax=ax[1])
ax[1].axhline(y=0, color="k")
ax[1].set(title="Cumulative excess deaths, since Covid-19 onset");
```

**TODO: Better formatting of x-axis. `az.plot_hdi` can't deal with time series inputs on the x-axis**

+++

**TODO: Some conclusion here!**

+++

## References

:::{bibliography}
:filter: docname in docnames
:::

+++

## Authors
- Authored by [Benjamin T. Vincent](https://github.com/drbenvincent) in July 2022.

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p aesara,aeppl,xarray
```

:::{include} ../page_footer.md
:::
