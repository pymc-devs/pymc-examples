---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: pymc_env
  language: python
  name: pymc_env
---

(excess_deaths)=
# Counterfactual inference: calculating excess deaths due to COVID-19

:::{post} July, 2022
:tags: counterfactuals, causal inference, time series, case study, Bayesian workflow, forecasting, causal impact, regression, posterior predictive
:category: intermediate
:author: Benjamin T. Vincent
:::

Causal reasoning and counterfactual thinking are really interesting but complex topics! Nevertheless, we can make headway into understanding the ideas through relatively simple examples. This notebook focuses on the concepts and the practical implementation of Bayesian causal reasoning using PyMC.

We do this using the sobering but important example of calculating excess deaths due to COVID-19. As such, the ideas in this notebook strongly overlap with Google's [CausalImpact](https://google.github.io/CausalImpact/CausalImpact.html) (see {cite:t}`google_causal_impact2015`). Practically, we will try to estimate the number of 'excess deaths' since the onset of COVID-19, using data from England and Wales. Excess deaths are defined as:

$$
\text{Excess deaths} = 
  \underbrace{\text{Reported Deaths}}_{\text{noisy measure of actual deaths}} 
  - \underbrace{\text{Expected Deaths}}_{\text{unmeasurable counterfactual}}
$$

Making a claim about excess deaths requires causal/counterfactual reasoning. While the reported number of deaths is nothing but a (maybe noisy and/or lagged) measure of a real observable fact in the world, _expected deaths_ is unmeasurable because these are never realised in our timeline. That is, the expected deaths is a counterfactual thought experiment where we can ask "What would/will happen if?"

+++

## Overall strategy
How do we go about this, practically? We will follow this strategy:
1. Import data on reported number of deaths from all causes (our outcome variable), as well as a few reasonable predictor variables: 
    - average monthly temperature
    - month of the year, which we use to model seasonal effects
    - and time which is used to model any underlying linear trend.
2. Split into `pre` and `post` covid datasets. This is an important step. We want to come up with a model based upon what we know _before_ COVID-19 so that we can construct our counterfactual predictions based on data before COVID-19 had any impact.
3. Estimate model parameters based on the `pre` dataset. 
4. [Retrodict](https://en.wikipedia.org/wiki/Retrodiction) the number of deaths expected by the model in the pre COVID-19 period. This is not a counterfactual, but acts to tell us how capable the model is at accounting for the already observed data.
5. Counterfactual inference - we use our model to construct a counterfactual forecast. What would we expect to see in the future if there was no COVID-19? This can be achieved by using the famous do-operator Practically, we do this with posterior prediction on out-of-sample data. 
6. Calculate the excess deaths by comparing the reported deaths with our counterfactual (expected number of deaths).

+++

## Modelling strategy
We could take many different approaches to the modelling. Because we are dealing with time series data, then it would be very sensible to use a time series modelling approach. For example, Google's [CausalImpact](https://google.github.io/CausalImpact/CausalImpact.html) uses a [Bayesian structural time-series](https://en.wikipedia.org/wiki/Bayesian_structural_time_series) model, but there are many alternative time series models we could choose. 

But because the focus of this case study is on the counterfactual reasoning rather than the specifics of time-series modelling, I chose the simpler approach of linear regression for time-series model (see {cite:t}`martin2021bayesian` for more on this).

+++ {"tags": []}

## Causal inference disclaimer

Readers should be aware that there are of course limits to the causal claims we can make here. If we were dealing with a marketing example where we ran a promotion for a period of time and wanted to make inferences about _excess sales_, then we could only make strong causal claims if we had done our due diligence in accounting for other factors which may have also taken place during our promotion period. 

Similarly, there are [many other things that changed in the UK since January 2020](https://en.wikipedia.org/wiki/2020_in_the_United_Kingdom#Events) (the well documented time of the first COVID-19 cases) in England and Wales. So if we wanted to be rock solid then we should account for other feasibly relevant factors.

Finally, we are _not_ claiming that $x$ people died directly from the COVID-19 virus. The beauty of the concept of excess deaths is that it captures deaths from all causes that are in excess of what we would expect. As such, it covers not only those who died directly from the COVID-19 virus, but also from all downstream effects of the virus and availability of care, for example.

```{code-cell} ipython3
import calendar
import os

import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns
import xarray as xr
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
```

Now let's define some helper functions

```{code-cell} ipython3
:tags: [hide-cell]

def ZeroSumNormal(name, *, sigma=None, active_dims=None, dims, model=None):
    model = pm.modelcontext(model=model)

    if isinstance(dims, str):
        dims = [dims]

    if isinstance(active_dims, str):
        active_dims = [active_dims]

    if active_dims is None:
        active_dims = dims[-1]

    def extend_axis(value, axis):
        n_out = value.shape[axis] + 1
        sum_vals = value.sum(axis, keepdims=True)
        norm = sum_vals / (pt.sqrt(n_out) + n_out)
        fill_val = norm - sum_vals / pt.sqrt(n_out)
        out = pt.concatenate([value, fill_val], axis=axis)
        return out - norm

    dims_reduced = []
    active_axes = []
    for i, dim in enumerate(dims):
        if dim in active_dims:
            active_axes.append(i)
            dim_name = f"{dim}_reduced"
            if name not in model.coords:
                model.add_coord(dim_name, length=len(model.coords[dim]) - 1, mutable=False)
            dims_reduced.append(dim_name)
        else:
            dims_reduced.append(dim)

    raw = pm.Normal(f"{name}_raw", sigma=sigma, dims=dims_reduced)
    for axis in active_axes:
        raw = extend_axis(raw, axis)
    return pm.Deterministic(name, raw, dims=dims)


def format_x_axis(ax, minor=False):
    # major ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y %b"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(which="major", linestyle="-", axis="x")
    # minor ticks
    if minor:
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%Y %b"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.grid(which="minor", linestyle=":", axis="x")
    # rotate labels
    for label in ax.get_xticklabels(which="both"):
        label.set(rotation=70, horizontalalignment="right")


def plot_xY(x, Y, ax):
    quantiles = Y.quantile((0.025, 0.25, 0.5, 0.75, 0.975), dim=("chain", "draw")).transpose()

    az.plot_hdi(
        x,
        hdi_data=quantiles.sel(quantile=[0.025, 0.975]),
        fill_kwargs={"alpha": 0.25},
        smooth=False,
        ax=ax,
    )
    az.plot_hdi(
        x,
        hdi_data=quantiles.sel(quantile=[0.25, 0.75]),
        fill_kwargs={"alpha": 0.5},
        smooth=False,
        ax=ax,
    )
    ax.plot(x, quantiles.sel(quantile=0.5), color="C1", lw=3)


# default figure sizes
figsize = (10, 5)

# create a list of month strings, for plotting purposes
month_strings = calendar.month_name[1:]
```

## Import data
For our purposes we will obtain number of deaths (per month) reported in England and Wales. This data is available from the Office of National Statistics dataset [Deaths registered monthly in England and Wales](https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/datasets/monthlyfiguresondeathsregisteredbyareaofusualresidence). I manually downloaded this data for the years 2006-2022 and aggregated it into a single `.csv` file. I also added the average UK monthly temperature data as a predictor, obtained from the [average UK temperature from the Met Office](https://www.metoffice.gov.uk/research/climate/maps-and-data/uk-and-regional-series) dataset.

```{code-cell} ipython3
try:
    df = pd.read_csv(os.path.join("..", "data", "deaths_and_temps_england_wales.csv"))
except FileNotFoundError:
    df = pd.read_csv(pm.get_data("deaths_and_temps_england_wales.csv"))

df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")

# split into separate dataframes for pre and post onset of COVID-19
pre = df[df.index < "2020"]
post = df[df.index >= "2020"]
```

## Visualise data

+++

### Reported deaths over time
Plotting the time series shows that there is clear seasonality in the number of deaths, and we can also take a guess that there may be an increase in the average number of deaths per year.

```{code-cell} ipython3
ax = sns.lineplot(data=df, x="date", y="deaths", hue="pre")
format_x_axis(ax)
```

### Seasonality

Let's take a closer look at the seasonal pattern (just of the pre-covid data) by plotting deaths as a function of month, and we will color code the year. This confirms our suspicion of a seasonal trend in numbers of deaths with there being more deaths in the winter season than the summer. We can also see a large number of deaths in January, followed by a slight dip in February which bounces back in March. This could be due to a combination of:
- `push-back` of deaths that actually occurred in December being registered in January
- or `pull-forward` where many of the vulnerable people who would have died in February ended up dying in January, potentially due to the cold conditions.

The colour coding supports our suspicion that there is a positive main effect of year - that the baseline number of deaths per year is increasing.

```{code-cell} ipython3
ax = sns.lineplot(data=pre, x="month", y="deaths", hue="year", lw=3)
ax.set(title="Pre COVID-19 data");
```

### Linear trend

Let's look at that more closely by plotting the total deaths over time, pre COVID-19. While there is some variability here, it seems like adding a linear trend as a predictor will capture some of the variance in reported deaths, and therefore make for a better model of reported deaths.

```{code-cell} ipython3
annual_deaths = pd.DataFrame(pre.groupby("year")["deaths"].sum()).reset_index()
sns.regplot(x="year", y="deaths", data=annual_deaths);
```

### Effects of temperature on deaths

Looking at the `pre` data alone, there is a clear negative relationship between monthly average temperature and the number of deaths. Over a wider range of temperatures it is clear that this deaths will have a U-shaped relationship with temperature. But the climate in England and Wales, we only see the lower side of this curve. Despite that, the relationship could plausibly be approximately quadratic, but for our purposes a linear relationship seems like a reasonable place to start.

```{code-cell} ipython3
fig, ax = plt.subplots(1, 2, figsize=figsize)
sns.regplot(x="temp", y="deaths", data=pre, scatter_kws={"s": 40}, order=1, ax=ax[0])
ax[0].set(title="Linear fit (pre COVID-19 data)")
sns.regplot(x="temp", y="deaths", data=pre, scatter_kws={"s": 40}, order=2, ax=ax[1])
ax[1].set(title="Quadratic fit (pre COVID-19 data)");
```

Let's examine the slope of this relationship, which will be useful in defining a prior for a temperature coefficient in our model.

```{code-cell} ipython3
# NOTE: results are returned from higher to lower polynomial powers
slope, intercept = np.polyfit(pre["temp"], pre["deaths"], 1)
print(f"{slope:.0f} deaths/degree")
```

Based on this, if we focus only on the relationship between temperature and deaths, we expect there to be 764 _fewer_ deaths for every $1^\circ C$ increase in average monthly temperature. So we can use this figure when it comes to defining a prior over the coefficient for the temperature effect.

+++

## Modelling

We are going to estimate reported deaths over time with an intercept, a linear trend, seasonal deflections (for each month), and average monthly temperature. So this is a pretty straightforward linear model. The only thing of note is that we transform the normally distributed monthly deflections to have a mean of zero in order to reduce the degrees of freedom of the model by one, which should help with parameter identifiability.

```{code-cell} ipython3
with pm.Model(coords={"month": month_strings}) as model:

    # observed predictors and outcome
    month = pm.MutableData("month", pre["month"].to_numpy(), dims="t")
    time = pm.MutableData("time", pre["t"].to_numpy(), dims="t")
    temp = pm.MutableData("temp", pre["temp"].to_numpy(), dims="t")
    deaths = pm.MutableData("deaths", pre["deaths"].to_numpy(), dims="t")

    # priors
    intercept = pm.Normal("intercept", 40_000, 10_000)
    month_mu = ZeroSumNormal("month mu", sigma=3000, dims="month")
    linear_trend = pm.TruncatedNormal("linear trend", 0, 50, lower=0)
    temp_coeff = pm.Normal("temp coeff", 0, 200)

    # the actual linear model
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

As part of the Bayesian workflow, we will plot our prior predictions to see what outcomes the model finds before having observed any data.

```{code-cell} ipython3
with model:
    idata = pm.sample_prior_predictive(random_seed=RANDOM_SEED)


fig, ax = plt.subplots(figsize=figsize)

plot_xY(pre.index, idata.prior_predictive["obs"], ax)
format_x_axis(ax)
ax.plot(pre.index, pre["deaths"], label="observed")
ax.set(title="Prior predictive distribution in the pre COVID-19 era")
plt.legend();
```

This seems reasonable:
- The _a priori_ number of deaths looks centred on the observed numbers.
- Given the priors, the predicted range of deaths is quite broad, and so is unlikely to over-constrain the model.
- The model does not predict negative numbers of deaths per month.

We can look at this in more detail with the Arviz prior predictive check (ppc) plot. Again we see that the distribution of the observations is centered on the actual observations but has more spread. This is useful as we know the priors are not too restrictive and are unlikely to systematically influence our posterior predictions upwards or downwards.

```{code-cell} ipython3
az.plot_ppc(idata, group="prior");
```

## Inference 
Draw samples for the posterior distribution, and remember we are doing this for the pre COVID-19 data only.

```{code-cell} ipython3
with model:
    idata.extend(pm.sample(random_seed=RANDOM_SEED))
```

```{code-cell} ipython3
az.plot_trace(idata, var_names=["~mu", "~month mu_raw"]);
```

Let's also look at the posterior estimates of the monthly deflections, in a different way to focus on the seasonal effect.

```{code-cell} ipython3
az.plot_forest(idata.posterior, var_names="month mu", figsize=figsize);
```

## Posterior predictive check

Another important aspect of the Bayesian workflow is to plot the model's posterior predictions, allowing us to see how well the model can retrodict the already observed data. It is at this point that we can decide whether the model is too simple (then we'd build more complexity into the model) or if it's fine.

```{code-cell} ipython3
with model:
    idata.extend(pm.sample_posterior_predictive(idata, random_seed=RANDOM_SEED))


fig, ax = plt.subplots(figsize=figsize)

az.plot_hdi(pre.index, idata.posterior_predictive["obs"], hdi_prob=0.5, smooth=False)
az.plot_hdi(pre.index, idata.posterior_predictive["obs"], hdi_prob=0.95, smooth=False)
ax.plot(pre.index, pre["deaths"], label="observed")
format_x_axis(ax)
ax.set(title="Posterior predictive distribution in the pre COVID-19 era")
plt.legend();
```

Let's do another check now, but focussing on the seasonal effect. We will replicate the plot that we had above of deaths as a function of month of the year. And in order to keep the plot from being a complete mess, we will just plot the posterior mean. As such this is not a posterior _predictive_ check, but a check of the posterior.

```{code-cell} ipython3
temp = idata.posterior["mu"].mean(dim=["chain", "draw"]).to_dataframe()
pre = pre.assign(deaths_predicted=temp["mu"].values)

fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=True)
sns.lineplot(data=pre, x="month", y="deaths", hue="year", ax=ax[0], lw=3)
ax[0].set(title="Observed")
sns.lineplot(data=pre, x="month", y="deaths_predicted", hue="year", ax=ax[1], lw=3)
ax[1].set(title="Model predicted mean");
```

The model is doing a pretty good job of capturing the properties of the data. On the right, we can clearly see the main effects of `month` and `year`. However, we can see that there is something interesting happening in the data (left) in January which the model is not capturing. This might be able to be captured in the model by adding an interaction between `month` and `year`, but this is left as an exercise for the reader.

+++

## Excess deaths: Pre-Covid

This step is not strictly necessary, but we can apply the excess deaths formula to the models' retrodictions for the `pre` period. This is useful because we can examine how good the model is.

```{code-cell} ipython3
:tags: [hide-input]

# convert deaths into an XArray object with a labelled dimension to help in the next step
deaths = xr.DataArray(pre["deaths"].to_numpy(), dims=["t"])

# do the calculation by taking the difference
excess_deaths = deaths - idata.posterior_predictive["obs"]

fig, ax = plt.subplots(figsize=figsize)
# the transpose is to keep arviz happy, ordering the dimensions as (chain, draw, t)
az.plot_hdi(pre.index, excess_deaths.transpose(..., "t"), hdi_prob=0.5, smooth=False)
az.plot_hdi(pre.index, excess_deaths.transpose(..., "t"), hdi_prob=0.95, smooth=False)
format_x_axis(ax)
ax.axhline(y=0, color="k")
ax.set(title="Excess deaths, pre COVID-19");
```

We can see that we have a few spikes here where the number of excess deaths is plausibly greater than zero. Such occasions are above and beyond what we could expect from: a) seasonal effects, b) the linearly increasing trend, c) the effect of cold winters. 

If we were interested, then we could start to generate hypotheses about what additional predictors may account for this. Some ideas could include the prevalence of the common cold, or minimum monthly temperatures which could add extra predictive information not captured by the mean.

We can also see that there is some additional temporal trend that the model is not quite capturing. There is some systematic low-frequency drift from the posterior mean from zero. That is, there is additional variance in the data that our predictors are not quite capturing which could potentially be caused by changes in the size of vulnerable cohorts over time.

But we are close to our objective of calculating excess deaths during the COVID-19 period, so we will move on as the primary purpose here is on counterfactual thinking, not on building the most comprehensive model of reported deaths ever.

+++

## Counterfactual inference
Now we will use our model to predict the reported deaths in the 'what if?' scenario of business as usual.

So we update the model with the `month` and time (`t`) and `temp` data from the `post` dataframe and run posterior predictive sampling to predict the number of reported deaths we would observe in this counterfactual scenario. We could also call this 'forecasting'.

```{code-cell} ipython3
with model:
    pm.set_data(
        {
            "month": post["month"].to_numpy(),
            "time": post["t"].to_numpy(),
            "temp": post["temp"].to_numpy(),
        }
    )
    counterfactual = pm.sample_posterior_predictive(
        idata, var_names=["obs"], random_seed=RANDOM_SEED
    )
```

```{code-cell} ipython3
:tags: [hide-input]

fig, ax = plt.subplots(figsize=figsize)

plot_xY(post.index, counterfactual.posterior_predictive["obs"], ax)
format_x_axis(ax, minor=True)
ax.plot(post.index, post["deaths"], label="reported deaths")
ax.set(title="Counterfactual: Posterior predictive forecast of deaths if COVID-19 had not appeared")
plt.legend();
```

We now have the ingredients needed to calculate excess deaths. Namely, the reported number of deaths, and the Bayesian counterfactual prediction of how many would have died if nothing had changed from the pre to post COVID-19 era.

+++

## Excess deaths: since Covid onset

+++

Now we'll use the predicted number of deaths under the counterfactual scenario and compare that to the reported number of deaths to come up with our counterfactual estimate of excess deaths.

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
:tags: [hide-input]

fig, ax = plt.subplots(2, 1, figsize=(figsize[0], 9), sharex=True)

# Plot the excess deaths
# The transpose is to keep arviz happy, ordering the dimensions as (chain, draw, t)
plot_xY(post.index, excess_deaths.transpose(..., "t"), ax[0])
format_x_axis(ax[0], minor=True)
ax[0].axhline(y=0, color="k")
ax[0].set(title="Excess deaths, since COVID-19 onset")

# Plot the cumulative excess deaths
plot_xY(post.index, cumsum.transpose(..., "t"), ax[1])
format_x_axis(ax[1], minor=True)
ax[1].axhline(y=0, color="k")
ax[1].set(title="Cumulative excess deaths, since COVID-19 onset");
```

And there we have it - we've done some Bayesian counterfactual inference in PyMC! In just a few steps we've:
- Built a simple linear regression model.
- Inferred the model parameters based on pre COVID-19 data, running prior and posterior predictive checks. We note that the model is pretty good, but as always there might be ways to improve the model in the future.
- Used the model to create counterfactual predictions of what would happen in the future (COVID-19 era) if nothing had changed.
- Calculated the excess deaths (and cumulative excess deaths) by comparing the reported deaths to our counterfactual expected number of deaths.

The bad news of course, is that as of the last data point (May 2022) the number of excess deaths in England and Wales has started to rise again.

+++

## References

:::{bibliography}
:filter: docname in docnames
:::

+++

## Authors
- Authored by [Benjamin T. Vincent](https://github.com/drbenvincent) in July 2022.
- Updated by Benjamin T. Vincent in February 2023 to run on PyMC v5

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,aeppl,xarray
```

:::{include} ../page_footer.md
:::
