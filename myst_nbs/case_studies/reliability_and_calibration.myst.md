---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3.9.0 ('pymc_ar_ex')
  language: python
  name: python3
---

(Reliability Statistics and Predictive Calibration)=
# Reliability Statistics and Predictive Calibration

:::{post} January, 2023
:tags: time-to-failure, prediction, calibration
:category: intermediate
:author: Nathaniel Forde
:::

```{code-cell} ipython3
import os

from io import StringIO

import aesara as at
import arviz as az
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from lifelines import KaplanMeierFitter, LogNormalFitter, WeibullFitter
from lifelines.utils import survival_table_from_events
from pymc.sampling_jax import sample_blackjax_nuts
from scipy.stats import lognorm, norm, weibull_min
```

```{code-cell} ipython3
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
%config InlineBackend.figure_format = 'retina'
```

## Reliability Statistics 

When we want to make inferences about likely failures on a production line. We may have large or small sample data set depending on the industry, nature of the goods or specificifty of the question we're seeking to answer. But in all cases there is a question of cost and a quantity of tolerable failures. In an industry where the line-item cost-profit ratio is clear, the estimation of failure rates is valuable. In an industry where failures are only tangentially related to a bottom line it is arguably even more crucial to correctly predict likely failures - otherwise the uncertainty propagates forward to make the revenue calculations next to impossible. 

A reliability study therefore has to account for the period in which a failure is important to observe, the cost of the failure and cost of running a mis-specified study. The requirements for precision in the definition of the question and the nature of the modelling exercise are paramount. 

There are more subtleties about reliability and failure time data such as censoring, accelerated life test data and degradation testing which we'll discuss briefly below. In this notebook we're going to focus on the prediction of failure times and compare the Bayesian notion of a calibrated prediction interval to some frequentist alternatives. 

### Types of Prediction

We might want to know:

- Time to failure of a new item
- Time until k failures in a future sample of m units
- Number of failures by time t in future sample of m units.

While there are non-parametric and descriptive methods that can be used to assess these kinds of question we're going to focus on the case where we have a probability model i.e. a lognormal distribution of failure times $F(t: \mathbf{\theta})$ parameterised by an unknown $\mathbf{\theta}$.

+++

### Example Failure Distribution

In the study of reliability statistics there is a focus on location-scale based distributions with long tails. In an ideal world we'd know exactly which distribution described our failure process and the prediction interval for the next failure could be defined exactly.

```{code-cell} ipython3
from scipy.stats import lognorm

mu, sigma = 6, 0.3


def plot_ln_pi(mu, sigma, xy=(700, 75), title="Exact Prediction Interval for Known Lognormal"):
    failure_dist = lognorm(s=sigma, scale=np.exp(mu))
    samples = failure_dist.rvs(size=1000, random_state=100)
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    axs = axs.flatten()
    axs[0].hist(samples, ec="black", color="slateblue", bins=30)
    axs[0].set_title(f"Failure Time Distribution: LN({mu}, {sigma})")
    count, bins_count = np.histogram(samples, bins=30)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    axs[1].plot(bins_count[1:], cdf, label="CDF", color="slateblue")
    axs[2].plot(bins_count[1:], 1 - cdf, label="Survival", color="slateblue")
    axs[2].legend()
    axs[1].legend()
    axs[1].set_title("Cumulative Density Function")
    axs[2].set_title("Survival Curve")

    lb = failure_dist.ppf(0.01)
    ub = failure_dist.ppf(0.99)
    axs[0].annotate(
        f"99% Prediction \nInterval: [{np.round(lb, 3)}, {np.round(ub, 3)}]",
        xy=(xy[0], xy[1] - 25),
        fontweight="bold",
    )
    axs[0].fill_betweenx(y=range(200), x1=lb, x2=ub, alpha=0.2, label="p99 PI", color="cyan")
    axs[1].fill_betweenx(y=range(2), x1=lb, x2=ub, alpha=0.2, label="p99 PI", color="cyan")
    axs[2].fill_betweenx(y=range(2), x1=lb, x2=ub, alpha=0.2, label="p99 PI", color="cyan")
    lb = failure_dist.ppf(0.025)
    ub = failure_dist.ppf(0.975)
    axs[0].annotate(
        f"95% Prediction \nInterval: [{np.round(lb, 3)}, {np.round(ub, 3)}]",
        xy=(xy[0], xy[1]),
        fontweight="bold",
    )
    axs[0].fill_betweenx(y=range(200), x1=lb, x2=ub, alpha=0.2, label="p95 PI", color="magenta")
    axs[1].fill_betweenx(y=range(2), x1=lb, x2=ub, alpha=0.2, label="p95 PI", color="magenta")
    axs[2].fill_betweenx(y=range(2), x1=lb, x2=ub, alpha=0.2, label="p95 PI", color="magenta")
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    plt.suptitle(title, fontsize=20)


plot_ln_pi(mu, sigma)
```

## Estimation of the Failure Distribution from Data

In the real world we rarely have such exact knowledge. Instead we start with altogether less clear data. We will first examine failure data about heat exchanges across three plants and pool the information to quantify the lifetime of the heat-exchanges over the three factories. 

The data is small deliberately so we can focus on the descriptive statistics involved in assessing time-to-failure data. In particular we'll estimate the empirical CDF and survival functions. We will then generalise this style of analysis to a larger data set afterwards. 

### Heat Exchange Data

**Note on Censored Data**:
See how the failure data flags whether or not an observation has been censored i.e. whether or not we have observed the full course of the life-time of the heat-exchanger. This is a crucial feature of failure time data. Too simple a statistical summary will be biased in its estimation of the prevalance of failure by the fact that our study has not seen out the full-course of every item's life-cycle. The most prevalent form of censoring is so called "Right censored" data where we have not seen the "failure" event for a subset of the observations. Our histories are incomplete due to prematurely ending the data collection. 

Left censoring (where we don't observe an item from the beginning of their history) and interval censoring (both left and right censoring) can also occur but are less common.

```{code-cell} ipython3
heat_exchange_df = pd.read_csv(
    StringIO(
        """Years Lower,Years Upper,Censoring Indicator,Count,Plant
0,1,Left,1,1
1,2,Interval,2,1
2,3,Interval,2,1
3, ,Right,95,1
0,1,Left,2,2
1,2,Interval,3,2
2, ,Right,95,2
0,1,Left,1,3
1, ,Right,99,3

"""
    )
)

heat_exchange_df["year_interval"] = (
    heat_exchange_df["Years Lower"].astype(str) + "," + heat_exchange_df["Years Upper"].astype(str)
)
heat_exchange_df["failed"] = np.where(
    heat_exchange_df["Censoring Indicator"] != "Right", heat_exchange_df["Count"], 0
)
heat_exchange_df["censored"] = np.where(
    heat_exchange_df["Censoring Indicator"] == "Right", heat_exchange_df["Count"], 0
)
heat_exchange_df["risk_set"] = [100, 99, 97, 0, 100, 98, 0, 100, 0]
heat_exchange_df
```

```{code-cell} ipython3
actuarial_table = heat_exchange_df.groupby(["Years Upper"])[["failed", "risk_set"]].sum()
actuarial_table = actuarial_table.tail(3)


def greenwood_variance(df):
    ### Used to estimate the variance in the CDF
    n = len(df)
    ps = [df.iloc[i]["p_hat"] / (df.iloc[i]["risk_set"] * df.iloc[i]["1-p_hat"]) for i in range(n)]
    s = [(df.iloc[i]["S_hat"] ** 2) * np.sum(ps[0 : i + 1]) for i in range(n)]
    return s


def logit_transform_interval(df):
    ### Used for robustness in the estimation of the Confidence intervals in the CDF
    df["logit_CI_95_lb"] = df["F_hat"] / (
        df["F_hat"]
        + df["S_hat"] * np.exp((1.960 * df["Standard_Error"]) / (df["F_hat"] * df["S_hat"]))
    )
    df["logit_CI_95_ub"] = df["F_hat"] / (
        df["F_hat"]
        + df["S_hat"] / np.exp((1.960 * df["Standard_Error"]) / (df["F_hat"] * df["S_hat"]))
    )
    df["logit_CI_95_lb"] = np.where(df["logit_CI_95_lb"] < 0, 0, df["logit_CI_95_lb"])
    df["logit_CI_95_ub"] = np.where(df["logit_CI_95_ub"] > 1, 1, df["logit_CI_95_ub"])
    return df


def make_actuarial_table(actuarial_table):
    ### Actuarial lifetables are used to describe the nature of the risk over time and estimate
    actuarial_table["p_hat"] = actuarial_table["failed"] / actuarial_table["risk_set"]
    actuarial_table["1-p_hat"] = 1 - actuarial_table["p_hat"]
    actuarial_table["S_hat"] = actuarial_table["1-p_hat"].cumprod()
    actuarial_table["CH_hat"] = -np.log(actuarial_table["S_hat"])
    ### The Estimate of the CDF function
    actuarial_table["F_hat"] = 1 - actuarial_table["S_hat"]
    actuarial_table["V_hat"] = greenwood_variance(actuarial_table)
    actuarial_table["Standard_Error"] = np.sqrt(actuarial_table["V_hat"])
    actuarial_table["CI_95_lb"] = (
        actuarial_table["F_hat"] - actuarial_table["Standard_Error"] * 1.960
    )
    actuarial_table["CI_95_lb"] = np.where(
        actuarial_table["CI_95_lb"] < 0, 0, actuarial_table["CI_95_lb"]
    )
    actuarial_table["CI_95_ub"] = (
        actuarial_table["F_hat"] + actuarial_table["Standard_Error"] * 1.960
    )
    actuarial_table["CI_95_ub"] = np.where(
        actuarial_table["CI_95_ub"] > 1, 1, actuarial_table["CI_95_ub"]
    )
    actuarial_table["ploting_position"] = actuarial_table["F_hat"].rolling(1).median()
    actuarial_table = logit_transform_interval(actuarial_table)
    return actuarial_table


actuarial_table_heat = make_actuarial_table(actuarial_table)
actuarial_table_heat = actuarial_table_heat.reset_index()
actuarial_table_heat.rename({"Years Upper": "t"}, axis=1, inplace=True)
actuarial_table_heat["t"] = actuarial_table_heat["t"].astype(int)
actuarial_table_heat
```

It's worth taking some time to walk through this example because it establishes estimates of some key quantities in time-to-failure modelling. 

First note how we're treating time as a series of discrete intervals. The data format is in discrete **period format**, since it records aggregate failures over time. We'll see below another format of failure data - the **item-period** format which records each individual item over all periods and their corresponding status. In this format the key quantities are the set of `failed` items and the `risk_set` in each period. Everything else is derived from these facts. 

First we've established across all three companies in three consecutive years the number of heat-exchanges that were produced and subsequently `failed`. This provides an estimate of the probability of failure in the year: `p_hat` and its inverse `1-p_hat` respectively. These are further combined over the course of the year to estimate the survival curve `S_hat` which can be further transformed to recover estimates of the cumulative hazard `CH_hat` and the cumulative density function `F_hat`. 

Next we want a quick a dirty way to quantify the extent of the uncertainty in our estimate of the CDF. For this purpose we use greenwood's formula to estimate the variance of our `V_hat` of our estimate `F_hat`. This gives us the standard error and the two varieties of confidence interval recommended in the literature. 

We' apply the same techniques to a larger dataset and plot some of these quantities below. 

### The Shock Absorbers Data: A Study in Frequentist Reliability Analysis

The shock absorbers data is in period format but it records a constantly decreasing risk set over time with one item being censored or failing at each time point i.e. removed from testing successfully (approved) or removed due to failure. This is a special case of the **period** format data.

```{code-cell} ipython3
shockabsorbers_df = pd.read_csv(
    StringIO(
        """Kilometers,Failure Mode,Censoring Indicator
6700,Mode1,Failed
6950,Censored,Censored
7820,Censored,Censored
8790,Censored,Censored
9120,Mode2,Failed
9660,Censored,Censored
9820,Censored,Censored
11310,Censored,Censored
11690,Censored,Censored
11850,Censored,Censored
11880,Censored,Censored
12140,Censored,Censored
12200,Mode1,Failed
12870,Censored,Censored
13150,Mode2,Failed
13330,Censored,Censored
13470,Censored,Censored
14040,Censored,Censored
14300,Mode1,Failed
17520,Mode1,Failed
17540,Censored,Censored
17890,Censored,Censored
18450,Censored,Censored
18960,Censored,Censored
18980,Censored,Censored
19410,Censored,Censored
20100,Mode2,Failed
20100,Censored,Censored
20150,Censored,Censored
20320,Censored,Censored
20900,Mode2,Failed
22700,Mode1,Failed
23490,Censored,Censored
26510,Mode1,Failed
27410,Censored,Censored
27490,Mode1,Failed
27890,Censored,Censored
28100,Censored,Censored
"""
    )
)

shockabsorbers_df["failed"] = np.where(shockabsorbers_df["Censoring Indicator"] == "Failed", 1, 0)
shockabsorbers_df["t"] = shockabsorbers_df["Kilometers"]
shockabsorbers_events = survival_table_from_events(
    shockabsorbers_df["t"], shockabsorbers_df["failed"]
).reset_index()
shockabsorbers_events.rename(
    {"event_at": "t", "observed": "failed", "at_risk": "risk_set"}, axis=1, inplace=True
)
actuarial_table_shock = make_actuarial_table(shockabsorbers_events)
actuarial_table_shock
```

### Maximum Likelihood Fits for Failure Data

+++

In addition to taking descriptive summaries of our data we can use the life-table data to estimate a univariate model fit to our distribution of failure times. Such a fit, if good, would enable us to have a clearer view of the predictive distribution a particular set of predictive intervals. Here we'll use the functions from the  `lifelines` package to estimate the MLE fit on right-censored data.

```{code-cell} ipython3
lnf = LogNormalFitter().fit(actuarial_table_shock["t"] + 1e-25, actuarial_table_shock["failed"])
lnf.print_summary()
```

Although it's tempting to take this model and run with it, we need to be cautious in the case of limited data. For instance in the heat-exchange data we have three years of data with a total of 11 failures. A too simple model can get this quite wrong. For the moment we'll focus on the shock-absorber data - its non-parametric description and a simple univariate fit to this data.

```{code-cell} ipython3
def plot_cdfs(actuarial_table, dist_fits=True, ax=None, title="", xy=(3000, 0.5), item_period=None):
    if item_period is None:
        lnf = LogNormalFitter().fit(actuarial_table["t"] + 1e-25, actuarial_table["failed"])
        wbf = WeibullFitter().fit(actuarial_table["t"] + 1e-25, actuarial_table["failed"])
    else:
        lnf = LogNormalFitter().fit(item_period["t"] + 1e-25, item_period["failed"])
        wbf = WeibullFitter().fit(item_period["t"] + 1e-25, item_period["failed"])
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(
        actuarial_table["t"],
        actuarial_table["F_hat"],
        "-o",
        color="black",
        label="Non-Parametric Estimate of CDF",
    )
    ax.plot(
        actuarial_table["t"],
        actuarial_table["CI_95_lb"],
        color="darkorchid",
        linestyle="--",
        label="Non-Parametric 95% CI based on Normal Approx",
    )
    ax.plot(actuarial_table["t"], actuarial_table["CI_95_ub"], color="darkorchid", linestyle="--")
    ax.fill_between(
        actuarial_table["t"],
        actuarial_table["CI_95_lb"],
        actuarial_table["CI_95_ub"],
        color="darkorchid",
        alpha=0.2,
    )
    ax.plot(
        actuarial_table["t"],
        actuarial_table["logit_CI_95_lb"],
        color="royalblue",
        linestyle="--",
        label="Non-Parametric 95% CI based on Logit Approx",
    )
    ax.plot(
        actuarial_table["t"], actuarial_table["logit_CI_95_ub"], color="royalblue", linestyle="--"
    )
    ax.fill_between(
        actuarial_table["t"],
        actuarial_table["logit_CI_95_lb"],
        actuarial_table["logit_CI_95_ub"],
        color="royalblue",
        alpha=0.2,
    )
    if dist_fits:
        lnf.plot_cumulative_density(ax=ax, color="crimson", alpha=0.8)
        wbf.plot_cumulative_density(ax=ax, color="cyan", alpha=0.8)
        ax.annotate(
            f"Lognormal Fit: mu = {np.round(lnf.mu_, 3)}, sigma = {np.round(lnf.sigma_, 3)} \nWeibull Fit: lambda = {np.round(wbf.lambda_, 3)}, rho = {np.round(wbf.rho_, 3)}",
            xy=(xy[0], xy[1]),
            fontsize=12,
            weight="bold",
        )
    ax.set_title(
        f"Estimates of the Cumulative Density Function \n derived from our {title} Failure Data",
        fontsize=20,
    )
    ax.set_ylabel("Fraction Failing")
    ax.set_xlabel("Time Scale")
    ax.legend()
    return ax
```

```{code-cell} ipython3
plot_cdfs(actuarial_table_shock, title="Shock Absorber")
```

This shows a good fit to the data and implies, as you might expect, that the failing fraction of shock absorbers increases with age as they wear out. But how do we quantify the prediction given an estimated model?

+++

## The Plug-in-Procedure for calculating Approximate Statistical Prediction Intervals

Since we've estimated a lognormal fit for the CDF of the shock absorbers data we can plot their approximate prediction interval. The interest here is likely to be in the lower bound of the prediction interval since we as manufacturers might want to be aware of warranty claims and the risk of exposure to refunds if the lower bound is too low.

```{code-cell} ipython3
plot_ln_pi(
    10.128,
    0.526,
    xy=(40000, 120),
    title="Plug-in Estimate of Shock Absorber Failure Prediction Interval",
)
```

### Bootstrap Calibration and Coverage Estimation

We want now to estimate the coverage implied by this prediction interval, and to do so we will bootstrap estimates for the lower and upper bounds of the 95% confidence interval and ultimately assess their coverage conditional on the MLE fit.

```{code-cell} ipython3
import random


def bootstrap(lb, ub):
    draws = actuarial_table_shock[["t", "failed"]].sample(replace=True, frac=1)
    draws.sort_values("t", inplace=True)
    ## Fit Lognormal Dist to
    lnf = LogNormalFitter().fit(draws["t"] + 1e-25, draws["failed"])
    ## Sample random choice from 95% percentile interval of bootstrapped dist
    # choices = draws['t'].values
    choices = np.linspace(draws["t"].min(), draws["t"].max(), draws.shape[0])
    future = random.choice(choices)
    ## Check if choice is contained within the MLE 95% PI
    contained = (future >= lb) & (future <= ub)
    ## Record 95% interval of bootstrapped dist
    lb = lognorm(s=lnf.sigma_, scale=np.exp(lnf.mu_)).ppf(0.025)
    ub = lognorm(s=lnf.sigma_, scale=np.exp(lnf.mu_)).ppf(0.975)
    return lb, ub, contained, future, lnf.sigma_, lnf.mu_


CIs = [bootstrap(8928, 70188) for i in range(1000)]
draws = pd.DataFrame(
    CIs, columns=["Lower Bound PI", "Upper Bound PI", "Contained", "future", "Sigma", "Mu"]
)
draws
```

Next we'll plot the bootstrapped data and the two estimates of coverage we achieve conditional on the MLE fit.

```{code-cell} ipython3
mosaic = """AABB
            CCCC"""
fig, axs = plt.subplot_mosaic(mosaic=mosaic, figsize=(20, 10))
mle_rv = lognorm(s=0.53, scale=np.exp(10.128))
axs = [axs[k] for k in axs.keys()]
axs[0].scatter(
    draws["Mu"],
    draws["Lower Bound PI"],
    c=draws["Contained"],
    cmap=cm.cool,
    alpha=0.3,
    label="Fits in MLE 95% CI",
)
axs[1].scatter(
    draws["Sigma"],
    draws["Lower Bound PI"],
    c=draws["Contained"],
    cmap=cm.cool,
    alpha=0.3,
    label="Fits in MLE 95% CI",
)
axs[0].set_title("Bootstrapped Mu against Bootstrapped 95% Lower Bound")
prop = draws["Contained"].sum() / len(draws)
axs[0].annotate(
    f"Estimated Prediction \nEmpirical Coverage Based on Sampling : {np.round(prop, 3)}",
    xy=(10.4, 16000),
    fontweight="bold",
)
axs[1].set_title("Bootstrapped Sigma against Bootstrapped 95% Lower Bound")
axs[0].legend()
axs[0].set_xlabel("Mu")
axs[1].set_xlabel("Sigma")
axs[0].set_ylabel("Estimated Lower 95% PI")
axs[1].legend()
axs[2].hist(
    mle_rv.cdf(draws["Lower Bound PI"]),
    bins=50,
    label="Bootstrap 95% LB",
    ec="k",
    color="royalblue",
    alpha=0.2,
)
axs[2].hist(
    mle_rv.cdf(draws["Upper Bound PI"]),
    bins=50,
    label="Bootstrap 95% UB",
    ec="k",
    color="darkorchid",
    alpha=0.2,
)
axs[2].hist(
    np.abs(mle_rv.cdf(draws["Lower Bound PI"]) - mle_rv.cdf(draws["Upper Bound PI"])),
    alpha=0.2,
    bins=50,
    color="slateblue",
    ec="black",
    label="Bootstrap Abs Diff",
)
axs[2].axvline(
    np.abs(mle_rv.cdf(draws["Lower Bound PI"]) - mle_rv.cdf(draws["Upper Bound PI"])).mean(),
    label="Expected Coverage",
)
axs[2].set_title("Difference in LB and UB | MLE(mu, sigma)")
axs[2].legend()

plug_in = np.abs(
    np.mean(mle_rv.cdf(draws["Lower Bound PI"])) - np.mean(mle_rv.cdf(draws["Upper Bound PI"]))
)
lb = np.round(draws["Lower Bound PI"].mean(), 3)
ub = np.round(draws["Upper Bound PI"].mean(), 3)
axs[2].annotate(
    f"Estimated Prediction Interval \n Coverage Based on Plug in Method : {np.round(plug_in, 3)} \n with [{lb, ub}]",
    xy=(0.6, 80),
    fontweight="bold",
);
```

### Bearing Cage Data: A Study in Bayesian Reliability Analysis

Next we'll look at a data set which has a slightly less clean parametric fit. The most obvious feature of this data is the small amount of failing records. The data is recorded in the **period** format with counts showing the extent of the `risk set` in each period. 

We want to spend some time with this example to show how the *frequentist* techniques which worked well to estimate the shock-absorbers data can be augmented in the case of the Bearing cage data. In particular we'll show how the issues arising can be resolved with a *Bayesian* approach.

```{code-cell} ipython3
bearing_cage_df = pd.read_csv(
    StringIO(
        """Hours,Censoring Indicator,Count
50,Censored,288
150,Censored,148
230,Failed,1
250,Censored,124
334,Failed,1
350,Censored,111
423,Failed,1
450,Censored,106
550,Censored,99
650,Censored,110
750,Censored,114
850,Censored,119
950,Censored,127
990,Failed,1
1009,Failed,1
1050,Censored,123
1150,Censored,93
1250,Censored,47
1350,Censored,41
1450,Censored,27
1510,Failed,1
1550,Censored,11
1650,Censored,6
1850,Censored,1
2050,Censored,2"""
    )
)


bearing_cage_df["t"] = bearing_cage_df["Hours"]
bearing_cage_df["failed"] = np.where(bearing_cage_df["Censoring Indicator"] == "Failed", 1, 0)
bearing_cage_df["censored"] = np.where(
    bearing_cage_df["Censoring Indicator"] == "Censored", bearing_cage_df["Count"], 0
)
bearing_cage_events = survival_table_from_events(
    bearing_cage_df["t"], bearing_cage_df["failed"], weights=bearing_cage_df["Count"]
).reset_index()
bearing_cage_events.rename(
    {"event_at": "t", "observed": "failed", "at_risk": "risk_set"}, axis=1, inplace=True
)
actuarial_table_bearings = make_actuarial_table(bearing_cage_events)
pd.options.display.float_format = "{:.5f}".format
actuarial_table_bearings
```

To estimate a univariate or non-parametric CDF we need to disaggregate the **period** format data into an **item-period** format.

+++

### Item Period Data Format

```{code-cell} ipython3
item_period = bearing_cage_df["Hours"].to_list() * bearing_cage_df["Count"].sum()
ids = [[i] * 25 for i in range(bearing_cage_df["Count"].sum())]
ids = [int(i) for l in ids for i in l]
item_period_bearing_cage = pd.DataFrame(item_period, columns=["t"])
item_period_bearing_cage["id"] = ids
item_period_bearing_cage["failed"] = np.zeros(len(item_period_bearing_cage))

## Censor appropriate number of ids
unique_ids = item_period_bearing_cage["id"].unique()
censored = bearing_cage_df[bearing_cage_df["Censoring Indicator"] == "Censored"]
i = 0
stack = []
for hour, count, idx in zip(censored["Hours"], censored["Count"], censored["Count"].cumsum()):
    temp = item_period_bearing_cage[
        item_period_bearing_cage["id"].isin(unique_ids[i:idx])
        & (item_period_bearing_cage["t"] == hour)
    ]
    stack.append(temp)
    i = idx

censored_clean = pd.concat(stack)

### Add  appropriate number of failings
stack = []
unique_times = censored_clean["t"].unique()
for id, fail_time in zip(
    [9999, 9998, 9997, 9996, 9995, 9994],
    bearing_cage_df[bearing_cage_df["failed"] == 1]["t"].values,
):
    temp = pd.DataFrame(unique_times[unique_times < fail_time], columns=["t"])
    temp["id"] = id
    temp["failed"] = 0
    temp = pd.concat([temp, pd.DataFrame({"t": [fail_time], "id": [id], "failed": [1]}, index=[0])])
    stack.append(temp)

failed_clean = pd.concat(stack).sort_values(["id", "t"])
censored_clean
item_period_bearing_cage = pd.concat([failed_clean, censored_clean])
## Transpose for more concise visual
item_period_bearing_cage.head(30).T
```

```{code-cell} ipython3
assert item_period_bearing_cage["id"].nunique() == 1703
assert item_period_bearing_cage["failed"].sum() == 6
assert item_period_bearing_cage[item_period_bearing_cage["t"] >= 1850]["id"].nunique() == 3
```

As we plot the empirical CDF we see that the y-axis only ever reaches as maximum height of 0.5. A naive MLE fit will go dramatically wrong in extrapolating outside the observed range of the data.

```{code-cell} ipython3
ax = plot_cdfs(
    actuarial_table_bearings,
    title="Bearings",
    dist_fits=False,
    xy=(20, 0.7),
    item_period=item_period_bearing_cage,
)
```

## Probability Plots: Comparing CDFs in a Restricted Linear Range

With this adjustment to the data format we compare the MLE fit against the empirical CDF. In the next section we'll use the technique of linearising the MLE fits so that can perform a visual "goodness of fit" check. These types of plots rely on a transformation that can be applied to the location and scale distributions to turn their CDF into a linear space.  

For both the Lognormal and Weibull fits we can represent their CDF in a linear space as a relationship between the logged value t and an appropriate $CDF^{-1}$.

```{code-cell} ipython3
def sev_ppf(p):
    return np.log(-np.log(1 - p))


mosaic = """AABB
            CCCC"""
fig, axs = plt.subplot_mosaic(mosaic=mosaic, figsize=(20, 10))
axs = [axs[k] for k in axs.keys()]
ax = axs[0]
ax2 = axs[1]
ax3 = axs[2]
ax.plot(
    np.log(actuarial_table_bearings["t"]),
    norm.ppf(actuarial_table_bearings["logit_CI_95_ub"]),
    "-o",
    label="Non-Parametric CI UB",
    color="slateblue",
)
ax.scatter(
    np.log(actuarial_table_bearings["t"]),
    norm.ppf(actuarial_table_bearings["ploting_position"]),
    label="Non-Parametric CDF",
    color="black",
)
ax.plot(
    np.log(actuarial_table_bearings["t"]),
    norm.ppf(actuarial_table_bearings["logit_CI_95_lb"]),
    "-o",
    label="Non-Parametric CI LB",
    color="slateblue",
)
for mu in np.linspace(10, 12, 3):
    for sigma in np.linspace(1.6, 1.9, 3):
        rv = lognorm(s=sigma, scale=np.exp(mu))
        ax.plot(
            np.log(actuarial_table_bearings["t"]),
            norm.ppf(rv.cdf(actuarial_table_bearings["t"])),
            "--",
            label=f"LN({np.round(mu, 3)}, {np.round(sigma, 3)})",
            color="grey",
        )

lnf = LogNormalFitter().fit(item_period_bearing_cage["t"], item_period_bearing_cage["failed"])
rv = lognorm(s=lnf.sigma_, scale=np.exp(lnf.mu_))
ax.plot(
    np.log(actuarial_table_bearings["t"]),
    norm.ppf(rv.cdf(actuarial_table_bearings["t"])),
    "--",
    label=f"MLE LN({np.round(lnf.mu_, 3)}, {np.round(lnf.sigma_, 3)})",
    color="RED",
)

for r in np.linspace(1, 2, 3):
    for s in np.linspace(12000, 25000, 2):
        rv = weibull_min(c=r, scale=s)
        ax2.plot(
            np.log(actuarial_table_bearings["t"]),
            sev_ppf(rv.cdf(actuarial_table_bearings["t"])),
            "--",
            label=f"Wb({np.round(s, 3)}, {np.round(r, 3)})",
            color="lightblue",
        )


wbf = WeibullFitter().fit(item_period_bearing_cage["t"], item_period_bearing_cage["failed"])
rv = weibull_min(c=wbf.rho_, scale=wbf.lambda_)
ax2.plot(
    np.log(actuarial_table_bearings["t"]),
    sev_ppf(rv.cdf(actuarial_table_bearings["t"])),
    "--",
    label=f"MLE Wb({np.round(wbf.lambda_, 3)}, {np.round(wbf.rho_, 3)})",
    color="red",
)

ax2.plot(
    np.log(actuarial_table_bearings["t"]),
    sev_ppf(actuarial_table_bearings["logit_CI_95_ub"]),
    "-o",
    label="Non-Parametric CI UB",
    color="slateblue",
)
ax2.scatter(
    np.log(actuarial_table_bearings["t"]),
    sev_ppf(actuarial_table_bearings["ploting_position"]),
    label="Non-Parametric CDF",
    color="black",
)
ax2.plot(
    np.log(actuarial_table_bearings["t"]),
    sev_ppf(actuarial_table_bearings["logit_CI_95_lb"]),
    "-o",
    label="Non-Parametric CI LB",
    color="slateblue",
)

ax3.plot(
    actuarial_table_bearings["t"],
    actuarial_table_bearings["logit_CI_95_ub"],
    "-o",
    label="Non-Parametric CI UB",
    color="slateblue",
)
ax3.scatter(
    actuarial_table_bearings["t"],
    actuarial_table_bearings["F_hat"],
    label="Non-Parametric CDF",
    color="black",
)
ax3.plot(
    actuarial_table_bearings["t"],
    actuarial_table_bearings["logit_CI_95_lb"],
    "-o",
    label="Non-Parametric CI LB",
    color="slateblue",
)
lnf.plot_cumulative_density(ax=ax3, color="cyan")
wbf.plot_cumulative_density(ax=ax3, color="darkorchid")


ax2.set_title("Linearizing the Weibull CDF", fontsize=20)
ax.set_title("Linearizing the Lognormal CDF", fontsize=20)
ax3.set_title("MLE CDF Fits", fontsize=20)
ax.legend()
ax.set_xlabel("Time")
ax2.set_xlabel("Time")
xticks = np.round(np.linspace(0, actuarial_table_bearings["t"].max(), 10), 1)
yticks = np.round(np.linspace(0, actuarial_table_bearings["F_hat"].max(), 10), 4)
ax.set_xticklabels(xticks)
ax.set_yticklabels(yticks)
ax2.set_xticklabels(xticks)
ax2.set_yticklabels([])
ax2.legend()
ax.set_ylabel("Fraction Failing");
```

## Bayesian Modelling

We've now seen how to model and visualise the parametric model fits to sparse reliability using a frequentist or MLE framework. We want to now show how the same style of inferences can be achieved in the Bayesian paradigm. 

As in the MLE paradigm we need to model the censored liklihood. For most log-location distributions we've seen above the likelihood is expressed as a function of a combination of the distribution pdf $\phi$ and cdf $\Phi$ applied as appropriately depending on whether or not the data point was fully observed in the time window or censored. 


$$ L(\mu, \sigma) =  \prod_{i = 1}^{n} \Bigg(\dfrac{1}{\sigma t_{i}} \phi\Bigg[ \dfrac{log(t_{i}) - \mu}{\sigma}  \Bigg] \Bigg)^{\delta_{i}} \cdot \Bigg(1 - \Phi \Bigg[ \dfrac{log(t_{i}) - \mu}{\sigma} \Bigg]   \Bigg)^{1-\delta}$$

where $\delta_{i}$ is an indicator for whether the observation is a faiure or a right censored observation. More complicated types of censoring can be included with similar modifications of the CDF depending on the nature of the censored observations.

+++

### Direct PYMC implementation of Weibull Survival

We'll first model the Weibull likelihood directly in terms of the parameters $\alpha, \beta$, and then consider an alternative parameterisation.

```{code-cell} ipython3
def weibull_lccdf(y, alpha, beta):
    """Log complementary cdf of Weibull distribution."""
    return -((y / beta) ** alpha)


item_period_max = item_period_bearing_cage.groupby("id")[["t", "failed"]].max()
y = item_period_max["t"].values
censored = ~item_period_max["failed"].values.astype(bool)

with pm.Model() as model:

    beta = pm.Uniform("beta", 100, 15_000)
    alpha = pm.TruncatedNormal("alpha", 2, 0.5, lower=0.02, upper=8)

    y_obs = pm.Weibull("y_obs", alpha=alpha, beta=beta, observed=y[~censored])
    y_cens = pm.Potential("y_cens", weibull_lccdf(y[censored], alpha, beta))
    idata = pm.sample_prior_predictive()
    idata.extend(pm.sample(random_seed=100, target_accept=0.95))
    idata.extend(pm.sample_posterior_predictive(idata))

pm.model_to_graphviz(model)
```

```{code-cell} ipython3
idata
```

```{code-cell} ipython3
az.plot_trace(idata);
```

```{code-cell} ipython3
az.summary(idata)
```

```{code-cell} ipython3
def ecdf(sample):

    # convert sample to a numpy array, if it isn't already
    sample = np.atleast_1d(sample)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob
```

```{code-cell} ipython3
alphas = az.extract_dataset(idata, group="posterior", num_samples=200)["alpha"].values
betas = az.extract_dataset(idata, group="posterior", num_samples=200)["beta"].values

mosaic = """AAAA
            BBCC"""
fig, axs = plt.subplot_mosaic(mosaic=mosaic, figsize=(20, 10))
axs = [axs[k] for k in axs.keys()]
ax = axs[0]
ax1 = axs[2]
ax2 = axs[1]
hist_data = []
for i in range(200):
    draws = pm.draw(pm.Weibull.dist(alpha=alphas[i], beta=betas[i]), 1000)
    qe, pe = ecdf(draws)
    lkup = dict(zip(pe, qe))
    hist_data.append([lkup[0.1], lkup[0.05]])
    ax.plot(qe, pe, color="slateblue", alpha=0.4)
hist_data = pd.DataFrame(hist_data, columns=["p10", "p05"])
draws = pm.draw(pm.Weibull.dist(alpha=np.mean(alphas), beta=np.mean(betas)), 1000)
qe, pe = ecdf(draws)
ax.plot(qe, pe, color="cyan", label="Expected CDF")
ax.plot(
    actuarial_table_bearings["t"],
    actuarial_table_bearings["logit_CI_95_ub"],
    "--",
    label="Non-Parametric CI UB",
    color="black",
)
ax.scatter(
    actuarial_table_bearings["t"],
    actuarial_table_bearings["F_hat"],
    label="Non-Parametric CDF",
    color="black",
    alpha=1,
)
ax.plot(
    actuarial_table_bearings["t"],
    actuarial_table_bearings["logit_CI_95_lb"],
    "--",
    label="Non-Parametric CI LB",
    color="black",
)
ax.set_xlim(0, 2500)
ax.set_title(
    "Bayesian Estimation of Uncertainty in the CDF \n and the non-parametric estimates", fontsize=20
)
ax.set_ylabel("Fraction Failing")
ax.set_xlabel("Time")
ax1.hist(hist_data["p10"], bins=30, ec="black", color="skyblue", alpha=0.4)
ax1.set_title("Distribution of 10% failure Time", fontsize=20)
ax2.hist(hist_data["p05"], bins=30, ec="black", color="cyan", alpha=0.4)
ax2.set_title("Distribution of 5% failure Time", fontsize=20)
ax.legend()
ax.set_ylim(0, 0.1);
```

## Authors

Nathaniel Forde

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p aesara,aeppl,xarray
```

:::{include} ../page_footer.md
:::
