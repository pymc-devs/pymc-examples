---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: pymc_examples_new
  language: python
  name: pymc_examples_new
myst:
  substitutions:
    extra_dependencies: lifelines
---

(frailty_models)=
# Frailty and Survival Regression Models

:::{post} November, 2023
:tags: frailty models, survival analysis, competing risks, model comparison
:category: intermediate, reference
:author: Nathaniel Forde
:::

+++

:::{include} ../extra_installs.md
:::

```{code-cell} ipython3
import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from lifelines import KaplanMeierFitter
from matplotlib import cm
from matplotlib.lines import Line2D
from scipy.stats import fisk, weibull_min
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'  # high resolution figures
az.style.use("arviz-darkgrid")
rng = np.random.default_rng(42)
```

The full generality and range of application for survival analysis is masked by the loaded semantics of medical jargon. It is obscured by the persistent anxiety of self-concern amidst life tracked across calendars and milestones. But survival analysis broadly construed is not about you, it's not even necessarily about survival.  

It requires an extra step in abstraction to move from the medical context towards seeing that time-to-event data is everywhere! Every task which has an implicit clock, every goal with a finish line, every reaper owed a toll - these are sources of time-to-event data. 

We will demonstrate how the concepts of survival based regression analysis, traditionally deployed in the medical setting, can be fruitfully applied to HR data and business process analysis. In particular, we'll look at the question of time-to-attrition in employee life-cycle data and model this phenomena as a function of employee survey responses recorded earlier in the year. 

### Survival Regression Models

The emphasis here is on the generality of the framework. We are describing the trajectory of state-transitions within time. Anywhere speed or efficiency matters, it is important to understand the inputs to time-to-event trajectories. This is the benefit of survival analysis - clearly articulated models which quantify the impact of demographic characteristics and treatment effects (in terms of speed) on the probability of state-transition. Movement between life and death, hired and fired, ill and cured, subscribed to churned. These state transitions are all tranparently and compellingly modelled using survival regression models. 

We will see two varieties of regression modelling with respect to time-to-event data: (1) Cox's Proportional Hazard approach and (2) the Accelerated Failure time models. Both models enable the analyst to combine and assess the impacts of different covariates on the survival time outcomes, but each does so in a slightly different manner. 

We will also show a hierarchical variant of survival modelling called frailty modelling, where we estimate the survival function using regression but allow for the inclusion of individual or groups specific "frailty" terms. These are a multiplicative factor applied to the estimation routine of an individual's survival curve allowing us to capture some of the unexplained heterogeneity in the population. Additionally we will show how to express stratified approaches to estimating the baseline hazards. Throughout we will draw on the discussion in {cite:t}`collett2014survival`.

+++

## Exploration of the Data

People Analytics is inherently about the understanding of efficiency and risk in business - survival analysis is uniquely suited to elucidating these dual concerns. Our example data is drawn from a HR themed case discussed in Keith McNulty's [Handbook of Regression Modelling in People Analytics](https://peopleanalytics-regression-book.org/survival.html) {cite:t}`mcknulty2020people`. 

The data describes survey responses to questions about job satisfaction and the respondents intention to seek employment elsewhere. Additionally the data has broad "demographic" information of the respondent and crucially indications of whether they `left` employment at the company and on which `month` after the survey we still have record of them at the company. We want to understand the probability of attrition over time as a function of the employee survey responses to help (a) manage the risk of being caught short-handed and (b) ensure efficiency through the maintenance of a suitably staffed company. 

It's important to note that this kind of data is invariably censored data, since it is always pulled at a point in time. So there are some people for whom which we do not see an exit event. They may never leave the company - but importantly at the point of measurement, we simply do not know if they will leave tomorrow... so the data is meaningfully censored at the point in time of measurement. Our modelling strategy needs to account for how that changes the probabilities in question as discussed in {ref}`GLM-truncated-censored-regression`.

```{code-cell} ipython3
try:
    retention_df = pd.read_csv(os.path.join("..", "data", "time_to_attrition.csv"))
except FileNotFoundError:
    retention_df = pd.read_csv(pm.get_data("time_to_attrition.csv"))


dummies = pd.concat(
    [
        pd.get_dummies(retention_df["gender"], drop_first=True),
        pd.get_dummies(retention_df["level"], drop_first=True),
        pd.get_dummies(retention_df["field"], drop_first=True),
    ],
    axis=1,
).rename({"M": "Male"}, axis=1)

retention_df = pd.concat([retention_df, dummies], axis=1).sort_values("Male").reset_index(drop=True)
retention_df.head()
```

We've added dummy-encoding of some of the categorical variables for use in regression models below. We drop the first encoded class because this avoids identification issues in the estimation procedure. Additionally this means that the coefficients estimated for each of these indicator variables have an interpretation relative to the dropped "reference" class.

First we'll look at a simple Kaplan Meier representation of the survival function estimated on our data. A survival function quantifies the probability that an event has not occurred before a given time i.e. the probability of employee attrition before a particular month. Naturally, different types of risk profile lead to different survival functions. Regression models, as is typical, help to parse the nature of that risk where the risk profile is too complicated to easily articulate. 

```{code-cell} ipython3
kmf = KaplanMeierFitter()
kmf.fit(retention_df["month"], event_observed=retention_df["left"])
kmf_hi = KaplanMeierFitter()
kmf_hi.fit(
    retention_df[retention_df["sentiment"] == 10]["month"],
    event_observed=retention_df[retention_df["sentiment"] == 10]["left"],
)
kmf_mid = KaplanMeierFitter()
kmf_mid.fit(
    retention_df[retention_df["sentiment"] == 5]["month"],
    event_observed=retention_df[retention_df["sentiment"] == 5]["left"],
)
kmf_low = KaplanMeierFitter()
kmf_low.fit(
    retention_df[retention_df["sentiment"] == 2]["month"],
    event_observed=retention_df[retention_df["sentiment"] == 2]["left"],
)

fig, axs = plt.subplots(1, 2, figsize=(20, 15))
axs = axs.flatten()
ax = axs[0]
for i in retention_df.sample(30).index[0:30]:
    temp = retention_df[retention_df.index == i]
    event = temp["left"].max() == 1
    level = temp["level"].unique()
    duration = temp["month"].max()
    color = np.where(level == "High", "red", np.where(level == "Medium", "slateblue", "grey"))
    ax.hlines(i, 0, duration, color=color)
    if event:
        ax.scatter(duration, i, color=color)
ax.set_title("Assorted Time to Attrition \n by Level", fontsize=20)
ax.set_yticklabels([])
from matplotlib.lines import Line2D

custom_lines = [
    Line2D([0], [0], color="red", lw=4),
    Line2D([0], [0], color="slateblue", lw=4),
    Line2D([0], [0], color="grey", lw=4),
]

ax.legend(custom_lines, ["High Sentiment", "Medium Sentiment", "Low Sentiment"])


kmf_hi.plot_survival_function(ax=axs[1], label="KM estimate for High Sentiment", color="red")
kmf_mid.plot_survival_function(
    ax=axs[1], label="KM estimate for Medium Sentiment", color="slateblue"
)
kmf_low.plot_survival_function(ax=axs[1], label="KM estimate for Low Sentiment", color="grey")
kmf.plot_survival_function(
    ax=axs[1], label="Overall KM estimate", color="cyan", at_risk_counts=True
)
axs[1].set_xlabel("Time in Months")
axs[1].set_title("Kaplan Meier Fits by Level", fontsize=20);
```

Here we've used the Kaplan Meier non-parametric estimate of the survival curve within levels of the `sentiment` variable to show how the anticipated levels of attrition over a 12 month period are modified by the levels of `sentiment` expressed by the participants at the outset of the trial period. This is just exploratory data analysis of the survival curves, but we want to understand how a probabilistic model can recover such survival curves and what are the appropriate interpretation of the probabilistic model. The lower the sentiment the faster attrition occurs. 

+++

## Data Preperation for Survival Regression

The idea behind Cox Proportional Hazard regression models is, put crudely, to treat the temporal component of risk seriously. We imagine a latent baseline hazard of occurrence over the time-interval. Michael Betancourt [asks](https://betanalpha.github.io/assets/case_studies/survival_modeling.html) that we think of the hazard as "the accumulation of some stimulating resource" that precedes the occurrence of an event. In failure modelling it can be imagined as sporadic increasing wear and tear. In the context of HR dyanamics it could be imagined as increasing frustration is the work-environment. In philosophy it could viewed as an articulation of the sorites paradox; how do chances change over time, as sand is piled higher, for us to identify a collection of individual grains as a heap?. This term is often denoted:

$$ \lambda_{0}(t)$$

It is combined multiplicatively in the Cox Regression with a linear covariate representation of the individual case: 

$$ \lambda_{0}(t) \cdot exp(\beta_{1}X_{1} + \beta_{2}X_{2}... \beta_{k}X_{k}) $$

and represents the baseline hazard at each point in time when the predictor variables are set at their baseline/reference levels. Which is to say any unit increase over 0 to any covariate $X_{i}$ in the regression model changes the baseline hazard. In our case we are looking at data with granularity of monthly entries. So we need to understand how the risk of attrition changes over the next 12 months subsequent to the date of the annual survey and how the covariate profile of each individual changes the baseline hazard.

These models can be estimated using the approach of Bayesian estimation outlined by Austin Rochford in {ref}`survival_analysis`. In what follows we build on his examples. First we specify the temporal dimension of risk, in our case we have intervals of one month over a year - representing time since the date of the survey response. 


```{code-cell} ipython3
intervals = np.arange(12)
intervals
```

We then arrange our data into a structure to show if and when each individual in the data set experienced an event of attrition. The columns here are indicators for each month and the rows represent each individual in data set. The values show a 1 if the employee left the company in that month and a 0 otherwise. 

```{code-cell} ipython3
n_employees = retention_df.shape[0]
n_intervals = len(intervals)
last_period = np.floor((retention_df.month - 0.01) / 1).astype(int)
employees = np.arange(n_employees)
quit = np.zeros((n_employees, n_intervals))
quit[employees, last_period] = retention_df["left"]

pd.DataFrame(quit)
```

As outlined in {ref}`Reliability Statistics and Predictive Calibration` the hazard function, the cumulative density function and the survival function of a event time distribution are all intimately related. In particular each of these can be described in relation to the set of individuals at risk at any given time in the sequence. The pool of those individuals at risk changes over time as people experience events of attrition. This changes the conditional hazard over time - with knock on implications for the implied survival function. To account for this in our estimation strategy we need to configure our data to flag who is at risk and when. 

```{code-cell} ipython3
exposure = np.greater_equal.outer(retention_df.month.to_numpy(), intervals) * 1
exposure[employees, last_period] = retention_df.month - intervals[last_period]
pd.DataFrame(exposure)
```

A 0 in this data structure means the employee has already quit and no longer exists in the "at-risk" pool at that point in time. Whereas a 1 in this structure means the pool is in the risk pool and should be used to calculate the instantenous hazard at that interval. 

With these structures we are now in a position to estimate our model. Following Austin Rochford's example we again use the Poisson trick to estimate the Proportional hazard model. This might be somewhat surprising because the Cox Proportional Hazard model is normally advertised as a semi-parametric model which needs to be estimated using a partial likelihood due to the piecewise nature of the baseline hazard component. 

The trick is to see that Poisson regression for event counts and CoxPH regression are linked through the parameters which determine the event-rate. In the case of predicting counts we need a latent risk of a event indexed to time by an offset for each time-interval. This ensures that the likelihood term for a kind of Poisson regression is similar enough to the likelihood under consideration in the Cox Proportional Hazard regression that we can substitute one for the other. In other words the Cox Proportional hazard model can be estimated as GLM using a Poisson likelihood where we specify an "off-set" or intercept modification for each point on the time-interval. Using Wilkinson notation we can write: 

$$ CoxPH(left, month) \sim gender + level $$

is akin to 

$$ left \sim glm(gender + level + (1 | month)) \\ \text{ where link is } Poisson  $$

which we estimate using the structures defined above and PyMC as follows: 

+++

## Fit Basic Cox Model with Fixed Effects

We'll set up a model factory function to fit the basic Cox proportional hazards model with different covariate specifications. We want to assess the differences in the model implications between a model that measures the intention to quit and one that does not. 

```{code-cell} ipython3
preds = [
    "sentiment",
    "Male",
    "Low",
    "Medium",
    "Finance",
    "Health",
    "Law",
    "Public/Government",
    "Sales/Marketing",
]
preds2 = [
    "sentiment",
    "intention",
    "Male",
    "Low",
    "Medium",
    "Finance",
    "Health",
    "Law",
    "Public/Government",
    "Sales/Marketing",
]


def make_coxph(preds):
    coords = {"intervals": intervals, "preds": preds, "individuals": range(len(retention_df))}

    with pm.Model(coords=coords) as base_model:
        X_data = pm.MutableData("X_data_obs", retention_df[preds], dims=("individuals", "preds"))
        lambda0 = pm.Gamma("lambda0", 0.01, 0.01, dims="intervals")

        beta = pm.Normal("beta", 0, sigma=1, dims="preds")
        lambda_ = pm.Deterministic(
            "lambda_",
            pt.outer(pt.exp(pm.math.dot(beta, X_data.T)), lambda0),
            dims=("individuals", "intervals"),
        )

        mu = pm.Deterministic("mu", exposure * lambda_, dims=("individuals", "intervals"))

        obs = pm.Poisson("obs", mu, observed=quit, dims=("individuals", "intervals"))
        base_idata = pm.sample(
            target_accept=0.95, random_seed=100, idata_kwargs={"log_likelihood": True}
        )

    return base_idata, base_model


base_idata, base_model = make_coxph(preds)
base_intention_idata, base_intention_model = make_coxph(preds2)
```

```{code-cell} ipython3
compare = az.compare({"sentiment": base_idata, "intention": base_intention_idata}, ic="waic")
compare
```

```{code-cell} ipython3
az.plot_compare(compare);
```

```{code-cell} ipython3
pm.model_to_graphviz(base_model)
```

We can see here how the structure of the model, while slightly different from a typical regression model, incorporates all the same elements. The observed variables are combined in a weighted sum that is fed forward to modify the outcome(s). In our case the outcomes are the hazards - or conditional risk at a specific point in time. It is this collection of estimates that serve as our view of the evolving nature of risk in the period. An obvious question then is how do the predictive variables contribute to the evolution of risk.

A secondary question is how does the instance by instance view of hazard translate into a view of the probability of survival over time? How can we move between the hazard based perspective to the survival base one?

+++

### Interpreting the Model Coefficients

We'll focus first on the differential implications for the input variables in our two models. The beta parameter estimates are recorded on the scale of the log hazard rate. See first how the `intention` predictor (a score measuring the survey participant's intention to quit) shifts the magnitude and sign of the parameter estimates achieved in the model which failed to include this variable. This is a simple but poignant reminder to ensure that we measure the right thing, and that the features/variables which go into our model compose a story about the data generating process whether we pay attention or not. 

```{code-cell} ipython3
m = (
    az.summary(base_idata, var_names=["beta"])
    .reset_index()[["index", "mean"]]
    .rename({"mean": "expected_hr"}, axis=1)
)
m1 = (
    az.summary(base_intention_idata, var_names=["beta"])
    .reset_index()[["index", "mean"]]
    .rename({"mean": "expected_intention_hr"}, axis=1)
)
m = m.merge(m1, left_on="index", right_on="index", how="outer")
m["exp(expected_hr)"] = np.exp(m["expected_hr"])
m["exp(expected_intention_hr)"] = np.exp(m["expected_intention_hr"])
m
```

Each individual model coefficient records an estimate of the impact on the log hazard ratio entailed by a unit increase in the input variable. Note how we have exponentiated the coefficients to return them to scale of the hazard ratio. For a predictor variable $X$ with coefficient $\beta$, the interpretation is as follows:

- If $exp(\beta)$ > 1: An increase in X is associated with an increased hazard (risk) of the event occurring.
- If $exp(\beta)$ < 1: An increase in X is associated with a decreased hazard (lower risk) of the event occurring.
- If $exp(\beta)$ = 1: X has no effect on the hazard rate.

So our case we can see that having an occupation in  the fields of Finance or Health would seem to induce a roughly 25% increase in the hazard risk of the event occuring over the baseline hazard. Interestingly we can see that the inclusion of the `intention` predictor seems to be important as a unit increase of the `intention` metric moves the dial similarly - and intention is a 0-10 scale. 

These are not time-varying - they enter __once__ into the weighted sum that modifies the baseline hazard. This is the proportional hazard assumption - that while the baseline hazard can change over time the difference in hazard induced by different levels in the covariates remains constant over time. The Cox model is popular because it allows us to estimate a changing hazard at each time-point and incorporates the impact of the demographic predictors multiplicatively across the period. The proportional hazards assumption does not always hold, and we'll see some adjustments below that can help deal with violations of the proportional hazards assumption. 

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(base_idata["posterior"]["lambda0"].mean(("draw", "chain")), color="black")
az.plot_hdi(
    range(12),
    base_idata["posterior"]["lambda0"],
    color="lightblue",
    ax=ax,
    hdi_prob=0.99,
    fill_kwargs={"label": "Baseline Hazard 99%", "alpha": 0.3},
    smooth=False,
)
az.plot_hdi(
    range(12),
    base_idata["posterior"]["lambda0"],
    color="lightblue",
    ax=ax,
    hdi_prob=0.50,
    fill_kwargs={"label": "Baseline Hazard 50%", "alpha": 0.8},
    smooth=False,
)
ax.legend()
ax.set_xlabel("Time")
ax.set_title("Expected Baseline Hazard", fontsize=20);
```

This is the baseline stimulus - the growing, sporadically shifting hazard that spurs the occurrence of attrition. We build regression models incorporating a slew of control variables and treatment indicators to evaluate what if any effect they have on changing the baseline hazard over time. Survival regression modelling is a transparent tool for analysing the impact of demographic and behavioural features of risk over time. Note the sharp increase at the end of an annual cycle.

+++

### Predicting Marginal Effects of CoxPH regression

We can make these interpretations a little more concrete by deriving the marginal effects on sample/fictional data. Now we define the relationship between the survival and cumulative hazard measures as a function of the baseline hazard. 

```{code-cell} ipython3
def cum_hazard(hazard):
    """Takes arviz.InferenceData object applies
    cumulative sum along baseline hazard"""
    return hazard.cumsum(dim="intervals")


def survival(hazard):
    """Takes arviz.InferenceData object transforms
    cumulative hazard into survival function"""
    return np.exp(-cum_hazard(hazard))


def get_mean(trace):
    """Takes arviz.InferenceData object marginalises
    over the chain and draw"""
    return trace.mean(("draw", "chain"))
```

The cumulative hazard smoothes out the jumpy nature of the base hazard function, giving us a cleaner picture of the degree of increased risk over time. This is (in turn) translated into our survival function, which nicely expresses the risk on the 0-1 scale. Next we set up a function to derive the survival and cumulative hazard functions for each individual conditional of their risk profile. 

```{code-cell} ipython3
def extract_individual_hazard(idata, i, retention_df, intention=False):
    beta = idata.posterior["beta"]
    if intention:
        intention_posterior = beta.sel(preds="intention")
    else:
        intention_posterior = 0
    hazard_base_m1 = idata["posterior"]["lambda0"]

    full_hazard_idata = hazard_base_m1 * np.exp(
        beta.sel(preds="sentiment") * retention_df.iloc[i]["sentiment"]
        + np.where(intention, intention_posterior * retention_df.iloc[i]["intention"], 0)
        + beta.sel(preds="Male") * retention_df.iloc[i]["Male"]
        + beta.sel(preds="Low") * retention_df.iloc[i]["Low"]
        + beta.sel(preds="Medium") * retention_df.iloc[i]["Medium"]
        + beta.sel(preds="Finance") * retention_df.iloc[i]["Finance"]
        + beta.sel(preds="Health") * retention_df.iloc[i]["Health"]
        + beta.sel(preds="Law") * retention_df.iloc[i]["Law"]
        + beta.sel(preds="Public/Government") * retention_df.iloc[i]["Public/Government"]
        + beta.sel(preds="Sales/Marketing") * retention_df.iloc[i]["Sales/Marketing"]
    )

    cum_haz_idata = cum_hazard(full_hazard_idata)
    survival_idata = survival(full_hazard_idata)
    return full_hazard_idata, cum_haz_idata, survival_idata, hazard_base_m1


def plot_individuals(retention_df, idata, individuals=[1, 300, 700], intention=False):
    fig, axs = plt.subplots(1, 2, figsize=(20, 7))
    axs = axs.flatten()
    colors = ["slateblue", "magenta", "darkgreen"]
    for i, c in zip(individuals, colors):
        haz_idata, cum_haz_idata, survival_idata, base_hazard = extract_individual_hazard(
            idata, i, retention_df, intention
        )
        axs[0].plot(get_mean(survival_idata), label=f"individual_{i}", color=c)
        az.plot_hdi(range(12), survival_idata, ax=axs[0], fill_kwargs={"color": c})
        axs[1].plot(get_mean(cum_haz_idata), label=f"individual_{i}", color=c)
        az.plot_hdi(range(12), cum_haz_idata, ax=axs[1], fill_kwargs={"color": c})
        axs[0].set_title("Individual Survival Functions", fontsize=20)
        axs[1].set_title("Individual Cumulative Hazard Functions", fontsize=20)
    az.plot_hdi(
        range(12),
        survival(base_hazard),
        color="lightblue",
        ax=axs[0],
        fill_kwargs={"label": "Baseline Survival"},
    )
    axs[0].plot(
        get_mean(survival(base_hazard)),
        color="black",
        linestyle="--",
        label="Expected Baseline Survival",
    )
    az.plot_hdi(
        range(12),
        cum_hazard(base_hazard),
        color="lightblue",
        ax=axs[1],
        fill_kwargs={"label": "Baseline Hazard"},
    )
    axs[1].plot(
        get_mean(cum_hazard(base_hazard)),
        color="black",
        linestyle="--",
        label="Expected Baseline Hazard",
    )
    axs[0].legend()
    axs[0].set_ylabel("Probability of Survival")
    axs[1].set_ylabel("Cumulative Hazard Risk")
    axs[0].set_xlabel("Time")
    axs[1].set_xlabel("Time")
    axs[1].legend()


#### Next set up test-data input to explore the relationship between levels of the variables.
test_df = pd.DataFrame(np.zeros((3, 15)), columns=retention_df.columns)
test_df["sentiment"] = [1, 5, 10]
test_df["intention"] = [1, 5, 10]
test_df["Medium"] = [0, 0, 0]
test_df["Finance"] = [0, 0, 0]
test_df["M"] = [1, 1, 1]
test_df
```

### The Intention Model

If we plot the marginal effects due to increases in the `intention` variable - in the model equipped to evaluate it, we see a sharp division in the individual predicted survival curves as implied by the significant and substantial parameter estimate seen in the coefficient table above for the `intention` variable.

```{code-cell} ipython3
plot_individuals(test_df, base_intention_idata, [0, 1, 2], intention=True)
```

Focus here on the plot on the right. The baseline cumulative hazard is represented in blue, where each subsequent curve represents the cumulative hazard for individuals with increasing scores on the `intention` metric i.e. with increased hazard. This pattern is inverted on the plot on the left - where instead we see how probability of survival decreases over time more sharply for those individuals which high `intention` values.

+++

### The Sentiment Model

If we submit the same test to a model unable to account for intention most of the weight falls on the differences specified between the sentiment recorded by the survey participant. Here we also see a seperation in the survival curves, but the effect is much less pronounced. 

```{code-cell} ipython3
plot_individuals(test_df, base_idata, [0, 1, 2], intention=False)
```

One major observation to note here is how much work is done by the baseline hazard in each model. In the model which can account for the influence of the `intention` metric the baseline hazard is lower. Suggesting the baseline hazard has to do more work. Other combinations of test-data and input specifications can be used to experiment with the conditional implications of the CoxPh model in this way. 

+++

### Make Predictions for Individual Characteristics

It's all well and good to use marginal effects analysis to better understand the impact of particular variables, but how can we use it to predict the likely trajectories within our pool of polled employees? Here we simply re-apply the model to our observed data set where each participant is characterised in some sense by the observable inputs of our model. 

We can use these characteristics to predict the survival curves of our current or future employee base and make interventions where necessary to mitigate the implied risk of attrition for these and similar employee risk profiles.  

```{code-cell} ipython3
def create_predictions(retention_df, idata, intention=False):
    cum_haz = {}
    surv = {}
    for i in range(len(retention_df)):
        haz_idata, cum_haz_idata, survival_idata, base_hazard = extract_individual_hazard(
            idata, i, retention_df, intention=intention
        )
        cum_haz[i] = get_mean(cum_haz_idata)
        surv[i] = get_mean(survival_idata)
    cum_haz = pd.DataFrame(cum_haz)
    surv = pd.DataFrame(surv)
    return cum_haz, surv


cum_haz_df, surv_df = create_predictions(retention_df, base_idata, intention=False)
surv_df
```

### Sample Survival Curves and their Marginal Expected Survival Trajectory.

We now plot these individual risk profiles and marginalise across the predicted survival curves.

```{code-cell} ipython3
cm_subsection = np.linspace(0, 1, 120)
colors_m = [cm.Purples(x) for x in cm_subsection]
colors = [cm.spring(x) for x in cm_subsection]


fig, axs = plt.subplots(1, 2, figsize=(20, 7))
axs = axs.flatten()
cum_haz_df.plot(legend=False, color=colors, alpha=0.05, ax=axs[1])
axs[1].plot(cum_haz_df.mean(axis=1), color="black", linewidth=4)
axs[1].set_title(
    "Individual Cumulative Hazard \n & Marginal Expected Cumulative Hazard", fontsize=20
)

surv_df.plot(legend=False, color=colors_m, alpha=0.05, ax=axs[0])
axs[0].plot(surv_df.mean(axis=1), color="black", linewidth=4)
axs[0].set_title("Individual Survival Curves \n  & Marginal Expected Survival Curve", fontsize=20)
axs[0].annotate(
    f"Expected Attrition by 6 months: {100*np.round(1-surv_df.mean(axis=1).iloc[6], 2)}%",
    (2, 0.5),
    fontsize=14,
    fontweight="bold",
);
```

The marginal survival curve here is a summary statistic just like measuring the average in simpler cases. It is characteristic of your sample data (the individuals in your sample), and as such you should only take it as an indicative or generalisable measure in so far as you're happy to say that your sample data is proportionally representative of the different characteristic features of risk in your population. Survival __modelling__ is not a substitute for sound experimental design, but it can be used to analyse experimental data.

In the HR context we might be interested in the time-to-attrition metrics under the impact of a management training programme, or lead time to production code in the context of a software development team when adopting agile practices or new tooling. Understanding policies that effect efficiency is good, understanding the rate at which policies effect efficiency is better.  

+++

## Accelerated Failure Time Models

Next we examine a parametric family of regression based survival models called accelerated failure time models (AFTs). These are regression models that seek to describe the survival function of interest with appeal to one or other of the canonical statistical distributions that can be neatly characterised with a set of location and scale parameters e.g. the Weilbull distribution, the Log-Logistic distribution and the LogNormal distribution to name a few. One advantage of these family of distributions is that we have access to more flexible hazard functions without having to explicitly parameterise the time-interval. 

See here for example how the log-logistic distribution exhibits a non-monotonic hazard function whereas the Weibull hazard is necessarily monotonic. This is an important observation if your theory of the case allows for rising and falling risks of event occurrence.

```{code-cell} ipython3
fig, axs = plt.subplots(2, 2, figsize=(20, 7))
axs = axs.flatten()


def make_loglog_haz(alpha, beta):
    ## This is the Log Logistic distribution
    dist = fisk(c=alpha, scale=beta)
    t = np.log(np.linspace(1, 13, 100))  # Time values
    pdf_values = dist.pdf(t)
    sf_values = dist.sf(t)
    haz_values = pdf_values / sf_values
    axs[0].plot(t, haz_values)
    axs[2].plot(t, sf_values)


def make_weibull_haz(alpha, beta):
    dist = weibull_min(c=alpha, scale=beta)
    t = np.linspace(1, 13, 100)  # Time values
    pdf_values = dist.pdf(t)
    sf_values = dist.sf(t)
    haz_values = pdf_values / sf_values
    axs[1].plot(t, haz_values)
    axs[3].plot(t, sf_values)


[make_loglog_haz(4, b) for b in np.linspace(0.5, 2, 4)]
[make_loglog_haz(a, 2) for a in np.linspace(0.2, 7, 4)]
[make_weibull_haz(25, b) for b in np.linspace(10, 15, 4)]
[make_weibull_haz(a, 3) for a in np.linspace(2, 7, 7)]
axs[0].set_title("Log-Logistic Hazard Function", fontsize=15)
axs[2].set_title("Log-Logistic Survival Function", fontsize=15)
axs[1].set_title("Weibull Hazard Function", fontsize=15)
axs[3].set_title("Weibull Survival Function", fontsize=15);
```

AFT models incorporate the explanatory variables in a regression model so that they act multiplicatively on the time scale effecting the rate at which an individual proceeds along the time axis. As such the model can be interpreted directly as parameterised by the speed of progression towards the event of interest. The Survival function of AFT models are generally specified as: 

$$ S_{i}(t) = S_{0}\Bigg(\frac{t}{exp(\alpha_{i}x_{i} + \alpha_{2}x_{2} ... \alpha_{p}x_{p})} \Bigg) $$

where $S_{0}$ is the baseline survival, but the model is often represented in log-linear form: 

$$ log (T_{i}) = \mu + \alpha_{i}x_{i} + \alpha_{2}x_{2} ... \alpha_{p}x_{p} + \sigma\epsilon_{i} $$

where we have the baseline survival function $S_{0} = P(exp(\mu + \sigma\epsilon_{i}) \geq t)$ modified by additional covariates. The details are largely important for the estimation strategies, but they show how the impact of risk can be decomposed here just as in the CoxPH model. The effects of the covariates are additive on the log-scale towards the acceleration factor induced by the individual's risk profile.

Below we'll estimate two AFT models: the weibull model and the Log-Logistic model. Ultimately we're just fitting a censored parametric distribution but we've allowed that that one of the parameters of each distribution is specified as a linear function of the explainatory variables. So the log likelihood term is just: 

$$ log(L) = \sum_{i}^{n} \Big[ c_{i}log(f(t)) + (1-c_{i})log(S(t))) \Big]  $$ 

where $f$ is the distribution pdf function , $S$ is the survival fucntion and $c$ is an indicator function for whether the observation is censored - meaning it takes a value in $\{0, 1\}$ depending on whether the individual is censored. Both $f$, $S$ are parameterised by some vector of parameters $\mathbf{\theta}$.  In the case of the Log-Logistic model we estimate it by transforming our time variable to a log-scale and fitting a logistic likelihood with parameters $\mu, s$. The resulting parameter fits can be adapted to recover the log-logistic survival function as we'll show below. In the case of the Weibull model the parameters are denote $\alpha, \beta$ respectively.

```{code-cell} ipython3
coords = {
    "intervals": intervals,
    "preds": [
        "sentiment",
        "intention",
        "Male",
        "Low",
        "Medium",
        "Finance",
        "Health",
        "Law",
        "Public/Government",
        "Sales/Marketing",
    ],
}

X = retention_df[
    [
        "sentiment",
        "intention",
        "Male",
        "Low",
        "Medium",
        "Finance",
        "Health",
        "Law",
        "Public/Government",
        "Sales/Marketing",
    ]
].copy()
y = retention_df["month"].values
cens = retention_df.left.values == 0.0


def logistic_sf(y, μ, s):
    return 1.0 - pm.math.sigmoid((y - μ) / s)


def weibull_lccdf(x, alpha, beta):
    """Log complementary cdf of Weibull distribution."""
    return -((x / beta) ** alpha)


def make_aft(y, weibull=True):
    with pm.Model(coords=coords, check_bounds=False) as aft_model:
        X_data = pm.MutableData("X_data_obs", X)
        beta = pm.Normal("beta", 0.0, 1, dims="preds")
        mu = pm.Normal("mu", 0, 1)

        if weibull:
            s = pm.HalfNormal("s", 5.0)
            eta = pm.Deterministic("eta", pm.math.dot(beta, X_data.T))
            reg = pm.Deterministic("reg", pt.exp(-(mu + eta) / s))
            y_obs = pm.Weibull("y_obs", beta=reg[~cens], alpha=s, observed=y[~cens])
            y_cens = pm.Potential("y_cens", weibull_lccdf(y[cens], alpha=s, beta=reg[cens]))
        else:
            s = pm.HalfNormal("s", 5.0)
            eta = pm.Deterministic("eta", pm.math.dot(beta, X_data.T))
            reg = pm.Deterministic("reg", mu + eta)
            y_obs = pm.Logistic("y_obs", mu=reg[~cens], s=s, observed=y[~cens])
            y_cens = pm.Potential("y_cens", logistic_sf(y[cens], reg[cens], s=s))

        idata = pm.sample_prior_predictive()
        idata.extend(
            pm.sample(target_accept=0.95, random_seed=100, idata_kwargs={"log_likelihood": True})
        )
        idata.extend(pm.sample_posterior_predictive(idata))
    return idata, aft_model


weibull_idata, weibull_aft = make_aft(y)
## Log y to ensure we're estimating a log-logistic random variable
loglogistic_idata, loglogistic_aft = make_aft(np.log(y), weibull=False)
```

```{code-cell} ipython3
compare = az.compare({"weibull": weibull_idata, "loglogistic": loglogistic_idata}, ic="waic")
compare
```

```{code-cell} ipython3
az.plot_compare(compare);
```

### Deriving Individual Survival Predictions from AFT models

From above we can see how the regression equation is calculated and enters into the Weibull likelihood as the $\beta$ term and the logistic distribution as the $\mu$ parameter. In both cases the $s$ parameter remains free to determine the shape of the distribution. Recall from above that the regression equation enters into the survival function as a denominator for the sequence of time-points $t$

$$ S_{i}(t) = S_{0}\Bigg(\frac{t}{exp(\alpha_{i}x_{i} + \alpha_{2}x_{2} ... \alpha_{p}x_{p})}\Bigg) $$

So the smaller the weighted sum the greater the **acceleration factor** induced by the individual's risk profile. 

### Weibull

The estimated parameters fit for each individual case can be directly fed into the Weibull survival function as the $\beta$ term to recover the predicted trajectories. 


```{code-cell} ipython3
fig, axs = plt.subplots(1, 2, figsize=(20, 7))
axs = axs.flatten()
#### Using the fact that we've already stored expected value for the regression equation
reg = az.summary(weibull_idata, var_names=["reg"])["mean"]
t = np.arange(1, 13, 1)
s = az.summary(weibull_idata, var_names=["s"])["mean"][0]
axs[0].hist(reg, bins=30, ec="black", color="slateblue")
axs[0].set_title(
    "Histogram of Acceleration Factors in the individual Weibull fits \n across our sample"
)
axs[1].plot(
    t,
    weibull_min.sf(t, s, scale=reg.iloc[0]),
    label=r"Individual 1 - $\beta$: " + f"{reg.iloc[0]}," + r"$\alpha$: " + f"{s}",
)
axs[1].plot(
    t,
    weibull_min.sf(t, s, scale=reg.iloc[1000]),
    label=r"Individual 1000 - $\beta$: " + f"{reg.iloc[1000]}," + r"$\alpha$: " + f"{s}",
)
axs[1].set_title("Comparing Impact of Individual Factor \n on Survival Function")
axs[1].legend();
```

```{code-cell} ipython3
diff = reg.iloc[1000] - reg.iloc[0]
pchange = np.round(100 * (diff / reg.iloc[1000]), 2)
print(
    f"In this case we could think of the relative change in acceleration \n factor between the individuals as representing a {pchange}% increase"
)
```

```{code-cell} ipython3
reg = az.summary(weibull_idata, var_names=["reg"])["mean"]
s = az.summary(weibull_idata, var_names=["s"])["mean"][0]
t = np.arange(1, 13, 1)
weibull_predicted_surv = pd.DataFrame(
    [weibull_min.sf(t, s, scale=reg.iloc[i]) for i in range(len(reg))]
).T

weibull_predicted_surv
```

### Log Logistic

In the case of the Logistic fit, we have derived parameter estimates that need to be transformed to recover the log-logistic survival curves that we aimed to estimate.

```{code-cell} ipython3
reg = az.summary(loglogistic_idata, var_names=["reg"])["mean"]
s = az.summary(loglogistic_idata, var_names=["s"])["mean"][0]
temp = retention_df
t = np.log(np.arange(1, 13, 1))
## Transforming to the Log-Logistic scale
alpha = np.round((1 / s), 3)
beta = np.round(np.exp(reg) ** s, 3)

fig, axs = plt.subplots(1, 2, figsize=(20, 7))
axs = axs.flatten()
axs[0].hist(reg, bins=30, ec="black", color="slateblue")
axs[0].set_title("Histogram of beta terms in the individual Log Logistic fits")
axs[1].plot(
    np.exp(t),
    fisk.sf(t, c=alpha, scale=beta.iloc[0]),
    label=r"$\beta$: " + f"{beta.iloc[0]}," + r"$\alpha$: " + f"{alpha}",
)
axs[1].plot(
    np.exp(t),
    fisk.sf(t, c=alpha, scale=beta.iloc[1000]),
    label=r"$\beta$: " + f"{beta.iloc[1000]}," + r"$\alpha$: " + f"{alpha}",
)
axs[1].set_title("Comparing Impact of Individual Factor \n on Survival Function")
axs[1].legend();
```

```{code-cell} ipython3
diff = beta.iloc[1000] - beta.iloc[0]
pchange = np.round(100 * (diff / beta.iloc[1000]), 2)
print(
    f"In this case we could think of the relative change in acceleration \n factor between the individuals as representing a {pchange}% increase"
)
```

```{code-cell} ipython3
loglogistic_predicted_surv = pd.DataFrame(
    [fisk.sf(t, c=alpha, scale=beta.iloc[i]) for i in range(len(reg))]
).T
loglogistic_predicted_surv
```

Both models fit comparable estimates for these two individuals. We'll see now how the marginal survival function compares across our entire sample of indivduals. 

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(20, 7))
ax.plot(
    loglogistic_predicted_surv.iloc[:, [1, 300]], label=["LL-Individual 1", "LL-Individual 300"]
)
ax.plot(
    loglogistic_predicted_surv.mean(axis=1),
    label="LL Marginal Survival Curve",
    linestyle="--",
    color="black",
    linewidth=4.5,
)
ax.plot(weibull_predicted_surv.iloc[:, [1, 300]], label=["W-Individual 1", "W-Individual 300"])
ax.plot(
    weibull_predicted_surv.mean(axis=1),
    label="W Marginal Survival Curve",
    linestyle="dotted",
    color="black",
    linewidth=4.5,
)
ax.plot(surv_df.iloc[:, [1, 300]], label=["CoxPH-Individual 1", "CoxPH-Individual 300"])
ax.plot(
    surv_df.mean(axis=1),
    label="CoxPH Marginal Survival Curve",
    linestyle="-.",
    color="black",
    linewidth=4.5,
)
ax.set_title(
    "Comparison predicted Individual Survival Curves and \n Marginal (expected) Survival curve across Sample",
    fontsize=25,
)
kmf.plot_survival_function(ax=ax, label="Overall KM estimate", color="black")
ax.set_xlabel("Time in Month")
ax.set_ylabel("Probability")
ax.legend();
```

Above we've plotted a sample of individual predicted survival functions from each model. Additionally we've plotted the marginal survival curve predicted by averaging row-wise across the sample of individuals in our data set. This marginal quantity is often a useful benchmark for comparing change over differing periods. It is a measure that can be compared year on year and time over time. 

+++

## Fit Model with Shared Frailty terms by Individual

One of the most compelling patterns in Bayesian regression modelling more generally is the ability to incorporate hierarchical structure. The analogue of the hierarchical survival model is the individual frailty survival model. But "frailities" do not need to be specified only at an individual level - so called "shared" frailities can be deployed at a group level e.g. across the `field`. 

In the above CoxPH models we fit the data to a standard regression formulation using indicator variables for different levels of the `field` variable which gets included in the weighted sum of the linear combination. With frailty models we instead allow the individual or group frailty term to enter into our model as a multiplicative factor over and above the combination of the baseline hazard with the weighted demographic characteristics of risk. This allows us to capture an estimate of the heterogenous effects accruing to being that particular individual or within that particular group. In our context these terms seeks to explain the "overly" long-term loyalty of some employees to a company despite other offers on the market. Additionally we can stratify baseline hazards e.g. for gender to capture varying degrees of risk over time as a function of their covariate profile. So our equation now becomes:

$$ \lambda_{i}(t) = z_{i}exp(\beta X)\lambda_{0}^{g}(t) $$

which can be estimated in the Bayesian fashion as seen below. Note how we must set a prior on the $z$ term which enters the equation multiplicatively. To set such a prior we reason that the individual heterogeneity will not induce more than 30% speed-up/slow-down in time to attrition and we select a gamma distribution as a prior over our frailty term. 

```{code-cell} ipython3
opt_params = pm.find_constrained_prior(
    pm.Gamma,
    lower=0.80,
    upper=1.30,
    mass=0.90,
    init_guess={"alpha": 1.7, "beta": 1.7},
)

opt_params
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(20, 6))
ax.hist(
    pm.draw(pm.Gamma.dist(alpha=opt_params["alpha"], beta=opt_params["beta"]), 1000),
    ec="black",
    color="royalblue",
    bins=30,
    alpha=0.4,
)
ax.set_title("Draws from Gamma constrained around Unity", fontsize=20);
```

```{code-cell} ipython3
preds = [
    "sentiment",
    "intention",
    "Low",
    "Medium",
    "Finance",
    "Health",
    "Law",
    "Public/Government",
    "Sales/Marketing",
]
preds3 = ["sentiment", "Low", "Medium"]


def make_coxph_frailty(preds, factor):
    frailty_idx, frailty_labels = pd.factorize(factor)
    df_m = retention_df[retention_df["Male"] == 1]
    df_f = retention_df[retention_df["Male"] == 0]
    coords = {
        "intervals": intervals,
        "preds": preds,
        "frailty_id": frailty_labels,
        "gender": ["Male", "Female"],
        "women": df_f.index,
        "men": df_m.index,
        "obs": range(len(retention_df)),
    }

    with pm.Model(coords=coords) as frailty_model:
        X_data_m = pm.MutableData("X_data_m", df_m[preds], dims=("men", "preds"))
        X_data_f = pm.MutableData("X_data_f", df_f[preds], dims=("women", "preds"))
        lambda0 = pm.Gamma("lambda0", 0.01, 0.01, dims=("intervals", "gender"))
        sigma_frailty = pm.Normal("sigma_frailty", opt_params["alpha"], 1)
        mu_frailty = pm.Normal("mu_frailty", opt_params["beta"], 1)
        frailty = pm.Gamma("frailty", mu_frailty, sigma_frailty, dims="frailty_id")

        beta = pm.Normal("beta", 0, sigma=1, dims="preds")

        ## Stratified baseline hazards
        lambda_m = pm.Deterministic(
            "lambda_m",
            pt.outer(pt.exp(pm.math.dot(beta, X_data_m.T)), lambda0[:, 0]),
            dims=("men", "intervals"),
        )
        lambda_f = pm.Deterministic(
            "lambda_f",
            pt.outer(pt.exp(pm.math.dot(beta, X_data_f.T)), lambda0[:, 1]),
            dims=("women", "intervals"),
        )
        lambda_ = pm.Deterministic(
            "lambda_",
            frailty[frailty_idx, None] * pt.concatenate([lambda_f, lambda_m], axis=0),
            dims=("obs", "intervals"),
        )

        mu = pm.Deterministic("mu", exposure * lambda_, dims=("obs", "intervals"))

        obs = pm.Poisson("outcome", mu, observed=quit, dims=("obs", "intervals"))
        frailty_idata = pm.sample(random_seed=101)

    return frailty_idata, frailty_model


frailty_idata, frailty_model = make_coxph_frailty(preds, range(len(retention_df)))
pm.model_to_graphviz(frailty_model)
```

Fitting the above model allows us to pull out the gender specific view on the baseline hazard. This kind of model specification can help account for failures of the proportional hazards assumption allowing for the expression of time-varying risk induced by different levels of the covariates. We can also allow for shared frailty terms across groups as here in the case of allowing group effect based on the `field` of work. Often however this is not too distinct from including the field as a fixed effect in your model as we did above in the first CoxPH model, but here we allow that the coefficient estimates are drawn from the same distribution. The variance characteristics of this distribution may be of independent interest. 

The greater the variance here - the worse the base model is at capturing the observed state-transitions. In thinking about the evolving hazard in the context of the sorites paradox, you might argue that the greater the heterogeniety in the individual frailty terms the less well-specified model, the poorer our understanding of the state-transition in question - leading to the semantic ambiguity of when sand becomes a heap and greater uncertainty around when an employee is likely to leave. 

Next we'll fit a mode with frailties across the `field` grouping. These are called shared frailties.

```{code-cell} ipython3
shared_frailty_idata, shared_frailty_model = make_coxph_frailty(preds3, retention_df["field"])
```

```{code-cell} ipython3
pm.model_to_graphviz(shared_frailty_model)
```

The comparison between shared and individual frailty terms allows us to see how the inclusion of more covariates and individual frailty term absorbs the variance in the baseline hazard and shrinks the magnitude of the latent hazard.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(20, 6))
base_m = shared_frailty_idata["posterior"]["lambda0"].sel(gender="Male")
base_f = shared_frailty_idata["posterior"]["lambda0"].sel(gender="Female")
az.plot_hdi(range(12), base_m, ax=ax, color="lightblue", fill_kwargs={"alpha": 0.5}, smooth=False)
az.plot_hdi(range(12), base_f, ax=ax, color="red", fill_kwargs={"alpha": 0.3}, smooth=False)
get_mean(base_m).plot(ax=ax, color="darkred", label="Male Baseline Hazard Shared Frailty")
get_mean(base_f).plot(ax=ax, color="blue", label="Female Baseline Hazard Shared Frailty")

base_m_i = frailty_idata["posterior"]["lambda0"].sel(gender="Male")
base_f_i = frailty_idata["posterior"]["lambda0"].sel(gender="Female")
az.plot_hdi(range(12), base_m_i, ax=ax, color="cyan", fill_kwargs={"alpha": 0.5}, smooth=False)
az.plot_hdi(range(12), base_f_i, ax=ax, color="magenta", fill_kwargs={"alpha": 0.3}, smooth=False)
get_mean(base_m_i).plot(ax=ax, color="cyan", label="Male Baseline Hazard Individual Frailty")
get_mean(base_f_i).plot(ax=ax, color="magenta", label="Female Baseline Hazard Individual Frailty")


ax.legend()
ax.set_title("Stratified Baseline Hazards");
```

Let us to pull out and inspect the individual frailty terms:

```{code-cell} ipython3
frailty_terms = az.summary(frailty_idata, var_names=["frailty"])
frailty_terms.head()
```

and the shared terms across the groups. Where we can see how working in either Finance or Health seems to drive up the chances of attrition after controlling for other demographic information.

```{code-cell} ipython3
axs = az.plot_posterior(shared_frailty_idata, var_names=["frailty"])
axs = axs.flatten()
for ax in axs:
    ax.axvline(1, color="red", label="No change")
    ax.legend()
plt.suptitle("Shared Frailty Estimates across the Job Area", fontsize=30);
```

Shared frailty models such as this one are important in, for instance, medical trials where we want to measure the differences across institutions that are implementing a trial protocol. But similarly in the HR context we might imagine examining the the differential frailty terms across different manager/team dynamics. 

For now we'll leave that suggestion aside and focus on the individual frailty model.

```{code-cell} ipython3
ax = az.plot_forest(
    [base_idata, base_intention_idata, weibull_idata, frailty_idata],
    model_names=["coxph_sentiment", "coxph_intention", "weibull_sentiment", "frailty_intention"],
    var_names=["beta"],
    combined=True,
    figsize=(20, 15),
    r_hat=True,
)

ax[0].set_title("Parameter Estimates: Various Models", fontsize=20);
```

We can now pull apart the frailty estimates and compare them to the demographic information we know about each individual. Since we modelled the data without the intention variable it's interesting to see how the model tries to compensate for the impact of stated intention with the individual frailty term. 

```{code-cell} ipython3
temp = retention_df.copy()
temp["frailty"] = frailty_terms.reset_index()["mean"]
(
    temp.groupby(["Male", "sentiment", "intention"])[["frailty"]]
    .mean()
    .reset_index()
    .pivot(index=["Male", "sentiment"], columns="intention", values="frailty")
    .style.background_gradient(cmap="OrRd", axis=None)
    .set_precision(3)
)
```

The above heatmap suggests that the model over weights the impact of low sentiment and low intention score particularly. The frailty term(s) compensate by adding a reduction in the rate of the multiplicative increase in the hazard term. There is a general pattern that the model overweights the risk which is "corrected" downwards by the frailty terms. This makes a kind of sense as it's a little strange to see such low sentiment coupled with no intent to quit. Indicating that the respondent's answers might not reflect their considered opinion. The effect is similarly pronounced where intention to quit is higher, which also makes sense in this context too. 

+++

## Interrogating the Cox Frailty Model

As before we'll want to pull out the individual predicted survival functions and cumulative hazard functions. This can be done similarly to the analysis above, but here we include the mean frailty term to predict the individual hazard. 

```{code-cell} ipython3
def extract_individual_frailty(i, retention_df, intention=False):
    beta = frailty_idata.posterior["beta"]
    if intention:
        intention_posterior = beta.sel(preds="intention")
    else:
        intention_posterior = 0
    hazard_base_m = frailty_idata["posterior"]["lambda0"].sel(gender="Male")
    hazard_base_f = frailty_idata["posterior"]["lambda0"].sel(gender="Female")
    frailty = frailty_idata.posterior["frailty"]
    if retention_df.iloc[i]["Male"] == 1:
        hazard_base = hazard_base_m
    else:
        hazard_base = hazard_base_f

    full_hazard_idata = hazard_base * (
        frailty.sel(frailty_id=i).mean().item()
        * np.exp(
            beta.sel(preds="sentiment") * retention_df.iloc[i]["sentiment"]
            + np.where(intention, intention_posterior * retention_df.iloc[i]["intention"], 0)
            + beta.sel(preds="Low") * retention_df.iloc[i]["Low"]
            + beta.sel(preds="Medium") * retention_df.iloc[i]["Medium"]
            + beta.sel(preds="Finance") * retention_df.iloc[i]["Finance"]
            + beta.sel(preds="Health") * retention_df.iloc[i]["Health"]
            + beta.sel(preds="Law") * retention_df.iloc[i]["Law"]
            + beta.sel(preds="Public/Government") * retention_df.iloc[i]["Public/Government"]
            + beta.sel(preds="Sales/Marketing") * retention_df.iloc[i]["Sales/Marketing"]
        )
    )

    cum_haz_idata = cum_hazard(full_hazard_idata)
    survival_idata = survival(full_hazard_idata)
    return full_hazard_idata, cum_haz_idata, survival_idata, hazard_base


def plot_individual_frailty(retention_df, individuals=[1, 300, 700], intention=False):
    fig, axs = plt.subplots(1, 2, figsize=(20, 7))
    axs = axs.flatten()
    colors = ["slateblue", "magenta", "darkgreen"]
    for i, c in zip(individuals, colors):
        haz_idata, cum_haz_idata, survival_idata, base_hazard = extract_individual_frailty(
            i, retention_df, intention
        )
        axs[0].plot(get_mean(survival_idata), label=f"individual_{i}", color=c)
        az.plot_hdi(range(12), survival_idata, ax=axs[0], fill_kwargs={"color": c})
        axs[1].plot(get_mean(cum_haz_idata), label=f"individual_{i}", color=c)
        az.plot_hdi(range(12), cum_haz_idata, ax=axs[1], fill_kwargs={"color": c})
        axs[0].set_title("Individual Survival Functions", fontsize=20)
        axs[1].set_title("Individual Cumulative Hazard Functions", fontsize=20)
    az.plot_hdi(
        range(12),
        survival(base_hazard),
        color="lightblue",
        ax=axs[0],
        fill_kwargs={"label": "Baseline Survival"},
    )
    az.plot_hdi(
        range(12),
        cum_hazard(base_hazard),
        color="lightblue",
        ax=axs[1],
        fill_kwargs={"label": "Baseline Hazard"},
    )
    axs[0].legend()
    axs[1].legend()


plot_individual_frailty(retention_df, [0, 1, 2], intention=True)
```

In these plots we see a stark difference in the predicted survival functions for each individual explainted by the measure of their stated `intention` to leave. We can see this by examining the covariate profile of the three individuals.

```{code-cell} ipython3
retention_df.iloc[0:3, :]
```

```{code-cell} ipython3
def create_predictions(retention_df, intention=False):
    cum_haz = {}
    surv = {}
    for i in range(len(retention_df)):
        haz_idata, cum_haz_idata, survival_idata, base_hazard = extract_individual_frailty(
            i, retention_df, intention
        )
        cum_haz[i] = get_mean(cum_haz_idata)
        surv[i] = get_mean(survival_idata)
    cum_haz = pd.DataFrame(cum_haz)
    surv = pd.DataFrame(surv)
    return cum_haz, surv


cum_haz_frailty_df, surv_frailty_df = create_predictions(retention_df, intention=True)
surv_frailty_df
```

```{code-cell} ipython3
cm_subsection = np.linspace(0, 1, 120)
colors_m = [cm.Purples(x) for x in cm_subsection]
colors = [cm.spring(x) for x in cm_subsection]


fig, axs = plt.subplots(1, 2, figsize=(20, 7))
axs = axs.flatten()
cum_haz_frailty_df.plot(legend=False, color=colors, alpha=0.05, ax=axs[1])
axs[1].plot(cum_haz_frailty_df.mean(axis=1), color="black", linewidth=4)
axs[1].set_title(
    "Predicted Individual Cumulative Hazard \n & Expected Cumulative Hazard", fontsize=20
)

surv_frailty_df.plot(legend=False, color=colors_m, alpha=0.05, ax=axs[0])
axs[0].plot(surv_frailty_df.mean(axis=1), color="black", linewidth=4)
axs[0].set_title("Predicted Individual Survival Curves \n  & Expected Survival Curve", fontsize=20)
axs[0].annotate(
    f"Expected Attrition by 6 months: {np.round(1-surv_frailty_df.mean(axis=1).iloc[6], 3)}",
    (2, 0.5),
    fontsize=12,
    fontweight="bold",
);
```

Note the increased range of the survival curves induced by our additional frailty terms when compared to the above Cox model. 

+++

### Plotting the effects of the Frailty Terms

There are different ways to marginalise across the data, but we can also inspect the individual "frailties". These kinds of plots and investigations are most fruitful in the context of an ongoing policy shift. Where you want to determine the differential rates of response for those individuals undergoing/experiencing the policy shift first-hand versus those who are not. It helps to zero-in on the most impacted employees or participants in the study to figure out what if anything was driving their response, and if preventative measures need to be adopted to resolve a crisis.

```{code-cell} ipython3
beta_individual_all = frailty_idata["posterior"]["frailty"]
predicted_all = beta_individual_all.mean(("chain", "draw"))
predicted_all = predicted_all.sortby(predicted_all, ascending=False)
beta_individual = beta_individual_all.sel(frailty_id=range(500))
predicted = beta_individual.mean(("chain", "draw"))
predicted = predicted.sortby(predicted, ascending=False)
ci_lb = beta_individual.quantile(0.025, ("chain", "draw")).sortby(predicted)
ci_ub = beta_individual.quantile(0.975, ("chain", "draw")).sortby(predicted)
hdi = az.hdi(beta_individual, hdi_prob=0.5).sortby(predicted)
hdi2 = az.hdi(beta_individual, hdi_prob=0.8).sortby(predicted)
```

```{code-cell} ipython3
cm_subsection = np.linspace(0, 1, 500)
colors = [cm.cool(x) for x in cm_subsection]

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(
    2,
    2,
    height_ratios=(1, 7),
    left=0.1,
    right=0.9,
    bottom=0.1,
    top=0.9,
    wspace=0.05,
    hspace=0.05,
)
# Create the Axes.
ax = fig.add_subplot(gs[1, 0])
ax.set_yticklabels([])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histx.set_title("Expected Frailty Terms per Individual Risk Profile", fontsize=20)
ax_histx.hist(predicted_all, bins=30, color="slateblue")
ax_histx.set_yticklabels([])
ax_histx.tick_params(labelsize=8)
ax.set_ylabel("Individual Frailty Terms", fontsize=18)
ax.tick_params(labelsize=8)
ax.hlines(
    range(len(predicted)),
    hdi.sel(hdi="lower").to_array(),
    hdi.sel(hdi="higher").to_array(),
    color=colors,
    label="50% HDI",
    linewidth=0.8,
)
ax.hlines(
    range(len(predicted)),
    hdi2.sel(hdi="lower").to_array(),
    hdi2.sel(hdi="higher").to_array(),
    color="green",
    alpha=0.2,
    label="80% HDI",
    linewidth=0.8,
)
ax.set_xlabel("Multiplicative Effect of Individual Frailty", fontsize=18)
ax.legend()
ax.fill_betweenx(range(len(predicted)), 0.95, 1.0, alpha=0.4, color="grey")

ax1 = fig.add_subplot(gs[1, 1])
f_index = retention_df[retention_df["gender"] == "F"].index
index = retention_df.index
surv_frailty_df[list(range(len(f_index)))].plot(ax=ax1, legend=False, color="red", alpha=0.8)
surv_frailty_df[list(range(len(f_index), len(index), 1))].plot(
    ax=ax1, legend=False, color="royalblue", alpha=0.1
)
ax1_hist = fig.add_subplot(gs[0, 1])
f_index = retention_df[retention_df["gender"] == "F"].index
ax1_hist.hist(
    (1 - surv_frailty_df[list(range(len(f_index), len(index), 1))].iloc[6]),
    bins=30,
    color="royalblue",
    ec="black",
    alpha=0.8,
)
ax1_hist.hist(
    (1 - surv_frailty_df[list(range(len(f_index)))].iloc[6]),
    bins=30,
    color="red",
    ec="black",
    alpha=0.8,
)
ax1.set_xlabel("Time", fontsize=18)
ax1_hist.set_title(
    "Predicted Distribution of Attrition \n by 6 Months across all risk profiles", fontsize=20
)
ax1.set_ylabel("Survival Function", fontsize=18)
ax.scatter(predicted, range(len(predicted)), color="black", ec="black", s=30)

custom_lines = [Line2D([0], [0], color="red", lw=4), Line2D([0], [0], color="royalblue", lw=4)]
ax1.legend(custom_lines, ["Female", "Male"]);
```

Here we see a plot of the individual frailty terms and the differential multiplicative effect they contribute to each individual's predicted hazard. This is a powerful lens on the question of how much the observed covariates capture for each individual and how much of a corrective adjustment is implied by the frailty terms?

+++

## Conclusion

In this example we've seen how to model time-to-attrition in a employee lifecycle - we might also want to know how much time it will take to hire a replacement for the role! These applications of survival analysis can be applied routinely in industry wherever process efficiency is at issue. The better our understanding of risk over time, the better we can adapt to threats posed in heightened periods of risk. 

There are roughly two perspectives to be balanced: (i) the "actuarial" need to understand expected losses over the lifecycle, and (ii) the "diagnostic" needs to understand the causative factors that extend or reduce the lifecycle. Both are ultimately complementary as we need to "price in" differential flavours of risk that impact the expected bottom line whenever we plan for the future. Survival regression analysis neatly combines both these perspectives enabling the analyst to understand and take preventative action to offset periods of increased risk.

We've seen above a number of distinct regression modelling strategies for time-to-event data, but there are more flavours to explore: joint longitidunal models with a survival component, survival models with time-varying covariates, cure-rate models. The Bayesian perspective on these survival models is useful because we often have detailed results from prior years or experiments where our priors add useful perspective on the problem - allowing us to numerically encode that information to help regularise model fits for complex survival modelling. In the case of frailty models like the ones above - we've seen how priors can be added to the frailty terms to describe the influence of unobserved covariates which influence individual trajectories. Similarly the stratified approach to modelling baseline hazards allows us to carefully express trajectories of individual risk.  This can be especially important in the human centric disciplines where we seek to understand repeat measurments of the same individual time and again - accounting for the degree to which we can explain individual effects. Which is to say that while the framework of survival analysis suits a wide range of domains and problems, it nevertheless allows us to model, predict and infer aspects of specific and individual risk. 

+++

## Authors
- Authored by [Nathaniel Forde](https://nathanielf.github.io/) in November 2023 

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
