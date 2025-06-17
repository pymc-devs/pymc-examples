---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: default
  language: python
  name: python3
---

+++ {"papermill": {"duration": 0.016911, "end_time": "2020-03-27T06:09:14.400757", "exception": false, "start_time": "2020-03-27T06:09:14.383846", "status": "completed"}}

(bayesian_workflow)=

# The Bayesian Workflow

:::{post} Jun 16, 2025
:tags: workflow
:category: intermediate, how-to
:author: Thomas Wiecki, Chris Fonnesbeck
:::

```{code-cell} ipython3
---
papermill:
  duration: 2.069288
  end_time: '2020-03-27T06:09:16.527404'
  exception: false
  start_time: '2020-03-27T06:09:14.458116'
  status: completed
---
import warnings

import arviz as az
import load_covid_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

warnings.simplefilter("ignore")

sns.set_context("talk")

RANDOM_SEED = 8451997
sampler_kwargs = {"chains": 4, "cores": 4, "tune": 2000, "random_seed": RANDOM_SEED}
```

Strengths of Bayesian statistics that are critical here:
* Great flexibility to quickly and iteratively build statistical models
* Offers principled way of dealing with uncertainty
* Don't just want most likely outcome but distribution of all possible outcomes
* Allows expert information to guide model by using informative priors

In this section you'll learn:
* How to go from data to a model idea
* How to find priors for your model
* How to evaluate a model
* How to iteratively improve a model
* How to forecast into the future
* How powerful generative modeling can be

+++ {"papermill": {"duration": 0.009784, "end_time": "2020-03-27T06:09:16.547140", "exception": false, "start_time": "2020-03-27T06:09:16.537356", "status": "completed"}}

## Load data

First we'll load data on COVID-19 cases from the WHO. In order to ease analysis we will remove any days were confirmed cases was below 100 (as reporting is often very noisy in this time frame). It also allows us to align countries with each other for easier comparison.

```{code-cell} ipython3
---
papermill:
  duration: 1.663552
  end_time: '2020-03-27T06:09:18.220032'
  exception: false
  start_time: '2020-03-27T06:09:16.556480'
  status: completed
---
df = load_covid_data.load_data(drop_states=True, filter_n_days_100=2)
countries = df.country.unique()
n_countries = len(countries)
df = df.loc[lambda x: (x.days_since_100 >= 0)]
df.head()
```

## Bayesian Workflow

Next, we will start developing a model of the spread. These models will start out simple (and poor) but we will iteratively improve them. A good workflow to adopt when developing your own models is:

1. Plot the data
2. Build model
3. Run prior predictive check
4. Fit model
5. Assess convergence
6. Run posterior predictive check
7. Improve model

### 1. Plot the data

We will look at German COVID-19 cases. At first, we will only look at the first 30 days after Germany crossed 100 cases, later we will look at the full data.

```{code-cell} ipython3
country = "Germany"
date = "2020-07-31"
df_country = df.query(f'country=="{country}"').loc[:date].iloc[:30]

fig, ax = plt.subplots(figsize=(10, 8))
df_country.confirmed.plot(ax=ax)
ax.set(ylabel="Confirmed cases", title=country)
sns.despine()
```

Look at the above plot and think of what type of model you would build to model the data.

### 2. Build model

The above line kind of looks exponential. This matches with knowledge from epidemiology whereas early in an epidemic it grows exponentially.

```{code-cell} ipython3
# Get time-range of days since 100 cases were crossed
t = df_country.days_since_100.values
# Get number of confirmed cases for Germany
confirmed = df_country.confirmed.values

with pm.Model() as model_exp1:
    # Intercept
    a = pm.Normal("a", mu=0, sigma=100)

    # Slope
    b = pm.Normal("b", mu=0.3, sigma=0.3)

    # Exponential regression
    growth = a * (1 + b) ** t

    # Error term
    eps = pm.HalfNormal("eps", 100)

    # Likelihood
    pm.Normal("obs", mu=growth, sigma=eps, observed=confirmed)
```

```{code-cell} ipython3
pm.model_to_graphviz(model_exp1)
```

Just looking at the above model, what do you think? Is there anything you would have done differently?

+++

## 3. Run prior predictive check

Without even fitting the model to our data, we generate new potential data from our priors. Usually we have less intuition about the parameter space, where we define our priors, and more intution about what data we might expect to see. A prior predictive check thus allows us to make sure the model can generate the types of data we expect to see.

The process works as follows:

1. Pick a point from the prior $\theta_i$
2. Generate data set $x_i \sim f(\theta_i)$ where $f$ is our likelihood function (e.g. normal).
3. Rinse and repeat $n$ times.

```{code-cell} ipython3
with model_exp1:
    prior_pred = pm.sample_prior_predictive()
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(prior_pred.prior_predictive["obs"].values.squeeze().T, color="0.5", alpha=0.1)
ax.set(
    ylim=(-1000, 1000),
    xlim=(0, 10),
    title="Prior predictive",
    xlabel="Days since 100 cases",
    ylabel="Positive cases",
);
```

Does this look sensible? Why or why not? What is the prior predictive sample telling us?

+++

### What's wrong with this model?

Above you hopefully identified a few issues with this model:
1. Cases can't be negative
2. Cases can not start at 0, as we set it to start at above 100.
3. Case counts can't go down

Let's improve our model. The presence of negative cases is due to us using a Normal likelihood. Instead, let's use a `NegativeBinomial`, which is similar to `Poisson` which is commonly used for count-data but has an extra dispersion parameter that allows more flexiblity in modeling the variance of the data.

We will also change the prior of the intercept to be centered at 100 and tighten the prior of the slope.

The negative binomial distribution uses an overdispersion parameter, which we will describe using a gamma distribution. A companion package called `preliz`, a library for prior distribution elicitation, has a nice utility called `maxent` that will help us parameterize this prior, as the gamma distribution is not as intuitive to work with as the normal distribution.

```{code-cell} ipython3
import preliz as pz

gamma_params = pz.maxent(pz.Gamma(), lower=0.1, upper=20, mass=0.95)
gamma_params
```

```{code-cell} ipython3
plt.hist(pm.draw(pm.Gamma.dist(alpha=2, beta=0.2), 1000), bins=20);
```

```{code-cell} ipython3
t = df_country.days_since_100.values
confirmed = df_country.confirmed.values

with pm.Model() as model_exp2:
    # Intercept
    a = pm.Normal("a", mu=100, sigma=25)

    # Slope
    b = pm.Normal("b", mu=0.3, sigma=0.1)

    # Exponential regression
    growth = a * (1 + b) ** t

    alpha = pz.maxent(pz.Gamma(), lower=0.1, upper=20, mass=0.95, plot=False).to_pymc("alpha")

    # Likelihood
    pm.NegativeBinomial("obs", growth, alpha=alpha, observed=confirmed)
```

```{code-cell} ipython3
with model_exp2:
    prior_pred = pm.sample_prior_predictive(random_seed=RANDOM_SEED)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(prior_pred.prior_predictive["obs"].values.squeeze().T, color="0.5", alpha=0.1)
ax.set(
    ylim=(-100, 1000),
    xlim=(0, 10),
    title="Prior predictive",
    xlabel="Days since 100 cases",
    ylabel="Positive cases",
);
```

```{code-cell} ipython3
with model_exp2:
    trace_exp2 = pm.sample(**sampler_kwargs)
```

That looks much better. However, we can include even more prior information. For example, we know that the intercept *can not* be below 100 because of how we sliced the data. We can thus create a prior that does not have probability mass below 100. For this, we use the PyMC `HalfNormal` distribution; we can apply the same for the slope which we know is not going to be negative.

```{code-cell} ipython3
t = df_country.days_since_100.values
confirmed = df_country.confirmed.values

with pm.Model() as model_exp3:
    # Intercept
    a0 = pm.HalfNormal("a0", sigma=25)
    a = pm.Deterministic("a", a0 + 100)

    # Slope
    b = pm.HalfNormal("b", sigma=0.2)

    # Exponential regression
    growth = a * (1 + b) ** t

    gamma_params = pm.find_constrained_prior(
        pm.Gamma, lower=0.1, upper=20, init_guess={"alpha": 6, "beta": 1}, mass=0.95
    )
    alpha = pm.Gamma("alpha", **gamma_params)

    # Likelihood
    pm.NegativeBinomial("obs", growth, alpha=alpha, observed=confirmed)

    prior_pred = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
```

```{code-cell} ipython3
sns.histplot(prior_pred.prior["a"].squeeze(), legend=False)
plt.title("Prior of a");
```

```{code-cell} ipython3
sns.histplot(prior_pred.prior["b"].squeeze(), legend=False)
plt.title("Prior of b");
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(az.extract(prior_pred.prior_predictive)["obs"], color="0.5", alpha=0.1)
ax.set(
    ylim=(0, 1000),
    xlim=(0, 10),
    title="Prior predictive",
    xlabel="Days since 100 cases",
    ylabel="Positive cases",
);
```

Note that even though the intercept parameter can not be below 100 now, we still see data generated at below hundred. Why? 

+++

## 4. Fit model

```{code-cell} ipython3
with model_exp3:
    # Inference button (TM)
    trace_exp3 = pm.sample(**sampler_kwargs)
```

## 5. Assess convergence

```{code-cell} ipython3
az.plot_trace(trace_exp3, var_names=["a", "b", "alpha"])
plt.tight_layout();
```

```{code-cell} ipython3
az.summary(trace_exp3, var_names=["a", "b", "alpha"])
```

```{code-cell} ipython3
az.plot_energy(trace_exp3);
```

### Model comparison

Let's quickly compare the two models we were able to sample from.

Model comparison requires the log-likelihoods of the respective models. For efficiency, these are not computed automatically, so we need to manually calculate them.

```{code-cell} ipython3
with model_exp2:
    pm.compute_log_likelihood(trace_exp2)

with model_exp3:
    pm.compute_log_likelihood(trace_exp3)
```

Now we can use the ArviZ `compare` function:

```{code-cell} ipython3
comparison = az.compare({"exp2": trace_exp2, "exp3": trace_exp3})
az.plot_compare(comparison)
```

It seems like bounding the priors did not result in better fit. This is not unexpected because our change in prior was very small. We will still continue with `model_exp3` because we have prior information that these parameters are bounded in this way.

+++

### 6. Run posterior predictive check

Similar to the prior predictive, we can also generate new data by repeatedly taking samples from the posterior and generating data using these parameters.

```{code-cell} ipython3
with model_exp3:
    # Draw sampels from posterior predictive
    post_pred = pm.sample_posterior_predictive(trace_exp3.posterior, random_seed=RANDOM_SEED)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(
    post_pred.posterior_predictive["obs"].sel(chain=0).values.squeeze().T, color="0.5", alpha=0.05
)
ax.plot(confirmed, color="r", label="data")
ax.set(
    xlabel="Days since 100 cases",
    ylabel="Confirmed cases (log scale)",
    # ylim=(0, 100_000),
    title=country,
    yscale="log",
);
```

OK, that does not look terrible, the data is at least inside of what the model can produce. Let's look at residuals for systematic errors:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 8))
resid = post_pred.posterior_predictive["obs"].sel(chain=0) - confirmed
ax.plot(resid.T, color="0.5", alpha=0.01)
ax.set(ylim=(-50_000, 200_000), ylabel="Residual", xlabel="Days since 100 cases");
```

What can you see?

+++

### Prediction and forecasting

We might also be interested in predicting on unseen or data, or, in the case time-series data like here, in forecasting. In `PyMC` you can do so easily using `pm.Data` nodes. What it allows you to do is define data to a PyMC model that you can later switch out for other data. That way, when you for example do posterior predictive sampling, it will generate samples into the future.

Let's change our model to use `pm.Data` instead.

```{code-cell} ipython3
with pm.Model() as model_exp4:
    # pm.Data needs to be in the model context so that we can
    # keep track of it.
    # Then, we can then use it like any other array.
    t_data = pm.Data("t", df_country.days_since_100.values)
    confirmed_data = pm.Data("confirmed", df_country.confirmed.values)

    # Intercept
    a0 = pm.HalfNormal("a0", sigma=25)
    a = pm.Deterministic("a", a0 + 100)

    # Slope
    b = pm.HalfNormal("b", sigma=0.2)

    # Exponential regression
    growth = a * (1 + b) ** t_data

    # Likelihood
    pm.NegativeBinomial(
        "obs", growth, alpha=pm.Gamma("alpha", mu=6, sigma=1), observed=confirmed_data
    )

    trace_exp4 = pm.sample(**sampler_kwargs)
```

```{code-cell} ipython3
with model_exp4:
    # Update our data containers.
    # Recall that because confirmed is observed, we do not
    # need to specify any data, as that is only needed
    # during inference. But do have to update it to match
    # the shape.
    pm.set_data({"t": np.arange(60), "confirmed": np.zeros(60, dtype="int")})

    post_pred = pm.sample_posterior_predictive(trace_exp4.posterior, random_seed=RANDOM_SEED)
```

As we held data back before, we can now see how the predictions of the model

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(
    post_pred.posterior_predictive["obs"].sel(chain=0).squeeze().values.T, color="0.5", alpha=0.05
)
ax.plot(df_country.confirmed.values, color="r", label="in-sample")
df_confirmed = df.query(f'country=="{country}"').loc[:date, "confirmed"]
ax.plot(
    np.arange(29, len(df_confirmed)),
    df_confirmed.iloc[29:].values,
    color="b",
    label="out-of-sample",
)
ax.set(xlabel="Days since 100 cases", ylabel="Confirmed cases", title=country, yscale="log")
ax.legend();
```

## 7. Improve model

+++

### Logistic model

<img src="https://s3-us-west-2.amazonaws.com/courses-images-archive-read-only/wp-content/uploads/sites/924/2015/11/25202016/CNX_Precalc_Figure_04_07_0062.jpg"/>

```{code-cell} ipython3
df_country = df.query(f'country=="{country}"').loc[:date]

with pm.Model() as logistic_model:
    t_data = pm.Data("t", df_country.days_since_100.values)
    confirmed_data = pm.Data("confirmed", df_country.confirmed.values)

    # Intercept
    a0 = pm.HalfNormal("a0", sigma=25)
    intercept = pm.Deterministic("intercept", a0 + 100)

    # Slope
    b = pm.HalfNormal("b", sigma=0.2)

    carrying_capacity = pm.Uniform("carrying_capacity", lower=1_000, upper=80_000_000)
    # Transform carrying_capacity to a
    a = carrying_capacity / intercept - 1

    # Logistic
    growth = carrying_capacity / (1 + a * pm.math.exp(-b * t_data))

    # Likelihood
    pm.NegativeBinomial(
        "obs", growth, alpha=pm.Gamma("alpha", mu=6, sigma=1), observed=confirmed_data
    )

    prior_pred = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(prior_pred.prior_predictive["obs"].squeeze().values.T, color="0.5", alpha=0.1)
ax.set(
    title="Prior predictive",
    xlabel="Days since 100 cases",
    ylabel="Positive cases",
    yscale="log",
);
```

```{code-cell} ipython3
with logistic_model:
    # Inference
    trace_logistic = pm.sample(**sampler_kwargs, target_accept=0.9)

    # Sample posterior predcitive
    pm.sample_posterior_predictive(trace_logistic, extend_inferencedata=True)
```

```{code-cell} ipython3
az.plot_trace(trace_logistic)
plt.tight_layout();
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(
    trace_logistic.posterior_predictive["obs"].sel(chain=0).squeeze().values.T,
    color="0.5",
    alpha=0.05,
)
ax.plot(df_confirmed.values, color="r")
ax.set(xlabel="Days since 100 cases", ylabel="Confirmed cases", title=country);
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 8))
resid = (
    trace_logistic.posterior_predictive["obs"].sel(chain=0).squeeze().values - df_confirmed.values
)
ax.plot(resid.T, color="0.5", alpha=0.01)
ax.set(ylabel="Residual", xlabel="Days since 100 cases");
```

What is the difference between the residuals from before?

#### Model comparison

In order to compare our models we first need to refit with the longer data we now have. Fortunately we can easily swap out the data because these are `pm.Data` now.

```{code-cell} ipython3
with model_exp4:
    pm.set_data({"t": df_country.days_since_100.values, "confirmed": df_country.confirmed.values})

    trace_exp4_full = pm.sample(**sampler_kwargs)
```

```{code-cell} ipython3
with model_exp4:
    pm.compute_log_likelihood(trace_exp4_full)

with logistic_model:
    pm.compute_log_likelihood(trace_logistic)

az.plot_compare(az.compare({"exp4": trace_exp4_full, "logistic": trace_logistic}))
```

As you can see, the logistic model provides a much better fit to the data. 

Although there is still some small bias in the residuals but overall we might think our model is quite good. Let's see how it does on a different country.

```{code-cell} ipython3
country = "US"
# df_country = df.loc[lambda x: (x.country == country)]
df_country = df.query(f'country=="{country}"').loc[:date]
df_confirmed = df_country["confirmed"]
fig, ax = plt.subplots(figsize=(10, 8))
df_country.confirmed.plot(ax=ax)
ax.set(ylabel="Confirmed cases", title=country)
sns.despine()
```

As you can see, the data looks quite different. Let's see how our logistic model fits this.

```{code-cell} ipython3
# df_confirmed = df.loc[lambda x: (x.country == country), 'confirmed']
df_confirmed = df.query(f'country=="{country}"').loc[:date, "confirmed"]

with pm.Model() as logistic_model:
    t_data = pm.Data("t", df_country.days_since_100.values)
    confirmed_data = pm.Data("confirmed", df_country.confirmed.values)

    # Intercept
    a0 = pm.HalfNormal("a0", sigma=25)
    intercept = pm.Deterministic("intercept", a0 + 100)

    # Slope
    b = pm.HalfNormal("b", sigma=0.2)

    carrying_capacity = pm.Uniform("carrying_capacity", lower=1_000, upper=100_000_000)
    # Transform carrying_capacity to a
    a = carrying_capacity / intercept - 1

    # Logistic
    growth = carrying_capacity / (1 + a * pm.math.exp(-b * t_data))

    # Likelihood
    pm.NegativeBinomial(
        "obs", growth, alpha=pm.Gamma("alpha", mu=6, sigma=1), observed=confirmed_data
    )
```

```{code-cell} ipython3
with logistic_model:
    trace_logistic_us = pm.sample(**sampler_kwargs)
```

Already we see some problems with sampling which should make us suspicious that this model might not be the best for this data.

```{code-cell} ipython3
az.plot_trace(trace_logistic_us)
plt.tight_layout();
```

```{code-cell} ipython3
with logistic_model:
    pm.sample_posterior_predictive(
        trace_logistic_us, extend_inferencedata=True, random_seed=RANDOM_SEED
    )
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(
    trace_logistic_us.posterior_predictive["obs"].sel(chain=0).squeeze().values.T,
    color="0.5",
    alpha=0.05,
)
ax.plot(df_confirmed.values, color="r")
ax.set(xlabel="Days since 100 cases", ylabel="Confirmed cases", title=country);
```

As you can see, the model is not a great fit to this data. Why? What assumptions does the model make about the spread of COVID-19?

+++

## Authors
- Originally authored by Thomas Wiecki in 2020
- Adapted and expanded by Chris Fonnesbeck in June 2025 

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w
```

:::{include} ../page_footer.md
:::
