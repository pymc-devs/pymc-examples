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
---

(ordinal_regression)=
# Regression Models with Ordered Categorical Outcomes

:::{post} April, 2023
:tags: ordinal regression, generalized linear model, 
:category: beginner, reference
:author: Nathaniel Forde
:::

```{code-cell} ipython3
:tags: []

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import statsmodels.api as sm

from scipy.stats import bernoulli
from statsmodels.miscmodels.ordinal_model import OrderedModel
```

```{code-cell} ipython3
:tags: []

%config InlineBackend.figure_format = 'retina'  # high resolution figures
az.style.use("arviz-darkgrid")
rng = np.random.default_rng(42)
```

## Ordered Categories: Known Distribution

We'll start by generating some fake data. Imagine an employee/manager relationship where part of the annual process involves conducting a 360 degree review. The manager gets a rating (1 - 10) by their team and HR collects these ratings. The HR manager wants to know which factors influence the manager's rating and what can move a manager who receives a 4 into a 5 or a 7 into an 8. They have a theory that the rating is largely a function of salary. 

Ordinal Regression is a statistical technique designed to model these kinds of relationships. 

```{code-cell} ipython3
:tags: []

def make_data():
    salary = np.random.normal(40, 10, 500)
    work_sat = np.random.beta(1, 0.4, 500)
    work_from_home = bernoulli.rvs(0.7, size=500)
    latent_rating = (
        0.08423 * salary + 0.2 * work_sat + 0.4 * work_from_home + np.random.normal(0, 1, 500)
    )
    explicit_rating = np.round(latent_rating, 0)
    df = pd.DataFrame(
        {
            "salary": salary,
            "work_sat": work_sat,
            "work_from_home": work_from_home,
            "latent_rating": latent_rating,
            "explicit_rating": explicit_rating,
        }
    )
    return df


try:
    df = pd.read_csv("../data/fake_employee_manger_rating.csv")
except FileNotFoundError:
    df = make_data()

K = len(df["explicit_rating"].unique())
df.head()
```

We've specified our data in such a way that there is an underlying latent sentiment which is continuous in scale that gets crudely discretised to represent the ordinal rating scale. We've specified the data in such a way that salary drives a fairly linear increase in the manager's rating. 

```{code-cell} ipython3
:tags: []

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs = axs.flatten()
ax = axs[0]
ax.scatter(df["salary"], df["explicit_rating"], label="Explicit Rating", color="blue", alpha=0.3)
axs[1].scatter(
    df["work_from_home"], df["latent_rating"], label="Latent Rating", color="red", alpha=0.3
)
axs[1].scatter(
    df["work_from_home"], df["explicit_rating"], label="Explicit Rating", c="blue", alpha=0.3
)
axs[2].scatter(df["work_sat"], df["latent_rating"], label="Latent Rating", color="red", alpha=0.3)
axs[2].scatter(
    df["work_sat"], df["explicit_rating"], label="Explicit Rating", color="blue", alpha=0.3
)
ax.scatter(df["salary"], df["latent_rating"], label="Latent Sentiment", color="red")
ax.set_title("Manager Ratings by Salary")
axs[1].set_title("Manager Ratings by WFH")
axs[2].set_title("Manager Ratings by Work Satisfaction")
ax.set_ylabel("Latent Sentiment")
ax.set_xlabel("Employee Salary")
axs[1].set_xlabel("Work From Home: Y/N")
axs[2].set_xlabel("Employee Work Satisfaction")
axs[1].legend();
```

We can see here however that if we fit this model with a simple OLS fit it implies values beyond the categorical scale, which might motivate spurious salary increases by an overzealous HR manager. The OLS approximation is not bad, but is limited in that it cannot account for the proper nature of the outcome variable. 

```{code-cell} ipython3
:tags: []

exog = sm.add_constant(df[["salary", "work_from_home", "work_sat"]])
mod = sm.OLS(df["explicit_rating"], exog)
results = mod.fit()
results.summary()

results.predict([1, 200, 1, 0.6])
fig, axs = plt.subplots(1, 2, figsize=(20, 6))
axs = axs.flatten()
ax = axs[1]
salaries = np.linspace(10, 125, 20)
predictions = [results.predict([1, i, 1, 0.6])[0] for i in salaries]
ax.plot(salaries, predictions, label="Implied Linear function of Salaries on Outcome")
ax.set_title("Out of bound Prediction based on Salary")
ax.axhline(10, linestyle="--", color="black")
ax.set_xlabel("Hypothetical Salary")
ax.set_ylabel("Manager Rating Scale")
ax.axhline(0, linestyle="--", color="black")
axs[0].hist(results.resid, ec="white")
axs[0].set_title("Simple OLS Residuals on Training Data");
```

## Ordinal Regression Models: The Idea

In this notebook we'll show how to fit regression models to outcomes with ordered categories. These types of models can be considered as an application logistic regression models with multiple thresholds on a latent continuous scale. 

+++

## Fit a variety of Model Specifications

```{code-cell} ipython3
:tags: []

def constrainedUniform(N, min=0, max=1):
    return pm.Deterministic(
        "cutpoints",
        pt.concatenate(
            [
                np.ones(1) * min,
                pt.extra_ops.cumsum(pm.Dirichlet("cuts_unknown", a=np.ones(N - 2)))
                * (min + (max - min)),
            ]
        ),
    )
```

```{code-cell} ipython3
:tags: [hide-output]

def make_model(priors, model_spec=1, constrained_uniform=False, logit=True):
    with pm.Model() as model:

        if constrained_uniform:
            cutpoints = constrainedUniform(K, 0, K)
        else:
            sigma = pm.Exponential("sigma", priors["sigma"])
            cutpoints = pm.Normal(
                "cutpoints",
                mu=priors["mu"],
                sigma=sigma,
                transform=pm.distributions.transforms.univariate_ordered,
            )

        if model_spec == 1:
            beta = pm.Normal("beta", priors["beta"][0], priors["beta"][1], size=1)
            mu = pm.Deterministic("mu", beta[0] * df.salary)
        elif model_spec == 2:
            beta = pm.Normal("beta", priors["beta"][0], priors["beta"][1], size=2)
            mu = pm.Deterministic("mu", beta[0] * df.salary + beta[1] * df.work_sat)
        else:
            beta = pm.Normal("beta", priors["beta"][0], priors["beta"][1], size=3)
            mu = pm.Deterministic(
                "mu", beta[0] * df.salary + beta[1] * df.work_sat + beta[2] * df.work_from_home
            )
        if logit:
            y_ = pm.OrderedLogistic("y", cutpoints=cutpoints, eta=mu, observed=df.explicit_rating)
        else:
            y_ = pm.OrderedProbit("y", cutpoints=cutpoints, eta=mu, observed=df.explicit_rating)
        idata = pm.sample(nuts_sampler="numpyro", idata_kwargs={"log_likelihood": True})
        idata.extend(pm.sample_posterior_predictive(idata))
    return idata, model


priors = {"sigma": 1, "beta": [0, 1], "mu": np.linspace(0, K, K - 1)}
idata1, model1 = make_model(priors, model_spec=1)
idata2, model2 = make_model(priors, model_spec=2)
idata3, model3 = make_model(priors, model_spec=3)
idata4, model4 = make_model(priors, model_spec=3, constrained_uniform=True)
idata5, model5 = make_model(priors, model_spec=3, constrained_uniform=True)
```

```{code-cell} ipython3
:tags: []

az.summary(idata3, var_names=["sigma", "cutpoints", "beta"])
```

```{code-cell} ipython3
:tags: []

pm.model_to_graphviz(model3)
```

### Extracting the Probabilities 

```{code-cell} ipython3
:tags: []

implied_probs = az.extract(idata3, var_names=["y_probs"])
implied_probs.shape
```

```{code-cell} ipython3
:tags: []

implied_probs[0].mean(axis=1)
```

```{code-cell} ipython3
:tags: []

fig, ax = plt.subplots(figsize=(20, 6))
for i in range(K):
    ax.hist(implied_probs[0, i, :], label=f"Cutpoint: {i}", ec="white", bins=20)
ax.set_xlabel("Probability")
ax.set_title("Probability by Interval of Manager Rating \n by Individual 0")
ax.legend()
```

```{code-cell} ipython3
:tags: []

implied_class = az.extract(idata3, var_names=["y"], group="posterior_predictive")
implied_class.shape
```

```{code-cell} ipython3
:tags: []

from scipy.stats import mode

mode(implied_class[0])
```

```{code-cell} ipython3
:tags: []

fig, ax = plt.subplots(figsize=(20, 6))
ax.hist(implied_class[0], ec="white", bins=20)
ax.set_title("Distribution of Allocated Intervals for Individual O");
```

## Compare Models: Parameter Fits and LOO

```{code-cell} ipython3
:tags: []

compare = az.compare(
    {
        "model_salary": idata1,
        "model_salary_worksat": idata2,
        "model_full": idata3,
        "constrained_uniform": idata4,
        "probit_full": idata5,
    }
)

az.plot_compare(compare)
compare
```

```{code-cell} ipython3
:tags: []

ax = az.plot_forest(
    [idata1, idata2, idata3, idata4, idata5],
    var_names=["sigma", "beta", "cutpoints"],
    combined=True,
    ridgeplot_overlap=4,
    figsize=(20, 10),
    r_hat=True,
    ridgeplot_alpha=0.3,
    model_names=[
        "rating ~ salary",
        "rating ~ salary + work_sat",
        "rating ~ salary + work_sat + work_from_home",
        "full_constrained_uniform",
        "full_constrained_probit",
    ],
)
ax[0].set_title("Model Parameter Estimates", fontsize=20);
```

```{code-cell} ipython3
:tags: []

az.summary(idata3, var_names=["cutpoints", "beta", "sigma"])
```

## Compare Cutpoints: Normal V Uniform Priors

```{code-cell} ipython3
:tags: []

def plot_fit(idata):
    posterior = idata.posterior.stack(sample=("chain", "draw"))
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    axs = axs.flatten()
    ax = axs[0]
    for i in range(K - 1):
        ax.axvline(posterior["cutpoints"][i].mean().values, color="k")
    for r in df["explicit_rating"].unique():
        temp = df[df["explicit_rating"] == r]
        ax.hist(temp["latent_rating"], ec="white")
    ax.set_title("Latent Sentiment with Estimated Cutpoints")
    axs[1].set_title("Posterior Predictive Checks")
    az.plot_ppc(idata, ax=axs[1])
    plt.show()


plot_fit(idata3)
```

```{code-cell} ipython3
:tags: []

az.plot_posterior(idata3, var_names=["beta"], ref_val=[0.08432, 0.2, 0.4]);
```

While the parameter estimates seem reasonable and the posterior predictive checks seem good too, the point to see here is that the cutpoints are unconstrained by the definition of the ordinal scale. They vary below 0 in the above model.

```{code-cell} ipython3
:tags: []

plot_fit(idata4)
```

```{code-cell} ipython3
:tags: []

az.plot_posterior(idata4, var_names=["beta"], ref_val=[0.08432, 0.2, 0.4]);
```

Again the parameters seem reasonable, and posterior predictive checks are sound. But now, having using the constrained uniform prior over the cutpoints our estimated cutpoints respect the definition of the ordinal scale. 

+++

## Comparison to Statsmodels

```{code-cell} ipython3
:tags: []

modf_logit = OrderedModel.from_formula(
    "explicit_rating ~ salary + work_sat + work_from_home", df, distr="logit"
)
resf_logit = modf_logit.fit(method="bfgs")
resf_logit.summary()
```

## Kruschke's IMDB movie Ratings Data

There are substantial reasons for using an ordinal regression model rather than trusting to alternatives. The temptation to treat the ordered category as a continuous metric will lead to false inferences. The details are discussed in Kruschke's paper on this topic. We'll briefly replicate his example about this phenomenon can appear in analysis of movies ratings data.

```{code-cell} ipython3
:tags: []

movies = pd.read_csv("../data/MoviesData.csv")
movies.head()
```

```{code-cell} ipython3
:tags: []

import pandas as pd

movies = pd.read_csv("../data/MoviesData.csv")


def pivot_movie(row):
    row_ratings = row[["n1", "n2", "n3", "n4", "n5"]]
    totals = []
    for c, i in zip(row_ratings.index, range(5)):
        totals.append(row_ratings[c] * [i])
    totals = [item for sublist in totals for item in sublist]
    movie = [row["Descrip"]] * len(totals)
    id = [row["ID"]] * len(totals)
    return pd.DataFrame({"rating": totals, "movie": movie, "movie_id": id})


movies_by_rating = pd.concat([pivot_movie(movies.iloc[i]) for i in range(len(movies))])
movies_by_rating.reset_index(inplace=True, drop=True)
movies_by_rating.shape
```

```{code-cell} ipython3
:tags: []

movies_by_rating.sample(100).head()
```

```{code-cell} ipython3
:tags: []

def constrainedUniform(N, group, min=0, max=1):
    return pm.Deterministic(
        f"cutpoints_{group}",
        pt.concatenate(
            [
                np.ones(1) * min,
                pt.extra_ops.cumsum(pm.Dirichlet(f"cuts_unknown_{group}", a=np.ones(N - 2)))
                * (min + (max - min)),
            ]
        ),
    )
```

We will fit this data with both an ordinal model and as a metric. This will show how the ordinal fit is subtantially more compelling. 

```{code-cell} ipython3
:tags: []

K = 5
movies_by_rating = movies_by_rating[movies_by_rating["movie_id"].isin([1, 2, 3, 4, 5, 6])]
indx, unique = pd.factorize(movies_by_rating["movie_id"])
priors = {"sigma": 1, "mu": [0, 1], "cut_mu": np.linspace(0, K, K - 1)}
coords = {"movie_id": unique}
ordered = False


def make_movies_model(ordered=False):
    with pm.Model(coords=coords) as model:

        for g in movies_by_rating["movie_id"].unique():
            if ordered:
                cutpoints = constrainedUniform(K, g, 0, K - 1)
                mu = pm.Normal(f"mu_{g}", priors["mu"][0], priors["mu"][1])
                y_ = pm.OrderedLogistic(
                    f"y_{g}",
                    cutpoints=cutpoints,
                    eta=mu,
                    observed=movies_by_rating[movies_by_rating["movie_id"] == g].rating.values,
                )
            else:
                mu = pm.Normal(f"mu_{g}", 3, 1)
                sigma = pm.HalfNormal(f"sigma_{g}", 1)
                y_ = pm.Normal(
                    f"y_{g}",
                    mu,
                    sigma,
                    observed=movies_by_rating[movies_by_rating["movie_id"] == g].rating.values,
                )

        idata = pm.sample_prior_predictive()
        idata.extend(pm.sample(nuts_sampler="numpyro", idata_kwargs={"log_likelihood": True}))
        idata.extend(pm.sample_posterior_predictive(idata))
    return idata, model


idata_ordered, model_ordered = make_movies_model(ordered=True)
idata_normal_metric, model_normal_metric = make_movies_model(ordered=False)
```

### Posterior Predictive Fit: Normal Metric Model

This is a horrific fit to the movies rating data for six movies.

```{code-cell} ipython3
:tags: []

az.plot_ppc(idata_normal_metric);
```

### Posterior Predictive Fit: Ordered Response Model

This shows a much nicer fit for each of the six movies. 

```{code-cell} ipython3
:tags: []

az.plot_ppc(idata_ordered);
```

### Compare Inferences between Models

Aside from the predictive fits, the inference drawns from the different modelling choices also vary quite significantly.

```{code-cell} ipython3
:tags: []

fig, axs = plt.subplots(2, 3, figsize=(15, 7))
axs = axs.flatten()
ordered_5 = az.extract(idata_ordered.posterior)["mu_5"]
ordered_6 = az.extract(idata_ordered.posterior)["mu_6"]
diff = ordered_5 - ordered_6
metric_5 = az.extract(idata_normal_metric.posterior)["mu_5"]
metric_6 = az.extract(idata_normal_metric.posterior)["mu_6"]
diff1 = metric_5 - metric_6
axs[0].hist(ordered_5, bins=30, ec="white", color="slateblue")
axs[1].hist(diff, ec="white", label="Ordered Fit", bins=30)
axs[2].hist(ordered_6, bins=30, ec="white", color="cyan")
axs[3].hist(metric_5, ec="white", bins=30, color="magenta")
axs[4].hist(diff1, ec="white", label="Metric Fit", bins=30, color="red")
axs[5].hist(metric_6, ec="white", bins=30, color="pink")
axs[1].set_title("Difference Between the \n Expected Movie Ratings")
axs[0].set_title("Posterior Estimate of Mu for Movie 5")
axs[2].set_title("Posterior Estimate of Mu for Movie 6")
axs[1].legend()
axs[4].legend();
```

## Authors
- Authored by [Nathaniel Forde](https://github.com/NathanielF) in April 2023 

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
