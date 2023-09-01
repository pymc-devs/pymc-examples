---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: pymc_examples_new
  language: python
  name: python3
---

(ordinal_regression)=
# Regression Models with Ordered Categorical Outcomes

:::{post} April, 2023
:tags: ordinal regression, generalized linear model, 
:category: beginner, reference
:author: Nathaniel Forde
:::

```{code-cell} ipython3
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
%config InlineBackend.figure_format = 'retina'  # high resolution figures
az.style.use("arviz-darkgrid")
rng = np.random.default_rng(42)
```

# Ordinal Scales and Survey Data

Like many areas of statistics the language of survey data comes with an overloaded vocabulary. When discussing survey design you will often hear about the contrast between *design* based and *model* based approaches to (i) sampling strategies and (ii) statistical inference on the associated data. We won't wade into the details about different sample strategies such as: simple random sampling, cluster random sampling or stratified random sampling using population weighting schemes. The literature on each of these is vast, but in this notebook we'll talk about when any why it's useful to apply model driven statistical inference to [Likert](https://en.wikipedia.org/wiki/Likert_scale) scaled survey response data and other kinds of ordered categorical data. 

+++

## Ordered Categories: Known Distribution

We'll start by generating some fake data. Imagine an employee/manager relationship where part of the annual process involves conducting a 360 degree review. The manager gets a rating (1 - 10) by their team and HR collects these ratings. The HR manager wants to know which factors influence the manager's rating and what can move a manager who receives a 4 into a 5 or a 7 into an 8. They have a theory that the rating is largely a function of salary. 

Ordinal Regression is a statistical technique designed to **model** these kinds of relationships and can be contrasted to a design based approach where the focus is to extract simple statistical summaries e.g. proportions, counts or ratios in the context of a survey design under strong guarantees about the error tolerance in those derived summaries.

```{code-cell} ipython3
def make_data():
    np.random.seed(100)
    salary = np.random.normal(40, 10, 500)
    work_sat = np.random.beta(1, 0.4, 500)
    work_from_home = bernoulli.rvs(0.7, size=500)
    work_from_home_calc = np.where(work_from_home, 1.4 * work_from_home, work_from_home)
    latent_rating = (
        0.08423 * salary + 0.2 * work_sat + work_from_home_calc + np.random.normal(0, 1, 500)
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
    df = df[["salary", "work_sat", "work_from_home", "latent_rating", "explicit_rating"]]
except FileNotFoundError:
    df = make_data()

K = len(df["explicit_rating"].unique())
df.head()
```

We've specified our data in such a way that there is an underlying latent sentiment which is continuous in scale that gets crudely discretised to represent the ordinal rating scale. We've specified the data in such a way that salary drives a fairly linear increase in the manager's rating. 

```{code-cell} ipython3
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

We can see here however that if we fit this model with a simple OLS fit it implies values beyond the categorical scale, which might motivate spurious salary increases by an overzealous HR manager. The OLS approximation is limited in that it cannot account for the proper nature of the outcome variable. 

```{code-cell} ipython3
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

This suggests a reason for the contrast between *design* and *model* based approaches to inference with survey data. The modelling approach often hides or buries assumptions which makes the model infeasible, and the conservative approach tends towards inference under a design philosophy that avoids the risk of model misspecification. 

+++

## Ordinal Regression Models: The Idea

In this notebook we'll show how to fit regression models to outcomes with ordered categories to avoid one type of model misspecification. These types of models can be considered as an application logistic regression models with multiple thresholds on a latent continuous scale. The idea is that there is a latent metric which can be partitioned by the extremity of the measure, but we observe only the indicator for which partition of the scale an individual resides. This is quite a natural perspective e.g. imagine the bundling of complexity that gets hidden in crude political classifications: liberal, moderate and conservative. You may have a range of views on any number of political issues, but they all get collapsed in the political calculus to finite set of (generally poor) choices. Which of the last 10 political choices pushed you from liberal to moderate?


The idea is to treat the outcome variable (our categorical) judgment as deriving from an underlying continuous measure. We see the outcomes we do just when some threshold on that continuous measure has been achieved. The primary inferential task of ordinal regression is to derive an estimate of those thresholds in the latent continuous space. 

In the data set above we've explicitly specified the relationship, and in the following steps we'll estimate a variety of ordinal regression models.

+++

## Fit a variety of Model Specifications

The model specification for ordinal regression models typically makes use of the the logit transformation and the cumulative probabilities implied. For $c$ outcome categories with probabilities $\pi_1, .... \pi_n$ the *cumulative logits* are defined:

$$ logit[P(Y \leq j)]  = log \frac{P(Y \leq j)}{1 - p(Y \leq j)}  = log \frac{\pi_1 + ... + \pi_j}{\pi_{j+1} + ... + \pi_n} \text{ where  j = 1, ..., c-1} $$

This gets employed in a regression context where we specify the factors which determine our latent outcome in a linear fashion:

$$ logit[P(Y \leq j)] = \alpha_{j} + \beta'x $$

which implies that:

$$ P(Y \leq j) = \frac{exp(\alpha_{j} + \beta'x)}{1 + exp(\alpha_{j} + \beta'x)} $$

and that the probability for belonging within a particular category $j$ is determined by the probability of being in the cell defined by: 

$$ P(Y = j) = \frac{exp(\alpha_{j} + \beta'x)}{1 + exp(\alpha_{j} + \beta'x)} - \frac{exp(\alpha_{j-1} + \beta'x)}{1 + exp(\alpha_{j-1} + \beta'x)} $$

One nice feature of ordinal regressions specified in this fashion is that the interpretation of the coefficients on the beta terms remain the same across each interval on the latent space. The interpretaiton of the model parameters is typical: a unit increase in $x_{k}$ corresponds to an increase in $Y_{latent}$ of $\beta_{k}$ Similar interpretation holds for probit regression specification too. However we must be careful about comparing the interpretation of coefficients across different model specifications with different variables. The above coefficient interpretation makes sense as conditional interpretation based on holding fixed precisely the variables in the model. Adding or removing variables changes the conditionalisation which breaks the comparability of the models due the phenomena of non-collapsability. We'll show below how it's better to compare the models on their predictive implications using the posterior predictive distribution. 

### Bayesian Particularities 

While Ordinal regression is often performed in a frequentist paradigm, the same techniques can be applied in a Bayesian setting with all the benefits of posterior probability distributions and posterior predictive simulations. In PyMC there are at least two ways we can go about specifying the priors over the this model. The first one uses a constrained Dirichlet distribution to define a prior over the thresholds. The second method, a little looser allows the specification of any prior distribution with suitable number of cutpoints applying a ordering transformation on the generated samples from the prior distribution. 

We'll show both in this notebook, but as we'll see, the definition of the Dirchlet ensures properties which make it a better fit for the constrained outcome space. In each approach we can include covariates as in more standard regression models. 

PyMC has both `OrderedLogistic` and `OrderedProbit` distributions available. 

```{code-cell} ipython3
def constrainedUniform(N, min=0, max=1):
    return pm.Deterministic(
        "cutpoints",
        pt.concatenate(
            [
                np.ones(1) * min,
                pt.extra_ops.cumsum(pm.Dirichlet("cuts_unknown", a=np.ones(N - 2))) * (max - min)
                + min,
            ]
        ),
    )
```

The above function, (brainchild of Dr Ben Vincent and Adrian Seyboldt), looks a little indimidating, but it's just a convenience function to specify a prior over the cutpoints in our $Y_{latent}$. The Dirichlet distribution is special in that draws from the distribution must sum to one. The above function ensures that each draw from the prior distribution is a cumulative share of the maximum category greater than the minimum of our ordinal categorisation. 

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
        elif model_spec == 3:
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
az.summary(idata3, var_names=["sigma", "cutpoints", "beta"])
```

```{code-cell} ipython3
pm.model_to_graphviz(model3)
```

### Extracting Individual Probabilities 

We can now for each individual manager's rating, look at the probability associated with each of the available categories. Across the posterior distributions of our cuts which section of the latent continous measure the employee is most likely to fall into.

```{code-cell} ipython3
implied_probs = az.extract(idata3, var_names=["y_probs"])
implied_probs.shape
```

```{code-cell} ipython3
implied_probs[0].mean(axis=1)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(20, 6))
for i in range(K):
    ax.hist(implied_probs[0, i, :], label=f"Cutpoint: {i}", ec="white", bins=20, alpha=0.4)
ax.set_xlabel("Probability")
ax.set_title("Probability by Interval of Manager Rating \n by Individual 0", fontsize=20)
ax.legend();
```

```{code-cell} ipython3
implied_class = az.extract(idata3, var_names=["y"], group="posterior_predictive")
implied_class.shape
```

```{code-cell} ipython3
from scipy.stats import mode

mode(implied_class[0])
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(20, 6))
ax.hist(implied_class[0], ec="white", bins=20, alpha=0.4)
ax.set_title("Distribution of Allocated Intervals for Individual O", fontsize=20);
```

## Compare Models: Parameter Fits and LOO

+++

One tool for ameliorating the risk of model misspecification is to compare amongst different candidate model to check for  predictive accuracy. 

```{code-cell} ipython3
compare = az.compare(
    {
        "model_salary": idata1,
        "model_salary_worksat": idata2,
        "model_full": idata3,
        "constrained_logit_full": idata4,
        "constrained_probit_full": idata5,
    }
)

az.plot_compare(compare)
compare
```

We can also compare the estimated parameters which govern each regression model to see how robust our model fit is to alternative parameterisation. 

```{code-cell} ipython3
ax = az.plot_forest(
    [idata1, idata2, idata3, idata4, idata5],
    var_names=["sigma", "beta", "cutpoints"],
    combined=True,
    ridgeplot_overlap=4,
    figsize=(20, 25),
    r_hat=True,
    ridgeplot_alpha=0.3,
    model_names=[
        "model_salary",
        "model_salary_worksat",
        "model_full",
        "constrained_logit_full",
        "constrained_probit_full",
    ],
)
ax[0].set_title("Model Parameter Estimates", fontsize=20);
```

```{code-cell} ipython3
az.summary(idata3, var_names=["cutpoints", "beta", "sigma"])
```

## Compare Cutpoints: Normal versus Uniform Priors

Note how the model with unconstrianed cutpoints allows the occurence of a threshold estimated to be below zero. This does not make much conceptual sense, but can lead to a plausible enough posterior predictive distribution.

```{code-cell} ipython3
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
az.plot_posterior(idata3, var_names=["beta"]);
```

```{code-cell} ipython3
az.summary(idata3, var_names=["cutpoints"])
```

While the parameter estimates seem reasonable and the posterior predictive checks seem good too, the point to see here is that the cutpoints are unconstrained by the definition of the ordinal scale. They vary below 0 in the above model.

However if we look at the model with the constrained Dirchlet prior: 

```{code-cell} ipython3
plot_fit(idata4)
```

```{code-cell} ipython3
az.plot_posterior(idata4, var_names=["beta"]);
```

Again the parameters seem reasonable, and posterior predictive checks are sound. But now, having using the constrained uniform prior over the cutpoints our estimated cutpoints respect the definition of the ordinal scale. 

```{code-cell} ipython3
az.summary(idata4, var_names=["cutpoints"])
```

## Comparison to Statsmodels

This type of model can also be estimated in the frequentist tradition using maximum likelihood methods.

```{code-cell} ipython3
modf_logit = OrderedModel.from_formula(
    "explicit_rating ~ salary + work_sat + work_from_home", df, distr="logit"
)
resf_logit = modf_logit.fit(method="bfgs")
resf_logit.summary()
```

We can also extract the threshold or cut point estimates, which match closely with the cutpoints above where used a normal distribution to represent the latent manager rating. 

```{code-cell} ipython3
num_of_thresholds = 8
modf_logit.transform_threshold_params(resf_logit.params[-num_of_thresholds:])
```

## Interrogating the Model's Implications

When we want to asses the implications of the model we can use the learned posterior estimates for the data generating equation, to simulate what proportions of survey results would have resulted in a rating over a particular threshold score. Here we allow for full uncertainty of the various beta-distributions to be represented under different working conditions and measure the proportion of employees who would give their manager a rating above a 7. 

```{code-cell} ipython3
betas_posterior = az.extract(idata4)["beta"]

fig, ax = plt.subplots(figsize=(20, 10))
calc_wfh = [
    df.iloc[i]["salary"] * betas_posterior[0, :]
    + df.iloc[i]["work_sat"] * betas_posterior[1, :]
    + 1 * betas_posterior[2, :]
    for i in range(500)
]
calc_not_wfh = [
    df.iloc[i]["salary"] * betas_posterior[0, :]
    + df.iloc[i]["work_sat"] * betas_posterior[1, :]
    + 0 * betas_posterior[2, :]
    for i in range(500)
]
sal = np.random.normal(25, 5, 500)
calc_wfh_and_low_sal = [
    sal[i] * betas_posterior[0, :]
    + df.iloc[i]["work_sat"] * betas_posterior[1, :]
    + 1 * betas_posterior[2, :]
    for i in range(500)
]

### Use implied threshold on latent score to predict proportion of ratings above 7
prop_wfh = np.sum([np.mean(calc_wfh[i].values) > 6.78 for i in range(500)]) / 500
prop_not_wfh = np.sum([np.mean(calc_not_wfh[i].values) > 6.78 for i in range(500)]) / 500
prop_wfh_low = np.sum([np.mean(calc_wfh_and_low_sal[i].values) > 6.78 for i in range(500)]) / 500

for i in range(500):
    if i == 499:
        ax.hist(calc_wfh[i], alpha=0.6, color="skyblue", ec="black", label="WFH")
        ax.hist(calc_not_wfh[i], alpha=0.3, color="grey", ec="black", label="Not WFH")
        ax.hist(
            calc_wfh_and_low_sal[i], alpha=0.4, color="red", ec="black", label="WFH + Low Salary"
        )
    else:
        ax.hist(calc_wfh[i], alpha=0.6, color="skyblue", ec="black")
        ax.hist(calc_wfh_and_low_sal[i], alpha=0.4, color="red", ec="black")
        ax.hist(calc_not_wfh[i], alpha=0.3, color="grey", ec="black")
ax.set_title("Implied of Effect of Work from Home", fontsize=20)
ax.annotate(
    f"Expected Proportion > 7: \nWFH:{prop_wfh} \nWFH + LOW: {prop_wfh_low} \nNot WFH {prop_not_wfh}",
    xy=(-0.5, 1000),
    fontsize=20,
    fontweight="bold",
)
ax.legend();
```

## Liddell and Kruschke's IMDB movie Ratings Data

There are substantial reasons for using an ordinal regression model rather than trusting to alternative model specifications. For instance, the temptation to treat the ordered category as a continuous metric will lead to false inferences. The details are discussed in the Liddell and Kruschke paper {cite:p}`LIDDELL2018328` on this topic. We'll briefly replicate their example about how this phenomenon can appear in analysis of movies ratings data.

```{code-cell} ipython3
try:
    movies = pd.read_csv("../data/MoviesData.csv")
except FileNotFoundError:
    movies = pd.DataFrame(pm.get_data("MoviesData.csv"))
```

```{code-cell} ipython3
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
movies_by_rating.sample(100).head()
```

```{code-cell} ipython3
def constrainedUniform(N, group, min=0, max=1):
    return pm.Deterministic(
        f"cutpoints_{group}",
        pt.concatenate(
            [
                np.ones(1) * min,
                pt.extra_ops.cumsum(pm.Dirichlet(f"cuts_unknown_{group}", a=np.ones(N - 2)))
                * (max - min)
                + min,
            ]
        ),
    )
```

We will fit this data with both an ordinal model and as a metric. This will show how the ordinal fit is subtantially more compelling. 

```{code-cell} ipython3
:tags: [hide-output]

K = 5
movies_by_rating = movies_by_rating[movies_by_rating["movie_id"].isin([1, 2, 3, 4, 5, 6])]
indx, unique = pd.factorize(movies_by_rating["movie_id"])
priors = {"sigma": 1, "mu": [0, 1], "cut_mu": np.linspace(0, K, K - 1)}


def make_movies_model(ordered=False):
    with pm.Model() as model:
        for g in movies_by_rating["movie_id"].unique():
            if ordered:
                cutpoints = constrainedUniform(K, g, 0, K - 1)
                mu = pm.Normal(f"mu_{g}", 0, 1)
                y_ = pm.OrderedLogistic(
                    f"y_{g}",
                    cutpoints=cutpoints,
                    eta=mu,
                    observed=movies_by_rating[movies_by_rating["movie_id"] == g].rating.values,
                )
            else:
                mu = pm.Normal(f"mu_{g}", 0, 1)
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
axs = az.plot_ppc(idata_normal_metric)
axs = axs.flatten()
for ax in axs:
    ax.set_xlim(0, 5);
```

### Posterior Predictive Fit: Ordered Response Model

This shows a much nicer fit for each of the six movies. 

```{code-cell} ipython3
az.plot_ppc(idata_ordered);
```

Since this is real data and we don't know the true data generating process it's impossible to say which is the correct model but I hope you'll agree that the posterior predictive checks strongly support the claim that the ordered categorical fit is a stronger candidate. The simpler point to make here is just that the implications of the metric models are wrong and if we hope to make sound inferences about what perhaps drives good movie ratings, then we'd better be sure not to introduce noise into the modelling exercise with a poor choice of likelihood function. 

+++

### Compare Model Fits

```{code-cell} ipython3
y_5_compare = az.compare({"ordered": idata_ordered, "metric": idata_normal_metric}, var_name="y_5")
y_5_compare
```

```{code-cell} ipython3
y_6_compare = az.compare({"ordered": idata_ordered, "metric": idata_normal_metric}, var_name="y_6")
y_6_compare
```

```{code-cell} ipython3
az.plot_compare(y_5_compare)
```

```{code-cell} ipython3
az.plot_compare(y_6_compare)
```

### Compare Inferences between Models

Aside from the predictive fits, the inferences drawn from the different modelling choices also vary quite significantly. Imagine being a movie executive trying to decide whether to commit to a sequel, then relative movie performance rating against competitor  benchmarks might be a pivotal feature of this decision, and difference induced by the analyst's choice of model can have an outsized effect on that choice. 

```{code-cell} ipython3
mosaic = """
AC
DE
BB
"""
fig, axs = plt.subplot_mosaic(mosaic, figsize=(15, 7))
axs = [axs[k] for k in axs.keys()]
axs
ordered_5 = az.extract(idata_ordered.posterior_predictive)["y_5"].mean(axis=0)
ordered_6 = az.extract(idata_ordered.posterior_predictive)["y_6"].mean(axis=0)
diff = ordered_5 - ordered_6
metric_5 = az.extract(idata_normal_metric.posterior_predictive)["y_5"].mean(axis=0)
metric_6 = az.extract(idata_normal_metric.posterior_predictive)["y_6"].mean(axis=0)
diff1 = metric_5 - metric_6
axs[0].hist(ordered_5, bins=30, ec="white", color="slateblue", label="Ordered Fit Movie 5")
axs[4].plot(
    az.hdi(diff.unstack())["x"].values, [1, 1], "o-", color="slateblue", label="Ordered Fits"
)
axs[4].plot(
    az.hdi(diff1.unstack())["x"].values, [1.2, 1.2], "o-", color="magenta", label="Metric Fits"
)
axs[2].hist(ordered_6, bins=30, ec="white", color="slateblue", label="Ordered Fit Movie 6")
axs[3].hist(metric_5, ec="white", label="Metric Fit Movie 5", bins=30, color="magenta")
axs[1].hist(metric_6, ec="white", label="Metric Fit Movie 6", bins=30, color="magenta")
axs[4].set_title("Implied Differences Between the \n Expected Rating")
axs[4].set_ylim(0.8, 1.4)
axs[4].set_yticks([])
axs[0].set_title("Expected Posterior Predictive Estimate \n for Movies Ordered Fits")
axs[1].set_title("Expected Posterior Predictive Estimate \n for Movie Metric Fits")
axs[4].set_xlabel("Difference between Movie 5 and 6")
axs[1].legend()
axs[0].legend()
axs[2].legend()
axs[3].legend()
axs[4].legend();
```

There are many millions of dollars on the line when making the decision to put a movie into production. The return on that investment is a least partially a function of the movie's popularity which is both measured and influenced by the rating scales on Rotten Tomatoes and IMDB. Understanding the relative popularity of different movies therefore can shift huge amounts of money through hollywood, and the implied differences seen here really do matter. Similar considerations follow for considering more significant rating scales which measure happiness and depression. 

+++

# Conclusion

In this notebook we've seen how to build ordinal regression models with PyMC and motivated the modelling exercise using the interpretation of ordinal outcomes as the discrete outcomes of a latent continuous phenomena. We've seen how different model specifications can generate more or less interpretable estimates of the parameters underlying the model. We've also compared the ordinal regression approach to a more naive regression approach on ordinal data. The results strongly suggest that the ordinal regression avoids some of the inferential pitfalls that occur with the naive approach. In addition we've shown that the flexibility of the bayesian modelling work flow can provide assurances against the risk of model misspecification making it a viable and compelling approach for the analysis of ordinal data. 

+++

## Authors
- Authored by [Nathaniel Forde](https://github.com/NathanielF) in June 2023 

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
