---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(Bayesian Missing Data Imputation)=
# Bayesian Missing Data Imputation

:::{post} February, 2023
:tags: missing data, bayesian imputation, hierarchical
:category: advanced
:author: Nathaniel Forde
:::

```{code-cell} ipython3
import random

import arviz as az
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import scipy.optimize

from matplotlib.lines import Line2D
from pymc.sampling.jax import sample_blackjax_nuts, sample_numpyro_nuts
from scipy.stats import multivariate_normal
```

## Bayesian Imputation and Degrees of Missing-ness

The analysis of data with missing values is a gateway into the study of causal inference. 

One of the key features of any analysis plagued by missing data is the assumption which governs the nature of the missing-ness i.e. what is the reason for gaps in our data? Can we ignore them? Should we worry about why? In this notebook we'll see an example of how to handle missing data using maximum likelihood estimation and bayesian imputation techniques. This will open up questions about the assumptions governing inference in the presence of missing data, and inference in counterfactual cases.

We will make the discussion concrete by considering an example analysis of an employee satisfaction survey and how different work conditions contribute to the responses and non-responses we see in the data.

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'  # high resolution figures
az.style.use("arviz-darkgrid")
rng = np.random.default_rng(42)
```

## Missing Data Taxonomy

Rubin's famous taxonomy breaks out the question into a choice of three fundamental options:

 - Missing Completely at Random (MCAR)
 - Missing at Random (MAR)
 - Missing Not at Random (MNAR)

Each of these paradigms can be reduced to explicit definition in terms of the conditional probability regarding the **pattern of missing data**. The first pattern is the least concerning. The (MCAR) assumption states that the data are missing in a manner that is unrelated to both the observed and unobserved parts of the realised data. It is missing due to the haphazard circumstance of the world $\phi$.

$$  P(M =1 | Y_{obs}, Y_{miss}, \phi) = P(M =1 | \phi) $$

whereas the second pattern (MAR) allows that the reasons for missingness can be function of the observed data and circumstances of the world. Some times this is called a case of *ignorable* missingness because estimation can proceed in good faith on the basis of the observed data. There may be a loss of precision, but the inference should be sound.  

$$  P(M =1 | Y_{obs}, Y_{miss}, \phi) = P(M =1 | Y_{obs}, \phi) $$ 

The most nefarious sort of missing data is when the missingness is a function of something outside the observed data, and the equation cannot be reduced further. Efforts at imputation and estimation more generally may become more difficulty in this final case because of the risk of confounding. This is a case of *non-ignorable* missing-ness. 

$$  P(M =1 | Y_{obs}, Y_{miss}, \phi) $$

These assumptions are made before any analysis begins. They are inherently unverifiable. Your analysis will stand or fall depending on how plausible each assumption is in the context you seek to apply them. For example, an another type missing data results from systematic censoring as discussed in {ref}`GLM-truncated-censored-regression`. In such cases the reason for censoring governs the missing-ness pattern. 

## Employee Satisfaction Surveys

We'll follow the presentation of Craig Enders' *Applied Missing Data Analysis* {cite:t}`enders2022` and work with employee satisifaction data set. The data set comprises of a few composite measures reporting employee working conditions and satisfactions. Of particular note are empowerment (`empower`), work satisfaction (`worksat`) and two composite survey scores recording the employees leadership climate (`climate`), and the relationship quality with their supervisor `lmx`. 

The key question is what assumptions governs our patterns of missing data.

```{code-cell} ipython3
try:
    df_employee = pd.read_csv("../data/employee.csv")
except FileNotFoundError:
    df_employee = pd.read_csv(pm.get_data("employee.csv"))
df_employee.head()
```

```{code-cell} ipython3
# Percentage Missing
df_employee[["worksat", "empower", "lmx"]].isna().sum() / len(df_employee)
```

```{code-cell} ipython3
# Patterns of missing Data
df_employee[["worksat", "empower", "lmx"]].isnull().drop_duplicates().reset_index(drop=True)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(20, 7))
ax.hist(df_employee["empower"], bins=30, ec="black", color="cyan", label="Empowerment")
ax.hist(df_employee["lmx"], bins=30, ec="black", color="yellow", label="LMX")
ax.hist(df_employee["worksat"], bins=30, ec="black", color="green", label="Work Satisfaction")
ax.set_title("Employee Satisfaction Survey Results", fontsize=20)
ax.legend();
```

We see here the histograms of the employee metrics. It is the gaps in the data that we wish to impute to better understand the relationships between the variables and how gaps in one may be driven by values in the others. 

+++

## FIML: Full Information Maximum Likelihood 

This method of handling missing data is **not** an imputation method. It uses maximum likelihood estimation to estimate the parameters of the multivariate normal distribution that could be best said to generate our observed data. It's a little trickier than straight forward MLE approaches in that it respects the fact that we have missing data in our original data set, but fundamentally it's the same idea. We want to optimize the parameters of our multivariate normal distribution to best fit the observed data. 

The procedure works by partitioning the data into their patterns of "missing-ness" and treating each partition as contributing to the ultimate log-likelihood term that we want to maximise. We combine their contributions to estimate a fit for the multivariate normal distribution.  

```{code-cell} ipython3
data = df_employee[["worksat", "empower", "lmx"]]


def split_data_by_missing_pattern(data):
    # We want to extract our the pattern of missing-ness in our dataset
    # and save each sub-set of our data in a structure that can be used to feed into a log-likelihood function
    grouped_patterns = []
    patterns = data.notnull().drop_duplicates().values
    # A pattern is whether the values in each column e.g. [True, True, True] or [True, True, False]
    observed = data.notnull()
    for p in range(len(patterns)):
        temp = observed[
            (observed["worksat"] == patterns[p][0])
            & (observed["empower"] == patterns[p][1])
            & (observed["lmx"] == patterns[p][2])
        ]
        grouped_patterns.append([patterns[p], temp.index, data.iloc[temp.index].dropna(axis=1)])

    return grouped_patterns


def reconstitute_params(params_vector, n_vars):
    # Convenience numpy function to construct mirrored COV matrix
    # From flattened params_vector
    mus = params_vector[0:n_vars]
    cov_flat = params_vector[n_vars:]
    indices = np.tril_indices(n_vars)
    cov = np.empty((n_vars, n_vars))
    for i, j, c in zip(indices[0], indices[1], cov_flat):
        cov[i, j] = c
        cov[j, i] = c
    cov = cov + 1e-25
    return mus, cov


def optimise_ll(flat_params, n_vars, grouped_patterns):
    mus, cov = reconstitute_params(flat_params, n_vars)
    # Check if COV is positive definite
    if (np.linalg.eigvalsh(cov) < 0).any():
        return 1e100
    objval = 0.0
    for obs_pattern, _, obs_data in grouped_patterns:
        # This is the key (tricky) step because we're selecting the variables which pattern
        # the full information set within each pattern of "missing-ness"
        # e.g. when the observed pattern is [True, True, False] we want the first two variables
        # of the mus vector and we want only the covariance relations between the relevant variables from the cov
        # in the iteration.
        obs_mus = mus[obs_pattern]
        obs_cov = cov[obs_pattern][:, obs_pattern]
        ll = np.sum(multivariate_normal(obs_mus, obs_cov).logpdf(obs_data))
        objval = ll + objval
    return -objval


def estimate(data):
    n_vars = data.shape[1]
    # Initialise
    mus0 = np.zeros(n_vars)
    cov0 = np.eye(n_vars)
    # Flatten params for optimiser
    params0 = np.append(mus0, cov0[np.tril_indices(n_vars)])
    # Process Data
    grouped_patterns = split_data_by_missing_pattern(data)
    # Run the Optimiser.
    try:
        result = scipy.optimize.minimize(
            optimise_ll, params0, args=(n_vars, grouped_patterns), method="Powell"
        )
    except Exception as e:
        raise e
    mean, cov = reconstitute_params(result.x, n_vars)
    return mean, cov


fiml_mus, fiml_cov = estimate(data)


print("Full information Maximum Likelihood Estimate Mu:")
display(pd.DataFrame(fiml_mus, index=data.columns).T)
print("Full information Maximum Likelihood Estimate COV:")
pd.DataFrame(fiml_cov, columns=data.columns, index=data.columns)
```

### Sampling from the Implied Distribution

We can then sample from the implied distribution to estimate other features of interest and test against the observed data.

```{code-cell} ipython3
mle_fit = multivariate_normal(fiml_mus, fiml_cov)
mle_sample = mle_fit.rvs(10000)
mle_sample = pd.DataFrame(mle_sample, columns=["worksat", "empower", "lmx"])
mle_sample.head()
```

This allows us to compare the implied distributions against the observed data

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(20, 7))
ax.hist(
    mle_sample["empower"],
    bins=30,
    ec="black",
    color="cyan",
    alpha=0.2,
    label="Inferred Empowerment",
)
ax.hist(mle_sample["lmx"], bins=30, ec="black", color="yellow", alpha=0.2, label="Inferred LMX")
ax.hist(
    mle_sample["worksat"],
    bins=30,
    ec="black",
    color="green",
    alpha=0.2,
    label="Inferred Work Satisfaction",
)
ax.hist(data["empower"], bins=30, ec="black", color="cyan", label="Observed Empowerment")
ax.hist(data["lmx"], bins=30, ec="black", color="yellow", label="Observed LMX")
ax.hist(data["worksat"], bins=30, ec="black", color="green", label="Observed Work Satisfaction")
ax.set_title("Inferred from MLE fit: Employee Satisfaction Survey Results", fontsize=20)
ax.legend()
```

### The Correlation Between the Imputed Metrics Data

We can also calculate other features of interest from our sample. For instance, we might want to know about the correlations between variables in question. 

```{code-cell} ipython3
pd.DataFrame(mle_sample.corr(), columns=data.columns, index=data.columns)
```

### Bootstrapping Sensitivity Analysis

We may also want to validate the estimated parameters against bootstrapped samples under different speficiations of missing-ness. 

```{code-cell} ipython3
data_200 = df_employee[["worksat", "empower", "lmx"]].dropna().sample(200)
data_200.reset_index(inplace=True, drop=True)


sensitivity = {}
n_missing = np.linspace(30, 100, 5)  # Change or alter the range as desired
bootstrap_iterations = 100  # change to large number running a real analysis in this case
for n in n_missing:
    sensitivity[int(n)] = {}
    sensitivity[int(n)]["mus"] = []
    sensitivity[int(n)]["cov"] = []
    for i in range(bootstrap_iterations):
        temp = data_200.copy()
        for m in range(int(n)):
            i = random.choice(range(200))
            j = random.choice(range(3))
            temp.iloc[i, j] = np.nan
        try:
            fiml_mus, fiml_cov = estimate(temp)
            sensitivity[int(n)]["mus"].append(fiml_mus)
            sensitivity[int(n)]["cov"].append(fiml_cov)
        except Exception as e:
            next
```

Here we plot the maximum likelihood parameter estimates against various missing data regimes. This approach can be applied for any imputation methodology.

```{code-cell} ipython3
fig, axs = plt.subplots(1, 3, figsize=(20, 7))
for n in sensitivity.keys():
    temp = pd.DataFrame(sensitivity[n]["mus"], columns=["worksat", "empower", "lmx"])
    for col, ax in zip(temp.columns, axs):
        ax.hist(
            temp[col], alpha=0.1, ec="black", label=f"Missing: {np.round(n/200, 2)}, Mean: {col}"
        )
        ax.legend()
        ax.set_title(f"Bootstrap Distribution for Mean:\n{col}")
```

```{code-cell} ipython3
fig, axs = plt.subplots(2, 3, figsize=(20, 14))
axs = axs.flatten()
for n in sensitivity.keys():
    length = len(sensitivity[n]["cov"])
    temp = pd.DataFrame(
        [sensitivity[n]["cov"][i][np.tril_indices(3)] for i in range(length)],
        columns=[
            "var(worksat)",
            "cov(worksat, empower)",
            "var(empower)",
            "cov(worksat, lmx)",
            "cov(lmx, empower)",
            "var(lmx)",
        ],
    )
    for col, ax in zip(temp.columns, axs):
        ax.hist(
            temp[col], alpha=0.1, ec="black", label=f"Missing: {np.round(n/200, 2)}, Mean: {col}"
        )
        ax.legend()
        ax.set_title(f"Bootstrap Distribution for Expected:\n{col}")
```

These plots show how under (MCAR) the parameter estimates of our multivariate normal distribution are quite robust to varying degrees of missing data. It's an instructive exercise to attempt a similar simulation exercise under alternative missing data regimes. 

+++

## Bayesian Imputation 

Next we'll apply bayesian methods to the same problem. But here we'll see direct imputation of the missing values using the posterior predictive distribution. The Bayesian approach to imputation is of a different flavour than we saw above. We're not just learning parameters of the data generating distribution (although we are doing that too), the bayesian process directly imputes the missing values for specific missing entries through the process of MCMC sampling. 

```{code-cell} ipython3
import pytensor.tensor as pt

with pm.Model() as model:
    # Priors
    mus = pm.Normal("mus", 0, 1, size=3)
    cov_flat_prior, _, _ = pm.LKJCholeskyCov("cov", n=3, eta=1.0, sd_dist=pm.Exponential.dist(1))
    # Create a vector of flat variables for the unobserved components of the MvNormal
    x_unobs = pm.Uniform("x_unobs", 0, 100, shape=(np.isnan(data.values).sum(),))

    # Create the symbolic value of x, combining observed data and unobserved variables
    x = pt.as_tensor(data.values)
    x = pm.Deterministic("x", pt.set_subtensor(x[np.isnan(data.values)], x_unobs))

    # Add a Potential with the logp of the variable conditioned on `x`
    pm.Potential("x_logp", pm.logp(rv=pm.MvNormal.dist(mus, chol=cov_flat_prior), value=x))
    idata = pm.sample_prior_predictive()
    idata = pm.sample()
    idata.extend(pm.sample(random_seed=120))
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)

pm.model_to_graphviz(model)
```

```{code-cell} ipython3
az.plot_posterior(idata, var_names=["mus", "cov"]);
```

```{code-cell} ipython3
az.summary(idata, var_names=["mus", "cov", "x_unobs"])
```

```{code-cell} ipython3
imputed_dims = data.shape
imputed = data.values.flatten()
imputed[np.isnan(imputed)] = az.summary(idata, var_names=["x_unobs"])["mean"].values
imputed = imputed.reshape(imputed_dims[0], imputed_dims[1])
imputed = pd.DataFrame(imputed, columns=[col + "_imputed" for col in data.columns])
imputed.head(10)
```

```{code-cell} ipython3
fig, axs = plt.subplots(1, 3, figsize=(20, 7))
axs = axs.flatten()
for col, col_i, ax in zip(data.columns, imputed.columns, axs):
    ax.hist(data[col], color="red", label=col, ec="black", bins=30)
    ax.hist(imputed[col_i], color="cyan", alpha=0.3, label=col_i, ec="black", bins=30)
    ax.legend()
    ax.set_title(f"Imputed Distribution and Observed for {col}")
```

```{code-cell} ipython3
pd.DataFrame(az.summary(idata, var_names=["cov_corr"])["mean"].values.reshape(3, 3))
```

These results agree with the FIML approach above and the results reported in Ender's *Applied Missing Data Analysis*.

+++

## Bayesian Imputation by Chained Equations

So far we've seen multivariate approaches to imputation which treat each of the variables in our dataset as a collection drawn from the same distribution. However, there is a more flexible approach which is often useful when there is a particular focal relationship that we're interested in analysing. 

Sticking with the employee data set we'll examine here the relationship between `lmx`, `climate`, `male` and `empower`, where our focus is on what drives empowerment. 
Recall that our gender variable `male` is fully specified and does not need to be imputed. So we have a joint distribution that can be decomposed:

$$ f(emp, lmx, climate, male) = f(emp | lmx, climate, male) \cdot f(lmx | climate, male) \cdot f(climate | male) \cdot f(male)^{*} $$

which can be split out into individual regression equations or more generally component models for each required conditional model. 

$$ empower = \alpha_{2} + \beta_{3}male + \beta_{4}climate + \beta_{5}lmx $$
$$ lmx = \alpha_{1} + \beta_{1}climate + \beta_{2}male $$
$$ climate = \alpha_{0} + \beta_{0}male $$

We can impute each of these equations in turn saving the imputed data set and feeding it forward into the next modelling exercise. This adds a little complexity because some of the variables will occur twice. Once as a predictor in our focal regression and once and as likelihood term in their own component model. 


+++

### PyMC Imputation

As we saw above we can use PyMC to impute the values of missing data by using a particular sampling distribution. In the case of chained equations this becomes a little trickier because we might want to use both the data for `lmx` as a regressor in one equation and observed data in our likelihood in another. 

It also matters how we specify the sampling distribution that will be used to impute our missing data. We'll show an example here where we use a uniform and normal sampling distribution alternatively for imputing the predictor terms in our in focal regression. 

```{code-cell} ipython3
data = df_employee[["lmx", "empower", "climate", "male"]]
lmx_mean = data["lmx"].mean()
lmx_min = data["lmx"].min()
lmx_max = data["lmx"].max()
lmx_sd = data["lmx"].std()

cli_mean = data["climate"].mean()
cli_min = data["climate"].min()
cli_max = data["climate"].max()
cli_sd = data["climate"].std()


priors = {
    "climate": {"normal": [lmx_mean, lmx_sd, lmx_sd], "uniform": [lmx_min, lmx_max]},
    "lmx": {"normal": [cli_mean, cli_sd, cli_sd], "uniform": [cli_min, cli_max]},
}


def make_model(priors, normal_pred_assumption=True):
    coords = {
        "alpha_dim": ["lmx_imputed", "climate_imputed", "empower_imputed"],
        "beta_dim": [
            "lmxB_male",
            "lmxB_climate",
            "climateB_male",
            "empB_male",
            "empB_climate",
            "empB_lmx",
        ],
    }
    with pm.Model(coords=coords) as model:
        # Priors
        beta = pm.Normal("beta", 0, 1, size=6, dims="beta_dim")
        alpha = pm.Normal("alphas", 10, 5, size=3, dims="alpha_dim")
        sigma = pm.HalfNormal("sigmas", 5, size=3, dims="alpha_dim")

        if normal_pred_assumption:
            mu_climate = pm.Normal(
                "mu_climate", priors["climate"]["normal"][0], priors["climate"]["normal"][1]
            )
            sigma_climate = pm.HalfNormal("sigma_climate", priors["climate"]["normal"][2])
            climate_pred = pm.Normal(
                "climate_pred", mu_climate, sigma_climate, observed=data["climate"].values
            )
        else:
            climate_pred = pm.Uniform("climate_pred", 0, 40, observed=data["climate"].values)

        if normal_pred_assumption:
            mu_lmx = pm.Normal("mu_lmx", priors["lmx"]["normal"][0], priors["lmx"]["normal"][1])
            sigma_lmx = pm.HalfNormal("sigma_lmx", priors["lmx"]["normal"][2])
            lmx_pred = pm.Normal("lmx_pred", mu_lmx, sigma_lmx, observed=data["lmx"].values)
        else:
            lmx_pred = pm.Uniform("lmx_pred", 0, 40, observed=data["lmx"].values)

        # Likelihood(s)
        lmx_imputed = pm.Normal(
            "lmx_imputed",
            alpha[0] + beta[0] * data["male"] + beta[1] * climate_pred,
            sigma[0],
            observed=data["lmx"].values,
        )
        climate_imputed = pm.Normal(
            "climate_imputed",
            alpha[1] + beta[2] * data["male"],
            sigma[1],
            observed=data["climate"].values,
        )
        empower_imputed = pm.Normal(
            "emp_imputed",
            alpha[2] + beta[3] * data["male"] + beta[4] * climate_pred + beta[5] * lmx_pred,
            sigma[2],
            observed=data["empower"].values,
        )

        idata = pm.sample_prior_predictive()
        idata.extend(pm.sample(random_seed=120))
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)
        return idata, model


idata_uniform, model_uniform = make_model(priors, normal_pred_assumption=False)
idata_normal, model_normal = make_model(priors, normal_pred_assumption=True)
pm.model_to_graphviz(model_uniform)
```

```{code-cell} ipython3
idata_normal
```

### Model Fits

Next we'll inspect the parameter fits for our regression models and observe how they're dependent on the prior specification in the imputation scheme. 

```{code-cell} ipython3
az.summary(idata_normal, var_names=["alphas", "beta", "sigmas"], stat_focus="median")
```

```{code-cell} ipython3
az.summary(idata_uniform, var_names=["alphas", "beta", "sigmas"], stat_focus="median")
```

We can see how the choice of sampling distribution has induced different parameter estimates on the beta coefficients across our two models. The two imputations broadly agrees at the level of the parameters, but they meaningfully differ in their implications. 

```{code-cell} ipython3
az.plot_forest(
    [idata_normal, idata_uniform],
    var_names=["beta"],
    kind="ridgeplot",
    model_names=["Gaussian Sampling Distribution", "Uniform Sampling Distribution"],
    figsize=(10, 8),
)
```

This difference has downstream effects on the posterior predictive distribution. We can see here how the sampling distribution for the predictor terms influences the posterior predictive fits on our focal regression equation.

### Posterior Predictive Distributions

```{code-cell} ipython3
az.plot_ppc(idata_uniform)
```

```{code-cell} ipython3
az.plot_ppc(idata_normal)
```

### Process the Posterior Predictive Distribution

Above we estimated a number of likelihood terms in a single PyMC model context. These likelihoods constrained the hyper-parameters which determined the imputation values of the missing terms in the variables used as predictors in our focal regression equation for `empower`. But we could also perform a more manual sequential imputation, where we model each of the subordinate regression equations seperately and extract the imputed values for each variable in turn and then run a simple regression on the imputed values for the focal regression equation. 

We show here how to extract the imputed values for each of the regression equations and augment the observed data.

```{code-cell} ipython3
def get_imputed(idata, data):
    imputed_data = data.copy()
    imputed_climate = az.extract(idata, group="posterior_predictive", num_samples=1000)[
        "climate_imputed"
    ].mean(axis=1)
    mask = imputed_data["climate"].isnull()
    imputed_data.loc[mask, "climate"] = imputed_climate.values[imputed_data[mask].index]

    imputed_lmx = az.extract(idata, group="posterior_predictive", num_samples=1000)[
        "lmx_imputed"
    ].mean(axis=1)
    mask = imputed_data["lmx"].isnull()
    imputed_data.loc[mask, "lmx"] = imputed_lmx.values[imputed_data[mask].index]

    imputed_emp = az.extract(idata, group="posterior_predictive", num_samples=1000)[
        "emp_imputed"
    ].mean(axis=1)
    mask = imputed_data["empower"].isnull()
    imputed_data.loc[mask, "empower"] = imputed_emp.values[imputed_data[mask].index]
    assert imputed_data.isnull().sum().to_list() == [0, 0, 0, 0]
    imputed_data.columns = ["imputed_" + col for col in imputed_data.columns]
    return imputed_data


imputed_data_uniform = get_imputed(idata_uniform, data)
imputed_data_normal = get_imputed(idata_normal, data)
imputed_data_normal.head(5)
```

We used the mean here to impute the expected value for each missing cell, but you could perform a kind of sensitivity analysis using the many plausible values in posterior predictive distribution

+++

### Plotting the Imputed Datasets

Now we'll plot the imputed values against their observed values to show how the different sampling distributions impact the pattern of imputation.

```{code-cell} ipython3
joined_uniform = pd.concat([imputed_data_uniform, data], axis=1)
joined_normal = pd.concat([imputed_data_normal, data], axis=1)
for col in ["lmx", "empower", "climate"]:
    joined_uniform[col + "_missing"] = np.where(joined_uniform[col].isnull(), 1, 0)
    joined_normal[col + "_missing"] = np.where(joined_normal[col].isnull(), 1, 0)


def rand_jitter(arr):
    stdev = 0.01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


fig, axs = plt.subplots(1, 3, figsize=(20, 8))
axs = axs.flatten()
ax = axs[0]
ax1 = axs[1]
ax2 = axs[2]

## Derived from MV norm fit.
z = multivariate_normal(
    [lmx_mean, joined_uniform["imputed_empower"].mean()], [[8.9, 5.4], [5.4, 19]]
).pdf(joined_uniform[["imputed_lmx", "imputed_empower"]])

ax.scatter(
    rand_jitter(joined_uniform["imputed_lmx"]),
    rand_jitter(joined_uniform["imputed_empower"]),
    c=joined_uniform["empower_missing"],
    cmap=cm.winter,
    ec="black",
    s=50,
)
ax.set_title("Relationship between LMX and Empowerment \n after Uniform Imputation", fontsize=20)
ax.tricontour(joined_uniform["imputed_lmx"], joined_uniform["imputed_empower"], z)
ax.set_xlabel("Leader-Member-Exchange")
ax.set_ylabel("Empowerment")


custom_lines = [
    Line2D([0], [0], color=cm.winter(0.0), lw=4),
    Line2D([0], [0], color=cm.winter(0.9), lw=4),
]
ax.legend(custom_lines, ["Observed", "Missing - Imputed Empowerment Values"])

z = multivariate_normal(
    [lmx_mean, joined_normal["imputed_empower"].mean()], [[8.9, 5.4], [5.4, 19]]
).pdf(joined_normal[["imputed_lmx", "imputed_empower"]])

ax2.scatter(
    rand_jitter(joined_normal["imputed_lmx"]),
    rand_jitter(joined_normal["imputed_empower"]),
    c=joined_normal["empower_missing"],
    cmap=cm.autumn,
    ec="black",
    s=50,
)
ax2.set_title("Relationship between LMX and Empowerment \n after Gaussian Imputation", fontsize=20)
ax2.tricontour(joined_normal["imputed_lmx"], joined_normal["imputed_empower"], z)
ax2.set_xlabel("Leader-Member-Exchange")
ax2.set_ylabel("Empowerment")
custom_lines = [
    Line2D([0], [0], color=cm.autumn(0.0), lw=4),
    Line2D([0], [0], color=cm.autumn(0.9), lw=4),
]
ax2.legend(custom_lines, ["Observed", "Missing - Imputed Empowerment Values"])

ax1.hist(
    joined_normal["imputed_empower"],
    label="Gaussian Imputed Empowerment",
    bins=30,
    color="slateblue",
    ec="black",
)
ax1.hist(
    joined_uniform["imputed_empower"],
    label="Uniform Imputed Empowerment",
    bins=30,
    color="cyan",
    ec="black",
)
ax1.hist(
    joined_normal["empower"], label="Observed Empowerment", bins=30, color="magenta", ec="black"
)
ax1.legend()
ax1.set_title("Imputed & Observed Empowerment", fontsize=20);
```

Ultimately our choice of sampling distribution leads to differently plausible imputations. The choice of which model to go with will driven by the assumptions which govern the reasons for missing-ness in our data. 

+++

## Hierarchical Structures and Data Imputation

Our employee dataset has more fine-grained structure than we've examined so far. In particular there are 100 or so teams which make up our employee pool, and we might wonder to what degree the propensity for satisfaction or incomplete survey scores are due to the local team environments? Could this be a factor in our patterns of missing data? We'll examine the reported empowerment scores by team and plot the regression lines by as predicted within each team by their reported `lmx` score. 

```{code-cell} ipython3
heatmap = df_employee.pivot("employee", "team", "empower").dropna(how="all")
heatmap = pd.concat(
    [heatmap[~heatmap[col].isnull()][col].reset_index(drop=True) for col in heatmap.columns], axis=1
)
with pd.option_context("format.precision", 2):
    display(heatmap.style.background_gradient(cmap="Blues"));
```

```{code-cell} ipython3
fits = []
x = np.linspace(0, 20, 100)
fig, ax = plt.subplots(figsize=(20, 7))
for team in df_employee["team"].unique():
    temp = df_employee[df_employee["team"] == team][["lmx", "empower"]].dropna()
    fit = np.polyfit(temp["lmx"], temp["empower"], 1)
    y = fit[0] * x + fit[1]
    fits.append(fit)
    ax.plot(x, y, alpha=0.6)
    ax.scatter(rand_jitter(temp["lmx"]), rand_jitter(temp["empower"]), color="black", ec="white")
ax.set_title("Simple Regression fits by Team \n Empower ~ LMX", fontsize=20)
ax.set_xlabel("Leader-Member-Exchange (LMX)")
ax.set_ylabel("Empowerment")
ax.set_ylim(0, 45);
```

There is enough spread in the regression lines to at least suggest that there is a heterogenous relationship between empowerment and the work environment as we look across different teams, but limited observations in each team. This is a perfect use case for a hierarchical bayesian model. 

```{code-cell} ipython3
team_idx, teams = pd.factorize(df_employee["team"], sort=True)
employee_idx, _ = pd.factorize(df_employee["employee"], sort=True)
coords = {"team": teams, "employee": np.arange(len(df_employee))}


with pm.Model(coords=coords) as hierarchical_model:
    # Priors
    company_beta_lmx = pm.Normal("company_beta_lmx", 0, 1)
    company_beta_male = pm.Normal("company_beta_male", 0, 1)
    company_alpha = pm.Normal("company_alpha", 20, 2)
    team_alpha = pm.Normal("team_alpha", 0, 1, dims="team")
    team_beta_lmx = pm.Normal("team_beta_lmx", 0, 1, dims="team")
    sigma = pm.HalfNormal("sigma", 4, dims="employee")

    # Imputed Predictors
    mu_lmx = pm.Normal("mu_lmx", 10, 5)
    sigma_lmx = pm.HalfNormal("sigma_lmx", 5)
    lmx_pred = pm.Normal("lmx_pred", mu_lmx, sigma_lmx, observed=df_employee["lmx"].values)

    # Combining Levels
    alpha_global = pm.Deterministic("alpha_global", company_alpha + team_alpha[team_idx])
    beta_global_lmx = pm.Deterministic(
        "beta_global_lmx", company_beta_lmx + team_beta_lmx[team_idx]
    )
    beta_global_male = pm.Deterministic("beta_global_male", company_beta_male)

    # Likelihood
    mu = pm.Deterministic(
        "mu",
        alpha_global + beta_global_lmx * lmx_pred + beta_global_male * df_employee["male"].values,
    )

    empower_imputed = pm.Normal(
        "emp_imputed",
        mu,
        sigma,
        observed=df_employee["empower"].values,
    )

    idata_hierarchical = pm.sample_prior_predictive()
    # idata_hierarchical.extend(pm.sample(random_seed=1200, target_accept=0.99))
    idata_hierarchical.extend(
        sample_blackjax_nuts(draws=20_000, random_seed=500, target_accept=0.99)
    )
    pm.sample_posterior_predictive(idata_hierarchical, extend_inferencedata=True)

pm.model_to_graphviz(hierarchical_model)
```

```{code-cell} ipython3
idata_hierarchical
```

### Some Convergence Checks

```{code-cell} ipython3
az.plot_trace(
    idata_hierarchical,
    var_names=["company_alpha", "team_alpha", "company_beta_lmx", "team_beta_lmx"],
    kind="rank_vlines",
);
```

```{code-cell} ipython3
az.plot_energy(idata_hierarchical, figsize=(20, 7));
```

### Inspecting the Model Fit

```{code-cell} ipython3
summary = az.summary(
    idata_hierarchical,
    var_names=[
        "company_alpha",
        "team_alpha",
        "company_beta_lmx",
        "company_beta_male",
        "team_beta_lmx",
    ],
)

summary
```

```{code-cell} ipython3
az.plot_ppc(
    idata_hierarchical, var_names=["emp_imputed_observed"], figsize=(20, 7), num_pp_samples=1000
)
```

### Heterogenous Patterns of Imputation

Just as when we consider questions of causal inference and we attend to the confounding influence of local factors, this is also required when we do imputation. We show here a selection of team specific intercept terms which suggest that belonging to a particular team can shift your empowerment above or below the grand mean of the company level intercept term. These local effects of environment are what we seek to account for when imputing missing values. 

```{code-cell} ipython3
ax = az.plot_forest(
    idata_hierarchical,
    var_names=["team_beta_lmx"],
    coords={"team": [1, 20, 22, 30, 50, 70, 76, 80, 100]},
    figsize=(20, 15),
    kind="ridgeplot",
    combined=True,
    ridgeplot_alpha=0.4,
    hdi_prob=True,
)
ax[0].axvline(0)
ax[0].set_title("Team Contribution to the marginal effect of LMX on Empowerment", fontsize=20);
```

The ability to capture this local variation impacts the pattern of imputed values too. 

```{code-cell} ipython3
imputed_data = df_employee[["lmx", "empower", "climate"]]

imputed_lmx = az.extract(idata_hierarchical, group="posterior_predictive", num_samples=1000)[
    "lmx_pred"
].mean(axis=1)
mask = imputed_data["lmx"].isnull()
imputed_data.loc[mask, "lmx"] = imputed_lmx.values[imputed_data[mask].index]

imputed_emp = az.extract(idata_hierarchical, group="posterior_predictive", num_samples=1000)[
    "emp_imputed"
].mean(axis=1)
mask = imputed_data["empower"].isnull()
imputed_data.loc[mask, "empower"] = imputed_emp.values[imputed_data[mask].index]
imputed_data.columns = ["imputed_" + col for col in imputed_data.columns]
joined = pd.concat([imputed_data, df_employee], axis=1)
joined["check"] = np.where(joined["empower"].isnull(), 1, 0)

mosaic = """AAAABB"""
fig, axs = plt.subplot_mosaic(mosaic, sharex=False, figsize=(20, 7))
axs = [axs[k] for k in axs.keys()]
axs[0].scatter(
    joined["imputed_lmx"],
    joined["imputed_empower"],
    c=joined["check"],
    cmap=cm.winter,
    ec="black",
    s=40,
)

z = multivariate_normal([10, joined["imputed_empower"].mean()], [[8.9, 5.4], [5.4, 19]]).pdf(
    joined[["imputed_lmx", "imputed_empower"]]
)
axs[0].tricontour(joined["imputed_lmx"], joined["imputed_empower"], z)

axs[1].hist(joined["imputed_empower"], ec="black", label="Imputed", color="limegreen", bins=30)
axs[1].hist(joined["empower"], ec="black", label="observed", color="blue", bins=30)
axs[1].set_title("Empowerment Distributions Imputed  \n with Team Informed Estimates", fontsize=20)
axs[0].set_xlabel("Leader Member Exchange - LMX")
axs[0].set_ylabel("Empowerment")
axs[0].set_title("Empowerment Imputed \n with Team Informed Estimates", fontsize=20)
axs[1].legend();
```

It's clear from the hierarchical model that the team specific information has allowed us to impute a wider range of empowerment values with a broader spread as a function of `lmx` and `male`. This is much more persuasive since all politics is local, and this latter model is informed by the conditions of work for each employee. As such, our hierarchical model is able to ascribe a more nuanced view of the probable empowerment values for the missing reports. The hierarchical imputation model "borrows information" in two ways (i) the individual team estimates are pulled toward the global estimates and (ii) the missing values are imputed with respect to our measures of the team dynamics. 

+++

## Conclusion

We've now seen multiple approaches to the imputation of missing data. We have focused on an example where the reason for the missing data is not immediately obvious given how different employees might very well have different reasons for under-specifying their relationship with management. However the techniques applied here are quite general. 

The multivariate normal approaches to imputation works surprisingly well in many cases, but the more cutting edge approach is the sequential specification of chained equations. The Bayesian approach here is **state of the art** because we are quite free to use more than simple regression models as the component models for our imputation equations. For each equation we can be liberal in our choice of likelihood terms and the priors we allow over the sampling distributions. We can also add hierarchical structure to respect natural clusters in our data in so far as they constrain the patterns of missing data.

This general point is important - the flexibility of the Bayesian approach can be tailored to the appropriate complexity of our theory about why our data is missing. Similar considerations apply to the estimation procedures involved in counterfactual inference. The more developed our theory for why the data is missing (why the world is as it is, and not another way), the more we need a flexible modelling framework to capture the subtleties of the theory. Bayesian modelling is a superb tool for this loop of theory construction and evaluation. 

+++

## Authors
- Authored by [Nathaniel Forde](https://nathanielf.github.io/) in February 2023 for [pymc-examples #500](https://github.com/pymc-devs/pymc-examples/pull/500)

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
