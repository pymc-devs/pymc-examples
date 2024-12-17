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

(lecture_09)=
# Modeling Events

:::{post} Jan 7, 2024
:tags: statistical rethinking, bayesian inference, logistic regression
:category: intermediate
:author: Dustin Stansbury
:::

This notebook is part of the PyMC port of the [Statistical Rethinking 2023](https://github.com/rmcelreath/stat_rethinking_2023) lecture series by Richard McElreath.

[Video - Lecture 09 - Modeling Events](https://www.youtube.com/watch?v=Zi6N3GLUJmw)# [Lecture 09 - Modeling Events](https://www.youtube.com/watch?v=Zi6N3GLUJmw)

```{code-cell} ipython3
# Ignore warnings
import warnings

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import statsmodels.formula.api as smf
import utils as utils
import xarray as xr

from matplotlib import pyplot as plt
from matplotlib import style
from scipy import stats as stats

warnings.filterwarnings("ignore")

# Set matplotlib style
STYLE = "statistical-rethinking-2023.mplstyle"
style.use(STYLE)
```

# UC Berkeley Admissions

## Dataset
- 4562 Graduate student applications
- Stratified by
  - Department
  - Gender

Goal is to identify gender discrimination by admissions officers

```{code-cell} ipython3
ADMISSIONS = utils.load_data("UCBadmit")
ADMISSIONS
```

## Modeling Events
- **Events**: discrete, unordered outcomes
- Observations are counts/aggregates
- Unknowns are probabilities $p$ of event ocurring, or odds of those probabilities $\frac{p}{1-p}$
- All things we stratify by interact -- generally never independent in real life
- Often deal with the Log Odds of $p = \log \left (\frac{p}{1-p} \right)$

+++

## Admissions: Drawing the owl 🦉
    
1. Estimands(s)
2. Scientific Models(s)
3. Statistical Models(s)
4. Analysis

+++

# 1. Estimand(s)

## Which path defines "discrimination"

### Direct Discrimination (Causal Effect)

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
utils.draw_causal_graph(
    edge_list=[("G", "D"), ("G", "A"), ("D", "A")],
    node_props={
        "G": {"label": "Gender, G", "color": "red"},
        "D": {"label": "Department, D"},
        "A": {"label": "Admission rate, A", "color": "red"},
    },
    edge_props={("G", "A"): {"color": "red"}},
    graph_direction="LR",
)
```

- aka "Institutional discrimination"
- Referees are biased for or against a particular group
- Usually the type we're interested in identifying if it exists
- Requires strong causal assumptions

Here, deparment, D is a mediator -- this is a common structure in social sciences, where categorical status (e.g. gender) effects some mediating context (e.g. occupation), both of which affect a target outcome (wage). Examples
- wage discrimination
- hiring
- scientific awards

+++

### Indirect Discrimination

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
utils.draw_causal_graph(
    edge_list=[("G", "D"), ("G", "A"), ("D", "A")],
    node_props={
        "G": {"label": "Gender, G", "color": "red"},
        "D": {"label": "Department, D"},
        "A": {"label": "Admission rate, A", "color": "red"},
    },
    edge_props={("G", "D"): {"color": "red"}, ("D", "A"): {"color": "red"}},
    graph_direction="LR",
)
```

- aka "Structural discrimination"
- e.g. Gender affects a person's interests, and therefore the department they will apply to
- Affects overall admission rates
- Requires strong causal assumptions

+++

### Total Discrimination

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
utils.draw_causal_graph(
    edge_list=[("G", "D"), ("G", "A"), ("D", "A")],
    node_props={
        "G": {"label": "Gender, G", "color": "red"},
        "D": {"label": "Department, D"},
        "A": {"label": "Admission rate, A", "color": "red"},
    },
    edge_props={
        ("G", "D"): {"color": "red"},
        ("G", "A"): {"color": "red"},
        ("D", "A"): {"color": "red"},
    },
    graph_direction="LR",
)
```

- aka "Experienced discrimination"
- through both direct and indirect routes
- Requires mild assumptions

+++

### Estimands & Estimators
- Each of the **different estimands require a different estimators**
- Often the thing we **can estimate** often isn't what we **want to estimate**
- e.g. Total discrimination may be easier to estimate, but is less actionable

+++

## Unobserved Confounds

```{code-cell} ipython3
utils.draw_causal_graph(
    edge_list=[("G", "D"), ("G", "A"), ("D", "A"), ("U", "D"), ("U", "A")],
    node_props={"U": {"style": "dashed"}, "unobserved": {"style": "dashed"}},
    graph_direction="LR",
)
```

It's always possible there are also confounds between the mediator and some unobserved confounds. We will ignore these for now.

+++

# 2. Scientific Model(s):

## Simulate the process
Below is a generative model of the review/admission process

```{code-cell} ipython3
np.random.seed(123)
n_samples = 1000

GENDER = stats.bernoulli.rvs(p=0.5, size=n_samples)

# Groundtruth parameters
# Gender 1 tends to apply to department 1
P_DEPARTMENT = np.where(GENDER == 0, 0.3, 0.8)

# Acceptance rates matrices -- Department x Gender
UNBIASED_ACCEPTANCE_RATES = np.array([[0.1, 0.1], [0.3, 0.3]])  # No *direct* gender bias

# Biased acceptance:
# - dept 0 accepts gender 0 at 50% of unbiased rate
# - dept 1 accepts gender 0 at 67% of unbiased rate
BIASED_ACCEPTANCE_RATES = np.array([[0.05, 0.1], [0.2, 0.3]])  # *direct* gender bias present

DEPARTMENT = stats.bernoulli.rvs(p=P_DEPARTMENT, size=n_samples).astype(int)


def simulate_admissions_data(bias_type):
    """Simulate admissions data using the global params above"""
    acceptance_rate = (
        BIASED_ACCEPTANCE_RATES if bias_type == "biased" else UNBIASED_ACCEPTANCE_RATES
    )
    acceptance = stats.bernoulli.rvs(p=acceptance_rate[DEPARTMENT, GENDER])

    return (
        pd.DataFrame(
            np.vstack([GENDER, DEPARTMENT, acceptance]).T,
            columns=["gender", "department", "accepted"],
        ),
        acceptance_rate,
    )


for bias_type in ["unbiased", "biased"]:

    fake_admissions, acceptance_rate = simulate_admissions_data(bias_type)

    gender_acceptance_counts = fake_admissions.groupby(["gender", "accepted"]).count()["department"]
    gender_acceptance_counts.name = None

    gender_department_counts = fake_admissions.groupby(["gender", "department"]).count()["accepted"]
    gender_department_counts.name = None

    observed_acceptance_rates = fake_admissions.groupby("gender").mean()["accepted"]
    observed_acceptance_rates.name = None

    print()
    print("-" * 30)
    print(bias_type.upper())
    print("-" * 30)
    print(f"Department Acceptance rate:\n{acceptance_rate}")
    print(f"\nGender-Department Frequency:\n{gender_department_counts}")
    print(f"\nGender-Acceptance Frequency:\n{gender_acceptance_counts}")
    print(f"\nOverall Acceptance Rates:\n{observed_acceptance_rates}")
```

### Simulated data properties

#### Both scenarios

- Gender 0 tends to apply to department 0
- Gender 1 tends to apply to department 1

#### Unbiased scenario:
- due to lower acceptance rates in dept 0 and tendency of gender 0 to apply to dept 0, gender 0 has a lower overall rejection rate compared to gender 1
- due to higher acceptance rates in dept 1 and tendency of gender 1 to apply to dept 1, gender 1 has a higher overall acceptance rate compared to gender 0
- even in the case of no (direct) gender discrimination, **there is still indirect discrimination based on tendency of genders to apply to different departments, and the unqual likelihood that each department accepts students.**

#### Biased scenario
- overall acceptance rates are lower (due to baseline reduction in gender 0 acceptance rates across both departments)
- in the scenario where there is actual department bias, **we see a similar overall _pattern_ of discrimination as the unbiased case due to the indirect effect.**

+++

# 3. Statistical Models

+++

## Modeling Events
- **We observe** counts of events
- **We estimate** probabilities -- or, rather, the log-odds of events ocurring


### Linear Models
Expected value is linear (additive) combination of parameters

$$
\begin{align*}
Y_i &\sim \mathcal{N}(\mu_i, \sigma) \\
\mu_i &= \alpha + \beta_X X_i + ...
\end{align*}
$$

b.c. Normal distribution is **unbounded**, so too is the expected value of the linear model.

### Event Models
Discrete events either occur, taking the value 1, or they do not, taking the value 0. This **puts bounds** on the expected value of an event. Namely the bounds are on the interval $(0, 1)$

## Generalized Linear Models & Link Functions

- Expected value is **some function $f(\mathbb{E}[\theta])$ of an additive combination of the parameters**.

$$
f(\mathbb{E}[\theta]) = \alpha + \beta_X X_i + ...
$$

- Generally, this function $f()$, called the **link function**, will have a specific form **that is associated with the likelihood distribution**. 
- the link function will have an **inverse link function** $ f^{-1}(X_i, \theta)$ such that
- to reiterate **link functions are matched to distributions**

$$
\mathbb{E}[\theta] = f^{-1}(\alpha + \beta_X X_i + ...)
$$

- In the linear regression case the likelihood model is Gaussian, and the associated link function (and inverse link function) is just the identity, $I()$. (left plot below)


```{code-cell} ipython3
_, axs = plt.subplots(1, 2, figsize=(8, 4))
xs = np.linspace(-3, 3, 100)
plt.sca(axs[0])
alpha = 0
beta = 0.75

plt.plot(xs, alpha + beta * xs)
plt.axhline([0], linestyle="--", color="gray")
plt.axhline([1], linestyle="--", color="gray")
plt.xlim([-3, 3])
plt.ylim([-0.5, 1.5])
plt.xlabel("x")
plt.ylabel("$I(\\alpha + \\beta X$)")
plt.title("Linear Model\n(continous outcomes)")

plt.sca(axs[1])
plt.plot(xs, utils.invlogit(alpha + beta * xs))
plt.axhline([0], linestyle="--", color="gray")
plt.axhline([1], linestyle="--", color="gray")
plt.xlim([-3, 3])
plt.ylim([-0.5, 1.5])
plt.xlabel("x")
plt.ylabel("$\\frac{1}{e^{-(\\alpha + \\beta X)}}$")
plt.title("Logistic Model\n(binary outcomes)");
```

## Logit Link Function and Logisitic Inverse function
### Binary outcomes

- **Logistic regression** used to model the probability of an event ocurring
- Likelihood function is the Bernouilli
- Associated link function is the **log-odds, or logit**: $f(p) = \frac{p}{1-p}$

$$
\begin{align*}
Y_i &\sim \text{Bernoulli}(p_i) \\
f(p_i) &= \alpha + \beta_X X_i + ...
\end{align*}
$$

- the inverse link function is the **inverse logit** aka **logistic** function (right plot above): $f^{-1}(X_i, \theta) = \frac{1}{1 + e^{-(\alpha + \beta_X X_i + ...)}}$
  - defining $\alpha + \beta_X X + ... = q$ (ignoring the data index $i$ for now), the derivation of the inverse logit is as follows
$$
\begin{align}
\log \left(\frac{p}{1-p}\right) &= \alpha + \beta_X X + ... = q \\
\frac{p}{1-p} &= e^{q} \\
p &= (1-p) e^{q} \\
p + p e^{q} &= e^{q} \\
p(1 + e^{q}) &= e^{q} \\
p &= \frac{ e^{q}}{(1 + e^{q})} \\
p &= \frac{ e^{q}}{(1 + e^{q})} \frac{e^{-q}}{e^{-q}}\\
p &= \frac{1}{(1 + e^{-q})} \\
\end{align}
$$

### logit is a harsh transform
Interpreting the log odds can be difficult at first, but in time becomes easier
- log-odds scale
- $\text{logit}(p)=0, p=0.5$
- $\text{logit}(p)=-\infty, p=0$
  - rule of thumb: $\text{logit}(p)<-4$ means event is unlikely (hardly ever)
- $\text{logit}(p)=\infty, p=1$
  - rule of thumb: $\text{logit}(p)>4$ means event is very likely (nearly always)

```{code-cell} ipython3
log_odds = np.linspace(-4, 4, 100)
utils.plot_line(log_odds, utils.invlogit(log_odds), label=None)
plt.axhline(0.5, linestyle="--", color="k", label="p=0.5")
plt.xlabel("$logit(p)$")
plt.ylabel("$p$")
plt.legend();
```

### Bayesian Updating for Logistic Regression
For the following simulation, we'll use a custom utility function `utils.simulate_2_parameter_bayesian_learning` for simulating general Bayeisan posterior update simulation. Here's the API for that function (for more details see `utils.py`)

```{code-cell} ipython3
help(utils.simulate_2_parameter_bayesian_learning_grid_approximation)
```

```{code-cell} ipython3
# Model function required for simulation


def logistic_model(x, alpha, beta):
    return utils.invlogit(alpha + beta * x)


# Posterior function required for simulation
def logistic_regression_posterior(x_obs, y_obs, alpha_grid, beta_grid, likelihood_prior_std=0.01):

    # Convert params to 1-d arrays
    if np.ndim(alpha_grid) > 1:
        alpha_grid = alpha_grid.ravel()

    if np.ndim(beta_grid):
        beta_grid = beta_grid.ravel()

    log_prior_intercept = stats.norm(0, 1).logpdf(alpha_grid)
    log_prior_slope = stats.norm(0, 1).logpdf(beta_grid)

    log_likelihood = np.array(
        [
            stats.bernoulli(p=utils.invlogit(a + b * x_obs)).logpmf(y_obs)
            for a, b in zip(alpha_grid, beta_grid)
        ]
    ).sum(axis=1)

    # Posterior is equal to the product of likelihood and priors (here a sum in log scale)
    log_posterior = log_likelihood + log_prior_intercept + log_prior_slope

    # Convert back to natural scale
    return np.exp(log_posterior - log_posterior.max())


# Generate data for demo
np.random.seed(123)
RESOLUTION = 100
N_DATA_POINTS = 128

# Ground truth parameters
ALPHA = -1
BETA = 2

x = stats.norm(0, 1).rvs(size=N_DATA_POINTS)
p_y = utils.invlogit(ALPHA + BETA * x)
y = stats.bernoulli.rvs(p=p_y)

alpha_grid = np.linspace(-3, 3, RESOLUTION)
beta_grid = np.linspace(-3, 3, RESOLUTION)

# Vary the sample size to show how the posterior adapts to more and more data
for n_samples in [0, 2, 4, 8, 16, 32, 64, 128]:
    # Run the simulation
    utils.simulate_2_parameter_bayesian_learning_grid_approximation(
        x_obs=x[:n_samples],
        y_obs=y[:n_samples],
        param_a_grid=alpha_grid,
        param_b_grid=beta_grid,
        true_param_a=ALPHA,
        true_param_b=BETA,
        model_func=logistic_model,
        posterior_func=logistic_regression_posterior,
        param_labels=["$\\alpha$", "$\\beta$"],
        data_range_x=(-2, 2),
        data_range_y=(-0.05, 1.05),
    )
```

## Priors for logistic regression

**logit link function is a harsh transform**
- Logit compresses parameter distributions
- $x > +4 \rightarrow $ event basically always occurs
- $x < -4 \rightarrow$ event basically never occurs

```{code-cell} ipython3
n_samples = 10000
fig, axs = plt.subplots(3, 2, figsize=(8, 8))
for ii, std in enumerate([10, 1.5, 1]):

    # Alpha prior distribution
    alphas = stats.norm(0, std).rvs(n_samples)
    plt.sca(axs[ii][0])
    az.plot_dist(alphas, color="C1")
    plt.xlim([-30, 30])
    plt.ylabel("density")
    if ii == 2:
        plt.xlabel("alpha")

    # Resulting event probability distribution
    ps = utils.invlogit(alphas)
    plt.sca(axs[ii][1])
    az.plot_dist(ps, color="C0")
    plt.xlim([0, 1])
    plt.ylabel("density")
    plt.title(f"$\\alpha \sim \mathcal{{N}}(0, {std})$")
    if ii == 2:
        plt.xlabel("p(event)")
```

### Prior Predictive Simulations

```{code-cell} ipython3
# Demonstrating the effect of alpha / beta on p(x)
from functools import partial


def gaussian_2d_pdf(xs, ys, mean=(0, 0), covariance=np.eye(2)):
    return np.array(stats.multivariate_normal(mean, covariance).pdf(np.vstack([xs, ys]).T))


pdf = partial(gaussian_2d_pdf)

xs = ys = np.linspace(-3, 3, 100)
# utils.plot_2d_function(xs, ys, pdf, cmap='gray_r')


def plot_logistic_prior_predictive(n_samples=20, prior_std=0.5):

    _, axs = plt.subplots(1, 2, figsize=(10, 4))
    plt.sca(axs[0])
    min_x, max_x = -prior_std, prior_std
    xs = np.linspace(min_x, max_x, 100)
    ys = xs

    pdf = partial(gaussian_2d_pdf, covariance=np.array([[prior_std**2, 0], [0, prior_std**2]]))
    utils.plot_2d_function(xs, ys, pdf, cmap="gray_r")

    plt.axis("square")
    plt.xlim([min_x, max_x])
    plt.ylim([min_x, max_x])

    # Sample some parameters from prior
    βs = stats.norm.rvs(0, prior_std, size=n_samples)
    αs = stats.norm.rvs(0, prior_std, size=n_samples)

    for b, a in zip(βs, αs):
        plt.scatter(a, b)

    plt.xlabel("$\\alpha$")
    plt.ylabel("$\\beta$")
    plt.title(f"Samples from prior with std={prior_std}")

    # Resulting sigmoid functions
    plt.sca(axs[1])
    min_x, max_x = -3, 3
    xs = np.linspace(min_x, max_x, 100)
    for a, b in zip(αs, βs):
        logit_p = a + xs * b
        p = utils.invlogit(logit_p)
        plt.plot(xs, p)

    plt.xlabel("x")
    plt.ylabel("p = invlogit(x)")
    plt.xlim([-3, 3])
    plt.ylim([-0.05, 1.05])
    plt.axhline(0.5, linestyle="--", color="k")
    plt.title(f"Resulting Logistic Models")
```

```{code-cell} ipython3
plot_logistic_prior_predictive(prior_std=0.5)
```

```{code-cell} ipython3
plot_logistic_prior_predictive(prior_std=1.0)
```

```{code-cell} ipython3
plot_logistic_prior_predictive(prior_std=10)
```

## Statistical models for admissions
Again, the estimator will depend on the estimand

+++

### Total Causal Effect

```{code-cell} ipython3
utils.draw_causal_graph(
    edge_list=[("G", "D"), ("G", "A"), ("D", "A")],
    node_props={
        "G": {"color": "red"},
        "A": {"color": "red"},
    },
    edge_props={
        ("G", "A"): {"color": "red"},
        ("D", "A"): {"color": "red"},
        ("G", "D"): {"color": "red"},
    },
    graph_direction="LR",
)
```


**Stratify by only Gender**. Don't stratify by Department b.c. it's a Pipe (mediator) that we do not want to block

$$
\begin{align*}
A_i &\sim \text{Bernoulli}(p=p_i) \\
\text{logit}(p_i) &= \alpha[G_i] \\
\alpha &= [\alpha_0, \alpha_1] \\
\alpha_j &\sim \text{Normal}(0, 1)
\end{align*}
$$

+++

### Direct Causal Effect

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
utils.draw_causal_graph(
    edge_list=[("G", "D"), ("G", "A"), ("D", "A")],
    node_props={
        "G": {"color": "red"},
        "A": {"color": "red"},
    },
    edge_props={("G", "A"): {"color": "red"}},
    graph_direction="LR",
)
```


**Stratify by Gender and Department** to block the Pipe

$$
\begin{align*}
A_i &\sim \text{Bernoulli}(p_i) \\
\text{logit}(p_i) &= \alpha[G_i, D_i] \\
\alpha &= \left[{\begin{array}{cc} \alpha_{0,0}, \alpha_{0,1} \\ \alpha_{1,0}, \alpha_{1,1} \end{array}}\right] \\
\alpha_{j,k} &\sim \text{Normal}(0, 1)
\end{align*}
$$

+++

### Fitting Total Causal Effect Model

```{code-cell} ipython3
def fit_total_effect_model_admissions(data):
    """Fit total effect model, stratifying by gender.

    Note
    ----
    We use the Binomial regression, so simulated observation-level
    data from above must be pre-aggregated. (see the function
    `aggregate_admissions_data` below). We could have
    used a Bernoulli likelihood (Logistic regression) for the simulated
    data, but given that we only have aggregates for the real
    UC Berkeley data we choose this more general formulation.
    """

    # Set up observations / coords
    n_admitted = data["admit"].values
    n_applications = data["applications"].values

    gender_coded, gender = pd.factorize(data["applicant.gender"].values)

    with pm.Model(coords={"gender": gender}) as total_effect_model:

        # Mutable data for running any gender-based counterfactuals
        gender_coded = pm.MutableData("gender", gender_coded, dims="obs_ids")

        alpha = pm.Normal("alpha", 0, 1, dims="gender")

        # Record the inverse linked probabilities for reporting
        pm.Deterministic("p_accept", pm.math.invlogit(alpha), dims="gender")

        p = pm.math.invlogit(alpha[gender_coded])

        likelihood = pm.Binomial(
            "accepted", n=n_applications, p=p, observed=n_admitted, dims="obs_ids"
        )
        total_effect_inference = pm.sample()

    return total_effect_model, total_effect_inference


def aggregate_admissions_data(raw_admissions_data):
    """Aggregate simulated data from `simulate_admissions_data`, which
    is in long format to short format, by so that it can be modeled
    using Binomial likelihood. We also recode column names to have the
    same fields as the UC Berkeley dataset so that we can use the
    same model-fitting functions.
    """

    # Aggregate applications, admissions, and rejections. Rename
    # columns to have the  same fields as UC Berkeley dataset
    applications = raw_admissions_data.groupby(["gender", "department"]).count().reset_index()
    applications.rename(columns={"accepted": "applications"}, inplace=True)

    admits = raw_admissions_data.groupby(["gender", "department"]).sum().reset_index()
    admits.rename(columns={"accepted": "admit"}, inplace=True)

    data = applications.merge(admits)
    data.loc[:, "reject"] = data.applications - data.admit

    # Code gender & department. For easier comparison to lecture,
    # we use 1-indexed gender/department indicators like McElreath
    data.loc[:, "applicant.gender"] = data.gender.apply(lambda x: 2 if x else 1)
    data.loc[:, "dept"] = data.department.apply(lambda x: 2 if x else 1)

    return data[["applicant.gender", "dept", "applications", "admit", "reject"]]


def plot_admissions_model_posterior(inference, figsize=(6, 2)):
    _, ax = plt.subplots(figsize=figsize)

    # Plot alphas
    az.plot_forest(inference, var_names=["alpha"], combined=True, hdi_prob=0.89, ax=ax)

    # Plot inver-linked acceptance probabilities
    _, ax = plt.subplots(figsize=figsize)
    az.plot_forest(inference, var_names=["p_accept"], combined=True, hdi_prob=0.89, ax=ax)

    return az.summary(inference)
```

### Fit Total Effect model on BIASED simulated admissions

```{code-cell} ipython3
SIMULATED_BIASED_ADMISSIONS, _ = simulate_admissions_data("biased")
SIMULATED_BIASED_ADMISSIONS = aggregate_admissions_data(SIMULATED_BIASED_ADMISSIONS)
SIMULATED_BIASED_ADMISSIONS
```

```{code-cell} ipython3
total_effect_model_simulated_biased, total_effect_inference_simulated_biased = (
    fit_total_effect_model_admissions(SIMULATED_BIASED_ADMISSIONS)
)
```

```{code-cell} ipython3
plot_admissions_model_posterior(total_effect_inference_simulated_biased, (6, 1.5))
```

### Fit Total Effect model on UNBIASED simulated admissions

```{code-cell} ipython3
SIMULATED_UNBIASED_ADMISSIONS, _ = simulate_admissions_data("unbiased")
SIMULATED_UNBIASED_ADMISSIONS = aggregate_admissions_data(SIMULATED_UNBIASED_ADMISSIONS)
SIMULATED_UNBIASED_ADMISSIONS
```

```{code-cell} ipython3
total_effect_model_simulated_unbiased, total_effect_inference_simulated_unbiased = (
    fit_total_effect_model_admissions(SIMULATED_UNBIASED_ADMISSIONS)
)
```

```{code-cell} ipython3
plot_admissions_model_posterior(total_effect_inference_simulated_unbiased, (6, 1.5))
```

## Fitting Direct Causal Effect Model

```{code-cell} ipython3
def fit_direct_effect_model_admissions(data):
    """Fit total effect model, stratifying by gender.

    Note
    ----
    We use the Binomial likelihood, so simulated observation-level data from
    above must be pre-aggregated. (see the function `aggregate_admissions_data`
    below). We could have used a Bernoulli likelihood for the simulated data,
    but given that we only have aggregates for the real UC Berkeley data we
    choose this more general formulation.
    """

    # Set up observations / coords
    n_admitted = data["admit"].values
    n_applications = data["applications"].values

    department_coded, department = pd.factorize(data["dept"].values)
    gender_coded, gender = pd.factorize(data["applicant.gender"].values)

    with pm.Model(coords={"gender": gender, "department": department}) as direct_effect_model:

        # Mutable data for running any gender-based or department-based counterfactuals
        gender_coded = pm.MutableData("gender", gender_coded, dims="obs_ids")
        department_coded = pm.MutableData("department", department_coded, dims="obs_ids")
        n_applications = pm.MutableData("n_applications", n_applications, dims="obs_ids")

        alpha = pm.Normal("alpha", 0, 1, dims=["department", "gender"])

        # Record inverse linked probabilities for reporting
        pm.Deterministic("p_accept", pm.math.invlogit(alpha), dims=["department", "gender"])

        p = pm.math.invlogit(alpha[department_coded, gender_coded])

        likelihood = pm.Binomial(
            "accepted", n=n_applications, p=p, observed=n_admitted, dims="obs_ids"
        )
        direct_effect_inference = pm.sample()

    return direct_effect_model, direct_effect_inference
```

### Fit Direct Effect model to BIASED simulated admissions data

```{code-cell} ipython3
direct_effect_model_simulated_biased, direct_effect_inference_simulated_biased = (
    fit_direct_effect_model_admissions(SIMULATED_BIASED_ADMISSIONS)
)
```

```{code-cell} ipython3
plot_admissions_model_posterior(direct_effect_inference_simulated_biased)
```

For comparison, here's the ground truth biased admission rates, which we're able to mostly recover:

```{code-cell} ipython3
# Department x Gender
BIASED_ACCEPTANCE_RATES
```

### Fit Direct Effect model to UNBIASED simulated admissions data

```{code-cell} ipython3
direct_effect_model_simulated_unbiased, direct_effect_inference_simulated_unbiased = (
    fit_direct_effect_model_admissions(SIMULATED_UNBIASED_ADMISSIONS)
)
```

```{code-cell} ipython3
plot_admissions_model_posterior(direct_effect_inference_simulated_unbiased)
```

For comparison, here's the ground truth unbiased admission rates, which were able to recover:

```{code-cell} ipython3
# Department x Gender
UNBIASED_ACCEPTANCE_RATES
```

# 4. Analyze the UC Berkeley Admissions data

+++

### Review of the counts dataset
- we'll use these counts data to model log odds of acceptance rate for gender/department
- note that we'll be using Binomial Regression, which is equivalent to Bernoulli regression, but operates on aggregate counts, as oppose to individual binary trials
  - The examples above were actually implemented as Binomial Regression so that we can re-use code and demonstrate general patterns of analysis
  - Either way, you'll get the same inference using both approaches

```{code-cell} ipython3
ADMISSIONS
```

### Fit Total Effect model to UC Berkeley admissions data

> Don't forget to look at diagnostics, which we'll skip here

```{code-cell} ipython3
total_effect_model, total_effect_inference = fit_total_effect_model_admissions(ADMISSIONS)
```

```{code-cell} ipython3
plot_admissions_model_posterior(total_effect_inference, figsize=(6, 1.5))
```

### Total Causal Effect

```{code-cell} ipython3
# Total Causal Effect
female_p_accept = total_effect_inference.posterior["p_accept"].sel(gender="female")
male_p_accept = total_effect_inference.posterior["p_accept"].sel(gender="male")
contrast = male_p_accept - female_p_accept

plt.subplots(figsize=(8, 3))
az.plot_dist(contrast)
plt.axvline(0, linestyle="--", color="black", label="No difference")
plt.xlabel("gender contrast (acceptance probability)")
plt.ylabel("density")
plt.title(
    "Total Causal Effect (Experienced Discrimination):\npositive values indicate male advantage"
)
plt.legend();
```

### Fit Direct Effect model to UC Berkeley admissions data

```{code-cell} ipython3
direct_effect_model, direct_effect_inference = fit_direct_effect_model_admissions(ADMISSIONS)
```

```{code-cell} ipython3
plot_admissions_model_posterior(direct_effect_inference, figsize=(6, 3))
```

```{code-cell} ipython3
# Fancier plot of department/gender acceptance probability distributions
_, ax = plt.subplots(figsize=(10, 5))
for ii, dept in enumerate(ADMISSIONS.dept.unique()):
    color = f"C{ii}"
    for gend in ADMISSIONS["applicant.gender"].unique():
        label = f"{dept}:{gend}"
        linestyle = "-." if gend == "female" else "-"
        post = direct_effect_inference.posterior["p_accept"].sel(department=dept, gender=gend)
        az.plot_dist(post, label=label, color=color, plot_kwargs={"linestyle": linestyle})
plt.xlabel("acceptance probability")
plt.legend(title="Deptartment:Gender");
```

### Direct Causal Effect

```{code-cell} ipython3
# Direct Causal Effect
_, ax = plt.subplots(figsize=(8, 4))

for ii, dept in enumerate(ADMISSIONS.dept.unique()):
    color = f"C{ii}"
    label = f"{dept}"

    female_p_accept = direct_effect_inference.posterior["p_accept"].sel(
        gender="female", department=dept
    )
    male_p_accept = direct_effect_inference.posterior["p_accept"].sel(
        gender="male", department=dept
    )

    contrast = male_p_accept - female_p_accept
    az.plot_dist(contrast, color=color, label=label)

plt.xlabel("gender contrast (acceptance probability)")
plt.ylabel("density")
plt.title(
    "Direct Causal Effect (Institutional Discrimination):\npositive values indicate male advantage"
)
plt.axvline(0, linestyle="--", color="black", label="No difference")
plt.legend(title="Department");
```

### Average Direct Effect
- The causal effect of gender **averaged across departments** (marginalize)
- Depends on the distribution of applications to each deparment
- This is easy to do as a simulation

+++

#### Post-stratification
- **Re-weight estimates** for the target population
- allows us to apply model fit from one university to estiamte causal impact at another university with a different distribution of departments
- Here, we use the empirical distribution for re-weighting estimates

```{code-cell} ipython3
# Use the empirical distribution of departments -- we'd update this for a different university
applications_per_dept = ADMISSIONS.groupby("dept").sum()["applications"].values
total_applications = applications_per_dept.sum()
department_weight = applications_per_dept / total_applications

female_alpha = direct_effect_inference.posterior.sel(gender="female")["alpha"]
male_alpha = direct_effect_inference.posterior.sel(gender="male")["alpha"]

weighted_female_alpha = female_alpha * department_weight
weighed_male_alpha = male_alpha * department_weight

contrast = weighed_male_alpha - weighted_female_alpha

_, ax = plt.subplots(figsize=(8, 4))
az.plot_dist(contrast)

plt.axvline(0, linestyle="--", color="k", label="No Difference")
plt.xlabel("causal effect of perceived gender")
plt.ylabel("density")
plt.title("Average Direct Causal of Gender (perception),\npositive values indicate male advantage")
plt.xlim([-0.3, 0.3])
plt.legend();
```

To verify the averaging process, we can look at the contrast of the `p_accept` samples from the posterior, which provides similar results. However, looking ath the posterior obviously wouldn't work for making predictions for an out-of-sample university however.

```{code-cell} ipython3
direct_posterior = direct_effect_inference.posterior

# Select by gender; note: p_accept already has link function applied
female_posterior = direct_posterior.sel(gender="female")[
    "p_accept"
]  # shape: (chain x draw x department)
male_posterior = direct_posterior.sel(gender="male")[
    "p_accept"
]  # shape: (chain x draw x department)
contrast = male_posterior - female_posterior  # shape: (chain x draw x department)

_, ax = plt.subplots(figsize=(8, 4))
# marginalize / collapse the contrast across all departments
az.plot_dist(contrast)
plt.axvline(0, linestyle="--", color="k", label="No Difference")
plt.xlim([-0.3, 0.3])
plt.xlabel("causal effect of perceived gender")
plt.ylabel("density")
plt.title("Posterior contrast of $p_{accept}$,\npositive values indicate male advantage")
plt.legend();
```

## Discrimination?
Hard to say
- Big structural effects
- Distribution of applicants could be due to discrimination
- **Confounds are likely**

+++

# BONUS: Survival Analysis
- Counts often modeled as time-to-event (i.e. Exponential or Gamma distribution)  -- Goal is to estimate event rate
- Tricky b.c. ignoring **censored data** can lead to inferential errors
  - **Left censored**: we don't know when the timer started
  - **Right censored** the observation period finished before the event had time to happen

+++

## Example: Austin Cat Adoption

- 20k cats
- Events: adopted (1) or not (0)
- Censoring mechanisms
  - death before adoption
  - escape
  - not adopted yet

**Goal**: determine if Black are adopted at a lower rate than non-Black cats.

```{code-cell} ipython3
CATS = utils.load_data("AustinCats")
CATS.head()
```

### Modeling outcome variable: `days_to_event`
Two go-to distributions for modeling time-to-event

**Exponential Distribution**:
- simpler of the two (single parameter)
- assumes constant rate
- maximum entropy distribution amongst the set of non-negative continuous distributions that have the same average rate.
- assumes a single thing to occur (e.g. part failure) before an event occurs (machine breakdown)

**Gamma Distribution**
- more complex of the two (two parameters)
- maximum entropy distribution amongst the set of distributions with the same mean and average log.
- assumes multiple things to happen (e.g. part failures) before an event occurs (machine breakdown)

```{code-cell} ipython3
_, axs = plt.subplots(1, 2, figsize=(10, 4))

days = np.linspace(0, 100, 100)

plt.sca(axs[0])
for scale in [20, 50]:
    expon = stats.expon(loc=0.01, scale=scale)
    utils.plot_line(days, expon.pdf(days), label=f"$\\lambda$=1/{scale}")
plt.xlabel("days")
plt.ylabel("density")
plt.legend()
plt.title("Exponential Distribution")


plt.sca(axs[1])
N = 8
alpha = 10
for scale in [2, 5]:
    gamma = stats.gamma(a=10, loc=alpha, scale=scale)
    utils.plot_line(days, gamma.pdf(days), label=f"$\\alpha$={alpha}, $\\beta$={scale}")
plt.xlabel("days")
plt.ylabel("density")
plt.legend()
plt.title("Gamma Distribution");
```

## Censored and un-censored observations
- Observed data use the Cumulative distribution; i.e. the **probability that the event occurred by time x**
- Unobserved (censored) data instead require the Complementary of the CDF; i.e. the **probability that the event hasn't happened yet**. 

```{code-cell} ipython3
xs = np.linspace(0, 5, 100)
_, axs = plt.subplots(1, 2, figsize=(10, 5))
plt.sca(axs[0])
cdfs = {}
for lambda_ in [0.5, 1]:
    cdfs[lambda_] = ys = 1 - np.exp(-lambda_ * xs)
    utils.plot_line(xs, ys, label=f"$\\lambda={lambda_}$")

plt.xlabel("x")
plt.ylabel("$1 - \\exp(-x)$")
plt.legend()
plt.title("CDF\np(event happened before or at time x| $\\lambda$)", fontsize=12)

plt.sca(axs[1])
for lambda_ in [0.5, 1]:
    ys = 1 - cdfs[lambda_]
    utils.plot_line(xs, ys, label=f"$\\lambda={lambda_}$")

plt.xlabel("x")
plt.ylabel("$\\exp(-x)$")
plt.legend()
plt.title("CCDF\np(event hasn't happened before or at time x| $\\lambda$)", fontsize=12);
```

## Statistical Model

+++

$$
\begin{align*}
D_i | A_i &= 1 \sim \text{Exponential}(\lambda_i) \\
D_i | A_i &= 0 \sim \text{ExponentialCCDF}(\lambda_i) \\
\lambda_i &= \frac{1}{\mu_i} \\
\log \mu_i &= \alpha_{\text{cat color}[i]} \\
\alpha_{Black, Other} &\sim \text{Exponential}(\gamma)
\end{align*}
$$

- $D_i | A_i = 1$ - observed adoptions
- $D_i | A_i = 1$ - not-yet-observed adoptions
- $\alpha_{\text{cat color}[i]}$ log average time-to-adoption for each cat color
- $\log \mu_i$ -- link function ensures $\alpha$s are positive
- $\lambda_i = \frac{1}{\mu_i}$ simplifies estimating average **time-to-adoption**

+++

### Finding reasonable hyperparameter for $\alpha$
We'll need to determine a reasonable data for the Exponential prior mean parameter $\gamma$. To do so, we'll look at the empirical distribution of time to adoption:

```{code-cell} ipython3
# determine reasonable prior for alpha
plt.subplots(figsize=(6, 3))
CATS[CATS.out_event == "Adoption"].days_to_event.hist(bins=100)
plt.xlim([0, 500]);
```

Using the above empirical historgram, we see that a majority of the probablity mass is between zero and 200, so let's use 50 as the expected wait time.

```{code-cell} ipython3
CAT_COLOR_ID, CAT_COLOR = pd.factorize(CATS.color.apply(lambda x: "Other" if x != "Black" else x))
ADOPTED_ID, ADOPTED = pd.factorize(
    CATS.out_event.apply(lambda x: "Other" if x != "Adoption" else "Adopted")
)
DAYS_TO_ADOPTION = CATS.days_to_event.values.astype(float)
LAMBDA = 50

with pm.Model(coords={"cat_color": CAT_COLOR}) as adoption_model:

    # Censoring
    right_censoring = DAYS_TO_ADOPTION.copy()
    right_censoring[ADOPTED_ID == 0] = DAYS_TO_ADOPTION.max()

    # Priors
    gamma = 1 / LAMBDA
    alpha = pm.Exponential("alpha", gamma, dims="cat_color")

    # Likelihood
    log_adoption_rate = 1 / alpha[CAT_COLOR_ID]
    pm.Censored(
        "adopted",
        pm.Exponential.dist(lam=log_adoption_rate),
        lower=None,
        upper=right_censoring,
        observed=DAYS_TO_ADOPTION,
    )
    adoption_inference = pm.sample()
```

## Poor black kitties 🐈‍⬛
It appears that black cats DO take longer to get adopted.

+++

#### Posterior Summary

```{code-cell} ipython3
_, ax = plt.subplots(figsize=(4, 2))
az.plot_forest(adoption_inference, var_names=["alpha"], combined=True, hdi_prob=0.89, ax=ax)
plt.xlabel("days to adoption");
```

#### Posterior distributions

```{code-cell} ipython3
for ii, cat_color in enumerate(["Black", "Other"]):
    color = "black" if cat_color == "Black" else "C0"
    posterior_alpha = adoption_inference.posterior.sel(cat_color=cat_color)["alpha"]
    az.plot_dist(posterior_alpha, color=color, label=cat_color)
plt.xlabel("waiting time")
plt.ylabel("density")
plt.legend();
```

```{code-cell} ipython3
days_until_adoption_ = np.linspace(1, 100)
n_posterior_samples = 25

for cat_color in ["Black", "Other"]:
    color = "black" if cat_color == "Black" else "C0"
    posterior = adoption_inference.posterior.sel(cat_color=cat_color)
    alpha_samples = posterior.alpha.sel(chain=0)[:n_posterior_samples].values
    for ii, alpha in enumerate(alpha_samples):
        label = cat_color if ii == 0 else None
        plt.plot(
            days_until_adoption_,
            np.exp(-days_until_adoption_ / alpha),
            color=color,
            alpha=0.25,
            label=label,
        )

plt.xlabel("days until adoption")
plt.ylabel("fraction")
plt.legend();
```

## Authors
* Ported to PyMC by Dustin Stansbury (2024)
* Based on Statistical Rethinking (2023) lectures by Richard McElreath

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,aeppl,xarray
```

:::{include} ../page_footer.md
:::
