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

(lecture_04)=
# Categories and Curves
:::{post} Jan 14, 2024
:tags: statistical rethinking, bayesian inference, categorical variables, splines
:category: intermediate
:author: Dustin Stansbury
:::

This notebook is part of the PyMC port of the [Statistical Rethinking 2023](https://github.com/rmcelreath/stat_rethinking_2023) lecture series by Richard McElreath.

[Video - Lecture 04 - Categories and Curves](https://youtu.be/F0N4b7K_iYQ)

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

style.use("fivethirtyeight")
az.style.use("arviz-darkgrid")

%config InlineBackend.figure_format = 'retina'
%load_ext autoreload
%autoreload 2
```

# Linear Regression & Drawing Inferences
- Can be used to approximate most anything, even nonlinear phenomena (e.g. GLMs)
- We need to incorporate causal thinking into...
  - ...how we compose statistical models
  - ...how we process and interpret results

+++

# Categories
- non-continous causes
- discrete, unordered types
- stratifying by category: fit a separate regression (e.g. line) to each

## Revisiting the Howell dataset

```{code-cell} ipython3
HOWELL = utils.load_data("Howell1")

# Adult data
ADULT_HOWELL = HOWELL[HOWELL.age >= 18]

# Split by the Sex Category
SEX = ["women", "men"]

plt.subplots(figsize=(5, 4))
for ii, label in enumerate(SEX):
    utils.plot_scatter(
        ADULT_HOWELL[ADULT_HOWELL.male == ii].height,
        ADULT_HOWELL[ADULT_HOWELL.male == ii].weight,
        color=f"C{ii}",
        label=label,
    )
plt.ylim([30, 65])
plt.legend();
```

```{code-cell} ipython3
# Draw the mediation graph
utils.draw_causal_graph(edge_list=[("H", "W"), ("S", "H"), ("S", "W")], graph_direction="LR")
```


## Think scientifically first

- How are height, weight, and sex **causally** related?
- How are height, weight, and sex **statistically** related?

### The cuases aren't in the data

Height should affect weight, not vice versa
- ✅ $H \rightarrow W$
- ❌ $H \leftarrow W$

Sex should affect height, not vice versa
- ❌ $H \rightarrow S$
- ✅ $H \leftarrow S$

```{code-cell} ipython3
# Split height by the Sex Category


def plot_height_weight_distributions(data):
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    plt.sca(axs[0])

    for ii, label in enumerate(SEX):
        utils.plot_scatter(
            data[data.male == ii].height,
            data[data.male == ii].weight,
            color=f"C{ii}",
            label=label,
        )
    plt.xlabel("height (cm)")
    plt.ylabel("weight (km)")
    plt.legend()
    plt.title("height vs weight")

    for vv, var in enumerate(["height", "weight"]):
        plt.sca(axs[vv + 1])
        for ii in range(2):
            az.plot_dist(
                data.loc[data.male == ii, var].values,
                color=f"C{ii}",
                label=SEX[ii],
                bw=1,
                plot_kwargs=dict(linewidth=3, alpha=0.6),
            )
        plt.title(f"{var} split by sex")
        plt.xlabel("height (cm)")
        plt.legend()


plot_height_weight_distributions(ADULT_HOWELL)
```

Causal graph defines a set of functional relationships

$$
\begin{align*}
H &= f_H(S) \\
W &= f_W(H, S)
\end{align*}
$$

Could also include unobservable causal influences $T$ on $S$ (see below graph):

$$
\begin{align*}
H &= f_H(S, U) \\
W &= f_W(H, S, V) \\
S &= f_S(T)
\end{align*}
$$

> Note: we use $T$ as an unobserved variable, rather than $W$ to avoid replication in the lecture.

```{code-cell} ipython3
utils.draw_causal_graph(
    edge_list=[("H", "W"), ("S", "H"), ("S", "W"), ("V", "W"), ("U", "H"), ("T", "S")],
    node_props={
        "T": {"style": "dashed"},
        "U": {"style": "dashed"},
        "V": {"style": "dashed"},
        "unobserved": {"style": "dashed"},
    },
    graph_direction="LR",
)
```

### Synthetic People

```{code-cell} ipython3
utils.draw_causal_graph(
    edge_list=[("H", "W"), ("S", "H"), ("S", "W"), ("V", "W"), ("U", "H"), ("T", "S")],
    node_props={
        "T": {"style": "dashed", "color": "lightgray"},
        "U": {"style": "dashed", "color": "lightgray"},
        "V": {"style": "dashed", "color": "lightgray"},
        "unobserved": {"style": "dashed", "color": "lightgray"},
    },
    edge_props={
        ("T", "S"): {"color": "lightgray"},
        ("U", "H"): {"color": "lightgray"},
        ("V", "W"): {"color": "lightgray"},
    },
    graph_direction="LR",
)
```

```{code-cell} ipython3
def simulate_sex_height_weight(
    S: np.ndarray,
    beta: np.ndarray = np.array([1, 1]),
    alpha: np.ndarray = np.array([0, 0]),
    female_mean_height: float = 150,
    male_mean_height: float = 160,
) -> np.ndarray:
    """
    Generative model for the effect of Sex on height & weight

    S: np.array[int]
        The 0/1 indicator variable sex. 1 means 'male'
    beta: np.array[float]
        Lenght 2 slope coefficient for each sex
    alpha: np.array[float]
        Length 2 intercept for each sex
    """
    N = len(S)
    H = np.where(S, male_mean_height, female_mean_height) + stats.norm(0, 5).rvs(size=N)
    W = alpha[S] + beta[S] * H + stats.norm(0, 5).rvs(size=N)

    return pd.DataFrame({"height": H, "weight": W, "male": S})


synthetic_sex = stats.bernoulli(p=0.5).rvs(size=100).astype(int)
synthetic_people = simulate_sex_height_weight(S=synthetic_sex)
plot_height_weight_distributions(synthetic_people)
synthetic_people.head()
```

### Think scientifically first

Different causal questions require different statistical models:

- Question 1: What's the causal effect of $H$ on $W$?
- Question 2: What's the **Total** Causal effect of $S$ on $W$?
- Question 3: What's the **Direct** Causal effect of $S$ on $W$?

Answering the last two questions requires different statistical models, but both will need stratification by $S$

+++

### From estimand to estimate

+++

#### Causal effect of $H$ on $W$ (Q1)

```{code-cell} ipython3
utils.draw_causal_graph(
    edge_list=[("H", "W"), ("S", "W"), ("S", "H")],
    node_props={"H": {"color": "red"}, "W": {"color": "red"}},
    edge_props={
        ("H", "W"): {"color": "red"},
    },
    graph_direction="LR",
)
```

#### **Total** Causal effect of $S$ on $W$ (Q2)

```{code-cell} ipython3
utils.draw_causal_graph(
    edge_list=[("H", "W"), ("S", "W"), ("S", "H")],
    node_props={"S": {"color": "red"}, "W": {"color": "red"}},
    edge_props={
        ("S", "H"): {"color": "red"},
        ("H", "W"): {"color": "red"},
        ("S", "W"): {"color": "red"},
    },
    graph_direction="LR",
)
```

#### **Direct** Causal effect of $S$ on $W$ (Q3)

```{code-cell} ipython3
utils.draw_causal_graph(
    edge_list=[("H", "W"), ("S", "W"), ("S", "H")],
    node_props={"S": {"color": "red"}, "W": {"color": "red"}},
    edge_props={("S", "W"): {"color": "red"}},
    graph_direction="LR",
)
```

**Stratify by S**: recover a different estimate for each value that $S$ can take

## Drawing the Causal Owl 🦉


Implement Categories via **Indicator Variables**

- generalizes code: can extend to any number of categories
- better for prior specification
- facilitates multi-level model specification

For categories $C = [C_1, C_2, ... C_D]$

$$
\begin{align*}
\alpha &= [\alpha_1, \alpha_2, ... \alpha_D] \\
y_i &\sim \text{Normal}(\mu_i, \sigma) \\
\mu_i &= \alpha_{C[i]}
\end{align*}
$$

For sex $S\in \{M, F\}$, we can model sex-specific  weight weight $W$ as

$$
\begin{align*}
W_i &\sim \text{Normal}(\mu_i, \sigma) \\
\mu_i &= \alpha_{S[i]} \\
\alpha &= [\alpha_F, \alpha_M] \\
\alpha_j &\sim \text{Normal}(60, 10) \\
\sigma &\sim \text{Uniform}(0, 10)
\end{align*}
$$



+++

## Testing

+++

### Total Causal Effect of Sex on Weight

```{code-cell} ipython3
utils.draw_causal_graph(
    edge_list=[("H", "W"), ("S", "W"), ("S", "H")],
    node_props={"S": {"color": "blue"}, "W": {"color": "red"}, "stratified": {"color": "blue"}},
    edge_props={
        ("S", "H"): {"color": "red"},
        ("H", "W"): {"color": "red"},
        ("S", "W"): {"color": "red"},
    },
    graph_direction="LR",
)
```

```{code-cell} ipython3
np.random.seed(12345)
n_simulations = 100
simulated_females = simulate_sex_height_weight(
    S=np.zeros(n_simulations).astype(int), beta=np.array((0.5, 0.6))
)

simulated_males = simulate_sex_height_weight(
    S=np.ones(n_simulations).astype(int), beta=np.array((0.5, 0.6))
)

simulated_delta = simulated_males - simulated_females
mean_simualted_delta = simulated_delta.mean()
az.plot_dist(simulated_delta["weight"].values)
plt.axvline(
    mean_simualted_delta["weight"],
    linestyle="--",
    color="k",
    label="Mean difference" + f"={mean_simualted_delta['weight']:1.2f}",
)
plt.xlabel("M - F")
plt.legend()
plt.title("simulated difference");
```

### Fit total effect on the synthetic sample
Stratify by $S$

```{code-cell} ipython3
def fit_total_effect_model(data):

    SEX_ID, SEX = pd.factorize(["M" if s else "F" for s in data["male"].values])

    with pm.Model(coords={"SEX": SEX}) as model:
        # Data
        S = pm.MutableData("S", SEX_ID)

        # Priors
        sigma = pm.Uniform("sigma", 0, 10)
        alpha = pm.Normal("alpha", 60, 10, dims="SEX")

        # Likelihood
        mu = alpha[S]
        pm.Normal("W_obs", mu, sigma, observed=data["weight"])

        inference = pm.sample()

    return inference, model


# Concatentate simulations and code sex
simulated_people = pd.concat([simulated_females, simulated_males])
simulated_total_effect_inference, simulated_total_effect_model = fit_total_effect_model(
    simulated_people
)
```

```{code-cell} ipython3
simulated_summary = az.summary(simulated_total_effect_inference, var_names=["alpha", "sigma"])
simulated_delta = (simulated_summary.iloc[1] - simulated_summary.iloc[0])["mean"]
print(f"Delta in average sex-specific weight: {simulated_delta:1.2f}")

simulated_summary
```

```{code-cell} ipython3
# Plotting helper functions


def plot_model_posterior(inference, effect_type: str = "Total"):
    np.random.seed(123)
    sex = ["F", "M"]
    posterior = inference.posterior

    fig, axs = plt.subplots(2, 2, figsize=(7, 7))

    # Posterior mean
    plt.sca(axs[0][0])
    for ii, s in enumerate(sex):
        posterior_mean = posterior["alpha"].sel(SEX=s).mean(dim="chain")
        az.plot_dist(posterior_mean, color=f"C{ii}", label=s, plot_kwargs=dict(linewidth=3))

    plt.xlabel("posterior mean weight (kg)")
    plt.ylabel("density")
    plt.legend()
    plt.title("Posterior $\\alpha_S$")

    # Posterior Predictive
    plt.sca(axs[0][1])
    posterior_prediction_std = posterior["sigma"].mean(dim=["chain"])
    posterior_prediction = {}

    for ii, s in enumerate(sex):
        posterior_prediction_mean = posterior.sel(SEX=s)["alpha"].mean(dim=["chain"])
        posterior_prediction[s] = stats.norm.rvs(
            posterior_prediction_mean, posterior_prediction_std
        )
        az.plot_dist(
            posterior_prediction[s], color=f"C{ii}", label=s, plot_kwargs=dict(linewidth=3)
        )

    plt.xlabel("posterior predicted weight (kg)")
    plt.ylabel("density")
    plt.legend()
    plt.title("Posterior Predictive")

    # Plost Contrasts
    ## Posterior Contrast
    plt.sca(axs[1][0])
    posterior_contrast = posterior.sel(SEX="M")["alpha"] - posterior.sel(SEX="F")["alpha"]
    az.plot_dist(posterior_contrast, color="k", plot_kwargs=dict(linewidth=3))
    plt.xlabel("$\\alpha_M$ - $\\alpha_F$ posterior mean weight contrast")
    plt.ylabel("density")
    plt.title("Posterior Contrast")

    ## Posterior Predictive Contrast
    plt.sca(axs[1][1])
    posterior_predictive_contrast = posterior_prediction["M"] - posterior_prediction["F"]
    n_draws = len(posterior_predictive_contrast)
    kde_ax = az.plot_dist(
        posterior_predictive_contrast, color="k", bw=1, plot_kwargs=dict(linewidth=3)
    )

    # Shade underneath posterior predictive contrast
    kde_x, kde_y = kde_ax.get_lines()[0].get_data()

    # Proportion of PPD contrast below zero
    neg_idx = kde_x < 0
    neg_prob = 100 * np.sum(posterior_predictive_contrast < 0) / n_draws
    plt.fill_between(
        x=kde_x[neg_idx],
        y1=np.zeros(sum(neg_idx)),
        y2=kde_y[neg_idx],
        color="C0",
        label=f"{neg_prob:1.0f}%",
    )

    # Proportion of PPD contrast above zero (inclusive)
    pos_idx = kde_x >= 0
    pos_prob = 100 * np.sum(posterior_predictive_contrast >= 0) / n_draws
    plt.fill_between(
        x=kde_x[pos_idx],
        y1=np.zeros(sum(pos_idx)),
        y2=kde_y[pos_idx],
        color="C1",
        label=f"{pos_prob:1.0f}%",
    )

    plt.xlabel("(M - F)\nposterior prediction contrast")
    plt.ylabel("density")
    plt.legend()
    plt.title("Posterior\nPredictive Contrast")
    plt.suptitle(f"{effect_type} Causal Effect of Sex on Weight", fontsize=18)


def plot_posterior_lines(data, inference, centered=False):
    plt.subplots(figsize=(6, 6))

    min_height = data.height.min()
    max_height = data.height.max()
    xs = np.linspace(min_height, max_height, 10)
    for ii, s in enumerate(["F", "M"]):
        sex_idx = data.male == ii
        utils.plot_scatter(
            xs=data[sex_idx].height, ys=data[sex_idx].weight, color=f"C{ii}", label=s
        )

        posterior_mean = inference.posterior.sel(SEX=s).mean(dim=("chain", "draw"))
        posterior_mean_alpha = posterior_mean["alpha"].values
        posterior_mean_beta = getattr(posterior_mean, "beta", pd.Series([0])).values

        if centered:
            pred_x = xs - data.height.mean()
        else:
            pred_x = xs

        ys = posterior_mean_alpha + posterior_mean_beta * pred_x
        utils.plot_line(xs, ys, label=None, color=f"C{ii}")

    # Model fit to both sexes simultaneously
    global_model = smf.ols("weight ~ height", data=data).fit()
    ys = global_model.params.Intercept + global_model.params.height * xs
    utils.plot_line(xs, ys, color="k", label="Unstratified\nModel")

    plt.axvline(
        data["height"].mean(), label="Average H", linewidth=0.5, linestyle="--", color="black"
    )
    plt.axhline(
        data["weight"].mean(), label="Average W", linewidth=1, linestyle="--", color="black"
    )
    plt.legend()
    plt.xlabel("height (cm), H")
    plt.ylabel("weight (kg), W");
```

```{code-cell} ipython3
plot_posterior_lines(simulated_people, simulated_total_effect_inference, centered=True)
```

```{code-cell} ipython3
plot_model_posterior(simulated_total_effect_inference)
```

## Analyze real sample

```{code-cell} ipython3
adult_howell_total_effect_inference, adult_howell_total_effect_model = fit_total_effect_model(
    ADULT_HOWELL
)
```

```{code-cell} ipython3
adult_howell_total_effect_summary = az.summary(
    adult_howell_total_effect_inference, var_names=["alpha"]
)
adult_howell_total_effect_delta = (
    adult_howell_total_effect_summary.iloc[1] - adult_howell_total_effect_summary.iloc[0]
)["mean"]
print(f"Delta in average sex-specific weight: {adult_howell_total_effect_delta:1.2f}")

adult_howell_summary = az.summary(adult_howell_total_effect_inference)
adult_howell_summary
```

### Always be contrasting
- need compare the **contrast** between categories 
- **never valid to calculate overlap** in distributions
  - this means **no comparing confidence intervals for p-values**
- Compute the difference of distributions -- the **contrast distribution**

```{code-cell} ipython3
plot_model_posterior(adult_howell_total_effect_inference)
```

```{code-cell} ipython3
plot_posterior_lines(ADULT_HOWELL, adult_howell_total_effect_inference, True)
```

### **Direct** causal effect of $S$ on $W$?
We need another model/estimator for this estimand. We stratify by both $S$ and $H$; by $H$ to block the path of $S$ though $H$

```{code-cell} ipython3
utils.draw_causal_graph(
    edge_list=[("H", "W"), ("S", "W"), ("S", "H")],
    node_props={
        "S": {"color": "blue"},
        "W": {"color": "red"},
        "H": {"color": "blue"},
        "stratified": {"color": "blue"},
    },
    edge_props={("S", "W"): {"color": "red"}},
    graph_direction="LR",
)
```

$$
\begin{align*}
W_i &\sim \text{Normal}(\mu_i, \sigma) \\
\mu_i &= \alpha_{S[i]} + \beta_{S[i]} (H_i - \bar H)
\end{align*}
$$

Where we've **centered the height**, meaning that 
- $\beta$ scales the difference of $H_i$ from the average height
- $\alpha$ is the weight when a person is the average height
- Global model fit to all data lies at the intersection of global average height and weight

+++

### Simulate some more people

```{code-cell} ipython3
ALPHA = 0.9
np.random.seed(1234)
n_synthetic_people = 200

synthetic_sex = stats.bernoulli.rvs(p=0.5, size=n_synthetic_people)
synthetic_people = simulate_sex_height_weight(
    S=synthetic_sex,
    beta=np.array([0.5, 0.5]),  # Same relationship between height & weight
    alpha=np.array([0.0, 10]),  # 10kg "boost for Males"
)
```

### Analyze the synthetic people

```{code-cell} ipython3
def fit_direct_effect_weight_model(data):

    SEX_ID, SEX = pd.factorize(["M" if s else "F" for s in data["male"].values])

    with pm.Model(coords={"SEX": SEX}) as model:
        # Data
        S = pm.MutableData("S", SEX_ID, dims="obs_ids")
        H = pm.MutableData("H", data["height"].values, dims="obs_ids")
        Hbar = pm.MutableData("Hbar", data["height"].mean())

        # Priors
        sigma = pm.Uniform("sigma", 0, 10)
        alpha = pm.Normal("alpha", 60, 10, dims="SEX")
        beta = pm.Uniform("beta", 0, 1, dims="SEX")  # postive slopes only

        # Likelihood
        mu = alpha[S] + beta[S] * (H - Hbar)
        pm.Normal("W_obs", mu, sigma, observed=data["weight"].values, dims="obs_ids")

        inference = pm.sample()

    return inference, model
```

```{code-cell} ipython3
direct_effect_simulated_inference, direct_effect_simulated_model = fit_direct_effect_weight_model(
    simulated_people
)
```

```{code-cell} ipython3
plot_posterior_lines(simulated_people, direct_effect_simulated_inference, centered=True)
```

- **Indirect effect**: M & F have specific slopes - in this simulation, they are the same slope, thus parallel lines
- **Direct effect**: There will be a delta, no matter the slope. -- in this simulation $S$=M are always 10kg heavier, thus blue is always above red

```{code-cell} ipython3
plot_model_posterior(direct_effect_simulated_inference, "Direct")
```

## Analyze the real sample

```{code-cell} ipython3
direct_effect_howell_inference, direct_effect_howell_model = fit_direct_effect_weight_model(
    ADULT_HOWELL
)
```

```{code-cell} ipython3
plot_posterior_lines(ADULT_HOWELL, direct_effect_howell_inference, True)
```

#### Contrasts

```{code-cell} ipython3
plot_model_posterior(direct_effect_howell_inference, "Direct")
```

## Contrast at each height

```{code-cell} ipython3
def plot_heightwise_contrast(model, inference):
    heights = np.linspace(130, 190, 100)
    ppds = {}
    for ii, s in enumerate(["F", "M"]):
        with model:
            pm.set_data(
                {"S": np.ones_like(heights).astype(int) * ii, "H": heights, "Hbar": heights.mean()}
            )
            ppds[s] = pm.sample_posterior_predictive(
                inference, extend_inferencedata=False
            ).posterior_predictive["W_obs"]

    ppd_contrast = ppds["M"] - ppds["F"]

    # Plot contours
    for prob in [0.5, 0.75, 0.95, 0.99]:
        az.plot_hdi(heights, ppd_contrast, hdi_prob=prob, color="gray")

    plt.axhline(0, linestyle="--", color="k")
    plt.xlabel("height, H (cm)")
    plt.ylabel("weight W contrast (M-F)")
    plt.xlim([130, 190])
```

```{code-cell} ipython3
plot_heightwise_contrast(direct_effect_howell_model, direct_effect_howell_inference)
```

When stratifying by Height, we see that Sex has very little, if any causal effect on height. i.e. **a lion's share of the causal effect on weight comes via height.**

```{code-cell} ipython3
# Try on the simulated data
plot_heightwise_contrast(direct_effect_simulated_model, direct_effect_simulated_inference)
```

we can see that in the simulated data, men are consistently heavier than women, which is aligned with the simulation

+++

# Curves from lines
Not all relationships are linear
- e.g. in the Howell dataset, we can see that if we include all ages, the relationship between Height and Weight is nonlinear
- linear models can fit curves
- **still not a mechanistic model**

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(4, 4))
HOWELL.plot(x="height", y="weight", kind="scatter", ax=ax);
```

## Polynomial Linear Models

$$\mu_i = \alpha + \sum_i^D \beta_i x^D $$

### Issues with Polynomial Models
- symmetric -- strange edge anomolies
- global models, so no local interpolation
- easy to overfit by increasing number of terms

### Quadratic Polynomial Model

$$\mu_i = \alpha + \beta_2 x + \beta_2 x^2 $$

```{code-cell} ipython3
def plot_polynomial_sample(degree, random_seed=123):
    np.random.seed(random_seed)
    xs = np.linspace(-1, 1, 100)
    ys = 0
    for d in range(1, degree + 1):
        beta_d = np.random.randn()
        ys += beta_d * xs**d

    utils.plot_line(xs, ys, color=f"C{degree}", label=f"Degree: {degree}")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("$\\mu$")


for degree in [1, 2, 3, 4]:
    plot_polynomial_sample(degree)
```

#### Simulate Bayesian Updating for Quadratic Polynomial Model
For the following simulation, we'll use a custom utility function `utils.simulate_2_parameter_bayesian_learning_grid_approximation` for simulating general Bayeisan posterior update simulation. Here's the API for that function (for more details see `utils.py`)

```{code-cell} ipython3
help(utils.simulate_2_parameter_bayesian_learning_grid_approximation)
```

##### Functions required for Bayesian learning simulation

```{code-cell} ipython3
def quadratic_polynomial_model(x, beta_1, beta_2):
    return beta_1 * x + beta_2 * x**2


def quadratic_polynomial_regression_posterior(
    x_obs, y_obs, beta_1_grid, beta_2_grid, likelihood_prior_std=1.0
):

    beta_1_grid = beta_1_grid.ravel()
    beta_2_grid = beta_2_grid.ravel()

    log_prior_beta_1 = stats.norm(0, 1).logpdf(beta_1_grid)
    log_prior_beta_2 = stats.norm(0, 1).logpdf(beta_2_grid)

    log_likelihood = np.array(
        [
            stats.norm(b1 * x_obs + b2 * x_obs**2, likelihood_prior_std).logpdf(y_obs)
            for b1, b2 in zip(beta_1_grid, beta_2_grid)
        ]
    ).sum(axis=1)

    log_posterior = log_likelihood + log_prior_beta_1 + log_prior_beta_2

    return np.exp(log_posterior - log_posterior.max())
```

##### Run the simulation

```{code-cell} ipython3
np.random.seed(123)
RESOLUTION = 100
N_DATA_POINTS = 64
BETA_1 = 2
BETA_2 = -2
INTERCEPT = 0

# Generate observations
x = stats.norm().rvs(size=N_DATA_POINTS)
y = INTERCEPT + BETA_1 * x + BETA_2 * x**2 + stats.norm.rvs(size=N_DATA_POINTS) * 0.5

beta_1_grid = np.linspace(-3, 3, RESOLUTION)
beta_2_grid = np.linspace(-3, 3, RESOLUTION)

# Vary the sample size to show how the posterior adapts to more and more data
for n_samples in [0, 2, 4, 8, 16, 32, 64]:

    utils.simulate_2_parameter_bayesian_learning_grid_approximation(
        x_obs=x[:n_samples],
        y_obs=y[:n_samples],
        param_a_grid=beta_1_grid,
        param_b_grid=beta_2_grid,
        true_param_a=BETA_1,
        true_param_b=BETA_2,
        model_func=quadratic_polynomial_model,
        posterior_func=quadratic_polynomial_regression_posterior,
        param_labels=["$\\beta_1$", "$\\beta_2$"],
        data_range_x=(-2, 3),
        data_range_y=(-3, 3),
    )
```

### Fitting N-th Order Polynomials to Height / Width Data

```{code-cell} ipython3
def fit_nth_order_polynomial(data, n=3):
    with pm.Model() as model:
        # Data
        H_std = pm.MutableData("H", utils.standardize(data.height.values), dims="obs_ids")

        # Priors
        sigma = pm.Uniform("sigma", 0, 10)
        alpha = pm.Normal("alpha", 0, 60)
        betas = []
        for ii in range(n):
            betas.append(pm.Normal(f"beta_{ii+1}", 0, 5))

        # Likelihood
        mu = alpha
        for ii, beta in enumerate(betas):
            mu += beta * H_std ** (ii + 1)

        mu = pm.Deterministic("mu", mu)

        pm.Normal("W_obs", mu, sigma, observed=data.weight.values, dims="obs_ids")

        inference = pm.sample(target_accept=0.95)

    return model, inference


polynomial_models = {}
for order in [2, 4, 6]:
    polynomial_models[order] = fit_nth_order_polynomial(HOWELL, n=order)
```

```{code-cell} ipython3
def plot_polynomial_model_posterior_predictive(model, inference, data, order):

    # Sample the posterior predictive for regions outside of the training data
    prediction_heights = np.linspace(30, 200, 100)
    with model:
        std_heights = (prediction_heights - data.height.mean()) / data.height.std()
        pm.set_data({"H": std_heights})
        ppd = pm.sample_posterior_predictive(inference, extend_inferencedata=False)

    plt.subplots(figsize=(4, 4))
    plt.scatter(data.height, data.weight)
    az.plot_hdi(
        prediction_heights,
        ppd.posterior_predictive["W_obs"],
        color="k",
        fill_kwargs=dict(alpha=0.1),
    )

    # Hack: use .5% HDI as proxy for posterior predictive mean
    az.plot_hdi(
        prediction_heights,
        ppd.posterior_predictive["W_obs"],
        hdi_prob=0.005,
        color="k",
        fill_kwargs=dict(alpha=1),
    )
    terms = "+".join([f"\\beta_{o} H_i^{o}" for o in range(1, order + 1)])
    plt.title(f"$\mu_i = \\alpha + {terms}$")


for order in [2, 4, 6]:
    model, inference = polynomial_models[order]
    plot_polynomial_model_posterior_predictive(model, inference, HOWELL, order)
```

### Thinking vs Fitting
- Linear models can fit anything (geocentric)
- Better off to use domain expertise to build more biologically plausible model e.g.

$$
\begin{align*}
\log W_i = \text{Normal}(\mu_i, \sigma) \\
\mu_i = \alpha + \beta (H - \bar H)
\end{align*}
$$

+++

## Splines
- "Wiggles" built from locally-fit smooth functions
- good alternative when you have little domain knowledge of the problem

$$
\mu_i = \alpha_0 + \alpha_1 B_1 + \alpha_2 B_2 + ... \alpha_S B_K
$$

where $B$ is a set of $K$ local kernel functions

+++

## Example: Cherry Blossom Blooms

```{code-cell} ipython3
BLOSSOMS = utils.load_data("cherry_blossoms")
BLOSSOMS.dropna(subset=["doy"], inplace=True)
plt.subplots(figsize=(10, 3))
plt.scatter(x=BLOSSOMS.year, y=BLOSSOMS.doy)
plt.xlabel("year")
plt.ylabel("day of first blossom")
BLOSSOMS.head()
```

```{code-cell} ipython3
from patsy import dmatrix


def generate_spline_basis(data, xdim="year", degree=2, n_bases=10):
    n_knots = n_bases - 1
    knots = np.quantile(data[xdim], np.linspace(0, 1, n_knots))
    return dmatrix(
        f"bs({xdim}, knots=knots, degree={degree}, include_intercept=True) - 1",
        {xdim: data[xdim], "knots": knots[1:-1]},
    )


# 4 spline basis for demo
demo_data = pd.DataFrame({"x": np.arange(100)})
demo_basis = generate_spline_basis(demo_data, "x", n_bases=4)

fig, axs = plt.subplots(2, 1, figsize=(10, 8))
plt.sca(axs[0])
for bi in range(demo_basis.shape[1]):
    plt.plot(demo_data.x, demo_basis[:, bi], color=f"C{bi}", label=f"Basis{bi+1}")
plt.legend()
plt.title("Demo Spline Basis")

# Arbitrarily-set weights for demo
basis_weights = [1, 2, -1, 0]

plt.sca(axs[1])
resulting_curve = np.zeros_like(demo_data.x)
for bi in range(demo_basis.shape[1]):
    weighted_basis = demo_basis[:, bi] * basis_weights[bi]
    resulting_curve = resulting_curve + weighted_basis
    plt.plot(
        demo_data.x, weighted_basis, color=f"C{bi}", label=f"{basis_weights[bi]} x Basis {bi+1}"
    )
plt.plot(demo_data.x, resulting_curve, label="Sum", color="k", linewidth=4)
plt.xlabel("x")
plt.legend()
plt.title("Sum of Weighted Bases");
```

```{code-cell} ipython3
# 10 spline basis for modeling blossoms data
blossom_basis = generate_spline_basis(BLOSSOMS)

fig, ax = plt.subplots(figsize=(10, 3))

for bi in range(blossom_basis.shape[1]):
    ax.plot(BLOSSOMS.year, blossom_basis[:, bi], color=f"C{bi}", label=f"Basis{bi+1}")
plt.legend()
plt.title("Basis functions, $B$ for the Cherry Blossoms Dataset");
```

#### Draw some samples from the prior

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 3))
n_samples = 5
spline_prior_sigma = 10
spline_prior = stats.norm(0, spline_prior_sigma)
for s in range(n_samples):
    sample = 0
    for bi in range(blossom_basis.shape[1]):
        sample += spline_prior.rvs() * blossom_basis[:, bi]
    ax.plot(BLOSSOMS.year, sample)
plt.title("Prior,  $\\alpha \\sim Normal(0, 10)$");
```

```{code-cell} ipython3
def fit_spline_model(data, xdim, ydim, n_bases=10):
    basis_set = generate_spline_basis(data, xdim, n_bases=n_bases).base
    with pm.Model() as spline_model:

        # Priors
        sigma = pm.Exponential("sigma", 1)
        alpha = pm.Normal("alpha", data[ydim].mean(), data[ydim].std())
        beta = pm.Normal("beta", 0, 25, shape=n_bases)

        # Likelihood
        mu = pm.Deterministic("mu", alpha + pm.math.dot(basis_set, beta.T))
        pm.Normal("ydim_obs", mu, sigma, observed=data[ydim].values)

        spline_inference = pm.sample(target_accept=0.95)

    _, ax = plt.subplots(figsize=(10, 3))
    plt.scatter(x=data[xdim], y=data[ydim])
    az.plot_hdi(
        data[xdim],
        spline_inference.posterior["mu"],
        color="k",
        hdi_prob=0.89,
        fill_kwargs=dict(alpha=0.3, label="Posterior Mean"),
    )
    plt.legend(loc="lower right")
    plt.xlabel(f"{xdim}")
    plt.ylabel(f"{ydim}")

    return spline_model, spline_inference, basis_set
```

### Cherry Blossoms Model

```{code-cell} ipython3
blossom_model, blossom_inference, blossom_basis = fit_spline_model(
    BLOSSOMS, "year", "doy", n_bases=20
)
```

```{code-cell} ipython3
summary = az.summary(blossom_inference, var_names=["alpha", "beta"])

beta_spline_mean = blossom_inference.posterior["beta"].mean(dim=("chain", "draw")).values
resulting_fit = np.zeros_like(BLOSSOMS.year)
for bi, beta in enumerate(beta_spline_mean):
    weighted_basis = beta * blossom_basis[:, bi]
    plt.plot(BLOSSOMS.year, weighted_basis, color=f"C{bi}")
    resulting_fit = resulting_fit + weighted_basis
plt.plot(
    BLOSSOMS.year,
    resulting_fit,
    label="Resulting Fit (excluding intercept term)",
    color="k",
    linewidth=4,
)
plt.legend()
plt.title("weighted spline bases\nfit to Cherry Blossoms dataset")

summary
```

### Return to Howell Dataset: use splines model for Height as a function of Age

```{code-cell} ipython3
# Fit spline model to Howell height data
fit_spline_model(HOWELL, "age", "height", n_bases=10);
```

### Weight as a function of age

```{code-cell} ipython3
# While we're at it, let's fit spline model to Howell weight data
fit_spline_model(HOWELL, "age", "weight", n_bases=10);
```

```{code-cell} ipython3
# ...how about height as a function of weight
fit_spline_model(HOWELL, "height", "weight", n_bases=10);
```

# BONUS: Full Luxury Bayes
- Approach: Program the whole generative shebang into a single model
- Includes multiple submodels (i.e. multiple likelihoods)

### Why would we do this?
- Model the system in aggregate
- Can run simulations (interventions) from full generative model to look at causal estimates.

```{code-cell} ipython3
utils.draw_causal_graph(edge_list=[("H", "W"), ("S", "H"), ("S", "W")], graph_direction="LR")
```

```{code-cell} ipython3
SEX_ID, SEX = pd.factorize(["M" if s else "F" for s in ADULT_HOWELL["male"].values])
with pm.Model(coords={"SEX": SEX}) as flb_model:

    # Data
    S = pm.MutableData("S", SEX_ID)
    H = pm.MutableData("H", ADULT_HOWELL.height.values)
    Hbar = pm.MutableData("Hbar", ADULT_HOWELL.height.mean())

    # Height Model
    ## Height priors
    tau = pm.Uniform("tau", 0, 10)
    h = pm.Normal("h", 160, 10, dims="SEX")
    nu = h[S]
    ## Height likelihood
    pm.Normal("H_obs", nu, tau, observed=ADULT_HOWELL.height.values)

    # Weight Model
    ## Weight priors
    alpha = pm.Normal("alpha", 60, 10, dims="SEX")
    beta = pm.Uniform("beta", 0, 1, dims="SEX")
    sigma = pm.Uniform("sigma", 0, 10)
    mu = alpha[S] + beta[S] * (H - Hbar)
    ## Weight likelihood
    pm.Normal("W_obs", mu, sigma, observed=ADULT_HOWELL.weight.values)

    flb_inference = pm.sample()

pm.model_to_graphviz(flb_model)
```

```{code-cell} ipython3
flb_summary = az.summary(flb_inference)
flb_summary
```

## Simulate interventions with `do` operator

```{code-cell} ipython3
from pymc.model.transform.conditioning import do
```

```{code-cell} ipython3
def plot_posterior_mean_contrast(contrast_type="weight"):
    Hbar = ADULT_HOWELL.height.mean()
    means = az.summary(flb_inference)["mean"]

    H_F = stats.norm(means["h[F]"], means["tau"]).rvs(1000)
    H_M = stats.norm(means["h[M]"], means["tau"]).rvs(1000)

    W_F = stats.norm(means["beta[F]"] * (H_F - Hbar), means["sigma"]).rvs(1000)
    W_M = stats.norm(means["beta[M]"] * (H_M - Hbar), means["sigma"]).rvs(1000)
    contrast = H_M - H_F if contrast_type == "height" else W_M - W_F

    az.plot_dist(contrast, color="k")
    plt.xlabel(f"Posterior mean {contrast_type} contrast")


def plot_causal_intervention_contrast(contrast_type, intervention_type="pymc_do_operator"):
    N = len(ADULT_HOWELL)
    if intervention_type == "pymc_do_operator":
        male_counterfactual_data = {"S": np.ones(N, dtype="int32")}
        female_counterfactual_data = {"S": np.zeros(N, dtype="int32")}

        if contrast_type == "weight":
            contrast_variable = "W_obs"
            mean_heights = ADULT_HOWELL.groupby("male")["height"].mean()
            male_counterfactual_data.update({"H": np.ones(N) * mean_heights[1]})
            female_counterfactual_data.update({"H": np.ones(N) * mean_heights[0]})
        else:
            contrast_variable = "H_obs"

        # p(Y| do(S=1))
        male_intervention_model = do(flb_model, male_counterfactual_data)

        # p(Y | do(S=0))
        female_intervention_model = do(flb_model, female_counterfactual_data)

        male_intervention_inference = pm.sample_posterior_predictive(
            flb_inference, model=male_intervention_model, predictions=True
        )
        female_intervention_inference = pm.sample_posterior_predictive(
            flb_inference, model=female_intervention_model, predictions=True
        )
        intervention_contrast = (
            male_intervention_inference.predictions - female_intervention_inference.predictions
        )
        contrast = intervention_contrast[contrast_variable]
    else:
        # Intervention by hand, like outlined in lecture
        Hbar = ADULT_HOWELL.height.mean()

        F_posterior = flb_inference.posterior.sel(SEX="F")
        M_posterior = flb_inference.posterior.sel(SEX="M")

        H_F = stats.norm.rvs(F_posterior["h"], F_posterior["tau"])
        H_M = stats.norm.rvs(M_posterior["h"], F_posterior["tau"])

        W_F = stats.norm.rvs(F_posterior["beta"] * (H_F - Hbar), F_posterior["sigma"])
        W_M = stats.norm.rvs(M_posterior["beta"] * (H_M - Hbar), M_posterior["sigma"])

        contrast = H_M - H_F if contrast_type == "height" else W_M - W_F

    pos_prob = 100 * np.sum(contrast >= 0) / np.product(contrast.shape)
    neg_prob = 100 - pos_prob

    kde_ax = az.plot_dist(contrast, color="k", plot_kwargs=dict(linewidth=3), bw=0.5)

    # Shade underneath posterior predictive contrast
    kde_x, kde_y = kde_ax.get_lines()[0].get_data()

    # Proportion of PPD contrast below zero
    neg_idx = kde_x < 0
    plt.fill_between(
        x=kde_x[neg_idx],
        y1=np.zeros(sum(neg_idx)),
        y2=kde_y[neg_idx],
        color="C0",
        label=f"{neg_prob:1.0f}%",
    )

    pos_idx = kde_x >= 0
    plt.fill_between(
        x=kde_x[pos_idx],
        y1=np.zeros(sum(pos_idx)),
        y2=kde_y[pos_idx],
        color="C1",
        label=f"{pos_prob:1.0f}%",
    )

    plt.axvline(0, color="k")
    plt.legend()
    plt.xlabel(f"{contrast_type} counterfactual contrast")


def plot_flb_contrasts(
    contrast_type="weight", intervention_type="pymc_do_operator", figsize=(8, 4)
):
    _, axs = plt.subplots(1, 2, figsize=figsize)
    plt.sca(axs[0])
    plot_posterior_mean_contrast(contrast_type)

    plt.sca(axs[1])
    plot_causal_intervention_contrast(contrast_type, intervention_type)
```

```{code-cell} ipython3
plot_flb_contrasts("weight")
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
