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

(lecture_16)=
# Gaussian Processes
:::{post} Jan 7, 2024
:tags: statistical rethinking, bayesian inference, gaussian processes
:category: intermediate
:author: Dustin Stansbury
:::

This notebook is part of the PyMC port of the [Statistical Rethinking 2023](https://github.com/rmcelreath/stat_rethinking_2023) lecture series by Richard McElreath.

[Video - Lecture 16 - Gaussian Processes](https://youtu.be/PIuqxOXHLBw)# [Lecture 16 - Gaussian Processes](https://youtu.be/Y2ZLt4iOrXU?si=NN2oor2yEz3YSoLY)

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

# Returning to Oceanic Technology Dataset

We'll now deal with some of the confounds we glossed over, now that we have some more advanced statistical tools

```{code-cell} ipython3
KLINE = utils.load_data("Kline2")
KLINE
```

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
utils.draw_causal_graph(
    edge_list=[("P", "T"), ("C", "T"), ("U", "T"), ("U", "P")],
    node_props={"U": {"style": "dashed"}, "unobserved": {"style": "dashed"}},
)
```

- Total tools in society $T$ (outcome)
- Populuation $P$ (treatment)
- Contact level $C$
- Unobserved confounds $U$, e.g.
  - materials
  - weather

There should be **spatial covariation**: 
- islands closer to one another should have similar unobserved confounds
- islands should share historical context
- we don't have to know the exact dymanics of the contexts, we can handle statistically at a macro scale


## Review of Techonological Innovation/Loss model

- Derived from steady state of difference equation $\Delta T = \alpha P^ \beta - \gamma T$
  - $\alpha$ is rate of innovation
  - $P$ is population
  - $\beta$ is elasticity rate (diminishing returns)
  - $\gamma$ is loss (e.g. forgetting) rate over time
- Provides the expected number of tools over the long-run:
  - $\hat T = \frac{\alpha P ^ \beta}{\gamma}$
  - use $\lambda= \hat T$ as the expected rate parameter for a $\text{Poisson}(\lambda)$ distribution
- **How can we encode spatial covariance in this model?**

## Let's start of with an easier model that ignores population

$$
\begin{align*}
    T_i &\sim \text{Poisson}(\lambda_i) \\
    \lambda_i &= \bar \alpha + \alpha_{S[i]} \\
    \begin{pmatrix}
        \alpha_1 \\
        \alpha_2 \\
        \vdots \\
        \alpha_{10}
    \end{pmatrix} &= \text{MVNormal}
    \left( 
        \begin{bmatrix}
            0 \\
            0 \\
            \vdots \\
            0
        \end{bmatrix}, \textbf{K}
    \right)
\end{align*}
$$

- $\alpha_{S[i]}$ are the variable intercepts for each society
  - These will have covariation based on their spatial distance from one another
  - closer societies have similar offsets
- $\mathbf{K}$ the covariance amongst societies
  - models covariance as a function distance
  - though $\mathbf{K}$ has many parameters (45 covariances), we should be able to leverage spatial regularity to estimate far fewer effective parameters. This is good b.c. we only have 10 data points 😅

+++

# Gaussian Processes (in the abstract)

- Uses a **kernel function** to generate covariance matrices
  - Requires far fewer parameters than the covariance matrix entries
  - Can generate covariance matrices of any dimenions (i.e. "infinite dimensional generalization of the `MNNormal`")
  - Can generate a prediction for any point


- Kernel function calculates the covariance of two points based on metric of comparison, e.g.
  - spatial distance
  - difference in time
  - difference in age

+++

## Different Kernel Functions

```{code-cell} ipython3
def plot_kernel_function(
    kernel_function, max_distance=1, resolution=100, label=None, ax=None, **line_kwargs
):
    """Helper to plot a kernel function"""
    X = np.linspace(0, max_distance, resolution)[:, None]
    covariance = kernel_function(X, X)
    distances = np.linspace(0, max_distance, resolution)
    if ax is not None:
        plt.sca(ax)
    utils.plot_line(distances, covariance[0, :], label=label, **line_kwargs)
    plt.xlim([0, max_distance])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("|X1-X2|")
    plt.ylabel("covariance")
    if label is not None:
        plt.legend()
```

### The Quadratic (L2) Kernel Function
- aka RGB
- aka Gaussian Kernel

$$
K(x_i, x_j) = \eta^2 \exp \left(- \frac{(x_i - x_j)^2}{\sigma^2} \right)
$$

```{code-cell} ipython3
def quadratic_distance_kernel(X0, X1, eta=1, sigma=0.5):
    # Use linear algebra identity: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 * x^T * y
    X0_norm = np.sum(X0**2, axis=-1)
    X1_norm = np.sum(X1**2, axis=-1)
    squared_distances = X0_norm[:, None] + X1_norm[None, :] - 2 * X0 @ X1.T
    rho = 1 / sigma**2
    return eta**2 * np.exp(-rho * squared_distances)


plot_kernel_function(quadratic_distance_kernel)
```

### Ornstein-Uhlenbeck Kernel

$$
\begin{align*}
K(x_i, x_j) &= \eta^2 \exp \left(- \frac{|x_i - x_j|}{\sigma} \right) \\
\sigma &= \frac{1}{\rho}
\end{align*}
$$

```{code-cell} ipython3
def ornstein_uhlenbeck_kernel(X0, X1, eta_squared=1, rho=4):
    distances = np.abs(X1[None, :] - X0[:, None])
    return eta_squared * np.exp(-rho * distances)


plot_kernel_function(ornstein_uhlenbeck_kernel)
```

### Periodic Kernel

$$
K(x_i, x_j) = \eta^2 \exp \left(- \frac{\sin^2((x_i - x_j) / p)}{\sigma^2} \right)
$$

with additional periodicity parameter, $p$

```{code-cell} ipython3
def periodic_kernel(X0, X1, eta=1, sigma=1, periodicity=0.5):
    distances = np.sin((X1[None, :] - X0[:, None]) * periodicity) ** 2
    rho = 2 / sigma**2
    return eta**2 * np.exp(-rho * distances)


plot_kernel_function(periodic_kernel, max_distance=6)
```

### Gaussian Process Prior

Using the Gaussian/L2 Kernel Function to generate our covariance $\mathbf{K}$, we can adjust a few parameters

- Varying $\sigma$ controls the bandwidth of the kernel function
  - smaller $\sigma$ makes covariance fall off quickly with space
  - larger $\sigma$ allow covariance to extend larger distances

- Varying $\eta$ controls the maximum degree of covariance

#### 1-D Examples
Below we draw functions from a 1-D Guassian process, varying either $\sigma$ or $\eta$ to demonstrate the effect of the parameters on the spatial correlation of the samples.

```{code-cell} ipython3
# Helper functions
def plot_gaussian_process(
    X, samples=None, mean=None, cov=None, X_obs=None, Y_obs=None, uncertainty_prob=0.89
):
    X = X.ravel()

    # Plot GP samples
    for ii, sample in enumerate(samples):
        label = "GP samples" if not ii else None
        utils.plot_line(X, sample, color=f"C{ii}", linewidth=1, label=label)

    # Add GP mean, if provided
    if mean is not None:
        mean = mean.ravel()
        utils.plot_line(X, mean, color="k", label="GP mean")

        # Add uncertainty around mean; requires covariance matrix
        if cov is not None:
            z = stats.norm.ppf(1 - (1 - uncertainty_prob) / 2)
            uncertainty = z * np.sqrt(np.diag(cov))
            plt.fill_between(
                X,
                mean + uncertainty,
                mean - uncertainty,
                alpha=0.1,
                color="gray",
                zorder=1,
                label="GP uncertainty",
            )

    # Add any training data points, if provided
    if X_obs is not None:
        utils.plot_scatter(X_obs, Y_obs, color="k", label="observations", zorder=100, alpha=1)

    plt.xlim([X.min(), X.max()])
    plt.ylim([-5, 5])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()


def plot_gaussian_process_prior(kernel_function, n_samples=3, figsize=(10, 5), resolution=100):
    X = np.linspace(-5, 5, resolution)[:, None]

    prior = gaussian_process_prior(X, kernel_function)
    samples = prior.rvs(n_samples)

    _, axs = plt.subplots(1, 2, figsize=figsize)
    plt.sca(axs[0])
    plot_gaussian_process(X, samples=samples)

    plt.sca(axs[1])
    plot_kernel_function(kernel_function, color="k")
    plt.title(f"kernel function")
    return axs


def gaussian_process_prior(X_pred, kernel_function):
    """Initializes a Gaussian Process prior distribution for provided Kernel function"""
    mean = np.zeros(X_pred.shape).ravel()
    cov = kernel_function(X_pred, X_pred)
    return stats.multivariate_normal(mean=mean, cov=cov, allow_singular=True)
```

#### Varying $\sigma$

```{code-cell} ipython3
from functools import partial

eta = 1
for sigma in [0.25, 0.5, 1]:
    kernel_function = partial(quadratic_distance_kernel, eta=eta, sigma=sigma)
    axs = plot_gaussian_process_prior(kernel_function, n_samples=5)
    axs[0].set_title(f"prior: $\\eta$={eta}; $\\sigma=${sigma}")
```

#### Varying $\eta$

```{code-cell} ipython3
sigma = 0.5
for eta in [0.25, 0.75, 1]:
    kernel_function = partial(quadratic_distance_kernel, eta=eta, sigma=sigma)
    axs = plot_gaussian_process_prior(kernel_function, n_samples=5)
    axs[0].set_title(f"prior: $\\eta$={eta}; $\\sigma=${sigma}")
```

### Gaussian Process Posterior

As the model observes data, it adjusts the posterior to account for that data while also incorporating the smoothness constraints of the prior.

```{code-cell} ipython3
def gaussian_process_posterior(
    X_obs, Y_obs, X_pred, kernel_function, sigma_y=1e-6, smoothing_factor=1e-6
):
    """Initializes a Gaussian Process Posterior distribution"""

    # Observation covariance
    K_obs = kernel_function(X_obs, X_obs)
    K_obs_noise = sigma_y**2 * np.eye(len(X_obs))
    K_obs += K_obs_noise
    K_obs_inv = np.linalg.inv(K_obs)

    # Prediction grid covariance
    K_pred = kernel_function(X_pred, X_pred)
    K_pred_smooth = smoothing_factor * np.eye(len(X_pred))
    K_pred += K_pred_smooth

    # Covariance between observations and prediction grid
    K_obs_pred = kernel_function(X_obs, X_pred)

    # Posterior
    posterior_mean = K_obs_pred.T.dot(K_obs_inv).dot(Y_obs)
    posterior_cov = K_pred - K_obs_pred.T.dot(K_obs_inv).dot(K_obs_pred)

    return stats.multivariate_normal(
        mean=posterior_mean.ravel(), cov=posterior_cov, allow_singular=True
    )


def plot_gaussian_process_posterior(
    X_obs,
    Y_obs,
    X_pred,
    kernel_function,
    sigma_y=1e-6,
    n_samples=3,
    figsize=(10, 5),
    resolution=100,
):
    X = np.linspace(-5, 5, resolution)[:, None]

    posterior = gaussian_process_posterior(X_obs, Y_obs, X, kernel_function, sigma_y=sigma_y)
    samples = posterior.rvs(n_samples)

    _, axs = plt.subplots(1, 2, figsize=figsize)
    plt.sca(axs[0])

    plot_gaussian_process(
        X_pred,
        samples=posterior.rvs(size=n_samples),
        mean=posterior.mean,
        cov=posterior.cov,
        X_obs=X_obs,
        Y_obs=Y_obs,
    )

    plt.sca(axs[1])
    plot_kernel_function(kernel_function, color="k")
    plt.title(f"kernel function")
    return axs
```

```{code-cell} ipython3
# Generate some training data
X_pred = np.linspace(-5, 5, 100)[:, None]
X_obs = np.linspace(-4, 4, 4)[:, None]
Y_obs = np.sin(X_obs) ** 2 + X_obs

# Initialize the kernel function
sigma_y = 0.25
sigma_kernel = 0.75
eta_kernel = 1
kernel_function = partial(quadratic_distance_kernel, eta=eta_kernel, sigma=sigma_kernel)

# Plot posterior
axs = plot_gaussian_process_posterior(
    X_obs, Y_obs, X_pred, kernel_function, sigma_y=sigma_y, n_samples=3
)
axs[0].set_title(f"posterior")
axs[1].set_title(f"kernel function:\n$\\eta$={eta}; $\\sigma=${sigma}");
```

## Distance-based model

$$
\begin{align}
T_i &\sim \text{Poisson}(\lambda_i) \\
\log(\lambda_i) &= \bar \alpha + \alpha_{S[i]} \\
\begin{pmatrix}
\alpha_1 \\
\alpha_2 \\
\vdots \\
\alpha_{10}
\end{pmatrix} &= \text{MVNormal}\left( 
\begin{bmatrix}
0 \\
0 \\
\vdots \\
0
\end{bmatrix}, \textbf{K}
\right) \\
k_{i,j} &= \eta^2\exp(-\rho^2D_{i,j}^2)) \\
\eta^2 &\sim \text{Exponential}(2) \\
\rho^2 &\sim \text{Exponential}(0.5)
\end{align}
$$

+++

### Islands Distance Matrix

```{code-cell} ipython3
# Distance matrix D_{i, j}
ISLANDS_DISTANCE_MATRIX = utils.load_data("islands_distance_matrix")
ISLANDS_DISTANCE_MATRIX.set_index(ISLANDS_DISTANCE_MATRIX.columns, inplace=True)
ISLANDS_DISTANCE_MATRIX.style.format(precision=1).background_gradient(cmap="viridis").set_caption(
    "Distance in Thousands of km"
)
```

```{code-cell} ipython3
# Data / coords
CULTURE_ID, CULTURE = pd.factorize(KLINE.culture.values)
ISLAND_DISTANCES = ISLANDS_DISTANCE_MATRIX.values.astype(float)
TOOLS = KLINE.total_tools.values.astype(int)
coords = {"culture": CULTURE}

with pm.Model(coords=coords) as distance_model:

    # Priors
    alpha_bar = pm.Normal("alpha_bar", 3, 0.5)
    eta_squared = pm.Exponential("eta_squared", 2)
    rho_squared = pm.Exponential("rho_squared", 0.5)

    # Gaussian Process
    kernel_function = eta_squared * pm.gp.cov.ExpQuad(input_dim=1, ls=rho_squared)
    GP = pm.gp.Latent(cov_func=kernel_function)
    alpha = GP.prior("alpha", X=ISLAND_DISTANCES, dims="culture")

    # Likelihood
    lambda_T = pm.math.exp(alpha_bar + alpha[CULTURE_ID])
    pm.Poisson("T", lambda_T, dims="culture", observed=TOOLS)
```

```{code-cell} ipython3
pm.model_to_graphviz(distance_model)
```

### Check model with prior-predictive simulation

```{code-cell} ipython3
def plot_predictive_covariance(predictive, n_samples=30, color="C0", label=None):

    eta_samples = predictive["eta_squared"].values[0, :n_samples] ** 0.5
    sigma_samples = 1 / predictive["rho_squared"].values[0, :n_samples] ** 0.5

    for ii, (eta, sigma) in enumerate(zip(eta_samples, sigma_samples)):
        label = label if ii == 0 else None

        kernel_function = partial(quadratic_distance_kernel, eta=eta, sigma=sigma)
        plot_kernel_function(
            kernel_function, color=color, label=label, alpha=0.5, linewidth=5, max_distance=7
        )
```

```{code-cell} ipython3
with distance_model:
    prior_predictive = pm.sample_prior_predictive(random_seed=12).prior

plot_predictive_covariance(prior_predictive, label="prior")
plt.ylim([0, 2])
plt.title("Prior Covariance Functions");
```

### Model the data

```{code-cell} ipython3
with distance_model:
    distance_inference = pm.sample(target_accept=0.99)
```

```{code-cell} ipython3
az.summary(distance_inference, var_names=["~alpha_rotated_"])
```

### Posterior Predictions

#### Compare the Prior to the Posterior

Ensure that we've learned something

```{code-cell} ipython3
plot_predictive_covariance(prior_predictive, color="k", label="prior")
plot_predictive_covariance(distance_inference.posterior, label="posterior")
plt.ylim([0, 2]);
```

```{code-cell} ipython3
def distance_to_covariance(distance, eta_squared, rho_squared):
    return eta_squared * np.exp(-rho_squared * distance**2)


def calculate_posterior_mean_covariance_matrix(inference):
    posterior_mean = inference.posterior.mean(dim=("chain", "draw"))

    posterior_eta_squared = posterior_mean["eta_squared"].values
    posterior_rho_squared = posterior_mean["rho_squared"].values

    print("Posterior Mean Kernel parameters:")
    print("eta_squared:", posterior_eta_squared)
    print("rho_squared:", posterior_rho_squared)

    model_covariance = np.zeros_like(ISLANDS_DISTANCE_MATRIX).astype(float)
    for ii in range(10):
        for jj in range(10):
            model_covariance[ii, jj] = distance_to_covariance(
                ISLAND_DISTANCES[ii, jj],
                eta_squared=posterior_eta_squared,
                rho_squared=posterior_rho_squared,
            )

    return model_covariance


def plot_posterior_mean_covariance_matrix(covariance_matrix, clim=(0, 0.4)):
    plt.matshow(covariance_matrix)
    plt.xticks(np.arange(10), labels=KLINE.culture, rotation=90)
    plt.yticks(np.arange(10), labels=KLINE.culture)
    plt.clim(clim)
    plt.grid(False)
    plt.colorbar()
    plt.title("posterior mean covariance matrix");
```

```{code-cell} ipython3
distance_model_covariance = calculate_posterior_mean_covariance_matrix(distance_inference)
plot_posterior_mean_covariance_matrix(distance_model_covariance)
```

```{code-cell} ipython3
def plot_kline_model_covarance(covariance_matric, min_alpha=0.01, alpha_gain=1, max_cov=None):
    plt.subplots(figsize=(10, 5))
    # Plot covariance
    max_cov = covariance_matric.max() if max_cov is None else max_cov
    for ii in range(10):
        for jj in range(10):
            if ii != jj:
                lat_a = KLINE.loc[ii, "lat"]
                lon_a = KLINE.loc[ii, "lon2"]

                lat_b = KLINE.loc[jj, "lat"]
                lon_b = KLINE.loc[jj, "lon2"]
                cov = covariance_matric[ii, jj]
                alpha = (min_alpha + (1 - min_alpha) * (cov / max_cov)) ** (1 / alpha_gain)
                plt.plot((lon_a, lon_b), (lat_a, lat_b), linewidth=3, color="k", alpha=alpha)

    plt.scatter(KLINE.lon2, KLINE.lat, s=KLINE.population * 0.01, zorder=10, alpha=0.9)

    # Annotate
    for ii in range(10):
        plt.annotate(
            KLINE.loc[ii, "culture"],
            (KLINE.loc[ii, "lon2"] + 1.5, KLINE.loc[ii, "lat"]),
            zorder=11,
            color="C1",
            fontsize=12,
            fontweight="bold",
        )
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.axis("tight")


plot_kline_model_covarance(distance_model_covariance, max_cov=0.4)
plt.title("Pure Spatial, nothing else");
```

## Stratify by population size

$$
\begin{align}
T_i &\sim \text{Poisson}(\lambda_i) \\
\lambda_i &= \frac{\bar \alpha P^\beta}{\gamma} \exp(\alpha_{S[i]}) \\
    \begin{pmatrix}
        \alpha_1 \\
        \alpha_2 \\
        \vdots \\
        \alpha_{10}
    \end{pmatrix} &= \text{MVNormal}
    \left( 
        \begin{bmatrix}
            0 \\
            0 \\
            \vdots \\
            0
        \end{bmatrix}, \textbf{K}
    \right) \\
    k_{i,j} &= \eta^2\exp(-\rho^2D_{i,j}^2)) \\
    \bar \alpha, \beta, \gamma &\sim \text{Exponential}(1) \\
    \eta^2 &\sim \text{Exponential}(2) \\
    \rho^2 &\sim \text{Exponential}(0.5)
\end{align}
$$

- We now fold our steady state equation into the intercept/baseline tools equation
- Include an additional $\exp$ multiplier to handle society-level offsets. Idea here being that
  - if $\alpha_{S[i]}=0$, then the society is average, and we fall back to the expected value of the difference equation i.e. multiplier is one
  - if $\alpha_{S[i]}>0$ then there is a boost in tool production. This boost is shared amongst spatially-local societies

```{code-cell} ipython3
with pm.Model(coords=coords) as distance_population_model:

    population = pm.MutableData("log_population", KLINE.logpop.values.astype(float))
    CULTURE_ID_ = pm.MutableData("CULTURE_ID", CULTURE_ID)

    # Priors
    alpha_bar = pm.Exponential("alpha_bar", 1)
    gamma = pm.Exponential("gamma", 1)
    beta = pm.Exponential("beta", 1)
    eta_squared = pm.Exponential("eta_squared", 2)
    rho_squared = pm.Exponential("rho_squared", 2)

    # Gaussian Process
    kernel_function = eta_squared * pm.gp.cov.ExpQuad(input_dim=1, ls=rho_squared)
    GP = pm.gp.Latent(cov_func=kernel_function)
    alpha = GP.prior("alpha", X=ISLAND_DISTANCES, dims="culture")

    # Likelihood
    lambda_T = (alpha_bar / gamma * population[CULTURE_ID_] ** beta) * pm.math.exp(
        alpha[CULTURE_ID_]
    )
    pm.Poisson("T", lambda_T, observed=TOOLS, dims="culture")
```

```{code-cell} ipython3
pm.model_to_graphviz(distance_population_model)
```

### Prior Predictive

```{code-cell} ipython3
with distance_population_model:
    prior_predictive = pm.sample_prior_predictive(random_seed=12).prior

plot_predictive_covariance(prior_predictive, label="prior")
plt.ylim([0, 1])
plt.title("Prior Covariance Functions");
```

```{code-cell} ipython3
with distance_population_model:
    distance_population_inference = pm.sample(target_accept=0.99)
```

```{code-cell} ipython3
plot_predictive_covariance(distance_inference.posterior, color="k", label="empty model")
plot_predictive_covariance(distance_population_inference.posterior, label="population")
plt.ylim([0, 1]);
```

```{code-cell} ipython3
distance_population_model_covariance = calculate_posterior_mean_covariance_matrix(
    distance_population_inference
)
plot_posterior_mean_covariance_matrix(distance_population_model_covariance)
```

We can see that by including population, the degree of covariance required to explain tool use is diminished, compared to the distance-only model. i.e. population has explained away a lot more of the variation.

```{code-cell} ipython3
plot_kline_model_covarance(distance_population_model_covariance, max_cov=0.4)
plt.title("Stratifying by Population");
```

#### Fit population-only model for comparison
I _think_ this is what McElreath is doing to construct the plot that combines population size, tool use and island covariance (next plot below). Specifically, we estimate a population-only model using the analytical solution, but includes no covariance amongst islands. We then compare add the covariance plot on top of that model to demonstrate that there is still some information left on the table regarding covariances, even when including population in the model.

```{code-cell} ipython3
ETA = 5.1  # Exponential hyperparmeter taken from Lecture 10 notes
with pm.Model() as population_model:

    # Note: raw population here, not log/standardized
    population = pm.MutableData("population", KLINE.logpop.values)

    # Priors
    # innovation rate
    alpha = pm.Exponential("alpha", eta)

    # Contact-level elasticity
    beta = pm.Exponential("beta", eta)

    # G lobal technology loss rate
    gamma = pm.Exponential("gamma", eta)

    # Likelihood using difference equation equilibrium as mean Poisson rate
    lamb = (alpha * (population**beta)) / gamma
    pm.Poisson("tools", lamb, observed=TOOLS)

    population_inference = pm.sample(tune=2000, target_accept=0.98)
    population_inference = pm.sample_posterior_predictive(
        population_inference, extend_inferencedata=True
    )
```

```{code-cell} ipython3
def plot_kline_model_population_covariance(
    covariance_matric, min_alpha=0.01, alpha_gain=1, max_cov=None
):

    # Plot covariancef
    max_cov = covariance_matric.max() if max_cov is None else max_cov
    for ii in range(10):
        for jj in range(10):
            if ii != jj:
                logpop_a = KLINE.loc[ii, "logpop"]
                tools_a = KLINE.loc[ii, "total_tools"]

                logpop_b = KLINE.loc[jj, "logpop"]
                tools_b = KLINE.loc[jj, "total_tools"]
                cov = covariance_matric[ii, jj]
                alpha = (min_alpha + (1 - min_alpha) * (cov / max_cov)) ** (1 / alpha_gain)
                plt.plot(
                    (logpop_a, logpop_b), (tools_a, tools_b), linewidth=3, color="k", alpha=alpha
                )

    plt.scatter(KLINE.logpop, KLINE.total_tools, s=KLINE.population * 0.01, zorder=10)

    # Annotate
    for ii in range(10):
        plt.annotate(
            KLINE.loc[ii, "culture"],
            (KLINE.loc[ii, "logpop"] + 0.1, KLINE.loc[ii, "total_tools"]),
            zorder=11,
            color="C1",
            fontsize=12,
            fontweight="bold",
        )
    plt.xlabel("log population")
    plt.ylabel("total tools")
    plt.axis("tight")


def plot_kline_population_only_model_posterior_predictive(inference):
    ppd = inference.posterior_predictive["tools"]
    az.plot_hdi(
        x=KLINE.logpop,
        y=ppd,
        color="C1",
        hdi_prob=0.89,
        fill_kwargs={"alpha": 0.1},
    )

    plt.plot(
        KLINE.logpop,
        ppd.mean(dim=("chain", "draw")),
        color="C1",
        linewidth=4,
        alpha=0.5,
        label="Population-only Model",
    )
    plt.legend()


plot_kline_model_population_covariance(distance_population_model_covariance, max_cov=0.4)
plot_kline_population_only_model_posterior_predictive(population_inference)
```

Though Population alone accounts for a lot of the variance in tool use (blue curve), there's still quite a bit of residual island-island similarity that explains tool use (i.e. some of the black lines still have some weight).

+++

# Phylogenetic Regression
## Dataset: Primates301

```{code-cell} ipython3
PRIMATES301 = utils.load_data("Primates301")
PRIMATES301.head()
```

- 301 Primate species
- Life history traits
- Body Mass $M$, Brain Size $B$, Social Group Size $G$
- Measurement error, unobserved confounding
- Missing data, **we'll fucus on the complete case analysis** in this lecture

#### Filter out 151 **Complete Cases**
Drop osbservations missing either Body Mass, Brain Size, or Group size

```{code-cell} ipython3
PRIMATES = PRIMATES301.query(
    "brain.notnull() and body.notnull() and group_size.notnull()", engine="python"
)
PRIMATES
```

#### There's lot's of Covariation amongst variables

```{code-cell} ipython3
from seaborn import pairplot

pairplot(
    PRIMATES.assign(
        log_brain_volume=lambda x: np.log(x["brain"]),
        log_body_mass=lambda x: np.log(x["body"]),
        log_group_size=lambda x: np.log(x["group_size"]),
    )[["log_brain_volume", "log_body_mass", "log_group_size"]]
);
```

There's no way from the sample alone to know what causes what

+++

## Causal Salad
- Throwing all predictors into a model, then interpreting their coefficients causally
- Using prediction methods as a stand-in for causality
- Throwing all factors into "salad" of regression terms
    - performing model selection base on prediction (we showed earlier in [Lecture 07 - Fitting Over & Under](<Lecture 07 - Fitting Over & Under.ipynb>) why this is bad to do)
    - and interpreting the resulting coefficients as causal effects (we've shown why this is bad many times)
- Controlling for phylogeny is important to do, but it's often done mindlessly (with the "salad" approach)
- Regression with phylogeny still requires a causal model

+++

## Different Hypotheses required different causal models (DAGs)

### Social Brain Hypothesis
- Group Size $G$ Effects Brain Size $B$

#### Body Mass $M$ is a shared cause  for both $M$ and $B$
This is the DAG we'll focus on in this Lecture

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
utils.draw_causal_graph(edge_list=[("G", "B"), ("M", "G"), ("M", "B")], graph_direction="LR")
```

#### Body Mass $M$ is a mediator for both $M$ and $B$

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
utils.draw_causal_graph(edge_list=[("G", "B"), ("G", "M"), ("M", "B")], graph_direction="LR")
```

#### Brain Size $B$ actually causes Group Size $G$; $M$ still a shared cause of both

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
utils.draw_causal_graph(edge_list=[("B", "G"), ("M", "G"), ("M", "B")], graph_direction="LR")
```

#### Unobserved confounds make naive regression on traits w/out controlling for history is dangerous
- History induces a number of unobserved shared confounds that we need to try to control for.
- phylogenetic history can work as a proxy for these shared confounds.
- We only have current measurements of outcome of history
  - actual phylogenies do not exist
  - different parts of a genome can have different histories


BUT, say that we _do_ have an inferred phylogeny (as we do in this lecture), we use this phylogeny, in conjunction with Gaussian Process Regression to model the similarity amongst species as a proxy for shared history

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
utils.draw_causal_graph(
    edge_list=[("G", "B"), ("M", "G"), ("M", "B"), ("u", "M"), ("u", "G"), ("u", "B"), ("h", "u")],
    node_props={
        "h": {"label": "history\n(phylogeny)"},
        "u": {"style": "dashed", "label": "confounds, u"},
        "unobserved": {"style": "dashed"},
    },
    graph_direction="LR",
)
```

## Phylogenetic Regression
- There's a long history of genetic evolution
- We only get to measure the current end state of that process
- In principle, common genetic histories should induce covariation amongst species
- We should hopefully be able to model this covariation to average over all the possible histories (micro state) that could have led to genetic similarity (macro state)

### Two conjoint problems

1. What is the history (phylogeny)?
2. How do we use it to model causes and control for confounds?


#### 1. What is the history (phylogeny)?

How do we estimate or identify a phylogeny?

##### Difficulties
- high degree of uncertainty
- processes are non-stationary
  - different parts of the genome have different histories
  - crossing-over effects
- available inferential tools
  - exploring tree space is difficult
  - repurposing software from other domains for studying biology
- phylogenies (just like social networks) do not exist.
  - They are abstractions we use to capture regularities in data
 
#### 2. How do we use it to model causes?

...say we've obtained a phylogeny, now what?


##### Approaches
- no default approach
- Gaussian Processes are default approach
  - use phylogeny as proxy for "genetic distance"


+++

## Two equivalent formulations of Linear Regression

You can always re-express a linear regression with a draw from a multi-variate Normal distribution

### Classical Linear Regression
The dataset $\textbf{B}$ is modeled as $D$ independent samples $B_i$ from a random a univariate normal.
$$
\begin{align}
B_i &\sim \text{Normal}(\mu_i, \sigma^2) \\
\mu_i &= \alpha + \beta_G G_i + \beta_M M_i \\
\alpha &\sim \text{Normal}(0, 1) \\
\beta_{G, M} &\sim \text{Normal}(0, 1) \\
\sigma &\sim \text{Exponential}(1)
\end{align}
$$

### $\text{MVNormal}$ Formulation
The dataset is modeled as a single $D$-vector $\textbf{B}$ sampled from a $\text{MVNormal}$. Has more of a Gaussian-process flavor to it; allows us to formulate the Linear regressoin in order to include covariation amongst the outputs.

$$
\begin{align}
\textbf{B} &\sim \text{MVNormal}(\mu, \textbf{K}) \\
\mathbf{K} &= \mathbf{I}\sigma^2 \\
\mu_i &= \alpha + \beta_G G_i + \beta_M M_i \\
\alpha &\sim \text{Normal}(0, 1) \\
\beta_{G, M} &\sim \text{Normal}(0, 1) \\
\sigma &\sim \text{Exponential}(1)
\end{align}
$$

+++

### Classic Linear Regression

```{code-cell} ipython3
# Preprocessing used in the Lecture
G = utils.standardize(np.log(PRIMATES.group_size.values))
M = utils.standardize(np.log(PRIMATES.body.values))
B = utils.standardize(np.log(PRIMATES.brain.values))
```

```{code-cell} ipython3
with pm.Model() as vanilla_lr_model:
    # Priors
    alpha = pm.Normal("alpha", 0, 1)
    beta_G = pm.Normal("beta_G", 0, 0.5)
    beta_M = pm.Normal("beta_M", 0, 0.5)
    sigma = pm.Exponential("sigma", 1)

    # Likelihood
    mu = alpha + beta_G * G + beta_M * M
    pm.Normal("B", mu=mu, sigma=sigma, observed=B)

    vanilla_lr_inference = pm.sample()
```

### $\text{MVNormal}$ Linear Regression

```{code-cell} ipython3
with pm.Model() as vector_lr_model:

    # Priors
    alpha = pm.Normal("alpha", 0, 1)
    beta_G = pm.Normal("beta_G", 0, 0.5)
    beta_M = pm.Normal("beta_M", 0, 0.5)
    sigma = pm.Exponential("sigma", 1)

    # Likelihood
    K = np.eye(len(B)) * sigma**2
    mu = alpha + beta_G * G + beta_M * M
    pm.MvNormal("B", mu=mu, cov=K, observed=B)

    vector_lr_inference = pm.sample()
```

#### Verify the two formulations provide the same parameter estimates
We're able to recover the same estimates as in the lecture

```{code-cell} ipython3
pm.summary(vanilla_lr_inference, round_to=2)
```

```{code-cell} ipython3
pm.summary(vector_lr_inference, round_to=2)
```

## From Model to Kernel
We'd like to **incorporate some residual $u_i$** into our linear regression that **adjusts the expecation in a way that encodes the shared history species**

$$
\begin{align}
\textbf{B} &\sim \text{MVNormal}(\mu, \textbf{K}) \\
\mathbf{K} &= \mathbf{I}\sigma^2 \\
\mu_i &= \alpha + \beta_G G_i + \beta_M M_i + u_i\\
\alpha &\sim \text{Normal}(0, 1) \\
\beta_{G, M} &\sim \text{Normal}(0, 1) \\
\sigma &\sim \text{Exponential}(1)
\end{align}
$$

### Phylogenetic distance as a proxy for species covariance
- Covariance falls off with **phylogenetic distance**
  - path length between leaf nodes in tree (e.g. see dendrogram below)
- Can use a Gaussian process in much the same way we did for island distances in the Oceanic Tools analysis

```{code-cell} ipython3
PRIMATES_DISTANCE_MATRIX301 = utils.load_data("Primates301_distance_matrix").values
# Filter out incomplete cases
PRIMATES_DISTANCE_MATRIX = PRIMATES_DISTANCE_MATRIX301[PRIMATES.index][:, PRIMATES.index]

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.matshow(PRIMATES_DISTANCE_MATRIX)
ax.grid(None)
fig.colorbar(im, orientation="vertical")
plt.xlabel("species ID")
plt.ylabel("species ID")
ax.set_title("Primate Phylogenetic Distance Matrix");
```

```{code-cell} ipython3
# Plot the distance matrix
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

dists = squareform(PRIMATES_DISTANCE_MATRIX)
linkage_matrix = linkage(dists, "single")
plt.subplots(figsize=(8, 20))
dendrogram(linkage_matrix, orientation="right", distance_sort=True, labels=PRIMATES.name.tolist())
plt.title("primate phylogeny")
plt.grid(False)
plt.xticks([]);
```

### Kernel Function as proxy for evolutionary dynamics

```{code-cell} ipython3
def brownian_motion_kernel(X0, X1, a=1, c=1):
    k = 1 / a * ((X0 - c) @ (X1 - c).T)[::-1, ::-1]
    return k


plt.subplots(figsize=(5, 5))
plot_kernel_function(
    partial(brownian_motion_kernel, a=22000), max_distance=150, label='Linear\n"Brownian Motion"'
)
plot_kernel_function(
    partial(ornstein_uhlenbeck_kernel, rho=1 / 20),
    max_distance=150,
    color="C1",
    label='Ornstein-Uhlenbeck\n"Damped Brownian Motion"',
)
plt.xlabel("phylogenetic distance");
```

#### Evolutionary model + tree structure = pattern of covariation observed at the tips

Common simple models of evolutionary dynamics:

- Brownian motion - implies in linear covariance kernel function
- Damped Brownian motion - implies an L1 / **Ornstein-Uhlenbeck** covariance kernel function

+++

### Phylogenetic "regression" model

$$
\begin{align}
\textbf{B} &\sim \text{MVNormal}(\mu, \textbf{K}) \\
\mathbf{K} &= \eta^2 \exp(-\rho D_{ij}) &\text{Ornstein-Uhlenbeck kernel}\\
\mu_i &= \alpha + \beta_G G_i + \beta_M M_i \\
\alpha &\sim \text{Normal}(0, 1) \\
\beta_{G, M} &\sim \text{Normal}(0, 1) \\
\eta^2 &\sim \text{HalfNormal}(1, 0.25) &\text{Maximum covariance prior} \\
\rho &\sim \text{HalfNormal}(3, 0.25) &\text{Covariance decline rate prior}
\end{align}
$$

+++

#### Prior Samples Ornstein-Uhlenbeck Kernel Model

```{code-cell} ipython3
eta_squared_samples = pm.draw(pm.TruncatedNormal.dist(mu=1, sigma=0.25, lower=0.01), 100) ** 0.5
rho_samples = pm.draw(pm.TruncatedNormal.dist(mu=3, sigma=0.25, lower=0.01), 100)

plt.subplots(figsize=(5, 5))
for ii, (eta_squared_, rho_) in enumerate(zip(eta_squared_samples, rho_samples)):
    label = "prior" if ii == 0 else None
    kernel_function = partial(ornstein_uhlenbeck_kernel, eta_squared=eta_squared_, rho=rho_)
    plot_kernel_function(
        kernel_function, color="C0", label=label, alpha=0.5, linewidth=2, max_distance=1
    )

plt.ylim([-0.05, 1.5])
plt.xlabel("phylogenetic distance")
plt.ylabel("covariance")
plt.title("Covariance Function Prior");
```

### Distance-only model

- Get the Gaussian-process part to work, then add details
- Set $\beta_M = \beta_G = 0$

```{code-cell} ipython3
# Rescale distances from 0 to 1
D = PRIMATES_DISTANCE_MATRIX / PRIMATES_DISTANCE_MATRIX.max()
assert D.max() == 1
```

#### PyMC5 Implementation

Below is an implementation that uses PyMC's Gaussian process module.

- In the Oceanic Tools models above, we didn't need the Gaussian process to have a mean function because all variables were standardized, thus we can use the default mean function of $0$
- For phylogenetic regression with we no longer want a zero mean function, but we want the mean function that depends on $M$ and $G$, but not a function of the phylogenetic distance matrix. We can implement a custom mean function in pymc to handle this
- Also, because our likelihood is a MVNormal, we can take advantage of conjugacy between the GP prior and the Normal likelihood, which returns a GP posterior. This means that we should use `pm.gp.Marginal` instead of `pm.gp.Latent`, which uses closed form solutions ot the posterior to speed up learning.

```{code-cell} ipython3
class MeanBodyMassSocialGroupSize(pm.gp.mean.Linear):
    """Custom mean function that separates covariates from phylogeny"""

    def __init__(self, alpha, beta_G, beta_M):
        self.alpha = alpha
        self.beta_G = beta_G
        self.beta_M = beta_M

    def __call__(self, X):
        return self.alpha + self.beta_G * G + self.beta_M * M
```

```{code-cell} ipython3
PRIMATE_ID, PRIMATE = pd.factorize(PRIMATES["name"], sort=False)
with pm.Model(coords={"primate": PRIMATE}) as intercept_only_phylogenetic_mvn_model:
    # Priors
    alpha = pm.Normal("alpha", 0, 1)
    sigma = pm.Exponential("sigma", 1)

    # Intercept-only model
    beta_M = 0
    beta_G = 0

    # Define the mean function
    mean_func = MeanBodyMassSocialGroupSize(alpha, beta_G, beta_M)

    # Phylogenetic distance covariance
    eta_squared = pm.TruncatedNormal("eta_squared", 1, 0.25, lower=0.01)
    rho = pm.TruncatedNormal("rho", 3, 0.25, lower=0.01)

    # For Ornstein-Uhlenbeck kernel we can use Matern 1/2 or Exponential covariance Function
    cov_func = eta_squared * pm.gp.cov.Matern12(1, ls=rho)
    # cov_func = eta_squared * pm.gp.cov.Exponential(1, ls=rho)

    # Init the GP
    gp = pm.gp.Marginal(mean_func=mean_func, cov_func=cov_func)

    # Likelihood
    gp.marginal_likelihood("B", X=D, y=B, noise=sigma)
    intercept_only_phylogenetic_mvn_inference = pm.sample(target_accept=0.9)
```

Below is an alternative implementation that builds the covariance function by hand, and directly models the dataset as a `MVNormal` with mean `alpha` and covariance defined by the kernel. I find that these direct `MVNormal` implementations track better with the lecture

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
def generate_L1_kernel_matrix(D, eta_squared, rho, smoothing=0.1):
    K = eta_squared * pm.math.exp(-rho * D)

    # Smooth the diagonal of the covariance matrix
    N = D.shape[0]
    K += np.eye(N) * smoothing
    return K


# with pm.Model() as intercept_only_phylogenetic_mvn_model:

#     # Priors
#     alpha = pm.Normal("alpha", 0, 1)

#     # Phylogenetic distance covariance
#     eta_squared = pm.TruncatedNormal("eta_squared", 1, 0.25, lower=.001)
#     rho = pm.TruncatedNormal("rho", 3, 0.25, lower=.001)

#     # Ornstein-Uhlenbeck kernel
#     K = pm.Deterministic('K', generate_L1_kernel_matrix(D, eta_squared, rho))
#     # K = pm.Deterministic('K', eta_squared * pm.math.exp(-rho * D))

#     # Likelihood
#     mu = pm.math.ones_like(B) * alpha
#     pm.MvNormal("B", mu=mu, cov=K, observed=B)
#     intercept_only_phylogenetic_mvn_inference = pm.sample(target_accept=.98)
```

```{code-cell} ipython3
az.summary(intercept_only_phylogenetic_mvn_inference, var_names=["alpha", "eta_squared", "rho"])
```

```{code-cell} ipython3
# Sample the prior for comparison
with intercept_only_phylogenetic_mvn_model:
    intercept_only_phylogenetic_mvn_prior = pm.sample_prior_predictive().prior

az.summary(intercept_only_phylogenetic_mvn_prior, var_names=["alpha", "eta_squared", "rho"])
```

### Compare the Prior and Posterior for distance-only model

```{code-cell} ipython3
def calculate_mean_covariance_matrix(inference):
    def distance_to_covariance(distance, eta_squared, rho):
        return eta_squared * np.exp(-rho * np.abs(distance))

    inference_mean = inference.mean(dim=("chain", "draw"))

    eta_squared = inference_mean["eta_squared"].values
    rho = inference_mean["rho"].values

    print("Mean Kernel parameters:")
    print("eta_squared:", eta_squared)
    print("rho:", rho)

    n_species = D.shape[0]
    model_covariance = np.zeros_like(D).astype(float)
    for ii in range(n_species):
        for jj in range(n_species):
            model_covariance[ii, jj] = distance_to_covariance(
                D[ii, jj], eta_squared=eta_squared, rho=rho
            )
    return model_covariance


def plot_mean_covariance_matrix(inference, title=None, max_cov=1.0):
    mean_cov = calculate_mean_covariance_matrix(inference)

    plt.imshow(mean_cov)
    # plt.imshow(inference['K'].mean(dim=('chain', 'draw')).values)
    plt.grid(None)
    plt.clim((0, max_cov))
    plt.colorbar()
    plt.xlabel("species ID")
    plt.ylabel("species ID")
    plt.title(title)
```

```{code-cell} ipython3
# Prior
plot_mean_covariance_matrix(
    intercept_only_phylogenetic_mvn_prior,
    title="Prior Covariance\nIntercept-only phylogenetic Model",
)
```

```{code-cell} ipython3
# Posterior
plot_mean_covariance_matrix(
    intercept_only_phylogenetic_mvn_inference.posterior,
    title="Prior Covariance\nIntercept-only phylogenetic Model",
)
```

#### Compare the Prior and Posterior kernel functions

```{code-cell} ipython3
def plot_predictive_kernel_function(predictive, n_samples=200, color="C0", label=None):

    eta_squared_samples = predictive["eta_squared"].values[0, :n_samples]
    rho_samples = predictive["rho"].values[0, :n_samples]

    for ii, (eta_squared_, rho_) in enumerate(zip(eta_squared_samples, rho_samples)):
        label = label if ii == 0 else None

        kernel_function = partial(ornstein_uhlenbeck_kernel, eta_squared=eta_squared_, rho=rho_)
        plot_kernel_function(
            kernel_function, color=color, label=label, alpha=0.1, linewidth=2, max_distance=1
        )

    plt.ylim([-0.05, 2.5])
    plt.xlabel("phylogenetic distance")
```

```{code-cell} ipython3
plot_predictive_kernel_function(intercept_only_phylogenetic_mvn_prior, label="prior", color="k")
plot_predictive_kernel_function(
    intercept_only_phylogenetic_mvn_inference.posterior, label="posterior B", color="C1"
)
```

We can see comparing the prior and posterior distirbution over kernel functions, that the model has learned from the data and attenuated our expectation on maximum covariance.

+++

### Fit the full Phylogentic model.

This model stratifies by Group size $G$ and Body Mass $M$

+++

Below is an implementation that uses PyMC's Gaussian process module.

```{code-cell} ipython3
with pm.Model(coords=coords) as full_phylogenetic_mvn_model:

    # Priors
    alpha = pm.Normal("alpha", 0, 1)
    sigma = pm.Exponential("sigma", 1)

    # Intercept-only model
    beta_M = pm.Normal("beta_M", 0, 1)
    beta_G = pm.Normal("beta_G", 0, 1)

    # Define the mean function
    mean_func = MeanBodyMassSocialGroupSize(alpha, beta_G, beta_M)

    # Phylogenetic distance covariance
    eta_squared = pm.TruncatedNormal("eta_squared", 1, 0.25, lower=0.01)
    rho = pm.TruncatedNormal("rho", 3, 0.25, lower=0.01)

    # For Ornstein-Uhlenbeck kernel we can use Matern 1/2 or Exponential covariance Function
    cov_func = eta_squared * pm.gp.cov.Matern12(1, ls=rho)
    # cov_func = eta_squared * pm.gp.cov.Exponential(1, ls=rho)

    # Init the GP
    gp = pm.gp.Marginal(mean_func=mean_func, cov_func=cov_func)

    # Likelihood
    gp.marginal_likelihood("B", X=D, y=B, noise=sigma)
    full_phylogenetic_mvn_inference = pm.sample(target_accept=0.9)
```

Below is an alternative implementation that builds the covariance function by hand, and directly models the dataset as a `MVNormal` with mean being the linear function of $G$ and $M$, and covariance defined by the kernel. I find that these `MVNormal` implementations track better with the lecture.

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
# with pm.Model() as full_phylogenetic_mvn_model:

#     # Priors
#     alpha = pm.Normal("alpha", 0, 1)
#     beta_G = pm.Normal("beta_G", 0, 0.5)
#     beta_M = pm.Normal("beta_M", 0, 0.5)

#     # Phylogenetic distance covariance
#     eta_squared = pm.TruncatedNormal("eta_squared", 1, .25, lower=.001)
#     rho = pm.TruncatedNormal("rho", 3, .25, lower=.001)
#     K = pm.Deterministic('K', generate_L1_kernel_matrix(D, eta_squared, rho))
#     # K = pm.Deterministic('K', eta_squared * pm.math.exp(-rho * D))

#     mu = alpha + beta_G * G + beta_M * M

#     # Likelihood
#     pm.MvNormal("B", mu=mu, cov=K, observed=B)
#     full_phylogenetic_mvn_inference = pm.sample(target_accept=.98)
```

```{code-cell} ipython3
az.summary(
    full_phylogenetic_mvn_inference, var_names=["alpha", "eta_squared", "rho", "beta_M", "beta_G"]
)
```

### Compare covariance terms in distance-only and full phylogenetic regression models

```{code-cell} ipython3
# Posterior
plot_mean_covariance_matrix(
    full_phylogenetic_mvn_inference.posterior,
    title="Prior Covariance\nIntercept-only phylogenetic Model",
)
```

```{code-cell} ipython3
plot_predictive_kernel_function(intercept_only_phylogenetic_mvn_prior, label="prior", color="k")
plot_predictive_kernel_function(
    intercept_only_phylogenetic_mvn_inference.posterior, label="posterior B", color="C1"
)
plot_predictive_kernel_function(full_phylogenetic_mvn_inference.posterior, label="posterior BMG")
```

We can see that when including regressors for body mass and group size, the amount of phylogenetic covaration used to explain brain size is greatly diminished, compared to the phylogenetic-distance-only model.

+++

The results for full-phylogenetic model reported here make sense to me too, given that we can explain away a lot of the variability in brain size by primarily body mass, after we've controlled for phylogenetic history; this in turn makes the phylogenetic history receive less weight in the model.

+++

### Influence of Group Size on Brain Size

```{code-cell} ipython3
utils.draw_causal_graph(
    edge_list=[("G", "B"), ("M", "G"), ("M", "B"), ("u", "M"), ("u", "G"), ("u", "B"), ("h", "u")],
    node_props={
        "h": {"label": "history\n(phylogeny)", "color": "lightgray", "fontcolor": "lightgray"},
        "u": {
            "style": "dashed",
            "label": "confounds, u",
            "color": "lightgray",
            "fontcolor": "lightgray",
        },
        "unobserved": {"style": "dashed", "color": "lightgray", "fontcolor": "lightgray"},
    },
    edge_props={
        ("G", "B"): {"color": "blue"},
        ("M", "G"): {"color": "red"},
        ("M", "B"): {"color": "red"},
        ("u", "B"): {"color": "lightgray"},
        ("u", "M"): {"color": "lightgray"},
        ("u", "G"): {"color": "lightgray"},
        ("h", "u"): {
            "color": "lightgray",
        },
    },
    graph_direction="LR",
)
```

- We ignore the confounds due to genetic history (light gray)
  - thus we don't try to control for potential confounds due to genetic history.
- To get <span style="color:blue">the direct effect of $G$ on $B$</span> we'll also stratify by $M$ to block <span style="color:red">the backdoor path through the fork passing through $M$ </span>.

```{code-cell} ipython3
with pm.Model() as ordinary_model:

    # Priors
    alpha = pm.Normal("alpha", 0, 1)
    beta_G = pm.Normal("beta_G", 0, 0.5)
    beta_M = pm.Normal("beta_M", 0, 0.5)

    # Independent species (equal variance)
    sigma = pm.Exponential("sigma", 1)
    K = np.eye(len(B)) * sigma

    mu = alpha + beta_G * G + beta_M * M

    pm.MvNormal("B", mu=mu, cov=K, observed=B)
    ordinary_inference = pm.sample()
```

```{code-cell} ipython3
az.summary(ordinary_inference, var_names=["alpha", "beta_M", "beta_G"])
```

```{code-cell} ipython3
def plot_primate_posterior_comparison(variable, ax, title=None):
    az.plot_dist(ordinary_inference.posterior[variable], color="k", label="ordinary", ax=ax)
    az.plot_dist(
        full_phylogenetic_mvn_inference.posterior[variable],
        color="C0",
        label="Ornstein-Uhlenbeck",
        ax=ax,
    )
    ax.axvline(0, color="k", linestyle="--")
    ax.set_xlabel(f"$\\{variable}$")
    ax.set_ylabel("density")
    ax.set_title(title)


_, axs = plt.subplots(1, 2, figsize=(10, 5))
plot_primate_posterior_comparison("beta_G", axs[0], "Effect of Group Size\non Brain Size")
plot_primate_posterior_comparison("beta_M", axs[1], "Effect of Body Mass\non Brain size")
```

By trying to control for potential confounds due to genetic history, we have nearly halved the estimated effect of Group size on Brain size $\beta_G$, though it's still mostly positive.

This estimator can also give us the direct effect of Body Mass $M$ on Brain Size $B$ (by blocking the Pipe through $G$). When we look at the $\beta_M$, we see that there is a slight increase in the effect of body mass on brain size when controlling for genetic history confounds, though the effect is large for both estimators indicating that body mass is a large driver of brain size.

+++

## Summary: Phylogenetic Regression

#### Potential Issues

- What about uncertainty in phylogeny?
  - better to perform phylogenic inference simultaneously -- i.e. estimate a posterior on phylogenies
- What about reciprocal causation?
  - feedback between organism and environment
  - there's no unidirectional cause
  - thus regression is likely not the best option
  - new were methods include differential equations to pose the problem as a multi-objective optimization problem

+++

## Summary: Gaussian Processes
- Provides means for (local) partial pooling for continuous groups/categories
- General approximation engine -- good for prediction
- Robust to overfitting
- Sensitive to priors: prior predictive simiulation is very important

+++

## Authors
* Ported to PyMC by Dustin Stansbury (2024)
* Based on Statistical Rethinking (2023) lectures by Richard McElreath

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,aeppl,xarray
```

:::{include} ../page_footer.md
:::
