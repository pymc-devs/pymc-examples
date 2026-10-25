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

(MvGaussianRandomWalk)=
# Multivariate Gaussian Random Walk
:::{post} Feb 2, 2023
:tags: linear model, regression, time series 
:category: beginner
:author: Lorenzo Itoniazzi, Chris Fonnesbeck
:::

```{code-cell} ipython3
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from scipy.linalg import cholesky
from sklearn.preprocessing import StandardScaler

%matplotlib inline
```

```{code-cell} ipython3
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-variat")
```

This notebook shows how to [fit a correlated time series](https://en.wikipedia.org/wiki/Curve_fitting) using multivariate [Gaussian random walks](https://en.wikipedia.org/wiki/Random_walk#Gaussian_random_walk) (GRWs). In particular, we perform a Bayesian [regression](https://en.wikipedia.org/wiki/Regression_analysis) of the time series data against a model dependent on GRWs.

We generate data as the 3-dimensional time series

$$
\mathbf y = \alpha_{i[\mathbf t]} +\beta_{i[\mathbf t]} *\frac{\mathbf t}{300} +\xi_{\mathbf t},\quad \mathbf t = [0,1,...,299], 
$$ (eqn:model)

where 
- $i\mapsto\alpha_{i}$ and $i\mapsto\beta_{i}$, $i\in\{0,1,2,3,4\}$, are two 3-dimensional Gaussian random walks for two correlation matrices $\Sigma_\alpha$ and $\Sigma_\beta$,
- we define the index 
$$
i[t]= j\quad\text{for}\quad t = 60j,60j+1,...,60j+59, \quad\text{and}\quad j = 0,1,2,3,4,
$$ 
- $*$ means that we multiply the $j$-th column of the $3\times300$ matrix with the $j$-th entry of the vector for each $j=0,1,...,299$, and 
- $\xi_{\mathbf t}$ is a $3\times300$ matrix with iid normal entries $N(0,\sigma^2)$.


So the series $\mathbf y$ changes due to the GRW $\alpha$ in five occasions, namely steps $0,60,120,180,240$. Meanwhile  $\mathbf y$ changes at steps $1,60,120,180,240$ due to the increments of the GRW $\beta$ and  at every step  due to the weighting of  $\beta$ with $\mathbf t/300$. Intuitively, we have a noisy ($\xi$) system that is shocked five times over a period of 300 steps, but the impact of the $\beta$ shocks gradually becomes more significant at every step. 

## Data generation

We first set the dimensions of the problem and draw the true random walks $\alpha$ and $\beta$.

```{code-cell} ipython3
D = 3  # Dimension of random walks
N = 300  # Number of steps
sections = 5  # Number of sections
period = N // sections  # Number steps in each section

Sigma_alpha = rng.standard_normal((D, D))
Sigma_alpha = Sigma_alpha.T.dot(Sigma_alpha)  # Construct covariance matrix for alpha
L_alpha = cholesky(Sigma_alpha, lower=True)  # Obtain its Cholesky decomposition

Sigma_beta = rng.standard_normal((D, D))
Sigma_beta = Sigma_beta.T.dot(Sigma_beta)  # Construct covariance matrix for beta
L_beta = cholesky(Sigma_beta, lower=True)  # Obtain its Cholesky decomposition

# Gaussian random walks:
alpha_true = np.cumsum(L_alpha.dot(rng.standard_normal((D, sections))), axis=1).T
beta_true = np.cumsum(L_beta.dot(rng.standard_normal((D, sections))), axis=1).T
sigma_true = 0.1

t = np.arange(N)[:, None] / N
t_section = np.repeat(np.arange(sections), period)  # Section index of each time step
```

We write the regression model in {eq}`eqn:model` as a function that takes the time steps $\mathbf t$, the section index $i[\mathbf t]$, and optionally the observed series $\mathbf y$, and returns a PyMC model. It imposes priors on the GRWs $\alpha$ and $\beta$, on the standard deviation $\sigma$ and hyperpriors on the Cholesky matrices. We use the LKJ prior {cite:p}`lewandowski2009generating` for the Cholesky matrices (see this {func}`link for the documentation <pymc.LKJCholeskyCov>` and also the PyMC notebook {doc}`/case_studies/LKJ` for some usage examples.) The same function generates the data and fits the model.

```{code-cell} ipython3
def make_model(t, t_section, y=None):
    coords = {"y_": ["y_0", "y_1", "y_2"], "steps": np.arange(len(t))}
    D = len(coords["y_"])
    sections = t_section.max() + 1

    with pm.Model(coords=coords) as model:
        # Hyperpriors on Cholesky matrices
        chol_alpha, *_ = pm.LKJCholeskyCov(
            "chol_cov_alpha", n=D, eta=2, sd_dist=pm.HalfCauchy.dist(2.5), compute_corr=True
        )
        chol_beta, *_ = pm.LKJCholeskyCov(
            "chol_cov_beta", n=D, eta=2, sd_dist=pm.HalfCauchy.dist(2.5), compute_corr=True
        )

        # Priors on Gaussian random walks
        init_dist = pm.MvNormal.dist(mu=np.zeros(D), cov=100 * np.eye(D))
        alpha = pm.MvGaussianRandomWalk(
            "alpha", mu=np.zeros(D), chol=chol_alpha, init_dist=init_dist, shape=(sections, D)
        )
        beta = pm.MvGaussianRandomWalk(
            "beta", mu=np.zeros(D), chol=chol_beta, init_dist=init_dist, shape=(sections, D)
        )

        # Deterministic construction of the correlated random walk
        regression = alpha[t_section] + beta[t_section] * t

        # Prior on noise ξ
        sigma = pm.HalfNormal("sigma", 1.0)

        # Likelihood
        pm.Normal("y", mu=regression, sigma=sigma, observed=y, dims=("steps", "y_"))
    return model
```

```{code-cell} ipython3
# fix the random walks and the noise to their true values and take a single draw of y
true_model = pm.do(
    model=make_model(t=t, t_section=t_section),
    vars_to_interventions={"alpha": alpha_true, "beta": beta_true, "sigma": sigma_true},
)
prior_draw = pm.sample_prior_predictive(draws=1, model=true_model, random_seed=rng)
y = prior_draw.prior["y"].isel(chain=0, draw=0).to_numpy()

# Plot the correlated series
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(t, y, ".", markersize=2, label=("y_0 data", "y_1 data", "y_2 data"))
ax.set(title="Three Correlated Series", xlabel="Time")
ax.legend()
plt.show();
```

## Model
First we standardize the data and the time parameter with scikit-learn's `StandardScaler`, so we can rescale the predictions to match the unscaled data after sampling.

```{code-cell} ipython3
# Standardize y and t
y_scaler = StandardScaler()
t_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)
t_scaled = t_scaler.fit_transform(t)
```

We build the model from the standardized time steps and series.

```{code-cell} ipython3
model = make_model(t=t_scaled, t_section=t_section, y=y_scaled)

pm.model_to_graphviz(model)
```

## Inference
We now sample from the posterior, then draw from the posterior predictive distribution.

```{code-cell} ipython3
with model:
    idata = pm.sample(random_seed=rng, target_accept=0.9)
```

```{code-cell} ipython3
with model:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=rng)
```

We now display the energy plot using {func}`arviz.plot_energy` for a visual check for the model's convergence. Then, using {func}`arviz.plot_ppc_dist`,  we plot the distribution of the {doc}`posterior predictive samples </diagnostics_and_criticism/posterior_predictive>` against the observed data $\mathbf y$. This plot provides a general idea of the accuracy of the model (note that the values of $\mathbf y$ actually correspond to the scaled version of $\mathbf y$).

```{code-cell} ipython3
az.plot_energy(idata)
az.plot_ppc_dist(idata);
```

+++ {"jupyter": {"outputs_hidden": true}}

## Posterior visualisation
The graphs above look good. Now we plot the observed 3-dimensional series against the average predicted 3-dimensional series, or in other words, we plot the data against the estimated regression curve from the model {eq}`eqn:model`.

```{code-cell} ipython3
# Compute the predicted mean of the multivariate GRWs
alpha_mean = idata.posterior["alpha"].mean(dim=("chain", "draw"))
beta_mean = idata.posterior["beta"].mean(dim=("chain", "draw"))

# Compute the predicted mean of the correlated series
y_pred = y_scaler.inverse_transform(
    alpha_mean[t_section].to_numpy() + beta_mean[t_section].to_numpy() * t_scaled
)

# Plot the predicted mean
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(t, y, ".", markersize=2, label=("y_0 data", "y_1 data", "y_2 data"))
plt.gca().set_prop_cycle(None)
ax.plot(t, y_pred, label=("y_0 pred", "y_1 pred", "y_2 pred"))
ax.set_xlabel("Time")
ax.legend()
ax.set_title("Predicted Mean of Three Correlated Series");
```

Finally, we plot the data against the posterior predictive samples.

```{code-cell} ipython3
# Rescale the posterior predictive samples
ppc = idata.posterior_predictive["y"].stack(sample=("chain", "draw", "steps"))
ppc = ppc.transpose("sample", "y_")
ppc_y = ppc.copy(data=y_scaler.inverse_transform(ppc)).unstack("sample")

ppc_mean = ppc_y.mean(dim=["chain", "draw"])
hdi_50 = az.hdi(ppc_y, prob=0.5, dim=["chain", "draw"])
hdi_95 = az.hdi(ppc_y, prob=0.95, dim=["chain", "draw"])

fig, ax = plt.subplots(figsize=(12, 6))
for i, series in enumerate(ppc_y["y_"].to_numpy()):
    color = f"C{i}"
    ax.plot(t, y[:, i], ".", color=color, markersize=3, label=f"{series} data")
    ax.plot(t, ppc_mean.sel(y_=series), color=color, linewidth=1)
    for hdi, fill_alpha in [(hdi_95, 0.2), (hdi_50, 0.4)]:
        ax.fill_between(
            t[:, 0],
            hdi.sel(y_=series, ci_bound="lower"),
            hdi.sel(y_=series, ci_bound="upper"),
            color=color,
            alpha=fill_alpha,
            linewidth=0,
        )
ax.set_xlabel("Time")
ax.legend()
ax.set_title("Posterior Predictive Mean, 50% and 95% HDI, and the Three Correlated Series");
```

## Authors
* updated to best practices by Lorenzon Itoniazzi in October, 2021 ([pymc-examples#195](https://github.com/pymc-devs/pymc-examples/pull/195))
* updated to v5 by Chris Fonnesbeck in February, 2023 
* updated to v6 by Jesse Grabowski in October, 2026

+++

## References

:::{bibliography}
:filter: docname in docnames
:::

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,xarray
```

:::{include} ../page_footer.md
:::
