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

+++ {"id": "FDW0_THqg8LC"}

(dependent_density_regression)=
# Dependent density regression
:::{post} 2017
:tags: mixture model, nonparametric
:category: intermediate
:author: Austin Rochford
:::

In another [example](dp_mix.ipynb), we showed how to use Dirichlet processes to perform Bayesian nonparametric density estimation.  This example expands on the previous one, illustrating dependent density regression.

Just as Dirichlet process mixtures can be thought of as infinite mixture models that select the number of active components as part of inference, dependent density regression can be thought of as infinite [mixtures of experts](https://en.wikipedia.org/wiki/Committee_machine) that select the active experts as part of inference.  Their flexibility and modularity make them powerful tools for performing nonparametric Bayesian Data analysis.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: wSEx-eTag8LD
outputId: a962b5ff-d107-47f8-b413-5dc0480648bf
---
from io import StringIO

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import requests
import seaborn as sns

from matplotlib import animation as ani
from matplotlib import pyplot as plt

print(f"Running on PyMC v{pm.__version__}")
```

```{code-cell} ipython3
:id: 0iVlIVjig8LE

%config InlineBackend.figure_format = 'retina'
plt.rc("animation", writer="ffmpeg")
blue, *_ = sns.color_palette()
az.style.use("arviz-darkgrid")
SEED = 1972917  # from random.org; for reproducibility
np.random.seed(SEED)
```

+++ {"id": "3VHUk32Mg8LE"}

We will use the LIDAR data set from Larry Wasserman's excellent book, [_All of Nonparametric Statistics_](http://www.stat.cmu.edu/~larry/all-of-nonpar/).  We standardize the data set to improve the rate of convergence of our samples.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: cVuo7yrRg8LE
outputId: bc357830-c080-453c-ff24-8154c328817b
---
DATA_URI = "http://www.stat.cmu.edu/~larry/all-of-nonpar/=data/lidar.dat"


def standardize(x):
    return (x - x.mean()) / x.std()


response = requests.get(DATA_URI, verify=False)
df = pd.read_csv(StringIO(response.text), sep=r"\s{1,3}", engine="python").assign(
    std_range=lambda df: standardize(df.range), std_logratio=lambda df: standardize(df.logratio)
)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 206
id: i30x-q2Cg8LE
outputId: 791768de-d65e-47f8-9aa2-ffecea186946
---
df.head()
```

+++ {"id": "tylbzDhcg8LE"}

We plot the LIDAR data below.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 628
id: HuFM6Wq8g8LE
outputId: 4240b043-428a-4923-9a48-3e1f24461842
---
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(df.std_range, df.std_logratio, color=blue)

ax.set_xticklabels([])
ax.set_xlabel("Standardized range")

ax.set_yticklabels([])
ax.set_ylabel("Standardized log ratio");
```

+++ {"id": "2mYwxtSfg8LE"}

This data set has a two interesting properties that make it useful for illustrating dependent density regression.

1. The relationship between range and log ratio is nonlinear, but has locally linear components.
2. The observation noise is [heteroskedastic](https://en.wikipedia.org/wiki/Heteroscedasticity); that is, the magnitude of the variance varies with the range.

The intuitive idea behind dependent density regression is to reduce the problem to many (related) density estimates, conditioned on fixed values of the predictors.  The following animation illustrates this intuition.

```{code-cell} ipython3
:id: di7x_3pvg8LE

fig, (scatter_ax, hist_ax) = plt.subplots(ncols=2, figsize=(16, 6))

scatter_ax.scatter(df.std_range, df.std_logratio, color=blue, zorder=2)

scatter_ax.set_xticklabels([])
scatter_ax.set_xlabel("Standardized range")

scatter_ax.set_yticklabels([])
scatter_ax.set_ylabel("Standardized log ratio")

bins = np.linspace(df.std_range.min(), df.std_range.max(), 25)

hist_ax.hist(df.std_logratio, bins=bins, color="k", lw=0, alpha=0.25, label="All data")

hist_ax.set_xticklabels([])
hist_ax.set_xlabel("Standardized log ratio")

hist_ax.set_yticklabels([])
hist_ax.set_ylabel("Frequency")

hist_ax.legend(loc=2)

endpoints = np.linspace(1.05 * df.std_range.min(), 1.05 * df.std_range.max(), 15)

frame_artists = []

for low, high in zip(endpoints[:-1], endpoints[2:]):
    interval = scatter_ax.axvspan(low, high, color="k", alpha=0.5, lw=0, zorder=1)
    *_, bars = hist_ax.hist(
        df[df.std_range.between(low, high)].std_logratio, bins=bins, color="k", lw=0, alpha=0.5
    )

    frame_artists.append((interval,) + tuple(bars))

animation = ani.ArtistAnimation(fig, frame_artists, interval=500, repeat_delay=3000, blit=True)
plt.close()
# prevent the intermediate figure from showing
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 641
id: SyWtHa72g8LE
outputId: c48bcfec-aa82-41ec-ce9a-32667117125e
---
from IPython.display import HTML

HTML(animation.to_html5_video())
```

+++ {"id": "i3B2R7-vg8LE"}

As we slice the data with a window sliding along the x-axis in the left plot, the empirical distribution of the y-values of the points in the window varies in the right plot.  An important aspect of this approach is that the density estimates that correspond to close values of the predictor are similar.

In the previous example, we saw that a Dirichlet process estimates a probability density as a mixture model with infinitely many components.  In the case of normal component distributions,

$$y \sim \sum_{i = 1}^{\infty} w_i \cdot N(\mu_i, \tau_i^{-1}),$$

where the mixture weights, $w_1, w_2, \ldots$, are generated by a [stick-breaking process](https://en.wikipedia.org/wiki/Dirichlet_process#The_stick-breaking_process).

Dependent density regression generalizes this representation of the Dirichlet process mixture model by allowing the mixture weights and component means to vary conditioned on the value of the predictor, $x$.  That is,

$$y\ |\ x \sim \sum_{i = 1}^{\infty} w_i\ |\ x \cdot N(\mu_i\ |\ x, \tau_i^{-1}).$$

In this example, we will follow Chapter 23 of [_Bayesian Data Analysis_](http://www.stat.columbia.edu/~gelman/book/) and use a probit stick-breaking process to determine the conditional mixture weights, $w_i\ |\ x$.  The probit stick-breaking process starts by defining

$$v_i\ |\ x = \Phi(\alpha_i + \beta_i x),$$

where $\Phi$ is the cumulative distribution function of the standard normal distribution.  We then obtain $w_i\ |\ x$ by applying the stick breaking process to $v_i\ |\ x$.  That is,

$$w_i\ |\ x = v_i\ |\ x \cdot \prod_{j = 1}^{i - 1} (1 - v_j\ |\ x).$$

For the LIDAR data set, we use independent normal priors $\alpha_i \sim N(0, 5^2)$ and $\beta_i \sim N(0, 5^2)$.  We now express this this model for the conditional mixture weights using `PyMC`.

```{code-cell} ipython3
:id: 5EgbxpkUg8LE

def norm_cdf(z):
    return 0.5 * (1 + pt.erf(z / np.sqrt(2)))


def stick_breaking(v):
    return v * pt.concatenate(
        [pt.ones_like(v[:, :1]), pt.extra_ops.cumprod(1 - v[:, :-1], axis=1)], axis=1
    )
```

```{code-cell} ipython3
:id: qtZS8sing8LE

N = len(df)
K = 20

std_range = df.std_range.values
std_logratio = df.std_logratio.values

with pm.Model(coords={"N": np.arange(N), "K": np.arange(K) + 1}) as model:
    alpha = pm.Normal("alpha", 0, 5, dims="K")
    beta = pm.Normal("beta", 0, 5, dims="K")
    x = pm.Data("x", std_range, dims="N")
    v = norm_cdf(alpha + pt.outer(x, beta))
    w = pm.Deterministic("w", stick_breaking(v), dims=["N", "K"])
```

+++ {"id": "TKt9RzIVg8LF"}

We have defined `x` as a `pm.Data` container in order to use `PyMC`'s posterior prediction capabilities later.

While the dependent density regression model theoretically has infinitely many components, we must truncate the model to finitely many components (in this case, twenty) in order to express it using `PyMC`.  After sampling from the model, we will verify that truncation did not unduly influence our results.

Since the LIDAR data seems to have several linear components, we use the linear models

$$
\begin{align*}
\mu_i\ |\ x
    & \sim \gamma_i + \delta_i x \\
\gamma_i
    & \sim N(0, 10^2) \\
\delta_i
    & \sim N(0, 10^2)
\end{align*}
$$

for the conditional component means.

```{code-cell} ipython3
:id: qMLOhLHsg8LF

with model:
    gamma = pm.Normal("gamma", 0, 3, dims="K")
    delta = pm.Normal("delta", 0, 3, dims="K")
    mu = pm.Deterministic("mu", gamma + pt.outer(x, delta), dims=("N", "K"))
```

+++ {"id": "4dcBWBbvg8LF"}

Finally, we specify a `NormalMixture` likelihood function, using the weights we have modeled above.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 487
id: ag8Lwc9sg8LF
outputId: 85b8d803-d144-4073-8e5d-7f3ffd35e48a
---
with model:
    sigma = pm.HalfNormal("sigma", 3, dims="K")
    y = pm.Data("y", std_logratio, dims="N")
    obs = pm.NormalMixture("obs", w, mu, sigma=sigma, observed=y, dims="N")

pm.model_to_graphviz(model)
```

+++ {"id": "gUPThEEEg8LF"}

We now sample from the dependent density regression model using a Metropolis sampler. The default NUTS sampler has a difficult time sampling the stick-breaking model, so we will employ a `CompoundSampler`, using a slice sampler for `alpha` and `beta` while leaving NUTS for the rest of the parameters.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 70
  referenced_widgets: [e2c19d27c2d24df69b2570d2580009a1, 6d10b9e9b680495386f1803d8994c2fb]
id: FSYdNHFUg8LF
outputId: 829d4ee8-c971-4962-aa71-265f93eeb356
---
with model:
    trace = pm.sample(random_seed=SEED, step=pm.Slice([alpha, beta]), tune=5_000, cores=2)
```

We can see from the R-hat diagnostics below (all near 1.0) that the model is reasonably well converged.

```{code-cell} ipython3
az.summary(trace, var_names=["beta"])
```

+++ {"id": "io6KXPdgg8LF"}

To verify that truncation did not unduly influence our results, we plot the largest posterior expected mixture weight for each component.  (In this model, each point has a mixture weight for each component, so we plot the maximum mixture weight for each component across all data points in order to judge if the component exerts any influence on the posterior.)

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 628
id: L_yuCm6Fg8LF
outputId: dda7fd9e-b609-4a23-8dc1-d298353c7182
---
fig, ax = plt.subplots(figsize=(8, 6))

max_mixture_weights = trace.posterior["w"].mean(("chain", "draw")).max("N")
ax.bar(max_mixture_weights.coords.to_index(), max_mixture_weights)

ax.set_xlim(1 - 0.5, K + 0.5)
ax.set_xticks(np.arange(0, K, 2) + 1)
ax.set_xlabel("Mixture component")

ax.set_ylabel("Largest posterior expected\nmixture weight");
```

+++ {"id": "6Pq0WqBbg8LF"}

Since only six mixture components have appreciable posterior expected weight for any data point, we can be fairly certain that truncation did not unduly influence our results.  (If most components had appreciable posterior expected weight, truncation may have influenced the results, and we would have increased the number of components and sampled again.)

Visually, it is reasonable that the LIDAR data has three linear components, so these posterior expected weights seem to have identified the structure of the data well.  We now sample from the posterior predictive distribution to get a better understand the model's performance.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 33
  referenced_widgets: [64628338cd314dcf998fdcdec5e64a2c, 8283e2190c4d45a6926da9d95273d376]
id: -tAIHunXg8LF
outputId: 733df6c3-aa98-44b6-bace-cc2075cee2a9
---
lidar_pp_x = np.linspace(std_range.min() - 0.05, std_range.max() + 0.05, 100)

with model:
    pm.set_data(
        {"x": lidar_pp_x, "y": np.zeros_like(lidar_pp_x)}, coords={"N": np.arange(len(lidar_pp_x))}
    )

    pm.sample_posterior_predictive(
        trace, predictions=True, extend_inferencedata=True, random_seed=SEED
    )
```

+++ {"id": "UecH3-jAg8LF"}

Below we plot the posterior expected value and the 95% posterior credible interval.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 628
id: m2ZWtSuQg8LF
outputId: ade722fc-744c-4b8a-8bf6-ec4fe55ce657
---
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(df.std_range, df.std_logratio, color=blue, zorder=10, label=None)

low, high = np.percentile(az.extract(trace.predictions)["obs"].T, [2.5, 97.5], axis=0)
ax.fill_between(
    lidar_pp_x, low, high, color="k", alpha=0.35, zorder=5, label="95% posterior credible interval"
)

ax.plot(
    lidar_pp_x,
    trace.predictions["obs"].mean(("chain", "draw")).values,
    c="k",
    zorder=6,
    label="Posterior expected value",
)

ax.set_xticklabels([])
ax.set_xlabel("Standardized range")

ax.set_yticklabels([])
ax.set_ylabel("Standardized log ratio")

ax.legend(loc=1)
ax.set_title("LIDAR Data");
```

+++ {"id": "0vFYLTpZg8LF"}

The model has fit the linear components of the data well, and also accommodated its heteroskedasticity.  This flexibility, along with the ability to modularly specify the conditional mixture weights and conditional component densities, makes dependent density regression an extremely useful nonparametric Bayesian model.

To learn more about dependent density regression and related models, consult [_Bayesian Data Analysis_](http://www.stat.columbia.edu/~gelman/book/), [_Bayesian Nonparametric Data Analysis_](http://www.springer.com/us/book/9783319189673), or [_Bayesian Nonparametrics_](https://www.google.com/webhp?sourceid=chrome-instant&ion=1&espv=2&ie=UTF-8#q=bayesian+nonparametrics+book).

This example first appeared [here](http://austinrochford.com/posts/2017-01-18-ddp-pymc3.html).

+++ {"id": "CxDFNZDtg8LF"}

## Authors
* authored by Austin Rochford in 2017
* updated to PyMC v5 by Christopher Fonnesbeck in September 2024

+++ {"id": "e41HT-6Og8LF"}

## References

:::{bibliography}
:filter: docname in docnames
:::

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: NMqJeLTAg8LF
outputId: 2a8b67c1-2922-4aff-82b2-392d66190951
---
%load_ext watermark
%watermark -n -u -v -iv -w
```
