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

(Multi-output-GPs_Coregion)=
# Multi-output Gaussian Processes: Coregionalization models using Hamadard product

:::{post} October, 2022
:tags: gaussian process, multi-output
:category: intermediate
:author: Danh Phan, Bill Engels, Chris Fonnesbeck
:::

+++

This notebook shows how to implement the **Intrinsic Coregionalization Model** (ICM) and the **Linear Coregionalization Model** (LCM) using a Hamadard product between the Coregion kernel and input kernels. Multi-output Gaussian Process is discussed in [this paper](https://papers.nips.cc/paper/2007/hash/66368270ffd51418ec58bd793f2d9b1b-Abstract.html) by {cite:t}`bonilla2007multioutput`. For further information about ICM and LCM, please check out the [talk](https://www.youtube.com/watch?v=ttgUJtVJthA&list=PLpTp0l_CVmgwyAthrUmmdIFiunV1VvicM) on Multi-output Gaussian Processes by Mauricio Alvarez, and [his slides](http://gpss.cc/gpss17/slides/multipleOutputGPs.pdf) with more references at the last page.

The advantage of Multi-output Gaussian Processes is their capacity to simultaneously learn and infer many outputs which have the same source of uncertainty from inputs. In this example, we model the average spin rates of several pitchers in different games from a baseball dataset.

```{code-cell} ipython3
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from pymc.gp.util import plot_gp_dist
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
```

## Preparing the data
The baseball dataset contains the average spin rate of several pitchers on different game dates.

```{code-cell} ipython3
# get data
try:
    df = pd.read_csv("../data/fastball_spin_rates.csv")
except FileNotFoundError:
    df = pd.read_csv(pm.get_data("fastball_spin_rates.csv"))

print(df.shape)
df.head()
```

```{code-cell} ipython3
print(
    f"There are {df['pitcher_name'].nunique()} pitchers, in {df['game_date'].nunique()} game dates"
)
```

```{code-cell} ipython3
# Standardise average spin rate
df["avg_spin_rate"] = (df["avg_spin_rate"] - df["avg_spin_rate"].mean()) / df["avg_spin_rate"].std()
df["avg_spin_rate"].describe()
```

#### Top N popular pitchers

```{code-cell} ipython3
# Get top N popular pitchers by who attended most games
n_outputs = 5  # Top 5 popular pitchers
top_pitchers = df.groupby("pitcher_name")["game_date"].count().nlargest(n_outputs).reset_index()
top_pitchers = top_pitchers.reset_index().rename(columns={"index": "output_idx"})
top_pitchers
```

```{code-cell} ipython3
# Filter the data with only top N pitchers
adf = df.loc[df["pitcher_name"].isin(top_pitchers["pitcher_name"])].copy()
print(adf.shape)
adf.head()
```

```{code-cell} ipython3
adf["avg_spin_rate"].describe()
```

#### Create a game date index

```{code-cell} ipython3
# There are 142 game dates from 01 Apr 2021 to 03 Oct 2021.
adf.loc[:, "game_date"] = pd.to_datetime(adf.loc[:, "game_date"])
game_dates = adf.loc[:, "game_date"]
game_dates.min(), game_dates.max(), game_dates.nunique(), (game_dates.max() - game_dates.min())
```

```{code-cell} ipython3
# Create a game date index
dates_idx = pd.DataFrame(
    {"game_date": pd.date_range(game_dates.min(), game_dates.max())}
).reset_index()
dates_idx = dates_idx.rename(columns={"index": "x"})
dates_idx.head()
```

#### Create training data

```{code-cell} ipython3
adf = adf.merge(dates_idx, how="left", on="game_date")
adf = adf.merge(top_pitchers[["pitcher_name", "output_idx"]], how="left", on="pitcher_name")
adf.head()
```

```{code-cell} ipython3
adf = adf.sort_values(["output_idx", "x"])
X = adf[
    ["x", "output_idx"]
].values  # Input data includes the index of game dates, and the index of pitchers
Y = adf["avg_spin_rate"].values  # Output data includes the average spin rate of pitchers
X.shape, Y.shape
```

#### Visualise training data

```{code-cell} ipython3
# Plot average spin rates of top pitchers
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
legends = []
for pitcher in top_pitchers["pitcher_name"]:
    cond = adf["pitcher_name"] == pitcher
    ax.plot(adf.loc[cond, "x"], adf.loc[cond, "avg_spin_rate"], "-o")
    legends.append(pitcher)
plt.title("Average spin rates of top 5 popular pitchers")
plt.xlabel("The index of game dates")
plt.ylim([-1.5, 4.0])
plt.legend(legends, loc="upper center");
```

## Intrinsic Coregionalization Model (ICM)

The Intrinsic Coregionalization Model (ICM) is a particular case of the Linear Coregionalization Model (LCM) with one input kernel, for example:

$$ K_{ICM} = B \otimes K_{ExpQuad} $$

Where $B(o,o')$ is the output kernel, and $K_{ExpQuad}(x,x')$ is an input kernel.

$$ B = WW^T +  diag(kappa) $$

```{code-cell} ipython3
def get_icm(input_dim, kernel, W=None, kappa=None, B=None, active_dims=None):
    """
    This function generates an ICM kernel from an input kernel and a Coregion kernel.
    """
    coreg = pm.gp.cov.Coregion(input_dim=input_dim, W=W, kappa=kappa, B=B, active_dims=active_dims)
    icm_cov = kernel * coreg  # Use Hadamard Product for separate inputs
    return icm_cov
```

```{code-cell} ipython3
with pm.Model() as model:
    # Priors
    ell = pm.Gamma("ell", alpha=2, beta=0.5)
    eta = pm.Gamma("eta", alpha=3, beta=1)
    kernel = eta**2 * pm.gp.cov.ExpQuad(input_dim=2, ls=ell, active_dims=[0])
    sigma = pm.HalfNormal("sigma", sigma=3)

    # Get the ICM kernel
    W = pm.Normal("W", mu=0, sigma=3, shape=(n_outputs, 2), initval=np.random.randn(n_outputs, 2))
    kappa = pm.Gamma("kappa", alpha=1.5, beta=1, shape=n_outputs)
    B = pm.Deterministic("B", pt.dot(W, W.T) + pt.diag(kappa))
    cov_icm = get_icm(input_dim=2, kernel=kernel, B=B, active_dims=[1])

    # Define a Multi-output GP
    mogp = pm.gp.Marginal(cov_func=cov_icm)
    y_ = mogp.marginal_likelihood("f", X, Y, sigma=sigma)
```

```{code-cell} ipython3
pm.model_to_graphviz(model)
```

```{code-cell} ipython3
%%time
with model:
    gp_trace = pm.sample(2000, chains=1)
```

#### Prediction

```{code-cell} ipython3
# Prepare test data
M = 200  # number of data points
x_new = np.linspace(0, 200, M)[
    :, None
]  # Select 200 days (185 previous days, and add 15 days into the future).
X_new = np.vstack([x_new for idx in range(n_outputs)])
output_idx = np.vstack([np.repeat(idx, M)[:, None] for idx in range(n_outputs)])
X_new = np.hstack([X_new, output_idx])
```

```{code-cell} ipython3
%%time
with model:
    preds = mogp.conditional("preds", X_new)
    gp_samples = pm.sample_posterior_predictive(gp_trace, var_names=["preds"], random_seed=42)
```

```{code-cell} ipython3
f_pred = gp_samples.posterior_predictive["preds"].sel(chain=0)


def plot_predictive_posteriors(f_pred, top_pitchers, M, X_new):
    fig, axes = plt.subplots(n_outputs, 1, figsize=(12, 15))

    for idx, pitcher in enumerate(top_pitchers["pitcher_name"]):
        # Prediction
        plot_gp_dist(
            axes[idx],
            f_pred[:, M * idx : M * (idx + 1)],
            X_new[M * idx : M * (idx + 1), 0],
            palette="Blues",
            fill_alpha=0.1,
            samples_alpha=0.1,
        )
        # Training data points
        cond = adf["pitcher_name"] == pitcher
        axes[idx].scatter(adf.loc[cond, "x"], adf.loc[cond, "avg_spin_rate"], color="r")
        axes[idx].set_title(pitcher)
    plt.tight_layout()


plot_predictive_posteriors(f_pred, top_pitchers, M, X_new)
```

It can be seen that the average spin rate of Rodriguez Richard decreases significantly from the 75th game dates. Besides, Kopech Michael's performance improves after a break of several weeks in the middle, while Hearn Taylor has performed better recently.

```{code-cell} ipython3
az.plot_trace(gp_trace)
plt.tight_layout()
```

## Linear Coregionalization Model (LCM)

The LCM is a generalization of the ICM with two or more input kernels, so the LCM kernel is basically a sum of several ICM kernels. The LMC allows several independent samples from GPs with different covariances (kernels).

In this example, in addition to an `ExpQuad` kernel, we add a `Matern32` kernel for input data.

$$ K_{LCM} = B \otimes K_{ExpQuad} + B \otimes K_{Matern32} $$

```{code-cell} ipython3
def get_lcm(input_dim, active_dims, num_outputs, kernels, W=None, B=None, name="ICM"):
    """
    This function generates a LCM kernel from a list of input `kernels` and a Coregion kernel.
    """
    if B is None:
        kappa = pm.Gamma(f"{name}_kappa", alpha=5, beta=1, shape=num_outputs)
        if W is None:
            W = pm.Normal(
                f"{name}_W",
                mu=0,
                sigma=5,
                shape=(num_outputs, 1),
                initval=np.random.randn(num_outputs, 1),
            )
    else:
        kappa = None

    cov_func = 0
    for idx, kernel in enumerate(kernels):
        icm = get_icm(input_dim, kernel, W, kappa, B, active_dims)
        cov_func += icm
    return cov_func
```

```{code-cell} ipython3
with pm.Model() as model:
    # Priors
    ell = pm.Gamma("ell", alpha=2, beta=0.5, shape=2)
    eta = pm.Gamma("eta", alpha=3, beta=1, shape=2)
    kernels = [pm.gp.cov.ExpQuad, pm.gp.cov.Matern32]
    sigma = pm.HalfNormal("sigma", sigma=3)

    # Define a list of covariance functions
    cov_list = [
        eta[idx] ** 2 * kernel(input_dim=2, ls=ell[idx], active_dims=[0])
        for idx, kernel in enumerate(kernels)
    ]

    # Get the LCM kernel
    cov_lcm = get_lcm(input_dim=2, active_dims=[1], num_outputs=n_outputs, kernels=cov_list)

    # Define a Multi-output GP
    mogp = pm.gp.Marginal(cov_func=cov_lcm)
    y_ = mogp.marginal_likelihood("f", X, Y, sigma=sigma)
```

```{code-cell} ipython3
pm.model_to_graphviz(model)
```

```{code-cell} ipython3
%%time
with model:
    gp_trace = pm.sample(2000, chains=1)
```

### Prediction

```{code-cell} ipython3
%%time
with model:
    preds = mogp.conditional("preds", X_new)
    gp_samples = pm.sample_posterior_predictive(gp_trace, var_names=["preds"], random_seed=42)
```

```{code-cell} ipython3
plot_predictive_posteriors(f_pred, top_pitchers, M, X_new)
```

```{code-cell} ipython3
az.plot_trace(gp_trace)
plt.tight_layout()
```

## Acknowledgement
This work is supported by 2022 [Google Summer of Codes](https://summerofcode.withgoogle.com/) and [NUMFOCUS](https://numfocus.org/).

+++

## Authors
* Authored by [Danh Phan](https://github.com/danhphan), [Bill Engels](https://github.com/bwengals), [Chris Fonnesbeck](https://github.com/fonnesbeck) in November, 2022 ([pymc-examples#454](https://github.com/pymc-devs/pymc-examples/pull/454))

+++

## References
:::{bibliography}
:filter: docname in docnames
:::

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,aeppl,xarray
```

:::{include} ../page_footer.md
:::
