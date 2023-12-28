---
jupytext:
  notebook_metadata_filter: substitutions
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
myst:
  substitutions:
    extra_dependencies: seaborn
---

(factor_analysis)=
# Factor analysis

:::{post} 19 Mar, 2022
:tags: factor analysis, matrix factorization, PCA 
:category: advanced, how-to
:author: Chris Hartl,  Christopher Krapu, Oriol Abril-Pla, Erik Werner
:::

+++

Factor analysis is a widely used probabilistic model for identifying low-rank structure in multivariate data as encoded in latent variables. It is very closely related to principal components analysis, and differs only in the prior distributions assumed for these latent variables. It is also a good example of a linear Gaussian model as it can be described entirely as a linear transformation of underlying Gaussian variates. For a high-level view of how factor analysis relates to other models, you can check out [this diagram](https://www.cs.ubc.ca/~murphyk/Bayes/Figures/gmka.gif) originally published by Ghahramani and Roweis.

+++

:::{include} ../extra_installs.md
:::

```{code-cell} ipython3
import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import scipy as sp
import seaborn as sns
import xarray as xr

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from numpy.random import default_rng
from xarray_einstats import linalg
from xarray_einstats.stats import XrContinuousRV

print(f"Running on PyMC v{pm.__version__}")
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")

np.set_printoptions(precision=3, suppress=True)
RANDOM_SEED = 31415
rng = default_rng(RANDOM_SEED)
```

## Simulated data generation

+++

To work through a few examples, we'll first generate some data. The data will not follow the exact generative process assumed by the factor analysis model, as the latent variates will not be Gaussian. We'll assume that we have an observed data set with $N$ rows and $d$ columns which are actually a noisy linear function of $k_{true}$ latent variables.

```{code-cell} ipython3
n = 250
k_true = 4
d = 10
```

The next code cell generates the data via creating latent variable arrays `M` and linear transformation `Q`. Then, the matrix product $QM$ is perturbed with additive Gaussian noise controlled by the variance parameter `err_sd`.

```{code-cell} ipython3
err_sd = 2
M = rng.binomial(1, 0.25, size=(k_true, n))
Q = np.hstack([rng.exponential(2 * k_true - k, size=(d, 1)) for k in range(k_true)]) * rng.binomial(
    1, 0.75, size=(d, k_true)
)
Y = np.round(1000 * Q @ M + rng.standard_normal(size=(d, n)) * err_sd) / 1000
```

Because of the way we have generated the data, the covariance matrix expressing correlations between columns of $Y$ will be equal to $QQ^T$. The fundamental assumption of PCA and factor analysis is that $QQ^T$ is not full rank. We can see hints of this if we plot the covariance matrix:

```{code-cell} ipython3
plt.figure(figsize=(4, 3))
sns.heatmap(np.corrcoef(Y));
```

If you squint long enough, you may be able to glimpse a few places where distinct columns are likely linear functions of each other.

+++

## Model
Probabilistic PCA (PPCA) and factor analysis (FA) are a common source of topics on [PyMC Discourse](https://discourse.pymc.io/). The posts linked below handle different aspects of the problem including:
* [Minibatched FA for large datasets](https://discourse.pymc.io/t/large-scale-factor-analysis-with-minibatch-advi/246)
* [Handling missing data in FA](https://discourse.pymc.io/t/dealing-with-missing-data/252)
* [Identifiability in FA / PPCA](https://discourse.pymc.io/t/unique-solution-for-probabilistic-pca/1324/14)

+++

### Direct implementation

+++

The model for factor analysis is the probabilistic matrix factorization

$X_{(d,n)}|W_{(d,k)}, F_{(k,n)} \sim \mathcal{N}(WF, \Psi)$

with $\Psi$ a diagonal matrix. Subscripts denote the dimensionality of the matrices. Probabilistic PCA is a variant that sets $\Psi = \sigma^2 I$. A basic implementation (taken from [this gist](https://gist.github.com/twiecki/c95578a6539d2098be2d83575e3d15fe)) is shown in the next cell. Unfortunately, it has undesirable properties for model fitting.

```{code-cell} ipython3
k = 2

coords = {"latent_columns": np.arange(k), "rows": np.arange(n), "observed_columns": np.arange(d)}

with pm.Model(coords=coords) as PPCA:
    W = pm.Normal("W", dims=("observed_columns", "latent_columns"))
    F = pm.Normal("F", dims=("latent_columns", "rows"))
    sigma = pm.HalfNormal("sigma", 1.0)
    X = pm.Normal("X", mu=W @ F, sigma=sigma, observed=Y, dims=("observed_columns", "rows"))

    trace = pm.sample(tune=2000, random_seed=rng)  # target_accept=0.9
```

At this point, there are already several warnings regarding failed convergence checks. We can see further problems in the trace plot below. This plot shows the path taken by each sampler chain for a single entry in the matrix $W$ as well as the average evaluated over samples for each chain.

```{code-cell} ipython3
for i in trace.posterior.chain.values:
    samples = trace.posterior["W"].sel(chain=i, observed_columns=3, latent_columns=1)
    plt.plot(samples, label="Chain {}".format(i + 1))
    plt.axhline(samples.mean(), color=f"C{i}")
plt.legend(ncol=4, loc="upper center", fontsize=12, frameon=True), plt.xlabel("Sample");
```

Each chain appears to have a different sample mean and we can also see that there is a great deal of autocorrelation across chains, manifest as long-range trends over sampling iterations.

One of the primary drawbacks for this model formulation is its lack of identifiability. With this model representation, only the product $WF$ matters for the likelihood of $X$, so $P(X|W, F) = P(X|W\Omega, \Omega^{-1}F)$ for any invertible matrix $\Omega$. While the priors on $W$ and $F$ constrain $|\Omega|$ to be neither too large or too small, factors and loadings can still be rotated, reflected, and/or permuted *without changing the model likelihood*. Expect it to happen between runs of the sampler, or even for the parametrization to "drift" within run, and to produce the highly autocorrelated $W$ traceplot above.

### Alternative parametrization

This can be fixed by constraining the form of W to be:
  + Lower triangular
  + Positive and increasing values along the diagonal

We can adapt `expand_block_triangular` to fill out a non-square matrix. This function mimics `pm.expand_packed_triangular`, but while the latter only works on packed versions of square matrices (i.e. $d=k$ in our model, the former can also be used with nonsquare matrices.

```{code-cell} ipython3
def expand_packed_block_triangular(d, k, packed, diag=None, mtype="pytensor"):
    # like expand_packed_triangular, but with d > k.
    assert mtype in {"pytensor", "numpy"}
    assert d >= k

    def set_(M, i_, v_):
        if mtype == "pytensor":
            return pt.set_subtensor(M[i_], v_)
        M[i_] = v_
        return M

    out = pt.zeros((d, k), dtype=float) if mtype == "pytensor" else np.zeros((d, k), dtype=float)
    if diag is None:
        idxs = np.tril_indices(d, m=k)
        out = set_(out, idxs, packed)
    else:
        idxs = np.tril_indices(d, k=-1, m=k)
        out = set_(out, idxs, packed)
        idxs = (np.arange(k), np.arange(k))
        out = set_(out, idxs, diag)
    return out
```

We'll also define another function which helps create a diagonal matrix with increasing entries along the main diagonal.

```{code-cell} ipython3
def makeW(d, k, dim_names):
    # make a W matrix adapted to the data shape
    n_od = int(k * d - k * (k - 1) / 2 - k)

    # trick: the cumulative sum of z will be positive increasing
    z = pm.HalfNormal("W_z", 1.0, dims="latent_columns")
    b = pm.Normal("W_b", 0.0, 1.0, shape=(n_od,), dims="packed_dim")
    L = expand_packed_block_triangular(d, k, b, pt.ones(k))
    W = pm.Deterministic("W", L @ pt.diag(pt.extra_ops.cumsum(z)), dims=dim_names)
    return W
```

With these modifications, we remake the model and run the MCMC sampler again.

```{code-cell} ipython3
with pm.Model(coords=coords) as PPCA_identified:
    W = makeW(d, k, ("observed_columns", "latent_columns"))
    F = pm.Normal("F", dims=("latent_columns", "rows"))
    sigma = pm.HalfNormal("sigma", 1.0)
    X = pm.Normal("X", mu=W @ F, sigma=sigma, observed=Y, dims=("observed_columns", "rows"))
    trace = pm.sample(tune=2000, random_seed=rng)  # target_accept=0.9

for i in range(4):
    samples = trace.posterior["W"].sel(chain=i, observed_columns=3, latent_columns=1)
    plt.plot(samples, label="Chain {}".format(i + 1))

plt.legend(ncol=4, loc="lower center", fontsize=8), plt.xlabel("Sample");
```

$W$ (and $F$!) now have entries with identical posterior distributions as compared between sampler chains, although it's apparent that some autocorrelation remains.

Because the $k \times n$ parameters in F all need to be sampled, sampling can become quite expensive for very large `n`. In addition, the link between an observed data point $X_i$ and an associated latent value $F_i$ means that streaming inference with mini-batching cannot be performed.

This scalability problem can be addressed analytically by integrating $F$ out of the model. By doing so, we postpone any calculation for individual values of $F_i$ until later. Hence, this approach is often described as *amortized inference*. However, this fixes the prior on $F$, allowing for less modeling flexibility. In keeping with $F_{ij} \sim \mathcal{N}(0, I)$ we have:

$X|WF \sim \mathcal{N}(WF, \sigma^2 I).$

We can therefore write $X$ as

$X = WF + \sigma I \epsilon,$

where $\epsilon \sim \mathcal{N}(0, I)$.
Fixing $W$ but treating $F$ and $\epsilon$ as random variables, both $\sim\mathcal{N}(0, I)$, we see that $X$ is the sum of two multivariate normal variables, with means $0$ and covariances $WW^T$ and $\sigma^2 I$, respectively. It follows that

$X|W \sim \mathcal{N}(0, WW^T + \sigma^2 I )$.

```{code-cell} ipython3
with pm.Model(coords=coords) as PPCA_amortized:
    W = makeW(d, k, ("observed_columns", "latent_columns"))
    sigma = pm.HalfNormal("sigma", 1.0)
    cov = W @ W.T + sigma**2 * pt.eye(d)
    # MvNormal parametrizes covariance of columns, so transpose Y
    X = pm.MvNormal("X", 0.0, cov=cov, observed=Y.T, dims=("rows", "observed_columns"))
    trace_amortized = pm.sample(tune=30, draws=100, random_seed=rng)
```

Unfortunately sampling of this model is very slow, presumably because calculating the logprob of the `MvNormal` requires inverting the covariance matrix. However, the explicit integration of $F$ also enables batching the observations for faster computation of `ADVI` and `FullRankADVI` approximations.

```{code-cell} ipython3
coords["observed_columns2"] = coords["observed_columns"]
with pm.Model(coords=coords) as PPCA_amortized_batched:
    W = makeW(d, k, ("observed_columns", "latent_columns"))
    Y_mb = pm.Minibatch(
        Y.T, batch_size=50
    )  # MvNormal parametrizes covariance of columns, so transpose Y
    sigma = pm.HalfNormal("sigma", 1.0)
    cov = W @ W.T + sigma**2 * pt.eye(d)
    X = pm.MvNormal("X", 0.0, cov=cov, observed=Y_mb)
    trace_vi = pm.fit(n=50000, method="fullrank_advi", obj_n_mc=1).sample()
```

## Results
When we compare the posteriors calculated using MCMC and VI, we find that (for at least this specific parameter we are looking at) the two distributions are close, but they do differ in their mean. The two MCMC posteriors agree with each other quite well with each other and the ADVI estimate is not far off.

```{code-cell} ipython3
col_selection = dict(observed_columns=3, latent_columns=1)

ax = az.plot_kde(
    trace.posterior["W"].sel(**col_selection).values,
    label="MCMC posterior for the explicit model".format(0),
    plot_kwargs={"color": f"C{1}"},
)

az.plot_kde(
    trace_amortized.posterior["W"].sel(**col_selection).values,
    label="MCMC posterior for amortized inference",
    plot_kwargs={"color": f"C{2}", "linestyle": "--"},
)


az.plot_kde(
    trace_vi.posterior["W"].sel(**col_selection).squeeze().values,
    label="FR-ADVI posterior for amortized inference",
    plot_kwargs={"alpha": 0},
    fill_kwargs={"alpha": 0.5, "color": f"C{0}"},
)


ax.set_title(rf"PDFs of $W$ estimate at {col_selection}")
ax.legend(loc="upper left", fontsize=10);
```

### Post-hoc identification of F

The matrix $F$ is typically of interest for factor analysis, and is often used as a feature matrix for dimensionality reduction. However, $F$ has been
marginalized away in order to make fitting the model easier; and now we need it back. Transforming

$X|WF \sim \mathcal{N}(WF, \sigma^2)$

into

$(W^TW)^{-1}W^T X|W,F \sim \mathcal{N}(F, \sigma^2(W^TW)^{-1})$

we have represented $F$ as the mean of a multivariate normal distribution with a known covariance matrix. Using the prior $ F \sim \mathcal{N}(0, I) $ and updating according to the expression for the [conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior), we find

$ F | X, W \sim \mathcal{N}(\mu_F, \Sigma_F) $,

where

$\mu_F = \left(I + \sigma^{-2} W^T W\right)^{-1} \sigma^{-2} W^T X$

$\Sigma_F = \left(I + \sigma^{-2} W^T W\right)^{-1}$

+++

For each value of $W$ and $X$ found in the trace, we now draw a sample from this distribution.

```{code-cell} ipython3
# configure xarray-einstats
def get_default_dims(dims, dims2):
    proposed_dims = [dim for dim in dims if dim not in {"chain", "draw"}]
    assert len(proposed_dims) == 2
    if dims2 is None:
        return proposed_dims


linalg.get_default_dims = get_default_dims
```

```{code-cell} ipython3
post = trace_vi.posterior
obs = trace.observed_data

WW = linalg.matmul(
    post["W"], post["W"], dims=("latent_columns", "observed_columns", "latent_columns")
)

# Constructing an identity matrix following https://einstats.python.arviz.org/en/latest/tutorials/np_linalg_tutorial_port.html
I = xr.zeros_like(WW)
idx = xr.DataArray(WW.coords["latent_columns"])
I.loc[{"latent_columns": idx, "latent_columns2": idx}] = 1
Sigma_F = linalg.inv(I + post["sigma"] ** (-2) * WW)
X_transform = linalg.matmul(
    Sigma_F,
    post["sigma"] ** (-2) * post["W"],
    dims=("latent_columns2", "latent_columns", "observed_columns"),
)
mu_F = xr.dot(X_transform, obs["X"], dims="observed_columns").rename(
    latent_columns2="latent_columns"
)
Sigma_chol = linalg.cholesky(Sigma_F)
norm_dist = XrContinuousRV(sp.stats.norm, xr.zeros_like(mu_F))  # the zeros_like defines the shape
F_sampled = mu_F + linalg.matmul(
    post["sigma"] * Sigma_F,
    norm_dist.rvs(),
    dims=(("latent_columns", "latent_columns2"), ("latent_columns", "rows")),
)
```

### Comparison to original data

To check how well our model has captured the original data, we will test how well we can reconstruct it from the sampled $W$ and $F$ matrices. For each element of $Y$ we plot the mean and standard deviation of $X = W F$, which is the reconstructed value based on our model.

```{code-cell} ipython3
X_sampled_amortized = linalg.matmul(
    post["W"],
    F_sampled,
    dims=("observed_columns", "latent_columns", "rows"),
)
reconstruction_mean = X_sampled_amortized.mean(dim=("chain", "draw")).values
reconstruction_sd = X_sampled_amortized.std(dim=("chain", "draw")).values
plt.errorbar(
    Y.ravel(),
    reconstruction_mean.ravel(),
    yerr=reconstruction_sd.ravel(),
    marker=".",
    ls="none",
    alpha=0.3,
)

slope, intercept, r_value, p_value, std_err = sp.stats.linregress(
    Y.ravel(), reconstruction_mean.ravel()
)
plt.plot(
    [Y.min(), Y.max()],
    np.array([Y.min(), Y.max()]) * slope + intercept,
    "k--",
    label=f"Linear regression for the mean, R²={r_value**2:.2f}",
)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], "k:", label="Perfect reconstruction")

plt.xlabel("Elements of Y")
plt.ylabel("Model reconstruction")
plt.legend(loc="upper left");
```

We find that our model does a decent job of capturing the variation in the original data, despite only using two latent factors instead of the actual four.

+++

## Authors
* Authored by [chartl](https://github.com/chartl) on May 6, 2019
* Updated by [Christopher Krapu](https://github.com/ckrapu) on April 4, 2021
* Updated by Oriol Abril-Pla to use PyMC v4 and xarray-einstats on March, 2022
* Updated by Erik Werner on Dec, 2023 ([pymc-examples#612](https://github.com/pymc-devs/pymc-examples/pull/612))

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w
```

:::{include} ../page_footer.md
:::
