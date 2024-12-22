---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: default
  language: python
  name: python3
---

# Heteroskedastic Gaussian Processes

+++

We can typically divide the sources of uncertainty in our models into two categories. "Aleatoric" uncertainty (from the Latin word for dice or randomness) arises from the intrinsic variability of our system. "Epistemic" uncertainty (from the Greek word for knowledge) arises from how our observations are placed throughout the domain of interest.

Gaussian Process (GP) models are a powerful tool to capture both of these sources of uncertainty. By considering the distribution of all functions that satisfy the conditions specified by the covariance kernel and the data, these models express low epistemic uncertainty near the observations and high epistemic uncertainty farther away. To incorporate aleatoric uncertainty, the standard GP model assumes additive white noise with constant magnitude throughout the domain. However, this "homoskedastic" model can do a poor job of representing your system if some regions have higher variance than others. Among other fields, this is particularly common in the experimental sciences, where varying experimental parameters can affect both the magnitude and the variability of the response. Explicitly incorporating the dependence (and inter-dependence) of noise on the inputs and outputs can lead to a better understanding of the mean behavior as well as a more informative landscape for optimization, for example.

This notebook will work through several approaches to heteroskedastic modeling with GPs. We'll use toy data that represents (independent) repeated measurements at a range of input values on a system where the magnitude of the noise increases with the response variable. We'll start with simplistic modeling approaches such as fitting a GP to the mean at each point weighted by the variance at each point (which may be useful if individual measurements are taken via a method with known uncertainty), contrasting this with a typical homoskedastic GP. We'll then construct a model that uses one latent GP to model the response mean and a second (independent) latent GP to model the response variance. To improve the efficiency and scalability of this model, we'll re-formulate it in a sparse framework. Finally, we'll use a coregionalization kernel to allow correlation between the noise and the mean response.

+++

## Data

```{code-cell} ipython3
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from scipy.spatial.distance import pdist

warnings.filterwarnings("ignore", category=UserWarning)

%config InlineBackend.figure_format ='retina'
%load_ext watermark
```

```{code-cell} ipython3
SEED = 2020
rng = np.random.default_rng(SEED)
az.style.use("arviz-darkgrid")
```

```{code-cell} ipython3
def signal(x):
    return x / 2 + np.sin(2 * np.pi * x) / 5


def noise(y):
    return np.exp(y) / 20


X = np.linspace(0.1, 1, 20)[:, None]
X = np.vstack([X, X + 2])
X_ = X.flatten()
y = signal(X_)
sigma_fun = noise(y)

y_err = rng.lognormal(np.log(sigma_fun), 0.1)
y_obs = rng.normal(y, y_err, size=(5, len(y)))
y_obs_ = y_obs.T.flatten()
X_obs = np.tile(X.T, (5, 1)).T.reshape(-1, 1)
X_obs_ = X_obs.flatten()
idx = np.tile(np.array([i for i, _ in enumerate(X_)]), (5, 1)).T.flatten()

Xnew = np.linspace(-0.15, 3.25, 100)[:, None]
Xnew_ = Xnew.flatten()
ynew = signal(Xnew)

plt.plot(X, y, "C0o")
plt.errorbar(X_, y, y_err, color="C0");
```

## Helper and plotting functions

```{code-cell} ipython3
def get_ell_prior(points):
    """Calculates mean and sd for InverseGamma prior on lengthscale"""
    distances = pdist(points[:, None])
    distinct = distances != 0
    ell_l = distances[distinct].min() if sum(distinct) > 0 else 0.1
    ell_u = distances[distinct].max() if sum(distinct) > 0 else 1
    ell_sigma = max(0.1, (ell_u - ell_l) / 6)
    ell_mu = ell_l + 3 * ell_sigma
    return ell_mu, ell_sigma


ell_mu, ell_sigma = [stat for stat in get_ell_prior(X_)]
```

```{code-cell} ipython3
def plot_inducing_points(ax):
    yl = ax.get_ylim()
    yu = -np.subtract(*yl) * 0.025 + yl[0]
    ax.plot(Xu, np.full(Xu.shape, yu), "xk", label="Inducing Points")
    ax.legend(loc="upper left")


def get_quantiles(samples, quantiles=[2.5, 50, 97.5]):
    return [np.percentile(samples, p, axis=0) for p in quantiles]


def plot_mean(ax, mean_samples):
    """Plots the median and 95% CI from samples of the mean

    Note that, although each individual GP exhibits a normal distribution at each point
    (by definition), we are sampling from a mixture of GPs defined by the posteriors of
    our hyperparameters. As such, we use percentiles rather than mean +/- stdev to
    represent the spread of predictions from our models.
    """
    l, m, u = get_quantiles(mean_samples)
    ax.plot(Xnew, m, "C0", label="Median")
    ax.fill_between(Xnew_, l, u, facecolor="C0", alpha=0.5, label="95% CI")

    ax.plot(Xnew, ynew, "--k", label="Mean Function")
    ax.plot(X, y, "C1.", label="Observed Means")
    ax.set_title("Mean Behavior")
    ax.legend(loc="upper left")


def plot_var(ax, var_samples):
    """Plots the median and 95% CI from samples of the variance"""
    if var_samples.squeeze().ndim == 1:
        ax.plot(Xnew, var_samples, "C0", label="Median")
    else:
        l, m, u = get_quantiles(var_samples)
        ax.plot(Xnew, m, "C0", label="Median")
        ax.fill_between(Xnew.flatten(), l, u, facecolor="C0", alpha=0.5, label="95% CI")
    ax.plot(Xnew, noise(signal(Xnew_)) ** 2, "--k", label="Noise Function")
    ax.plot(X, y_err**2, "C1.", label="Observed Variance")
    ax.set_title("Variance Behavior")
    ax.legend(loc="upper left")


def plot_total(ax, mean_samples, var_samples=None, bootstrap=True, n_boots=100):
    """Plots the overall mean and variance of the aggregate system

    We can represent the overall uncertainty via explicitly sampling the underlying normal
    distributrions (with `bootstrap=True`) or as the mean +/- the standard deviation from
    the Law of Total Variance. For systems with many observations, there will likely be
    little difference, but in cases with few observations and informative priors, plotting
    the percentiles will likely give a more accurate representation.
    """

    if (var_samples is None) or (var_samples.squeeze().ndim == 1):
        samples = mean_samples
        l, m, u = get_quantiles(samples)
        ax.plot(Xnew, m, "C0", label="Median")
    elif bootstrap:
        # Estimate the aggregate behavior using samples from each normal distribution in the posterior
        samples = (
            rng.normal(
                mean_samples.values.T[..., None],
                np.sqrt(var_samples.values).T[..., None],
                (*mean_samples.values.T.shape, n_boots),
            )
            .reshape(len(Xnew_), -1)
            .T
        )
        l, m, u = get_quantiles(samples)
        ax.plot(Xnew, m, "C0", label="Median")
    else:
        m = mean_samples.mean(axis=0)
        ax.plot(Xnew, m, "C0", label="Mean")
        sd = np.sqrt(mean_samples.var(axis=0) + var_samples.mean(axis=0))
        l, u = m - 2 * sd, m + 2 * sd

    ax.fill_between(Xnew.flatten(), l, u, facecolor="C0", alpha=0.5, label="Total 95% CI")

    ax.plot(Xnew, ynew, "--k", label="Mean Function")
    ax.plot(X_obs, y_obs_, "C1.", label="Observations")
    ax.set_title("Aggregate Behavior")
    ax.legend(loc="upper left")
```

## Homoskedastic GP

+++

First let's fit a standard homoskedastic GP using PyMC's `Marginal Likelihood` implementation. Here and throughout this notebook we'll use an informative prior for length scale as suggested by [Michael Betancourt](https://betanalpha.github.io/assets/case_studies/gp_part3/part3.html#4_adding_an_informative_prior_for_the_length_scale). We could use `pm.find_MAP()` and `.predict`for even faster inference and prediction, with similar results, but for direct comparison to the other models we'll use NUTS and `.conditional` instead, which run fast enough.

```{code-cell} ipython3
with pm.Model() as model_hm:
    ell = pm.InverseGamma("ell", mu=ell_mu, sigma=ell_sigma)
    eta = pm.Gamma("eta", alpha=2, beta=1)
    cov = eta**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ell)

    gp_hm = pm.gp.Marginal(cov_func=cov)

    sigma = pm.Exponential("sigma", lam=1)

    ml_hm = gp_hm.marginal_likelihood("ml_hm", X=X_obs, y=y_obs_, noise=sigma)

    trace_hm = pm.sample(random_seed=SEED)

with model_hm:
    mu_pred_hm = gp_hm.conditional("mu_pred_hm", Xnew=Xnew)
    noisy_pred_hm = gp_hm.conditional("noisy_pred_hm", Xnew=Xnew, pred_noise=True)
    pm.sample_posterior_predictive(
        trace_hm,
        var_names=["mu_pred_hm", "noisy_pred_hm"],
        extend_inferencedata=True,
        predictions=True,
    )
```

```{code-cell} ipython3
_, axs = plt.subplots(1, 3, figsize=(18, 4))
mu_samples = az.extract(trace_hm.predictions["mu_pred_hm"])["mu_pred_hm"]
noisy_samples = az.extract(trace_hm.predictions["noisy_pred_hm"])["noisy_pred_hm"]
plot_mean(axs[0], mu_samples.T)
plot_var(axs[1], noisy_samples.var(dim=["sample"]))
plot_total(axs[2], noisy_samples.T)
```

Here we've plotted our understanding of the mean behavior with the corresponding epistemic uncertainty on the left, our understanding of the variance or aleatoric uncertainty in the middle, and integrate all sources of uncertainty on the right. This model captures the mean behavior well, but we can see that it overestimates the noise in the lower regime while underestimating the noise in the upper regime, as expected.

+++

## Variance-weighted GP

+++

The simplest approach to modeling a heteroskedastic system is to fit a GP on the mean at each point along the domain and supply the standard deviation as weights.

```{code-cell} ipython3
with pm.Model() as model_wt:
    ell = pm.InverseGamma("ell", mu=ell_mu, sigma=ell_sigma)
    eta = pm.Gamma("eta", alpha=2, beta=1)
    cov = eta**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ell)

    gp_wt = pm.gp.Marginal(cov_func=cov)

    # Using the observed noise now
    ml_wt = gp_wt.marginal_likelihood("ml_wt", X=X, y=y, noise=y_err)

    trace_wt = pm.sample(return_inferencedata=True, random_seed=SEED)

with model_wt:
    mu_pred_wt = gp_wt.conditional("mu_pred_wt", Xnew=Xnew)
    pm.sample_posterior_predictive(
        trace_wt, var_names=["mu_pred_wt"], extend_inferencedata=True, predictions=True
    )
```

```{code-cell} ipython3
_, axs = plt.subplots(1, 3, figsize=(18, 4))
mu_samples = az.extract(trace_wt.predictions["mu_pred_wt"])["mu_pred_wt"]

plot_mean(axs[0], mu_samples.T)
axs[0].errorbar(X_, y, y_err, ls="none", color="C1", label="STDEV")
plot_var(axs[1], mu_samples.var(dim=["sample"]))
plot_total(axs[2], mu_samples.T)
```

This approach captured slightly more nuance in the overall uncertainty than the homoskedastic GP, but still underestimated the variance within both the observed regimes. Note that the variance displayed by this model is purely epistemic: our understanding of the mean behavior is weighted by the uncertainty in our observations, but we didn't include a component to account for aleatoric noise.

+++

## Heteroskedastic GP: latent variance model

+++

Now let's model the mean and the log of the variance as separate GPs through PyMC's `Latent` implementation, feeding both into a `Normal` likelihood. Note that we add a small amount of diagonal noise to the individual covariances in order to stabilize them for inversion.

The `Latent` parameterization takes signifiantly longer to sample than the `Marginal` model, so we are going to accerelate the sampling with the Nutpie NUTS sampler.

```{code-cell} ipython3
with pm.Model() as model_ht:
    ell = pm.InverseGamma("ell", mu=ell_mu, sigma=ell_sigma)
    eta = pm.Gamma("eta", alpha=2, beta=1)
    cov = eta**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ell) + pm.gp.cov.WhiteNoise(sigma=1e-6)

    gp_ht = pm.gp.Latent(cov_func=cov)
    mu_f = gp_ht.prior("mu_f", X=X_obs)

    sigma_ell = pm.InverseGamma("sigma_ell", mu=ell_mu, sigma=ell_sigma)
    sigma_eta = pm.Gamma("sigma_eta", alpha=2, beta=1)
    sigma_cov = sigma_eta**2 * pm.gp.cov.ExpQuad(
        input_dim=1, ls=sigma_ell
    ) + pm.gp.cov.WhiteNoise(sigma=1e-6)

    sigma_gp = pm.gp.Latent(cov_func=sigma_cov)
    lg_sigma_f = sigma_gp.prior("lg_sigma_f", X=X_obs)
    sigma_f = pm.Deterministic("sigma_f", pm.math.exp(lg_sigma_f))

    lik_ht = pm.Normal("lik_ht", mu=mu_f, sigma=sigma_f, observed=y_obs_)

    trace_ht = pm.sample(
        # target_accept=0.95,
        chains=2,
        nuts_sampler="numpyro",
        random_seed=SEED,
    )

with model_ht:
    mu_pred_ht = gp_ht.conditional("mu_pred_ht", Xnew=Xnew)
    lg_sigma_pred_ht = sigma_gp.conditional("lg_sigma_pred_ht", Xnew=Xnew)
    pm.sample_posterior_predictive(
        trace_ht,
        var_names=["mu_pred_ht", "lg_sigma_pred_ht"],
        extend_inferencedata=True,
        predictions=True,
    )
```

```{code-cell} ipython3
_, axs = plt.subplots(1, 3, figsize=(18, 4))
mu_samples = az.extract(trace_ht.predictions["mu_pred_ht"])["mu_pred_ht"]
sigma_samples = np.exp(az.extract(trace_ht.predictions["lg_sigma_pred_ht"])["lg_sigma_pred_ht"])

plot_mean(axs[0], mu_samples.T)
plot_var(axs[1], sigma_samples.T**2)
plot_total(axs[2], mu_samples.T, sigma_samples.T**2)
```

That looks much better! We've accurately captured the mean behavior of our system, as well as the underlying trend in the variance (with appropriate uncertainty). Crucially, the aggregate behavior of the model integrates both epistemic *and* aleatoric uncertainty, and the ~5% of our observations fall outside the 2σ band are more or less evenly distributed across the domain. However, even with the Nutpie sampler, this took nearly an hour on a Ryzen 7040 laptop to sample only 4k NUTS iterations. Due to the expense of the requisite matrix inversions, GPs are notoriously inefficient for large data sets. Let's reformulate this model using a sparse approximation.

+++

### Sparse Heteroskedastic GP

+++

Sparse approximations to GPs use a small set of *inducing points* to condition the model, vastly improve speed of inference and somewhat improving memory consumption. PyMC doesn't have an implementation for sparse latent GPs yet, but we can throw together our own real quick using Bill Engel's [DTC latent GP example](https://gist.github.com/bwengals/a0357d75d2083657a2eac85947381a44). These inducing points can be specified in a variety of ways, such as via the popular k-means initialization or even optimized as part of the model, but since our observations are evenly distributed we can make do with simply a subset of our unique input values.

```{code-cell} ipython3
class SparseLatent:
    def __init__(self, cov_func):
        self.cov = cov_func

    def prior(self, name, X, Xu):
        Kuu = self.cov(Xu)
        self.L = pt.linalg.cholesky(pm.gp.util.stabilize(Kuu))

        self.v = pm.Normal(f"u_rotated_{name}", mu=0.0, sigma=1.0, shape=len(Xu))
        self.u = pm.Deterministic(f"u_{name}", self.L @ self.v)

        Kfu = self.cov(X, Xu)
        self.Kuiu = pt.slinalg.solve_triangular(
            self.L.T, pt.slinalg.solve_triangular(self.L, self.u, lower=True), lower=False
        )
        self.mu = pm.Deterministic(f"mu_{name}", Kfu @ self.Kuiu)
        return self.mu

    def conditional(self, name, Xnew, Xu):
        Ksu = self.cov(Xnew, Xu)
        mus = Ksu @ self.Kuiu
        tmp = pt.slinalg.solve_triangular(self.L, Ksu.T, lower=True)
        Qss = tmp.T @ tmp
        Kss = self.cov(Xnew)
        Lss = pt.linalg.cholesky(pm.gp.util.stabilize(Kss - Qss))
        mu_pred = pm.MvNormal(name, mu=mus, chol=Lss, shape=len(Xnew))
        return mu_pred
```

```{code-cell} ipython3
# Explicitly specify inducing points by downsampling our input vector
Xu = X[1::2]

with pm.Model() as model_hts:
    ell = pm.InverseGamma("ell", mu=ell_mu, sigma=ell_sigma)
    eta = pm.Gamma("eta", alpha=2, beta=1)
    cov = eta**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ell)

    mu_gp = SparseLatent(cov)
    mu_f = mu_gp.prior("mu", X_obs, Xu)

    sigma_ell = pm.InverseGamma("sigma_ell", mu=ell_mu, sigma=ell_sigma)
    sigma_eta = pm.Gamma("sigma_eta", alpha=2, beta=1)
    sigma_cov = sigma_eta**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=sigma_ell)

    lg_sigma_gp = SparseLatent(sigma_cov)
    lg_sigma_f = lg_sigma_gp.prior("lg_sigma_f", X_obs, Xu)
    sigma_f = pm.Deterministic("sigma_f", pm.math.exp(lg_sigma_f))

    lik_hts = pm.Normal("lik_hts", mu=mu_f, sigma=sigma_f, observed=y_obs_)
    trace_hts = pm.sample(
        nuts_sampler="numpyro",
        chains=2,
        random_seed=SEED,
    )

with model_hts:
    mu_pred = mu_gp.conditional("mu_pred", Xnew, Xu)
    lg_sigma_pred = lg_sigma_gp.conditional("lg_sigma_pred", Xnew, Xu)
    pm.sample_posterior_predictive(
        trace_hts,
        var_names=["mu_pred", "lg_sigma_pred"],
        extend_inferencedata=True,
        predictions=True,
    )
```

```{code-cell} ipython3
_, axs = plt.subplots(1, 3, figsize=(18, 4))
mu_samples = az.extract(trace_hts.predictions["mu_pred"])["mu_pred"]
sigma_samples = np.exp(az.extract(trace_hts.predictions["lg_sigma_pred"])["lg_sigma_pred"])

plot_mean(axs[0], mu_samples.T)
plot_inducing_points(axs[0])
plot_var(axs[1], sigma_samples.T**2)
plot_inducing_points(axs[1])
plot_total(axs[2], mu_samples.T, sigma_samples.T**2)
plot_inducing_points(axs[2])
```

That was ~8x faster with nearly indistinguishable results, and fewer divergences as well.

+++

## Heteroskedastic GP with correlated noise and mean response: Linear Model of Coregionalization

+++

So far, we've modeled the mean and noise of our system as independent. However, there may be scenarios where we expect them to be correlated, for example if higher measurement values are expected to have greater noise. Here, we'll explicitly model this correlation through a covariance function that is a Kronecker product of the spatial kernel we've used previously and a `Coregion` kernel, as suggested by Bill Engels [here](https://discourse.pymc.io/t/coregionalization-model-for-two-separable-multidimensional-gaussian-process/2550/4). This is an implementation of the Linear Model of Coregionalization, which treats each correlated GP as a linear combination of a small number of independent basis functions, which are themselves GPs. We first add a categorical dimension to the domain of our observations to indicate whether the mean or variance is being considered, then unpack the respective components before feeding them into a `Normal` likelihood as above.

```{code-cell} ipython3
def add_coreg_idx(x):
    return np.hstack([np.tile(x, (2, 1)), np.vstack([np.zeros(x.shape), np.ones(x.shape)])])


Xu = X[1::2]
Xu_c, X_obs_c, Xnew_c = [add_coreg_idx(x) for x in [Xu, X_obs, Xnew]]
```

```{code-cell} ipython3
with pm.Model() as model_htsc:
    ell = pm.InverseGamma("ell", mu=ell_mu, sigma=ell_sigma, initval=1.0)
    eta = pm.Gamma("eta", alpha=2, beta=1, initval=1.0)
    eq_cov = eta**2 * pm.gp.cov.Matern52(input_dim=1, active_dims=[0], ls=ell)

    D_out = 2  # two output dimensions, mean and variance
    rank = 2  # two basis GPs
    W = pm.Normal("W", mu=0, sigma=3, shape=(D_out, rank))
    kappa = pm.Gamma("kappa", alpha=1.5, beta=1, shape=D_out)
    coreg = pm.gp.cov.Coregion(input_dim=1, active_dims=[0], kappa=kappa, W=W)

    cov = pm.gp.cov.Kron([eq_cov, coreg])

    gp_LMC = SparseLatent(cov)
    LMC_f = gp_LMC.prior("LMC", X_obs_c, Xu_c)

    # First half of LMC_f is the mean, second half is the log-scale
    mu_f = LMC_f[: len(y_obs_)]
    lg_sigma_f = LMC_f[len(y_obs_) :]
    sigma_f = pm.Deterministic("sigma_f", pm.math.exp(lg_sigma_f))

    lik_htsc = pm.Normal("lik_htsc", mu=mu_f, sigma=sigma_f, observed=y_obs_)
    trace_htsc = pm.sample(
        # target_accept=0.9,
        chains=2,
        nuts_sampler="numpyro",
        return_inferencedata=True,
        random_seed=SEED,
    )

with model_htsc:
    c_mu_pred = gp_LMC.conditional("c_mu_pred", Xnew_c, Xu_c)
    pm.sample_posterior_predictive(
        trace_htsc, var_names=["c_mu_pred"], extend_inferencedata=True, predictions=True
    )
```

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
with pm.Model() as model_htsc:
    ell = pm.InverseGamma("ell", mu=ell_mu, sigma=ell_sigma, initval=1.0)
    # eta = pm.Gamma("eta", alpha=2, beta=1, initval=1.0)
    eta = pm.HalfNormal("eta", sigma=3.0)
    eq_cov = eta**2 * pm.gp.cov.Matern52(input_dim=2, active_dims=[0], ls=ell)

    D_out = 2  # two output dimensions, mean and variance
    rank = 2  # two basis GPs
    W = pm.Normal("W", mu=0, sigma=1, shape=(D_out, rank))
    kappa = pm.Gamma("kappa", alpha=1.5, beta=1, shape=D_out)
    B = pm.Deterministic("B", pt.dot(W, W.T) + pt.diag(kappa))
    cov_icm = get_icm(input_dim=2, kernel=eq_cov, B=B, active_dims=[1])

    gp_LMC = SparseLatent(cov_icm)
    LMC_f = gp_LMC.prior("LMC", X_obs_c, Xu_c)

    # First half of LMC_f is the mean, second half is the log-scale
    mu_f = LMC_f[: len(y_obs_)]
    lg_sigma_f = LMC_f[len(y_obs_) :]
    sigma_f = pm.Deterministic("sigma_f", pm.math.exp(lg_sigma_f))

    lik_htsc = pm.Normal("lik_htsc", mu=mu_f, sigma=sigma_f, observed=y_obs_)

    trace_htsc = pm.sample(
        target_accept=0.95,
        chains=2,
        nuts_sampler="numpyro",
        return_inferencedata=True,
        random_seed=SEED,
    )

    c_mu_pred = gp_LMC.conditional("c_mu_pred", Xnew_c, Xu_c)
    pm.sample_posterior_predictive(
        trace_htsc, var_names=["c_mu_pred"], extend_inferencedata=True, predictions=True
    )
```

```{code-cell} ipython3
az.plot_trace(trace_htsc.sel(chain=[0]), var_names=["ell", "eta", "kappa"])
```

```{code-cell} ipython3
trace_htsc.predictions["c_mu_pred"].shape
az.extract(trace_htsc.predictions)["c_mu_pred"].shape
```

```{code-cell} ipython3
# μ_samples = samples_htsc["c_mu_pred"][:, : len(Xnew)]
# σ_samples = np.exp(samples_htsc["c_mu_pred"][:, len(Xnew) :])
samples_htsc = az.extract(trace_htsc.predictions)["c_mu_pred"]
mu_samples = samples_htsc[: len(Xnew)]
sigma_samples = np.exp(samples_htsc[len(Xnew) :])

_, axs = plt.subplots(1, 3, figsize=(18, 4))
# plot_mean(axs[0], mu_samples.T)
# plot_inducing_points(axs[0])
# plot_var(axs[1], sigma_samples.T**2)
# axs[1].set_ylim(-0.01, 0.2)
# axs[1].legend(loc="upper left")
# plot_inducing_points(axs[1])
# plot_total(axs[2], mu_samples.T, sigma_samples.T**2)
# plot_inducing_points(axs[2])
plot_mean(axs[0], mu_samples.T)
plot_inducing_points(axs[0])
plot_var(axs[1], sigma_samples.T**2)
plot_inducing_points(axs[1])
plot_total(axs[2], mu_samples.T, sigma_samples.T**2)
plot_inducing_points(axs[2])
```

```{code-cell} ipython3
# μ_samples = samples_htsc[:, : len(Xnew)]
# σ_samples = np.exp(samples_htsc[:, len(Xnew) :])

# _, axs = plt.subplots(1, 3, figsize=(18, 4))
# plot_mean(axs[0], μ_samples)
# plot_inducing_points(axs[0])
# plot_var(axs[1], σ_samples**2)
# axs[1].set_ylim(-0.01, 0.2)
# axs[1].legend(loc="upper left")
# plot_inducing_points(axs[1])
# plot_total(axs[2], μ_samples, σ_samples**2)
# plot_inducing_points(axs[2])
```

We can look at the learned correlation between the mean and variance by inspecting the covariance matrix $\bf{B}$ constructed via $\mathbf{B} \equiv \mathbf{WW}^T+diag(\kappa)$:

```{code-cell} ipython3
with model_htsc:
    B_samples = pm.sample_posterior_predictive(trace_htsc, var_names=["W", "kappa"])
```

```{code-cell} ipython3
# Keep in mind that the first dimension in all arrays is the sampling dimension
W = az.extract(B_samples.posterior_predictive["W"])["W"].values.T
W_T = np.swapaxes(W, 1, 2)
WW_T = np.matmul(W, W_T)

kappa = az.extract(B_samples.posterior_predictive["kappa"])["kappa"].values.T
I = np.tile(np.identity(2), [kappa.shape[0], 1, 1])
# einsum is just a concise way of doing multiplication and summation over arbitrary axes
diag_kappa = np.einsum("ij,ijk->ijk", kappa, I)

B = WW_T + diag_kappa
B.mean(axis=0)
```

```{code-cell} ipython3
sd = np.sqrt(np.diagonal(B, axis1=1, axis2=2))
outer_sd = np.einsum("ij,ik->ijk", sd, sd)
correlation = B / outer_sd
print(f"2.5%ile correlation: {np.percentile(correlation,2.5,axis=0)[0,1]:0.3f}")
print(f"Median correlation: {np.percentile(correlation,50,axis=0)[0,1]:0.3f}")
print(f"97.5%ile correlation: {np.percentile(correlation,97.5,axis=0)[0,1]:0.3f}")
```

The model has inadvertently learned that the mean and noise are slightly negatively correlated, albeit with a wide credible interval.

+++

## Comparison

+++

The three latent approaches shown here varied in their complexity and efficiency, but ultimately produced very similar regression surfaces, as shown below. All three displayed a nuanced understanding of both aleatoric and epistemic uncertainties. It's worth noting that we had to increase `target_accept` from the default 0.8 to 0.95 to avoid an excessive number of divergences, but this has the downside of slowing down NUTS evaluations. Sampling times could be decreased by reducing `target_accept`, at the expense of potentially biased inference due to divergences, or by further reducing the number of inducing points used in the sparse approximations. Inspecting the convergence statistics for each method, all had low r_hat values of 1.01 or below but the LMC model showed low effective sample sizes for some parameters, in particular the `ess_tail` for the η and ℓ parameters. To have confidence in the 95% CI bounds for this model, we should run the sampling for more iterations, ideally at least until the smallest `ess_tail` is above 200 but the higher the better.

+++

### Regression surfaces

```{code-cell} ipython3
trace_ht.predictions["lg_sigma_pred_ht"].shape
```

```{code-cell} ipython3
_, axs = plt.subplots(1, 3, figsize=(18, 4))

mu_samples = trace_ht.predictions["mu_pred_ht"]
sigma_samples = np.exp(trace_ht.predictions["lg_sigma_pred_ht"])
plot_total(axs[0], mu_samples, sigma_samples**2)
axs[0].set_title("Latent")

mu_samples = trace_hts.predictions["mu_pred"]
sigma_samples = np.exp(trace_hts.predictions["lg_sigma_pred"])
plot_total(axs[1], mu_samples, sigma_samples**2)
axs[1].set_title("Sparse Latent")

mu_samples = trace_htsc.predictions["c_mu_pred"][:, :100]
sigma_samples = np.exp(trace_htsc.predictions["c_mu_pred"][:, 100:])
plot_total(axs[2], mu_samples, sigma_samples**2)
axs[2].set_title("Correlated Sparse Latent")

yls = [ax.get_ylim() for ax in axs]
yl = [np.min([l[0] for l in yls]), np.max([l[1] for l in yls])]
for ax in axs:
    ax.set_ylim(yl)

plot_inducing_points(axs[1])
plot_inducing_points(axs[2])

axs[0].legend().remove()
axs[1].legend().remove()
```

### Latent model convergence

```{code-cell} ipython3
display(az.summary(trace_ht).sort_values("ess_bulk").iloc[:5])
```

### Sparse Latent model convergence

```{code-cell} ipython3
display(az.summary(trace_hts).sort_values("ess_bulk").iloc[:5])
```

### Correlated Sparse Latent model convergence

```{code-cell} ipython3
display(az.summary(trace_htsc).sort_values("ess_bulk").iloc[:5])
```

* This notebook was written by John Goertz on 5 May, 2021.

```{code-cell} ipython3
%watermark -n -u -v -iv -w -p xarray
```
