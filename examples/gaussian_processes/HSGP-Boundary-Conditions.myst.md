---
jupytext:
  cell_metadata_filter: -all
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(hsgp-boundary-conditions)=
# Boundary Conditions in the Hilbert Space GP Approximation

:::{post} September 30, 2026
:tags: gaussian process, hilbert space approximation, forecasting
:category: intermediate
:author: Alexandre Andorra, Bill Engels, Ben Mares, Juan Orduz
:::

A Gaussian process (GP) prior describes a distribution over functions before we observe data.
Its covariance function controls how function values vary together.
For a **stationary** GP, this relationship depends on the separation between inputs, not their absolute positions.

The Hilbert space Gaussian process approximation (HSGP) replaces the GP with a finite sum of fixed curves, called **basis functions**.
Random weights combine these curves into possible functions.
This representation can reduce computation when the number of basis functions is much smaller than the number of observations.

One constraint on this approach is that these curves need a finite interval and a rule at each endpoint.
Those rules are **boundary conditions**, and they affect both the function and its uncertainty near the endpoints.
This notebook explains the four choices in {class}`pymc.gp.HSGP` and compares interpolation, known physical constraints, and short-range forecasts.

For HSGP background, see {ref}`hsgp`.
The approximation follows {cite:t}`solin2020Hilbert` and {cite:t}`riutort2022PracticalHilbertSpaceApproximate`.

```{code-cell} ipython3
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from pymc.gp.hsgp_approx import calc_eigenvalues, calc_eigenvectors
```

```{code-cell} ipython3
az.style.use("arviz-darkgrid")
plt.rcParams["figure.figsize"] = [11, 4]
plt.rcParams["figure.dpi"] = 100
%config InlineBackend.figure_format = "retina"

seed = 42
rng = np.random.default_rng(seed)
```

## Boundary conditions

### Dirichlet and Neumann conditions

Imagine a curve on a finite interval.
At both endpoints, we can force its height to zero or force its slope to zero.
The rule applies to every curve that the HSGP can represent, before and after we observe data.

- A **Dirichlet** condition fixes the function value to zero.
  Every possible curve passes through zero at that endpoint.
- A **Neumann** condition fixes the derivative, or slope, to zero.
  Every possible curve has a horizontal tangent there, but its height remains uncertain.

These are hard constraints, so no amount of data can override them

The models in this notebook all use a zero mean function, $\mu(x)=0$, so the prior expected value is zero at each input, but individual function draws need not equal zero.

With a nonzero mean, write $f(x)=\mu(x)+g(x)$, where $g(x)$ is the weighted basis sum.
Dirichlet fixes $g=0$ at the endpoint, so $f=\mu$ there.
Neumann fixes $g'=0$ at the endpoint, so $f'=\mu'$ there.


| `boundary` | Lower endpoint | Upper endpoint |
|---|---|---|
| `"dirichlet"` (default) | Zero value | Zero value |
| `"neumann"` | Zero slope | Zero slope |
| `"dirichlet-neumann"` | Zero value | Zero slope |
| `"neumann-dirichlet"` | Zero slope | Zero value |

### Approximation domain

In one dimension, the HSGP interval has a center and a half-width $L$.
PyMC centers the interval on the midpoint of the inputs supplied when it first builds the basis.
The notation $[-L, L]$ describes coordinates relative to that center.

So for example, inputs spanning $[0, 1]$, the center is $0.5$.
With $c = 1.5$, the half-width is $L = 0.75$, so the interval in original coordinates is $[-0.25, 1.25]$.
Note that the constraints act at $-0.25$ and $1.25$, not at the first and last observations.

The three hyperparameters have different roles:

- `L` sets the distance from the center to each endpoint.
  Alternatively, `c` sets $L$ as a multiple of the input half-range.
- `m` sets the number of basis functions in one dimension.
  More functions can represent finer variation, but they do not remove the endpoint constraint.
- `boundary` sets the endpoint behavior.
  It does not change the interval width.

If the endpoints only support the numerical approximation, place them beyond the inputs where you need accurate predictions.
If they represent known physical boundaries, place them at those boundaries instead.
The radial-profile model in **Radial profiles with physical constraints** places its interval endpoints at $r=0$ and $r=1$.
Its observations cover only $r\in[0.1,0.7]$.

### HSGP basis functions

HSGP essentially turns the Gaussian process into a linear regression with fixed curves instead of ordinary predictors.
It assigns a random coefficient to each curve and adds the weighted curves together:

$$
f(x) \approx \sum_{j \in J} \sqrt{S(\omega_j)}\,\beta_j\,\phi_j(x),
\qquad \beta_j \overset{\mathrm{iid}}{\sim} \mathcal{N}(0, 1).
$$

Here, $\phi_j$ is a basis function, $\beta_j$ is its independent standard Normal coefficient, and $\omega_j$ is its frequency.
The **spectral density** $S$ comes from the covariance function.
It controls the variance assigned to each frequency, so the chosen kernel still controls the scale and smoothness of variation.

The basis functions are eigenfunctions of the Laplacian.
In this one-dimensional example, they are sine or cosine curves that satisfy the chosen endpoint rules.
Happily, we do not need to solve this eigenfunction problem ourselves because PyMC supplies the basis.

For the sine and cosine bases in the table, the Laplacian eigenfunction equation is $-\phi_j''(x)=\lambda_j\phi_j(x)$.
The eigenvalue $\lambda_j$ is the squared frequency of basis function $j$.

If every basis function is zero at an endpoint, every weighted sum is also zero there (the same applies to zero derivatives).
This explains why learning the coefficients cannot remove the constraint.

For readers who want the formulae, the frequency and eigenvalue obey the same relation for all four choices:

$$
\omega_j = \frac{\pi j}{2L}, \qquad \lambda_j = \omega_j^2.
$$

The allowed indices $J$ and the sine or cosine basis differ:

| `boundary` | Basis function $\phi_j(x)$ | Indices $J$ for $m$ functions |
|---|---|---|
| `"dirichlet"` | $\sin(\omega_j(x + L))/\sqrt{L}$ | $1, 2, \ldots, m$ |
| `"neumann"` | $\cos(\omega_j(x + L))/\sqrt{L}$ for $j > 0$ | $0, 1, \ldots, m-1$ |
| `"dirichlet-neumann"` | $\sin(\omega_j(x + L))/\sqrt{L}$ | $\tfrac12, \tfrac32, \ldots, m-\tfrac12$ |
| `"neumann-dirichlet"` | $\cos(\omega_j(x + L))/\sqrt{L}$ | $\tfrac12, \tfrac32, \ldots, m-\tfrac12$ |

For the Neumann basis, $j = 0$ is a constant function with a different normalization: $\phi_0(x) = 1/\sqrt{2L}$.
Its random coefficient lets the overall level vary while the slope stays zero at both ends.
Its contribution to the variance at any input is $S(0)/(2L)$.

The plot below shows the first four basis functions for Dirichlet, Neumann, and Neumann–Dirichlet.
These are the **fixed curves** that the random coefficients combine, not sampled GP functions.

At each Dirichlet endpoint in the basis plot, all four basis functions equal zero.
At each Neumann endpoint, all four derivatives equal zero, but the function values differ.

Dirichlet–Neumann just flips the endpoint roles shown in the third panel.
The fitting comparisons use `"neumann-dirichlet"` rather than repeat both orientations.

```{code-cell} ipython3
BOUNDARIES = ["dirichlet", "neumann", "neumann-dirichlet"]

L_demo = np.array([1.0])
x_demo = np.linspace(-1.0, 1.0, 300)[:, None]

fig, axes = plt.subplots(1, 3, sharey=True)
for ax, boundary in zip(axes, BOUNDARIES):
    eigvals = calc_eigenvalues(L_demo, [4], boundary=boundary)
    phi = calc_eigenvectors(x_demo, L_demo, eigvals, [4], boundary=boundary).eval()
    ax.plot(x_demo, phi)
    ax.set(title=boundary, xlabel="$x$")
axes[0].set(ylabel=r"$\phi_j(x)$")
fig.suptitle("First four basis functions on $[-L, L]$, $L = 1$", y=1.02)
```

## Prior uncertainty near boundaries

A GP prior describes uncertainty before we observe data.
Its pointwise standard deviation measures how much possible function values vary at each input.
This is uncertainty about the latent function, not observation noise.

The prior-standard-deviation plot below uses the squared-exponential kernel (`ExpQuad`), $L=5$, $m=200$, and $\ell=1$.
Its horizontal axis spans the full approximation interval, including both constrained endpoints.
The lengthscale $\ell=1$ sets the distance over which function values remain strongly correlated.

Here, we set the exact GP's signal standard deviation to $1$.
Stationarity makes that standard deviation constant across inputs.

The HSGP constraints break that stationarity near the endpoints.
We compute the standard deviation directly from the basis weights, without sampling or fitting a model.

```{code-cell} ipython3
def prior_sd(boundary, X, L, m, ls):
    with pm.Model():
        gp = pm.gp.HSGP(m=[m], L=[L], boundary=boundary, cov_func=pm.gp.cov.ExpQuad(1, ls=ls))
        phi, sqrt_psd = gp.prior_linearized(X)
    return np.sqrt(((phi**2) * sqrt_psd**2).sum(axis=1).eval())


X_prior = np.linspace(-5.0, 5.0, 201)[:, None]
fig, ax = plt.subplots()
for boundary in BOUNDARIES:
    ax.plot(X_prior, prior_sd(boundary, X_prior, L=5.0, m=200, ls=1.0), label=boundary)
ax.axhline(1.0, color="k", ls="--", label="exact GP")
ax.set(xlabel="$x$", ylabel="prior sd", title=r"Prior sd, ExpQuad with $\ell = 1$, $L = 5$")
ax.legend()
```

Notice a few things:

1. Near $x=0$, all three HSGP curves approach the exact GP's standard deviation of $1$.
   For `ExpQuad` with $L=5$ and $\ell=1$, visible differences concentrate within about $1.5$ input units of an endpoint.
   But, this distance is not a universal cutoff for other kernels or lengthscales.
2. A Dirichlet endpoint has zero function variance because every possible function has the same value there.
   A Neumann endpoint fixes the slope instead, so its function value remains uncertain.
   For this `ExpQuad` calculation, the Neumann endpoint variance approaches $2$, so its standard deviation approaches $\sqrt{2}$.
3. Neither constraint reproduces the unconstrained GP at the endpoint.
   Increasing `m` refines the basis but cannot remove this difference.
   To approximate the unconstrained GP over a wider prediction range, increase the interval width and reconsider `m` together.

Reflecting an input across an interval endpoint helps describe how these constraints change covariance.
With `ExpQuad`, a nearby reflected input adds a covariance contribution for Neumann and subtracts one for Dirichlet.
More distant reflections contribute less when the interval spans many lengthscales.

In the above, the interval width is ten lengthscales, which makes the influence of the opposite endpoint small near either boundary.
Keep in mind that a boundary condition is just a modeling assumption near the edge, not an improvement in approximation accuracy!

+++

## Model fitting and evaluation

The interpolation, radial-profile, and forecasting comparisons use some fitting and evaluation helper functions specified below below.
Each comparison includes an exact GP without endpoint constraints as a reference.
Here, *exact* describes the covariance calculation, not a guarantee that the model matches the physical process.

The figures and tables use different inference methods:

- The **figures** use NUTS sampling to estimate uncertainty in the covariance parameters and latent function.
  The parameters are the lengthscale $\ell$, signal amplitude $\eta$, and observation-noise standard deviation $\sigma_\text{noise}$.
- The **tables** repeat each comparison over 100 simulated functions.
  For each function and model, a grid search selects the parameter values with the largest marginal likelihood.
  A closed-form Gaussian calculation then gives the conditional posterior at those selected values.

The table method is **type-II maximum likelihood**.
Unlike the NUTS fits, it does not integrate over parameter uncertainty, so the two methods need not produce identical uncertainty intervals.
This repetition helps us fairly compare methods across functions rather than judge them from one favorable draw.

The tables report root mean square error (RMSE), coverage, and the latent standard deviation at the last prediction input, labeled `sd(last)`.

The reported standard deviation measures uncertainty, not error.
A small value could reflect either precise information or an overzealous constraint.

You want to choose `m` according to the plausible lengthscale range, rather than making it arbitrarily large.
Very high frequencies can receive spectral weights that round to zero, which can cause invalid gradients during NUTS sampling. Here, the interpolation and forecasting fits use $m=15$. The radial-profile fit uses $m=6$ and endpoints fixed at $r=0$ and $r=1$.

Pro tip: use `pm.gp.hsgp_approx.approx_hsgp_hyperparams` to select initial values of `m` and interval width for your input and lengthscale ranges.
Include the intended prediction inputs, not just the observations.

The settings `m` and `L` control the largest basis frequency through $\omega_j=\pi j/(2L)$.
Increasing `L` without increasing `m` reduces the highest frequency represented by the basis.

```{code-cell} ipython3
def hyperpriors():
    ell = pm.InverseGamma("ell", mu=0.4, sigma=0.2)
    eta = pm.HalfNormal("eta", sigma=1.5)
    sigma = pm.HalfNormal("sigma", sigma=0.5)
    return eta**2 * pm.gp.cov.ExpQuad(1, ls=ell), sigma


NUTS = {
    "draws": 1000,
    "tune": 1000,
    "chains": 4,
    "target_accept": 0.99,
    "random_seed": seed,
    "progressbar": False,
}
PPC = {"random_seed": seed, "progressbar": False}


def stack(ds, name):
    return ds[name].stack(sample=("chain", "draw")).values


def fit_hsgp(boundary, X, y, X_new, m=15, c=1.5):
    with pm.Model():
        cov_func, sigma = hyperpriors()
        gp = pm.gp.HSGP(m=[m], c=c, boundary=boundary, cov_func=cov_func)
        f = gp.prior("f", X=X)
        pm.Normal("y", mu=f, sigma=sigma, observed=y)
        idata = pm.sample(**NUTS)
        gp.conditional("f_new", Xnew=X_new)
        ppc = pm.sample_posterior_predictive(idata, var_names=["f_new"], **PPC)
    return stack(ppc.posterior_predictive, "f_new")


def fit_exact(X, y, X_new):
    with pm.Model():
        cov_func, sigma = hyperpriors()
        gp = pm.gp.Marginal(cov_func=cov_func)
        gp.marginal_likelihood("y", X=X, y=y, sigma=sigma)
        idata = pm.sample(**NUTS)
        gp.conditional("f_new", Xnew=X_new)
        ppc = pm.sample_posterior_predictive(idata, var_names=["f_new"], **PPC)
    return stack(ppc.posterior_predictive, "f_new")


def plot_results(results, x, y, x_all, f_all, x_new, title, vlines=()):
    fig, ax = plt.subplots()
    ax.plot(x, y, "k.", ms=4, label="data")
    ax.plot(x_all, f_all, "k-", lw=1, label="truth")
    for name, s in results.items():
        mu, lo, hi = s.mean(1), np.quantile(s, 0.05, axis=1), np.quantile(s, 0.95, axis=1)
        ax.plot(x_new, mu, label=name)
        ax.fill_between(x_new, lo, hi, alpha=0.15)
    for v, lab in vlines:
        ax.axvline(v, color="gray", ls=":", label=lab)
    ax.set(title=title, xlabel="$x$")
    ax.legend(loc="upper left", fontsize=8)


# --- closed-form machinery for the replicated tables -------------------------------------
def k_expquad(a, b, eta, ell):
    return eta**2 * np.exp(-0.5 * ((a[:, None] - b[None, :]) / ell) ** 2)


FIRST_INDEX = {"dirichlet": 1.0, "neumann": 0.0, "dirichlet-neumann": 0.5, "neumann-dirichlet": 0.5}


def basis_np(boundary, x, L, m):
    """Same eigenbasis as `pymc.gp.hsgp_approx.calc_eigenvectors`, in numpy (fast for replication)."""
    j = FIRST_INDEX[boundary] + np.arange(m)
    omega = np.pi * j / (2 * L)
    trig = np.sin if boundary.startswith("dirichlet") else np.cos
    phi = trig(omega[None, :] * (x[:, None] + L)) / np.sqrt(L)
    phi[:, omega == 0] = 1.0 / np.sqrt(2 * L)
    return phi, omega


def hsgp_kernel(boundary, a, b, L, m, eta, ell):
    """Closed-form HSGP ExpQuad covariance between inputs a and b (already centred on the box)."""
    phi_a, omega = basis_np(boundary, a, L, m)
    phi_b, _ = basis_np(boundary, b, L, m)
    psd = eta**2 * np.sqrt(2 * np.pi) * ell * np.exp(-0.5 * (ell * omega) ** 2)
    return (phi_a * psd) @ phi_b.T


# self-check: the numpy basis is the library basis
for _b in FIRST_INDEX:
    _ev = calc_eigenvalues(np.array([0.75]), [8], boundary=_b)
    _phi = calc_eigenvectors(x_demo, np.array([0.75]), _ev, [8], boundary=_b).eval()
    np.testing.assert_allclose(_phi, basis_np(_b, x_demo[:, 0], 0.75, 8)[0], atol=1e-12)


def gp_posterior(K_xx, K_sx, K_ss, y, noise):
    A = K_xx + noise**2 * np.eye(len(y))
    Ainv_y = np.linalg.solve(A, y)
    mu = K_sx @ Ainv_y
    cov = K_ss - K_sx @ np.linalg.solve(A, K_sx.T)
    return mu, np.sqrt(np.clip(np.diag(cov), 0.0, None))


def log_marginal(K, y, noise):
    A = K + noise**2 * np.eye(len(y))
    with np.errstate(all="ignore"):
        sign, logdet = np.linalg.slogdet(A)
    if sign <= 0 or not np.isfinite(logdet):
        return -np.inf  # numerically singular: never the type-II optimum
    return -0.5 * (logdet + y @ np.linalg.solve(A, y))


ELL_GRID = np.exp(np.linspace(np.log(0.08), np.log(2.0), 18))
ETA_GRID = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
NOISE_GRID = np.array([0.15, 0.2, 0.3, 0.45, 0.6])


def fit_type2(kernel, x_obs, y, x_pred):
    """Type-II ML over the grid, then the closed-form posterior at x_pred. `kernel(a, b, eta, ell)`."""
    best, best_hp = -np.inf, None
    for ell in ELL_GRID:
        for eta in ETA_GRID:
            K = kernel(x_obs, x_obs, eta, ell)
            for noise in NOISE_GRID:
                v = log_marginal(K, y, noise)
                if v > best:
                    best, best_hp = v, (eta, ell, noise)
    eta, ell, noise = best_hp
    return gp_posterior(
        kernel(x_obs, x_obs, eta, ell),
        kernel(x_pred, x_obs, eta, ell),
        kernel(x_pred, x_pred, eta, ell),
        y,
        noise,
    )


def replicated_table(models, draw_truth, x_obs, x_pred, noise_true, n_rep=100):
    """models: {name: kernel(a, b, eta, ell)}; draw_truth() -> f on np.r_[x_obs, x_pred]."""
    rows = {name: [] for name in models}
    for _ in range(n_rep):
        f_all = draw_truth()
        f_obs, f_pred = f_all[: len(x_obs)], f_all[len(x_obs) :]
        y = f_obs + noise_true * rng.standard_normal(len(x_obs))
        for name, kernel in models.items():
            mu, sd = fit_type2(kernel, x_obs, y, x_pred)
            rows[name].append(
                (
                    np.sqrt(np.mean((mu - f_pred) ** 2)),
                    np.mean(np.abs(mu - f_pred) < 1.645 * sd),
                    sd[-1],
                )
            )
    print(f"{'model':>20} {'RMSE':>7} {'90% cov':>8} {'sd(last)':>10}   ({n_rep} replications)")
    for name, r in rows.items():
        r = np.array(r)
        print(f"{name:>20} {r[:, 0].mean():7.3f} {r[:, 1].mean():8.2f} {r[:, 2].mean():10.3f}")


def draw_gp(x, ell, eta=1.0):
    K = k_expquad(x, x, eta, ell)
    return np.linalg.cholesky(K + 1e-9 * np.eye(len(x))) @ rng.standard_normal(len(x))
```

## Interpolation within the observed range

We first simulate 60 noisy observations on $[0, 1]$ and predict the latent function over the same interval.
We use $c = 1.5$, which places the approximation boundaries at $-0.25$ and $1.25$.
The nearest observation is $0.25$ from a boundary, less than the generating lengthscale of $0.4$.

The interpolation comparison tests whether nearby observations reduce differences between the three HSGP boundary choices.
The exact GP supplies a fourth fitted model without endpoint constraints.
Compare the fitted curves and 90% intervals over $[0,1]$, then compare RMSE and coverage in the repeated-study table.

```{code-cell} ipython3
ell_true, noise_true = 0.4, 0.3
x = np.linspace(0.0, 1.0, 60)
f = draw_gp(x, ell_true)
y = f + noise_true * rng.standard_normal(len(x))
X = x[:, None]

results_in = {b: fit_hsgp(b, X, y, X) for b in BOUNDARIES}
results_in["exact GP"] = fit_exact(X, y, X)
plot_results(results_in, x, y, x, f, x, "In-sample fit: every boundary condition agrees")
```

```{code-cell} ipython3
box = {"L": 0.75, "m": 15}
models_in = {
    b: (lambda a, bb, eta, ell, b=b: hsgp_kernel(b, a - 0.5, bb - 0.5, eta=eta, ell=ell, **box))
    for b in BOUNDARIES
}
models_in["exact GP"] = k_expquad
replicated_table(models_in, lambda: draw_gp(np.r_[x, x], ell_true), x, x, noise_true)
```

For the 60-point interpolation data set, the observations constrain function values throughout $[0,1]$.
Agreement between these four fitted models does not establish accuracy for sparse data, larger noise, or different lengthscales.

For interpolation, start with the default `"dirichlet"` boundary and an interval that extends beyond the observation range.

+++

## Radial profiles with physical constraints

Consider a radial profile $f(r)$ on $[0, R]$, such as brightness or concentration around a source.
Symmetry about the origin gives $f'(0) = 0$.
After background subtraction, a known background level at the outer edge gives $f(R) = 0$.

The radial profile has outer radius $R=1$, and observations cover only $r\in[0.1,0.7]$.
Its unobserved regions are $[0,0.1)$ and $(0.7,1]$.
The goal is to reconstruct the unobserved ends using physical knowledge, not just to approximate an unconstrained GP.

A larger Dirichlet interval moves its zero-value constraints elsewhere, but it does not impose zero slope at the physical origin.
A Dirichlet weighted sum can happen to have zero slope there, but that property does not hold for every possible function.

With `"neumann-dirichlet"` and endpoints at $0$ and $1$, every represented function satisfies both physical constraints.
We build the basis with `prior_linearized` on a grid that spans the full physical domain.
We then use only observed rows in the likelihood.

The full-domain grid centers the basis at $r=0.5$ with $L=0.5$, which places the endpoints at $r=0$ and $r=1$.
For this grid, $L=0.5$ equals $c=1$ times the half-range.
Using `c=1` on the observed inputs alone would instead place the boundaries at $0.1$ and $0.7$.

We generate two sets of radial profiles.
Both sets satisfy $f'(0)=0$.
The first also fixes $f(1)=0$, while the second allows $f(1)\sim\mathcal{N}(0,0.3^2)$.

The HSGP still fixes $f(1)=0$ in both comparisons.
In the second comparison, its assumption conflicts with the simulated endpoint values.

```{code-cell} ipython3
def draw_constrained_profile(r, ell, eta=1.0, edge_sd=0.0):
    """GP draw on r with f'(0) = 0 exactly and f(1) ~ N(0, edge_sd^2) (edge_sd=0: exactly zero)."""

    def k(a, b):
        return eta**2 * np.exp(-0.5 * ((a - b) / ell) ** 2)

    def dk(a, b):  # cov(f(a), f'(b)) = d/db k(a, b)
        return k(a, b) * (a - b) / ell**2

    def ddk(a, b):  # cov(f'(a), f'(b))
        return k(a, b) * (1 / ell**2 - (a - b) ** 2 / ell**4)

    Kff = k(r[:, None], r[None, :])
    C = np.c_[dk(r, 0.0), k(r, 1.0)]  # cov(f(r), [f'(0), f(1)])
    Kcc = np.array([[ddk(0.0, 0.0), dk(1.0, 0.0)], [dk(1.0, 0.0), k(1.0, 1.0)]])
    target = np.array([0.0, edge_sd * rng.standard_normal()])  # f'(0) = 0, f(1) = small value
    mean = C @ np.linalg.solve(Kcc, target)
    Kc = Kff - C @ np.linalg.solve(Kcc, C.T)
    return mean + np.linalg.cholesky(Kc + 1e-9 * np.eye(len(r))) @ rng.standard_normal(len(r))


r_grid = np.linspace(0.0, 1.0, 101)
obs = (r_grid >= 0.1) & (r_grid <= 0.7)
f_r = draw_constrained_profile(r_grid, ell=0.5)
y_r = f_r[obs] + 0.2 * rng.standard_normal(obs.sum())


def fit_hsgp_on_grid(boundary, r_grid, obs, y, m=6):
    with pm.Model():
        cov_func, sigma = hyperpriors()
        gp = pm.gp.HSGP(m=[m], L=[0.5], boundary=boundary, cov_func=cov_func)
        phi, sqrt_psd = gp.prior_linearized(r_grid[:, None])
        beta = pm.Normal("beta", size=gp.n_basis_vectors)
        f_new = pm.Deterministic("f_new", phi @ (beta * sqrt_psd))
        pm.Normal("y", mu=f_new[obs], sigma=sigma, observed=y)
        idata = pm.sample(**NUTS)
    return stack(idata.posterior, "f_new")


RADIAL = ["dirichlet", "neumann", "neumann-dirichlet", "dirichlet-neumann"]
results_r = {b: fit_hsgp_on_grid(b, r_grid, obs, y_r) for b in RADIAL}
results_r["exact GP"] = fit_exact(r_grid[obs][:, None], y_r, r_grid[:, None])
plot_results(
    results_r,
    r_grid[obs],
    y_r,
    r_grid,
    f_r,
    r_grid,
    "Radial profile: $f'(0) = 0$, $f(R) = 0$, observed on $[0.1, 0.7]$ only ($m = 6$, $c = 1$)",
    vlines=[(0.0, "$r = 0$ (Neumann end)"), (1.0, "$r = R$ (Dirichlet end)")],
)
```

```{code-cell} ipython3
r_obs, r_pred = r_grid[obs], r_grid[~obs]  # metrics on the unobserved part of the domain
models_r = {
    b: (
        lambda a, bb, eta, ell, b=b: hsgp_kernel(b, a - 0.5, bb - 0.5, L=0.5, m=6, eta=eta, ell=ell)
    )
    for b in RADIAL
}
models_r["exact GP"] = k_expquad


def truth_radial(edge_sd):
    def draw():
        f_all = draw_constrained_profile(r_grid, ell=0.5, edge_sd=edge_sd)
        return np.r_[f_all[obs], f_all[~obs]]

    return draw


print("constraints hold exactly:")
replicated_table(models_r, truth_radial(0.0), r_obs, r_pred, noise_true=0.2)
print("\nf(R) only approximately zero (sd 0.3):")
replicated_table(models_r, truth_radial(0.3), r_obs, r_pred, noise_true=0.2)
```

When the generated profiles satisfy $f'(0)=0$ and $f(1)=0$, `"neumann-dirichlet"` supplies correct information about the unobserved endpoints.
When $f(1)\sim\mathcal{N}(0,0.3^2)$, that HSGP still assigns zero uncertainty to $f(1)$.
It cannot represent the nonzero endpoint values in the second set of simulated profiles.

The unconstrained exact GP in the radial-profile comparison does not impose $f'(0)=0$ or $f(1)=0$.
The comparison tests whether those two physical constraints improve reconstruction of the unobserved profile regions.
An exact GP can also impose these constraints through a suitable covariance or conditioning calculation.

Neither mixed condition nor Neumann is a safe default for an unknown physical boundary.
You should choose the condition from domain knowledge about the latent function, and check sensitivity when that knowledge is uncertain.

+++

## Forecasting near approximation boundaries

Let's now observe inputs on $[0, 1]$ and forecast to $1.2$, half a generating lengthscale beyond the last observation.
With $c = 1.5$, the upper approximation boundary is $1.25$, so the forecasts approach it without crossing it.

A Dirichlet endpoint forces the latent function toward zero with vanishing uncertainty at the boundary.
A Neumann endpoint allows uncertain function values but forces the slope to zero there.
Neither rule follows from the observed time series alone.

The exact stationary GP provides a reference without these artificial endpoint constraints.
The forecasting experiment tests predictions on $(1,1.2]$, not arbitrary forecast horizons.
With $c=1.5$, all prediction inputs remain below the upper approximation endpoint at $1.25$.

```{code-cell} ipython3
x_new = np.linspace(1.0, 1.2, 21)[1:]
x_all = np.r_[x, x_new]
f_all = draw_gp(x_all, ell_true)
f, f_new = f_all[: len(x)], f_all[len(x) :]
y = f + noise_true * rng.standard_normal(len(x))
X_new = x_new[:, None]

results_fc = {b: fit_hsgp(b, X, y, X_new) for b in ["dirichlet", "neumann"]}
results_fc["exact GP"] = fit_exact(X, y, X_new)
plot_results(
    results_fc,
    x,
    y,
    x_all,
    f_all,
    x_new,
    "Forecast past the data edge ($m = 15$, $c = 1.5$)",
    vlines=[(1.0, "last observation"), (1.25, "box edge $L$")],
)
```

```{code-cell} ipython3
for c in [1.5, 2.0]:
    L = c * 0.5
    models_fc = {
        f"{b} (c={c})": (
            lambda a, bb, eta, ell, b=b, L=L: hsgp_kernel(
                b, a - 0.5, bb - 0.5, L=L, m=15, eta=eta, ell=ell
            )
        )
        for b in ["dirichlet", "neumann"]
    }
    models_fc["exact GP"] = k_expquad
    replicated_table(models_fc, lambda: draw_gp(x_all, ell_true), x, x_new, noise_true)
    print()
```

The nominal intervals contain 90% of the model's probability for each latent function value.
Coverage tests whether those intervals contain the simulated truth about 90% of the time across repetitions.
A narrow interval is not better if it misses the truth too often.

The forecast table compares two interval widths, with the same training-input center at $0.5$:

1. With $c=1.5$, the approximation interval is $[-0.25,1.25]$.
   The last prediction input, $x=1.2$, lies only $0.05$ below the Dirichlet endpoint, where latent uncertainty must vanish.
   Neumann avoids that zero-value constraint, but it substitutes a zero-slope assumption.
2. With $c=2$, the interval is $[-0.5,1.5]$, so the last prediction input is $0.3$ below the upper endpoint.
   Compare its forecast means, 90% coverage, and `sd(last)` with the corresponding $c=1.5$ results.
   A wider interval can need more basis functions to preserve the same range of frequencies.
3. The two forecast configurations cover only $(1,1.2]$.
   Neither calculation tests predictions outside its approximation interval.
   Neumann's zero-slope constraint does not supply a model for an arbitrary trend beyond that interval.

Include the intended prediction range when using `approx_hsgp_hyperparams`.
For each proposed interval width and basis count, compare forecast means and 90% intervals at the same prediction inputs.
Use the repeated-study coverage to check whether narrow intervals miss the simulated function values too often.

+++

## Choosing a boundary condition

We reccommend separating the two goals: approximating an unconstrained GP and expressing known behavior at a physical endpoint.
Padding addresses the first goal.
A boundary condition can address the second only when its exact constraint matches the latent function.

| Your situation | Recommendation |
|---|---|
| Interpolation without known endpoint constraints | Start with `"dirichlet"` and enough padding, then check approximation accuracy |
| Known zero slope at a physical endpoint | Use Neumann at that endpoint and place the interval edge there |
| Known zero value at a physical endpoint | Use Dirichlet at that endpoint and place the interval edge there |
| Zero slope at one end and zero value at the other | Use the matching mixed condition, with endpoints in the correct order |
| Short-range forecasts near an artificial endpoint | Increase padding and reconsider `m`, then compare forecast means and uncertainty |
| Need the unconstrained GP near the current endpoint | Widen the interval because no boundary choice removes its own constraint |

The recommendations from `approx_hsgp_hyperparams` give initial values for the basis count and interval width, not accuracy guarantees.
Repeat the fit with a wider interval and enough basis functions to cover the same frequencies.
Compare posterior means and 90% intervals at the same prediction inputs.
If physical knowledge is uncertain, examine the consequences of an incorrect exact constraint before using it.

+++

## References

:::{bibliography}
:filter: docname in docnames
:::

+++

## Authors

- Authored by [Alexandre Andorra](https://github.com/AlexAndorra), [Bill Engels](https://github.com/bwengals), [Ben Mares](https://github.com/maresb) and [Juan Orduz](https://juanitorduz.github.io/) in September 2026 ([pymc-examples#896](https://github.com/pymc-devs/pymc-examples/pull/896)).
- Expanded narrative by [Chris Fonnesbeck](https://github.com/fonnesbeck) in November 2026

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor
```

:::{include} ../page_footer.md
:::
