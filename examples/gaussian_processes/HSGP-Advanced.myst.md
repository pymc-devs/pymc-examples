---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: pymc-examples
  language: python
  name: pymc-examples
---

(hsgp)=
# Gaussian Processes: HSGP Advanced Usage

:::{post} June 12, 2024
:tags: gaussian processes
:category: reference, intermediate
:author: Bill Engels, Alexandre Andorra
:::

+++

The Hilbert Space Gaussian processes approximation is a low-rank GP approximation that is particularly well-suited to usage in probabilistic programming languages like PyMC.  It approximates the GP using a pre-computed and fixed set of basis functions that don't depend on the form of the covariance kernel or its hyperparameters.  It's a _parametric_ approximation, so prediction in PyMC can be done as one would with a linear model via `pm.Data` or `pm.set_data`.  You don't need to define the `.conditional` distribution that non-parameteric GPs rely on.  This makes it _much_ easier to integrate an HSGP, instead of a GP, into your existing PyMC model.  Additionally, unlike many other GP approximations, HSGPs can be used anywhere within a model and with any likelihood function.  

It's also fast.  The computational cost for unapproximated GPs per MCMC step is $\mathcal{O}(n^3)$, where $n$ is the number of data points.  For HSGPs, it is $\mathcal{O}(mn + m)$, where $m$ is the number of basis vectors.  It's important to note that _sampling speeds_ are also very strongly determined by posterior geometry. 

The HSGP approximation does carry some restrictions:
1. It **can only be used with _stationary_ covariance kernels** such as the Matern family.  The {class}`~pymc.gp.HSGP` class is compatible with any `Covariance` class that implements the `power_spectral_density` method.  There is a special case made for the `Periodic` covariance, which is implemented in PyMC by The {class}`~pymc.gp.HSGPPeriodic`.
2. It **does not scale well with the input dimension**.  The HSGP approximation is a good choice if your GP is over a one dimensional process like a time series, or a two dimensional spatial point process.  It's likely not an efficient choice where the input dimension is larger than three. 
3. It **_may_ struggle with more rapidly varying processes**.  If the process you're trying to model changes very quickly relative to the extent of the domain, the HSGP approximation may fail to accurately represent it.  We'll show in later sections how to set the accuracy of the approximation, which involves a trade-off between the fidelity of the approximation and the computational complexity.
4. **For smaller data sets, the full unapproximated GP may still be more efficient**.

A secondary goal of this implementation is flexibility via an accessible implementation where the core computations are implemented in a modular way.  For basic usage, users can use the `.prior` and `.conditional` methods and essentially treat the HSGP class as a drop in replacement for `pm.gp.Latent`, the unapproximated GP.  More advanced users can bypass those methods and work with `.prior_linearized` instead, which exposes the HSGP as a parametric model.  For more complex models with multiple HSGPs, users can work directly with functions like `pm.gp.hsgp_approx.calc_eigenvalues` and `pm.gp.hsgp_approx.calc_eigenvectors`.

#### References:
- Original reference: [Solin & Sarkka, 2019](https://link.springer.com/article/10.1007/s11222-019-09886-w).
- HSGPs in probabilistic programming languages: [Riutort-Mayol et al., 2020](https://arxiv.org/abs/2004.11408).
- HSTPs (Student-t process): [Sellier & Dellaportas, 2023](https://proceedings.mlr.press/v202/sellier23a.html).
- Kronecker HSGPs: [Dan et al., 2022](https://arxiv.org/pdf/2210.11358.pdf)
- PyMC's {class}`~pymc.gp.HSGP` API

```{code-cell} ipython3
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt
```

```{code-cell} ipython3
az.style.use("arviz-whitegrid")
plt.rcParams["figure.figsize"] = [12, 5]
%config InlineBackend.figure_format = 'retina'
seed = sum(map(ord, "hsgp advanced"))
rng = np.random.default_rng(seed)
```

# Example 1: A hierarchical HSGP, a more custom model

+++

`````{admonition} Looking for a beginner's introduction?
:class: tip
This notebook is the second part of our HSGP tutorials. We strongly recommend you start by reading [the first part](HSGP-Basic.myst.md), which is a smoother introduction to HSGPs and covers more basic use-cases.

The following notebook does _not_ cover the theory of HSGPs and shows more advanced use-cases.
``````

+++

The {class}`~pymc.gp.HSGP` class and associated functions are also meant to be clear and hackable to enable building more complicated models.  In the following example we fit a hierarchical HSGP, where each of the individual GPs (indexed by $i$) can have different lengthscales. The model is:

$$
\begin{align}
f^\mu &\sim \mathcal{GP}\left(0 \,, K^\mu(x, x' \,; \eta^\mu, \ell^\mu) \right) \\
f_i &\sim \mathcal{GP}\left(f^\mu \,, K^\delta(x, x' \,; \eta^\delta, \ell^\delta_i) \right) \\
\end{align}
$$

There are two scale parameters $\eta^\mu$ and $\eta^\delta$. $\eta^\mu$ controls the overall scaling of the group GP, and $\eta^\delta$ controls the strength of the partial pooling of the $f_i$ to $f^\mu$. Each of the $i$ GPs can have its own lengthscale $\ell^\delta_i$. In the example below we simulate additive Gaussian noise, but this HSGP model will of course work with any likelihood anywhere within your model.

**Refer to this section if you're interested in:**
1. Seeing an example of a fast approximation to a Hierarchical GP.
2. Seeing how to construct more advanced and custom GP models.
3. Using HSGPs for prediction within larger PyMC models.

+++

## Simulate data

+++

Let's simulate a one-dimensional GP observed at 300 locations (200 used for training, the remaining 100 for testing):

```{code-cell} ipython3
# Generate wider range data
x_full = np.linspace(0, 15, 300)

# Split into training and test sets
n_train = 200
x_train = x_full[:n_train]
x_test = x_full[n_train:]

# Definfe the mean GP
eta_mu_true = 1.0
ell_mu_true = 1.5
cov_mu = eta_mu_true**2 * pm.gp.cov.Matern52(input_dim=1, ls=ell_mu_true)

# Define the delta GPs
n_gps = 10
eta_delta_true = 0.5
ell_delta_true = pm.draw(pm.Lognormal.dist(mu=np.log(ell_mu_true), sigma=0.5), draws=n_gps)

cov_deltas = [
    eta_delta_true**2 * pm.gp.cov.Matern52(input_dim=1, ls=ell_i) for ell_i in ell_delta_true
]

# Additive gaussian noise
sigma_noise = 0.5
noise_dist = pm.Normal.dist(mu=0.0, sigma=sigma_noise)
```

```{code-cell} ipython3
def generate_gp_samples(x, cov_mu, cov_deltas, noise_dist, rng):
    """
    Generate samples from a hierarchical Gaussian Process (GP).
    """
    n = len(x)
    # One draw from the mean GP
    f_mu_true = pm.draw(pm.MvNormal.dist(mu=np.zeros(n), cov=cov_mu(x[:, None])))

    # Draws from the delta GPs
    f_deltas = []
    for cov_delta in cov_deltas:
        f_deltas.append(pm.draw(pm.MvNormal.dist(mu=np.zeros(n), cov=cov_delta(x[:, None]))))
    f_delta = np.vstack(f_deltas)

    # The hierarchical GP
    f_true = f_mu_true[:, None] + f_delta.T

    # Observed values with noise
    n_gps = len(cov_deltas)
    y_obs = f_true + pm.draw(noise_dist, draws=n * n_gps, random_seed=rng).reshape(n, n_gps)

    return f_mu_true, f_true, y_obs
```

```{code-cell} ipython3
# Generate samples for the full data
f_mu_true_full, f_true_full, y_full = generate_gp_samples(
    x_full, cov_mu, cov_deltas, noise_dist, rng
)

f_mu_true_train = f_mu_true_full[:n_train]
f_mu_true_test = f_mu_true_full[n_train:]

f_true_train = f_true_full[:n_train]
f_true_test = f_true_full[n_train:]

y_train = y_full[:n_train]
y_test = y_full[n_train:]
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
colors_train = plt.cm.Blues(np.linspace(0.1, 0.9, n_gps))
colors_test = plt.cm.Greens(np.linspace(0.1, 0.9, n_gps))
ylims = [1.1 * np.min(y_full), 1.1 * np.max(y_full)]

axs[0].plot(x_train, f_mu_true_train, color="C1", lw=3)
axs[0].plot(x_test, f_mu_true_test, color="C1", lw=3, ls="--")
axs[0].axvline(x_train[-1], ls=":", lw=3, color="k", alpha=0.6)
axs[1].axvline(x_train[-1], ls=":", lw=3, color="k", alpha=0.6)

# Positioning text for "Training territory" and "Testing territory"
train_text_x = (x_train[0] + x_train[-1]) / 2
test_text_x = (x_train[-1] + x_test[-1]) / 2
text_y = ylims[0] + (ylims[1] - ylims[0]) * 0.9

# Adding text to the left plot
axs[0].text(
    train_text_x,
    text_y,
    "Training territory",
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=14,
    color="blue",
    alpha=0.7,
)
axs[0].text(
    test_text_x,
    text_y,
    "Testing territory",
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=14,
    color="green",
    alpha=0.7,
)

for i in range(n_gps):
    axs[0].plot(x_train, f_true_train[:, i], color=colors_train[i])
    axs[0].plot(x_test, f_true_test[:, i], color=colors_test[i])
    axs[1].scatter(x_train, y_train[:, i], color=colors_train[i], alpha=0.6)
    axs[1].scatter(x_test, y_test[:, i], color=colors_test[i], alpha=0.6)

axs[0].set(xlabel="x", ylim=ylims, title="True GPs\nMean GP in orange")
axs[1].set(xlabel="x", ylim=ylims, title="Observed data\nColor corresponding to GP");
```

## Build the model

+++

To build this model to allow different lengthscales per GP, we need to rewrite the power spectral density.  The one attached to the PyMC covariance classes, i.e. `pm.gp.cov.Matern52.power_spectral_density`, is vectorized over the _input dimension_, but we need one vectorized across _GPs_.

Fortunately, this one at least isn't too hard to adapt:

+++

### Adapting the power spectral density

```{code-cell} ipython3
def matern52_psd(omega, ls):
    """
    Calculate the power spectral density for the Matern52 covariance kernel.

    Inputs:
      - omega: The frequencies where the power spectral density is evaluated
      - ls: The lengthscales. Can be a scalar or a vector.
    """
    num = 2.0 * np.sqrt(np.pi) * pt.gamma(3.0) * pt.power(5.0, 5.0 / 2.0)
    den = 0.75 * pt.sqrt(np.pi)
    return (num / den) * ls * pt.power(5.0 + pt.outer(pt.square(omega), pt.square(ls)), -3.0)
```

Next, we build a function that constructs the hierarchical GP.  Notice that it assumes some names for the `dims`, but our goal is to provide a simple foundation that you can adapt to your specific use-case. You can see that this is a bit more deconstructed than `.prior_linearized`.

+++

### Coding the hierarchical GP

```{code-cell} ipython3
def hierarchical_HSGP(Xs, m, c, eta_mu, ell_mu, eta_delta, ell_delta):
    "Important: the Xs need to be 0-centered"
    L = pm.gp.hsgp_approx.set_boundary(Xs, c)
    eigvals = pm.gp.hsgp_approx.calc_eigenvalues(L, m)
    phi = pm.gp.hsgp_approx.calc_eigenvectors(Xs, L, eigvals, m)
    omega = pt.sqrt(eigvals)

    # calculate f_mu, the mean of the hierarchical gp
    beta = pm.Normal("f_mu_coeffs", mu=0.0, sigma=1.0, dims="m_ix")
    psd = matern52_psd(omega, ell_mu).flatten()
    f_mu = pm.Deterministic("f_mu", phi @ (beta * pt.sqrt(psd) * eta_mu))

    # calculate f_delta, the gp offsets
    beta = pm.Normal("f_delta_coeffs", mu=0.0, sigma=1.0, dims=("m_ix", "gp_ix"))
    psd = matern52_psd(omega, ell_delta)
    f_delta = phi @ (beta * pt.sqrt(psd) * eta_delta)

    # calculate total gp
    return pm.Deterministic("f", f_mu[:, None] + f_delta)
```

### Choosing the HSGP parameters

+++

Next, we use the heuristics to choose `m` and `c`:

```{code-cell} ipython3
m, c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range=[np.min(x_full), np.max(x_full)],
    lengthscale_range=[np.min(ell_delta_true), np.max(ell_delta_true)],
    cov_func="matern52",
)
print(f"m: {m}, c: {c:.2f}")
```

That actually looks a bit too low, especially `c`. We can actually check the computation by hand. The way we defined `hierarchical_HSGP`, it needs the 0-centered `x_train` data, called `Xs`, so we'll need to do that here (we'll also need to do that later when we define the model):

```{code-cell} ipython3
x_center = (np.max(x_train) + np.min(x_train)) / 2
Xs = x_train - x_center
```

Then we can use the `c` from above and check the implied `L`, which is the result of `set_boundary`:

```{code-cell} ipython3
pm.gp.hsgp_approx.set_boundary(Xs, c)
```

And this is indeed too low. How do we know? Well, thankfully, `L` has a pretty interpretable meaning in the HSGP decomposition. It is the boundary of the approximation, so we need to chose `L` such that the domain `[-L, L]` contains all points, not only in `x_train`, but in `x_full` (see [the first tutorial](HSGP-Basic.myst.md) for more details).

So we want $L > 15$ in this case, which means we need to increase `c` until we're satisfied:

```{code-cell} ipython3
pm.gp.hsgp_approx.set_boundary(Xs, 4.0)
```

Bingo!

One last thing we also talked about in the first turorial: increasing `c` requires increasing `m` to compensate for the loss of fidelity at smaller lengthscales. So let's err on the side of safety and choose:

```{code-cell} ipython3
m, c = 100, 4.0
```

### Setting up the model

```{code-cell} ipython3
coords = {
    "gp_ix": np.arange(n_gps),
    "m_ix": np.arange(m),
}
```

As discussed, you'll see we're handling the 0-centering of `X` before defining the GP. When you're using `pm.HSGP` or `prior_linearized`, you don't need to care about that, as it's done for you under the hood. But when using more advanced model like this one, you need to get your hands dirtier as you need to access lower-level functions of the package.

```{code-cell} ipython3
with pm.Model(coords=coords) as model:
    ## handle 0-centering correctly
    x_center = (np.max(x_train) + np.min(x_train)) / 2
    X = pm.Data("X", x_train[:, None])
    Xs = X - x_center

    ## Prior for the mean process
    eta_mu = pm.Gamma("eta_mu", 2, 2)
    log_ell_mu = pm.Normal("log_ell_mu")
    ell_mu = pm.Deterministic("ell_mu", pt.softplus(log_ell_mu))

    ## Prior for the offsets
    log_ell_delta_offset = pm.ZeroSumNormal("log_ell_delta_offset", dims="gp_ix")
    log_ell_delta_sd = pm.Gamma("log_ell_delta_sd", 2, 2)

    log_ell_delta = log_ell_mu + log_ell_delta_sd * log_ell_delta_offset
    ell_delta = pm.Deterministic("ell_delta", pt.softplus(log_ell_delta), dims="gp_ix")

    eta_delta = pm.Gamma("eta_delta", 2, 2)

    ## define full GP
    f = hierarchical_HSGP(Xs, [m], c, eta_mu, ell_mu, eta_delta, ell_delta)

    ## prior on observational noise
    sigma = pm.Exponential("sigma", scale=1)

    ## likelihood
    pm.Normal("y", mu=f, sigma=sigma, observed=y_train, shape=(X.shape[0], n_gps))
```

```{code-cell} ipython3
pm.model_to_graphviz(model)
```

## Prior predictive checks

+++

Now, what do these priors mean? Good question. As always, it's crucial to do **prior predictive checks**, especially for GPs, where amplitudes and lenghtscales can be very hard to infer:

```{code-cell} ipython3
with model:
    idata = pm.sample_prior_predictive()
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
def plot_gps(idata, f_mu_true, f_true, group="posterior", return_f=False):
    """
    Plot the underlying hierarchical GP and inferred GPs with posterior intervals.

    Parameters:
    - idata: InferenceData object containing the prior or posterior samples.
    - f_mu_true: The true mean function values.
    - f_true: The true function values for each group.
    - group: one of 'prior', 'posterior' or 'predictions'.
            Whether to plot the prior predictive, posterior predictive or out-of-sample predictions samples.
            Default posterior.
    """
    if group == "predictions":
        x = idata.predictions_constant_data.X.squeeze().to_numpy()
    else:
        x = idata.constant_data.X.squeeze().to_numpy()
    y_obs = idata.observed_data["y"].to_numpy()
    n_gps = f_true.shape[1]

    # Extract mean and standard deviation for 'f_mu' and 'f' from the posterior
    f_mu_post = az.extract(idata, group=group, var_names="f_mu")
    f_mu_mu = f_mu_post.mean(dim="sample")
    f_mu_sd = f_mu_post.std(dim="sample")

    f_post = az.extract(idata, group=group, var_names="f")
    f_mu = f_post.mean(dim="sample")
    f_sd = f_post.std(dim="sample")

    # Plot settings
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    colors = plt.cm.Set1(np.linspace(0.1, 0.9, n_gps))
    ylims = [1.1 * np.min(y_obs), 1.1 * np.max(y_obs)]

    # Plot true underlying GP
    axs[0].plot(x, f_mu_true, color="k", lw=3)
    for i in range(n_gps):
        axs[0].plot(x, f_true[:, i], color=colors[i], alpha=0.7)

    # Plot inferred GPs with uncertainty
    for i in range(n_gps):
        axs[1].fill_between(
            x,
            f_mu[:, i] - f_sd[:, i],
            f_mu[:, i] + f_sd[:, i],
            color=colors[i],
            alpha=0.3,
            edgecolor="none",
        )
    # Plot mean GP
    axs[1].fill_between(
        x,
        f_mu_mu - f_mu_sd,
        f_mu_mu + f_mu_sd,
        color="k",
        alpha=0.6,
        edgecolor="none",
    )

    # Set labels and titles
    for ax in axs:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    axs[0].set(ylim=ylims, title="True GPs for the 10 time-series\nMean GP in black")
    axs[1].set(ylim=ylims, title=r"Inferred GPs, $\pm 1 \sigma$ posterior intervals")

    if return_f:
        return f_mu_mu, f_mu_sd, f_mu, f_sd
```

```{code-cell} ipython3
plot_gps(idata, f_mu_true_train, f_true_train, group="prior");
```

Once we're satisfied with our priors, which is the case here, we can... sample the model!

+++

## Sampling & Convergence checks

```{code-cell} ipython3
with model:
    idata.extend(pm.sample(nuts_sampler="numpyro"))
```

```{code-cell} ipython3
idata.sample_stats.diverging.sum().data
```

```{code-cell} ipython3
az.summary(idata, var_names=["eta_mu", "ell_mu", "eta_delta", "ell_delta", "sigma"], round_to=2)
```

```{code-cell} ipython3
az.plot_trace(
    idata,
    var_names=["eta_mu", "ell_mu", "eta_delta", "ell_delta", "sigma"],
    lines=[
        ("eta_mu", {}, [eta_mu_true]),
        ("ell_mu", {}, [ell_mu_true]),
        ("eta_delta", {}, [eta_delta_true]),
        ("ell_delta", {}, [ell_delta_true]),
        ("sigma", {}, [sigma_noise]),
    ],
);
```

Everything went great here, that's really goos sign! Now let's see if the model could recover the true parameters.

+++

## Posterior checks

```{code-cell} ipython3
az.plot_posterior(
    idata,
    var_names=["eta_mu", "ell_mu", "eta_delta", "ell_delta", "sigma"],
    ref_val={
        "eta_mu": [{"ref_val": eta_mu_true}],
        "ell_mu": [{"ref_val": ell_mu_true}],
        "eta_delta": [{"ref_val": eta_delta_true}],
        "ell_delta": [{"gp_ix": i, "ref_val": ell_delta_true[i]} for i in range(n_gps)],
        "sigma": [{"ref_val": sigma_noise}],
    },
    grid=(5, 3),
    textsize=30,
);
```

Really good job -- the model recovered everything decently!

```{code-cell} ipython3
az.plot_forest(
    [idata.prior, idata.posterior],
    model_names=["Prior", "Posterior"],
    var_names=["eta_mu", "ell_mu", "eta_delta", "ell_delta", "sigma"],
    combined=True,
    figsize=(12, 6),
);
```

And we can see the GP parameters were well informed by the data. Let's close up this section by updating our prior predictive plot with the posterior of the inferred GPs:

```{code-cell} ipython3
plot_gps(idata, f_mu_true_train, f_true_train);
```

That looks great! Now we can go ahead and predict out of sample.

+++

## Out-of-sample predictions

```{code-cell} ipython3
with model:
    pm.set_data({"X": x_full[:, None]})

    idata.extend(
        pm.sample_posterior_predictive(idata, var_names=["f_mu", "f"], predictions=True),
    )
```

```{code-cell} ipython3
pred_f_mu_mu, pred_f_mu_sd, pred_f_mu, pred_f_sd = plot_gps(
    idata, f_mu_true_full, f_true_full, group="predictions", return_f=True
)
```

This looks good! And we can check our predictions make sense with another plot:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
fig, axs = plt.subplot_mosaic(
    [["True", "Data"], ["Preds", "Preds"]],
    layout="constrained",
    sharex=True,
    sharey=True,
    figsize=(12, 8),
)

axs["True"].plot(x_train, f_mu_true_train, color="C1", lw=3)
axs["True"].plot(x_test, f_mu_true_test, color="C1", lw=3, ls="--")
axs["True"].axvline(x_train[-1], ls=":", lw=3, color="k", alpha=0.6)
axs["True"].text(
    train_text_x,
    text_y,
    "Training territory",
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=14,
    color="blue",
    alpha=0.7,
)
axs["True"].text(
    test_text_x,
    text_y,
    "Testing territory",
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=14,
    color="green",
    alpha=0.7,
)

axs["Data"].axvline(x_train[-1], ls=":", lw=3, color="k", alpha=0.6)
axs["Preds"].axvline(x_train[-1], ls=":", lw=3, color="k", alpha=0.6)

# Plot mean GP
axs["Preds"].fill_between(
    x_full,
    pred_f_mu_mu - pred_f_mu_sd,
    pred_f_mu_mu + pred_f_mu_sd,
    color="C1",
    alpha=0.8,
    edgecolor="none",
)

for i in range(n_gps):
    axs["True"].plot(x_train, f_true_train[:, i], color=colors_train[i])
    axs["True"].plot(x_test, f_true_test[:, i], color=colors_test[i])
    axs["Data"].scatter(x_train, y_train[:, i], color=colors_train[i], alpha=0.6)
    axs["Data"].scatter(x_test, y_test[:, i], color=colors_test[i], alpha=0.6)
    # Plot inferred GPs with uncertainty
    axs["Preds"].fill_between(
        x_train,
        pred_f_mu[:n_train, i] - pred_f_sd[:n_train, i],
        pred_f_mu[:n_train, i] + pred_f_sd[:n_train, i],
        color=colors_train[i],
        alpha=0.3,
        edgecolor="none",
    )
    axs["Preds"].fill_between(
        x_test,
        pred_f_mu[n_train:, i] - pred_f_sd[n_train:, i],
        pred_f_mu[n_train:, i] + pred_f_sd[n_train:, i],
        color=colors_test[i],
        alpha=0.3,
        edgecolor="none",
    )

axs["True"].set(xlabel="x", ylim=ylims, title="True GPs\nMean GP in orange")
axs["Data"].set(xlabel="x", ylim=ylims, title="Observed data\nColor corresponding to GP")
axs["Preds"].set(
    xlabel="x",
    ylim=ylims,
    title="Predicted GPs, $\\pm 1 \\sigma$ posterior intervals\nMean GP in orange",
);
```

# Example 2: An HSGP that exploits Kronecker structure

+++ {"editable": true, "slideshow": {"slide_type": ""}}

This example is a multiple GP model like the previous one, but it assumes a different relationship between the GPs. **Instead of pooling towards a common mean GP, there is an additional covariance structure that specifies their relationship**.

For example, we may have time series measurements of temperature from multiple weather stations. The similarity over time should mostly depend only on the _distance_ between the weather stations. They all will likely have the same dynamics, or same covariance structure, over time. You can think of this as _local_ partial pooling.  

In the example below, we arrange the GPs along a single "spatial" axis, so it's a 1D problem and not 2D, and then allow them to share the same time covariance.  This might be clearer after taking a look at the simulated data below.  

Mathematically, this model uses the [Kronecker product](GP-Kron.myst.md), where the "space" and "time" dimensions are _separable_.
$$
K = K_{x} \otimes K_{t}
$$
If there are $n_t$ time points and $n_x$ GPs, then the resulting $K$ matrix will have dimension $n_x \cdot n_t \times n_x \cdot n_t$.  Using a regular GP, this would be $\mathcal{O}(n_t^3 n_x^3)$.  So, we can achieve a pretty massive speed-up by both taking advantage of Kronecker structure and using the HSGP approximation.  It isn't required that both of the dimensions (in this example, space and time) use the HSGP approximation.  It's possible to use either a vanilla GP or inducing points for the "spatial" covariance, and the HSGP approximation in time.  In the example below, both use the HSGP approximation.

**Refer to this section if you're interested in:**
1. Seeing an example of exploiting Kronecker structure and the HSGP approximation.
2. Seeing how to construct more advanced and custom GP models.

+++

## Data generation

```{code-cell} ipython3
n_gps, n_t = 30, 100
t = np.linspace(0, 10, n_t)
x = np.linspace(-5, 5, n_gps)

eta_true = 1.0
ell_x_true = 2.0
cov_x = eta_true**2 * pm.gp.cov.Matern52(input_dim=1, ls=ell_x_true)
Kx = cov_x(x[:, None])

ell_t_true = 2.0
cov_t = pm.gp.cov.Matern52(input_dim=1, ls=ell_t_true)
Kt = cov_t(t[:, None])

K = pt.slinalg.kron(Kx, Kt)
f_true = pm.draw(pm.MvNormal.dist(mu=np.zeros(n_gps * n_t), cov=K)).reshape(n_gps, n_t).T

# Additive gaussian noise
sigma_noise = 0.5
noise_dist = pm.Normal.dist(mu=0.0, sigma=sigma_noise)

y_obs = f_true + pm.draw(noise_dist, draws=n_t * n_gps, random_seed=rng).reshape(n_t, n_gps)
```

```{code-cell} ipython3
fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
colors = plt.cm.Blues(np.linspace(0.0, 0.9, n_gps))
ylims = [1.1 * np.min(y_obs), 1.1 * np.max(y_obs)]

for i in range(n_gps):
    axs[0].plot(t, f_true[:, i], color=colors[i], lw=2, alpha=0.7)
    axs[1].scatter(t, y_obs[:, i], color=colors[i], alpha=0.7)

for ax in axs:
    ax.set_xlabel("t")
    ax.set_ylabel("y")
axs[0].set(ylim=ylims, title="Underlying Kronecker GP")
axs[1].set(ylim=ylims, title="Observed data, color corresponding to GP");
```

## Kronecker GP specification

```{code-cell} ipython3
def kronecker_HSGP(Xs, m, c, cov_t, cov_x):
    Xs_t, Xs_x = Xs  # Xs needs to be 0-centered
    m_t, m_x = m
    c_t, c_x = c

    L_t = pm.gp.hsgp_approx.set_boundary(Xs_t, c_t)
    eigvals_t = pm.gp.hsgp_approx.calc_eigenvalues(L_t, [m_t])
    phi_t = pm.gp.hsgp_approx.calc_eigenvectors(Xs_t, L_t, eigvals_t, [m_t])
    omega_t = pt.sqrt(eigvals_t)

    sqrt_psd_t = pt.sqrt(cov_t.power_spectral_density(omega_t))
    chol_t = phi_t * sqrt_psd_t

    L_x = pm.gp.hsgp_approx.set_boundary(Xs_x, c_x)
    eigvals_x = pm.gp.hsgp_approx.calc_eigenvalues(L_x, [m_x])
    phi_x = pm.gp.hsgp_approx.calc_eigenvectors(Xs_x, L_x, eigvals_x, [m_x])
    omega_x = pt.sqrt(eigvals_x)

    sqrt_psd_x = pt.sqrt(cov_x.power_spectral_density(omega_x))
    chol_x = phi_x * sqrt_psd_x

    z = pm.Normal("beta", size=m_x * m_t)

    return (chol_x @ (chol_t @ pt.reshape(z, (m_t, m_x))).T).T
```

## PyMC Model

+++

Next, we use the heuristics to choose `m` and `c`:

```{code-cell} ipython3
m_t, c_t = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range=[np.min(t), np.max(t)], lengthscale_range=[1.0, 3.0], cov_func="matern52"
)
m_x, c_x = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range=[np.min(x), np.max(x)], lengthscale_range=[1.0, 3.0], cov_func="matern52"
)

print(f"m_t: {m_t}, c_t: {c_t:.2f}")
print(f"m_x: {m_x}, c_x: {c_x:.2f}")
```

```{code-cell} ipython3
with pm.Model() as model:
    ## handle 0-centering correctly
    xt_center = (np.max(t) + np.min(t)) / 2
    Xt = pm.Data("Xt", t[:, None])
    Xs_t = Xt - xt_center

    xx_center = (np.max(x) + np.min(x)) / 2
    Xx = pm.Data("Xx", x[:, None])
    Xs_x = Xx - xx_center

    ## covariance on time GP
    ell_t_params = pm.find_constrained_prior(
        pm.Lognormal, lower=0.5, upper=4.0, mass=0.95, init_guess={"mu": 1.0, "sigma": 1.0}
    )
    ell_t = pm.Lognormal("ell_t", **ell_t_params)
    cov_t = pm.gp.cov.Matern52(1, ls=ell_t)

    ## covariance on space GP
    ell_x_params = pm.find_constrained_prior(
        pm.Lognormal, lower=0.5, upper=4.0, mass=0.95, init_guess={"mu": 1.0, "sigma": 1.0}
    )
    ell_x = pm.Lognormal("ell_x", **ell_x_params)
    cov_x = pm.gp.cov.Matern52(1, ls=ell_x)

    ## Kronecker GP
    eta = pm.Gamma("eta", 2, 2)
    Xs, m, c = [Xs_t, Xs_x], [m_t, m_x], [c_t, c_x]
    f = kronecker_HSGP(Xs, m, c, cov_t, cov_x)
    f = pm.Deterministic("f", eta * f)

    # observational noise
    sigma = pm.Exponential("sigma", scale=1)

    # likelihood
    pm.Normal("y", mu=f, sigma=sigma, observed=y_obs)
```

```{code-cell} ipython3
pm.model_to_graphviz(model)
```

## Prior predictive checks

```{code-cell} ipython3
with model:
    idata = pm.sample_prior_predictive()
```

```{code-cell} ipython3
f_mu = az.extract(idata, group="prior", var_names="f").mean(dim="sample")
f_sd = az.extract(idata, group="prior", var_names="f").std(dim="sample")

fig, axs = plt.subplots(1, 2, figsize=(14, 4), sharex=True, sharey=True)
colors = plt.cm.Blues(np.linspace(0.0, 0.9, n_gps))
ylims = [1.1 * np.min(y_obs), 1.1 * np.max(y_obs)]

for i in range(n_gps):
    axs[0].plot(t, f_true[:, i], color=colors[i], lw=2, alpha=0.7)
    axs[1].fill_between(
        t,
        f_mu[:, i] - f_sd[:, i],
        f_mu[:, i] + f_sd[:, i],
        color=colors[i],
        alpha=0.4,
        edgecolor="none",
    )

for ax in axs:
    ax.set_xlabel("t")
    ax.set_ylabel("y")

axs[0].set(ylim=ylims, title="True Kronecker GP")
axs[1].set(ylim=ylims, title=r"Prior GPs, $\pm 1 \sigma$ posterior intervals");
```

## Sampling & Convergence checks

```{code-cell} ipython3
with model:
    idata.extend(pm.sample(nuts_sampler="numpyro"))
```

```{code-cell} ipython3
idata.sample_stats.diverging.sum().data
```

```{code-cell} ipython3
az.summary(idata, var_names=["eta", "ell_x", "ell_t", "sigma"], round_to=2)
```

```{code-cell} ipython3
az.plot_trace(
    idata,
    var_names=["eta", "ell_x", "ell_t", "sigma"],
    lines=[
        ("eta", {}, [eta_true]),
        ("ell_x", {}, [ell_x_true]),
        ("ell_t", {}, [ell_t_true]),
        ("sigma", {}, [sigma_noise]),
    ],
);
```

## Posterior predictive checks

```{code-cell} ipython3
f_mu = az.extract(idata, group="posterior", var_names="f").mean(dim="sample")
f_sd = az.extract(idata, group="posterior", var_names="f").std(dim="sample")

fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
colors = plt.cm.Blues(np.linspace(0.0, 0.9, n_gps))
ylims = [1.1 * np.min(y_obs), 1.1 * np.max(y_obs)]

for i in range(n_gps):
    axs[0].plot(t, f_true[:, i], color=colors[i], lw=2, alpha=0.7)
    axs[1].fill_between(
        t,
        f_mu[:, i] - f_sd[:, i],
        f_mu[:, i] + f_sd[:, i],
        color=colors[i],
        alpha=0.4,
        edgecolor="none",
    )

for ax in axs:
    ax.set_xlabel("t")
    ax.set_ylabel("y")

axs[0].set(ylim=ylims, title="True Kronecker GP")
axs[1].set(ylim=ylims, title=r"Prior GPs, $\pm 1 \sigma$ posterior intervals");
```

And isn't this beautiful?? Now go on, and HSGP-on!

+++

## Authors

* Created by [Bill Engels](https://github.com/bwengals) and [Alexandre Andorra](https://github.com/AlexAndorra) in 2024 ([pymc-examples#668](https://github.com/pymc-devs/pymc-examples/pull/668))

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p xarray
```

:::{include} ../page_footer.md
:::
