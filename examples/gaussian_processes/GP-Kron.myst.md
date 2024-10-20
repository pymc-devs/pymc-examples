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

(GP-Kron)=
# Kronecker Structured Covariances

:::{post} October, 2022
:tags: gaussian process
:category: intermediate
:author: Bill Engels, Raul-ing Average, Christopher Krapu, Danh Phan, Alex Andorra
:::

+++

PyMC contains implementations for models that have Kronecker structured covariances.  This patterned structure enables Gaussian process models to work on much larger datasets.  Kronecker structure can be exploited when
- The dimension of the input data is two or greater ($\mathbf{x} \in \mathbb{R}^{d}\,, d \ge 2$)
- The influence of the process across each dimension or set of dimensions is *separable*
- The kernel can be written as a product over dimension, without cross terms:

$$k(\mathbf{x}, \mathbf{x'}) = \prod_{i = 1}^{d} k(\mathbf{x}_{i}, \mathbf{x'}_i) \,.$$

The covariance matrix that corresponds to the covariance function above can be written with a *Kronecker product*

$$
\mathbf{K} = \mathbf{K}_2 \otimes \mathbf{K}_2 \otimes \cdots \otimes \mathbf{K}_d \,.
$$

These implementations support the following property of Kronecker products to speed up calculations, $(\mathbf{K}_1 \otimes \mathbf{K}_2)^{-1} = \mathbf{K}_{1}^{-1} \otimes \mathbf{K}_{2}^{-1}$, the inverse of the sum is the sum of the inverses.  If $K_1$ is $n \times n$ and $K_2$ is $m \times m$, then $\mathbf{K}_1 \otimes \mathbf{K}_2$ is $mn \times mn$.  For $m$ and $n$ of even modest size, this inverse becomes impossible to do efficiently.  Inverting two matrices, one $n \times n$ and another $m \times m$ is much easier.

This structure is common in spatiotemporal data.  Given that there is Kronecker structure in the covariance matrix, this implementation is exact -- not an approximation to the full Gaussian process.  PyMC contains two implementations that follow the same pattern as {class}`gp.Marginal <pymc.gp.Marginal>` and {class}`gp.Latent <pymc.gp.Latent>`.  For Kronecker structured covariances where the data likelihood is Gaussian, use {class}`gp.MarginalKron <pymc.gp.MarginalKron>`. For Kronecker structured covariances where the data likelihood is non-Gaussian, use {class}`gp.LatentKron <pymc.gp.LatentKron>`.  

Our implementations follow [Saatchi's Thesis](http://mlg.eng.cam.ac.uk/pub/authors/#Saatci). `gp.MarginalKron` follows "Algorithm 16" using the Eigendecomposition, and `gp.LatentKron` follows "Algorithm 14", and uses the Cholesky decomposition.

+++

## Using `MarginalKron` for a 2D spatial problem

The following is a canonical example of the usage of `gp.MarginalKron`.  Like `gp.Marginal`, this model assumes that the underlying GP is unobserved, but the sum of the GP and normally distributed noise are observed.  

For the simulated data set, we draw one sample from a Gaussian process with inputs in two dimensions whose covariance is Kronecker structured.  Then we use `gp.MarginalKron` to recover the unknown Gaussian process hyperparameters $\theta$ that were used to simulate the data.

+++

### Example

We'll simulate a two dimensional data set and display it as a scatter plot whose points are colored by magnitude.  The two dimensions are labeled `x1` and `x2`.  This could be a spatial dataset, for instance.  The covariance will have a Kronecker structure since the points lie on a two dimensional grid.

```{code-cell} ipython3
import arviz as az
import matplotlib as mpl
import numpy as np
import pymc as pm
```

```{code-cell} ipython3
az.style.use("arviz-whitegrid")
plt = mpl.pyplot
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
seed = sum(map(ord, "gpkron"))
rng = np.random.default_rng(seed)
```

```{code-cell} ipython3
# One dimensional column vectors of inputs
n1, n2 = (50, 30)
x1 = np.linspace(0, 5, n1)
x2 = np.linspace(0, 3, n2)

# make cartesian grid out of each dimension x1 and x2
X = pm.math.cartesian(x1[:, None], x2[:, None])

l1_true = 0.8
l2_true = 1.0
eta_true = 1.0

# Although we could, we don't exploit kronecker structure to draw the sample
cov = (
    eta_true**2
    * pm.gp.cov.Matern52(2, l1_true, active_dims=[0])
    * pm.gp.cov.Cosine(2, ls=l2_true, active_dims=[1])
)

K = cov(X).eval()
f_true = rng.multivariate_normal(np.zeros(X.shape[0]), K, 1).flatten()

sigma_true = 0.5
y = f_true + sigma_true * rng.standard_normal(X.shape[0])
```

The lengthscale along the `x2` dimension is longer than the lengthscale along the `x1` direction (`l1_true` < `l2_true`).

```{code-cell} ipython3
fig = plt.figure(figsize=(12, 6))
cmap = "terrain"
norm = mpl.colors.Normalize(vmin=-3, vmax=3)
plt.scatter(X[:, 0], X[:, 1], s=35, c=y, marker="o", norm=norm, cmap=cmap)
plt.colorbar()
plt.xlabel("x1"), plt.ylabel("x2")
plt.title("Simulated dataset");
```

There are 1500 data points in this data set.  Without using the Kronecker factorization, finding the MAP estimate would be much slower.

+++

Since the two covariances are a product, we only require one scale parameter `eta` to model the product covariance function.

```{code-cell} ipython3
# this implementation takes a list of inputs for each dimension as input
Xs = [x1[:, None], x2[:, None]]

with pm.Model() as model:
    # Set priors on the hyperparameters of the covariance
    ls1 = pm.Gamma("ls1", alpha=2, beta=2)
    ls2 = pm.Gamma("ls2", alpha=2, beta=2)
    eta = pm.HalfNormal("eta", sigma=2)

    # Specify the covariance functions for each Xi
    # Since the covariance is a product, only scale one of them by eta.
    # Scaling both overparameterizes the covariance function.
    cov_x1 = pm.gp.cov.Matern52(1, ls=ls1)  # cov_x1 must accept X1 without error
    cov_x2 = eta**2 * pm.gp.cov.Cosine(1, ls=ls2)  # cov_x2 must accept X2 without error

    # Specify the GP.  The default mean function is `Zero`.
    gp = pm.gp.MarginalKron(cov_funcs=[cov_x1, cov_x2])

    # Set the prior on the variance for the Gaussian noise
    sigma = pm.HalfNormal("sigma", sigma=2)

    # Place a GP prior over the function f.
    y_ = gp.marginal_likelihood("y", Xs=Xs, y=y, sigma=sigma)
```

```{code-cell} ipython3
with model:
    mp = pm.find_MAP(method="BFGS")
```

```{code-cell} ipython3
mp
```

Next we use the map point `mp` to extrapolate in a region outside the original grid.  We can also interpolate.  There is no grid restriction on the new inputs where predictions are desired. It's important to note that under the current implementation, having a grid structure in these points doesn't produce any efficiency gains.  The plot with the extrapolations is shown below.  The original data is marked with circles as before, but the extrapolated posterior mean is marked with squares.

```{code-cell} ipython3
x1new = np.linspace(5.1, 7.1, 20)
x2new = np.linspace(-0.5, 3.5, 40)
Xnew = pm.math.cartesian(x1new[:, None], x2new[:, None])

with model:
    mu, var = gp.predict(Xnew, point=mp, diag=True)
```

```{code-cell} ipython3
fig = plt.figure(figsize=(12, 6))
cmap = "terrain"
norm = mpl.colors.Normalize(vmin=-3, vmax=3)
m = plt.scatter(X[:, 0], X[:, 1], s=30, c=y, marker="o", norm=norm, cmap=cmap)
plt.colorbar(m)
plt.scatter(Xnew[:, 0], Xnew[:, 1], s=30, c=mu, marker="s", norm=norm, cmap=cmap)
plt.ylabel("x2"), plt.xlabel("x1")
plt.title("observed data 'y' (circles) with predicted mean (squares)");
```

## `LatentKron`

Like the `gp.Latent` implementation, the `gp.LatentKron` implementation specifies a Kronecker structured GP regardless of context.  **It can be used with any likelihood function, or can be used to model a variance or some other unobserved processes**.  The syntax follows that of `gp.Latent` exactly.  

### Model

To compare with `MarginalLikelihood`, we use same example as before where the noise is normal, but the GP itself is not marginalized out.  Instead, it is sampled directly using NUTS.  It is very important to note that `gp.LatentKron` does not require a Gaussian likelihood like `gp.MarginalKron`; rather, any likelihood is admissible.

Here though, we'll need to be more informative for our priors, at least those for the GP hyperparameters. This is a general rule when using GPs: **use as informative priors as you can**, as sampling lenghtscale and amplitude is a challenging business, so you want to make the sampler's work as easy as possible.

Here thankfully, we have a lot of information about our amplitude and lenghtscales -- we're the ones who created them ;) So we could fix them, but we'll show how you could code that prior knowledge in your own models, with, e.g, Truncated Normal distributions:

```{code-cell} ipython3
with pm.Model() as model:
    # Set priors on the hyperparameters of the covariance
    ls1 = pm.TruncatedNormal("ls1", lower=0.5, upper=1.5, mu=1, sigma=0.5)
    ls2 = pm.TruncatedNormal("ls2", lower=0.5, upper=1.5, mu=1, sigma=0.5)
    eta = pm.HalfNormal("eta", sigma=0.5)

    # Specify the covariance functions for each Xi
    cov_x1 = pm.gp.cov.Matern52(1, ls=ls1)
    cov_x2 = eta**2 * pm.gp.cov.Cosine(1, ls=ls2)

    # Specify the GP. The default mean function is `Zero`
    gp = pm.gp.LatentKron(cov_funcs=[cov_x1, cov_x2])

    # Place a GP prior over the function f
    f = gp.prior("f", Xs=Xs)

    # Set the prior on the variance for the Gaussian noise
    sigma = pm.HalfNormal("sigma", sigma=0.5)

    y_ = pm.Normal("y_", mu=f, sigma=sigma, observed=y)
```

```{code-cell} ipython3
pm.model_to_graphviz(model)
```

```{code-cell} ipython3
with model:
    idata = pm.sample(nuts_sampler="numpyro", target_accept=0.9, tune=1500, draws=1500)
```

```{code-cell} ipython3
idata.sample_stats.diverging.sum().data
```

### Posterior convergence

+++

The posterior distribution of the unknown lengthscale parameters, covariance scaling `eta`, and white noise `sigma` are shown below.  The vertical lines are the true values that were used to generate the original data set:

```{code-cell} ipython3
var_names = ["ls1", "ls2", "eta", "sigma"]
```

```{code-cell} ipython3
az.plot_posterior(
    idata,
    var_names=var_names,
    ref_val=[l1_true, l2_true, eta_true, sigma_true],
    grid=(2, 2),
    figsize=(12, 6),
);
```

We can see how challenging sampling can be in these situations. Here, all went well because we were careful with our choice of priors -- especially in this simulated case, where parameters don't have a real interpretation.

What does the trace plot looks like?

```{code-cell} ipython3
az.plot_trace(idata, var_names=var_names);
```

All good, so let's go ahead with out-of-sample predictions!

+++

### Out-of-sample predictions

```{code-cell} ipython3
x1new = np.linspace(5.1, 7.1, 20)[:, None]
x2new = np.linspace(-0.5, 3.5, 40)[:, None]
Xnew = pm.math.cartesian(x1new, x2new)
x1new.shape, x2new.shape, Xnew.shape
```

```{code-cell} ipython3
with model:
    fnew = gp.conditional("fnew", Xnew, jitter=1e-6)
```

```{code-cell} ipython3
with model:
    ppc = pm.sample_posterior_predictive(idata, var_names=["fnew"], compile_kwargs={"mode": "JAX"})
```

Below we show the original data set as colored circles, and the mean of the conditional samples as colored squares.  The results closely follow those given by the `gp.MarginalKron` implementation.

```{code-cell} ipython3
fig = plt.figure(figsize=(14, 7))
m = plt.scatter(X[:, 0], X[:, 1], s=30, c=y, marker="o", norm=norm, cmap=cmap)
plt.colorbar(m)
plt.scatter(
    Xnew[:, 0],
    Xnew[:, 1],
    s=30,
    c=np.mean(ppc.posterior_predictive["fnew"].sel(chain=0), axis=0),
    marker="s",
    norm=norm,
    cmap=cmap,
)
plt.ylabel("x2"), plt.xlabel("x1")
plt.title("observed data 'y' (circles) with mean of conditional, or predicted, samples (squares)");
```

Next we plot the original data set indicated with circles markers, along with four samples from the conditional distribution over `fnew` indicated with square markers.  As we can see, the level of variation in the predictive distribution leads to distinctly different patterns in the values of `fnew`.  However, these samples display the correct correlation structure - we see distinct sinusoidal patterns in the y-axis and proximal correlation structure in the x-axis. The patterns displayed in the observed data seamlessly blend into the conditional distribution.

```{code-cell} ipython3
fig, axs = plt.subplots(2, 2, figsize=(24, 16))
axs = axs.ravel()

for i, ax in enumerate(axs):
    ax.axis("off")
    ax.scatter(X[:, 0], X[:, 1], s=20, c=y, marker="o", norm=norm, cmap=cmap)
    ax.scatter(
        Xnew[:, 0],
        Xnew[:, 1],
        s=20,
        c=ppc.posterior_predictive["fnew"].sel(chain=0)[i],
        marker="s",
        norm=norm,
        cmap=cmap,
    )
    ax.set_title(f"Sample {i+1}", fontsize=24)
```

## Authors
* Authored by [Bill Engels](https://github.com/bwengals), 2018
* Updated by [Raul-ing Average](https://github.com/CloudChaoszero), March 2021
* Updated by [Christopher Krapu](https://github.com/ckrapu), July 2021
* Updated to PyMC 4.x by [Danh Phan](https://github.com/danhphan), November 2022
* Updated with some new plots and priors, by [Alex Andorra](https://github.com/AlexAndorra), April 2024

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,xarray
```

:::{include} ../page_footer.md
:::
