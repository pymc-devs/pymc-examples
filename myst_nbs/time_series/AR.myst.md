---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: 'Python 3.8.2 64-bit (''pymc'': conda)'
  language: python
  name: python38264bitpymcconda8b8223a2f9874eff9bd3e12d36ed2ca2
---

+++ {"ein.tags": ["worksheet-0"], "slideshow": {"slide_type": "-"}}

# Analysis of An $AR(1)$ Model in PyMC

```{code-cell} ipython3
---
ein.tags: [worksheet-0]
slideshow:
  slide_type: '-'
---
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")
```

+++ {"ein.tags": ["worksheet-0"], "slideshow": {"slide_type": "-"}}

Consider the following AR(1) process, initialized in the
infinite past:
$$
   y_t = \theta y_{t-1} + \epsilon_t,
$$
where $\epsilon_t \overset{iid}{\sim} {\cal N}(0,1)$.  Suppose you'd like to learn about $\theta$ from a a sample of observations $Y^T = \{ y_0, y_1,\ldots, y_T \}$.

First, let's generate some synthetic sample data. We simulate the 'infinite past' by generating 10,000 samples from an AR(1) process and then discarding the first 5,000:

```{code-cell} ipython3
---
ein.tags: [worksheet-0]
slideshow:
  slide_type: '-'
---
T = 10000
y = np.zeros((T,))

# true stationarity:
true_theta = 0.95
# true standard deviation of the innovation:
true_sigma = 2.0
# true process mean:
true_center = 0.0

for t in range(1, T):
    y[t] = true_theta * y[t - 1] + np.random.normal(loc=true_center, scale=true_sigma)

y = y[-5000:]
plt.plot(y, alpha=0.8)
plt.xlabel("Timestep")
plt.ylabel("$y$");
```

+++ {"ein.tags": ["worksheet-0"], "slideshow": {"slide_type": "-"}}

This generative process is quite straight forward to implement in PyMC:

```{code-cell} ipython3
---
ein.tags: [worksheet-0]
slideshow:
  slide_type: '-'
---
with pm.Model() as ar1:
    # assumes 95% of prob mass is between -2 and 2
    theta = pm.Normal("theta", 0.0, 1.0)
    # precision of the innovation term
    tau = pm.Exponential("tau", 0.5)
    # process mean
    center = pm.Normal("center", mu=0.0, sigma=1.0)

    likelihood = pm.AR1("y", k=theta, tau_e=tau, observed=y - center)

    trace = pm.sample(1000, tune=2000, init="advi+adapt_diag", random_seed=RANDOM_SEED)
    idata = pm.to_inference_data(trace)
```

We can see that even though the sample data did not start at zero, the true center of zero is captured rightly inferred by the model, as you can see in the trace plot below. Likewise, the model captured the true values of the autocorrelation parameter 𝜃 and the innovation term $\epsilon_t$ (`tau` in the model) -- 0.95 and 1 respectively).

```{code-cell} ipython3
az.plot_trace(
    idata,
    lines=[
        ("theta", {}, true_theta),
        ("tau", {}, true_sigma**-2),
        ("center", {}, true_center),
    ],
);
```

+++ {"ein.tags": ["worksheet-0"], "slideshow": {"slide_type": "-"}}

## Extension to AR(p)
We can instead estimate an AR(2) model using PyMC.

$$
 y_t = \theta_1 y_{t-1} + \theta_2 y_{t-2} + \epsilon_t.
$$

The `AR` distribution infers the order of the process thanks to the size the of `rho` argmument passed to `AR`. 

We will also use the standard deviation of the innovations (rather than the precision) to parameterize the distribution.

```{code-cell} ipython3
---
ein.tags: [worksheet-0]
slideshow:
  slide_type: '-'
---
with pm.Model() as ar2:
    theta = pm.Normal("theta", 0.0, 1.0, shape=2)
    sigma = pm.HalfNormal("sigma", 3)
    likelihood = pm.AR("y", theta, sigma=sigma, observed=y)

    trace = pm.sample(
        1000,
        tune=2000,
        random_seed=RANDOM_SEED,
    )
    idata = pm.to_inference_data(trace)
```

```{code-cell} ipython3
az.plot_trace(
    idata,
    lines=[
        ("theta", {"theta_dim_0": 0}, true_theta),
        ("theta", {"theta_dim_0": 1}, 0.0),
        ("sigma", {}, true_sigma),
    ],
);
```

+++ {"ein.tags": ["worksheet-0"], "slideshow": {"slide_type": "-"}}

You can also pass the set of AR parameters as a list.

```{code-cell} ipython3
---
ein.tags: [worksheet-0]
slideshow:
  slide_type: '-'
---
with pm.Model() as ar2_bis:
    beta0 = pm.Normal("theta0", mu=0.0, sigma=1.0)
    beta1 = pm.Uniform("theta1", -1, 1)
    sigma = pm.HalfNormal("sigma", 3)
    likelhood = pm.AR("y", [beta0, beta1], sigma=sigma, observed=y)

    trace = pm.sample(
        1000,
        tune=2000,
        random_seed=RANDOM_SEED,
    )
    idata = pm.to_inference_data(trace)
```

```{code-cell} ipython3
az.plot_trace(
    idata,
    lines=[("theta0", {}, true_theta), ("theta1", {}, 0.0), ("sigma", {}, true_sigma)],
);
```

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w
```
