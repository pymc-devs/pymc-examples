---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

+++ {"button": false, "new_sheet": false, "run_control": {"read_only": false}, "slideshow": {"slide_type": "slide"}}

# Inferring parameters of SDEs using a Euler-Maruyama scheme

_This notebook is derived from a presentation prepared for the Theoretical Neuroscience Group, Institute of Systems Neuroscience at Aix-Marseile University._

```{code-cell} ipython3
---
button: false
new_sheet: false
run_control:
  read_only: false
slideshow:
  slide_type: '-'
---
%pylab inline
import arviz as az
import pymc3 as pm
import scipy
import theano.tensor as tt

from pymc3.distributions.timeseries import EulerMaruyama
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")
```

+++ {"button": false, "nbpresent": {"id": "2325c7f9-37bd-4a65-aade-86bee1bff5e3"}, "new_sheet": false, "run_control": {"read_only": false}, "slideshow": {"slide_type": "slide"}}

## Toy model 1

Here's a scalar linear SDE in symbolic form

$ dX_t = \lambda X_t + \sigma^2 dW_t $

discretized with the Euler-Maruyama scheme

```{code-cell} ipython3
---
button: false
new_sheet: false
run_control:
  read_only: false
---
# parameters
λ = -0.78
σ2 = 5e-3
N = 200
dt = 1e-1

# time series
x = 0.1
x_t = []

# simulate
for i in range(N):
    x += dt * λ * x + sqrt(dt) * σ2 * randn()
    x_t.append(x)

x_t = array(x_t)

# z_t noisy observation
z_t = x_t + randn(x_t.size) * 5e-3
```

```{code-cell} ipython3
---
button: false
nbpresent:
  id: 0994bfef-45dc-48da-b6bf-c7b38d62bf11
new_sheet: false
run_control:
  read_only: false
slideshow:
  slide_type: subslide
---
figure(figsize=(10, 3))
subplot(121)
plot(x_t[:30], "k", label="$x(t)$", alpha=0.5), plot(z_t[:30], "r", label="$z(t)$", alpha=0.5)
title("Transient"), legend()
subplot(122)
plot(x_t[30:], "k", label="$x(t)$", alpha=0.5), plot(z_t[30:], "r", label="$z(t)$", alpha=0.5)
title("All time")
tight_layout()
```

+++ {"button": false, "new_sheet": false, "run_control": {"read_only": false}}

What is the inference we want to make? Since we've made a noisy observation of the generated time series, we need to estimate both $x(t)$ and $\lambda$.

+++ {"button": false, "new_sheet": false, "run_control": {"read_only": false}, "slideshow": {"slide_type": "subslide"}}

First, we rewrite our SDE as a function returning a tuple of the drift and diffusion coefficients

```{code-cell} ipython3
---
button: false
new_sheet: false
run_control:
  read_only: false
---
def lin_sde(x, lam):
    return lam * x, σ2
```

+++ {"button": false, "new_sheet": false, "run_control": {"read_only": false}, "slideshow": {"slide_type": "subslide"}}

Next, we describe the probability model as a set of three stochastic variables, `lam`, `xh`, and `zh`:

```{code-cell} ipython3
---
button: false
nbpresent:
  id: 4f90230d-f303-4b3b-a69e-304a632c6407
new_sheet: false
run_control:
  read_only: false
slideshow:
  slide_type: '-'
---
with pm.Model() as model:
    # uniform prior, but we know it must be negative
    lam = pm.Flat("lam")

    # "hidden states" following a linear SDE distribution
    # parametrized by time step (det. variable) and lam (random variable)
    xh = EulerMaruyama("xh", dt, lin_sde, (lam,), shape=N, testval=x_t)

    # predicted observation
    zh = pm.Normal("zh", mu=xh, sigma=5e-3, observed=z_t)
```

+++ {"button": false, "nbpresent": {"id": "287d10b5-0193-4ffe-92a7-362993c4b72e"}, "new_sheet": false, "run_control": {"read_only": false}, "slideshow": {"slide_type": "subslide"}}

Once the model is constructed, we perform inference, i.e. sample from the posterior distribution, in the following steps:

```{code-cell} ipython3
---
button: false
new_sheet: false
run_control:
  read_only: false
---
with model:
    trace = pm.sample(2000, tune=1000)
```

+++ {"button": false, "new_sheet": false, "run_control": {"read_only": false}, "slideshow": {"slide_type": "subslide"}}

Next, we plot some basic statistics on the samples from the posterior,

```{code-cell} ipython3
---
button: false
nbpresent:
  id: 925f1829-24cb-4c28-9b6b-7e9c9e86f2fd
new_sheet: false
run_control:
  read_only: false
---
figure(figsize=(10, 3))
subplot(121)
plot(percentile(trace[xh], [2.5, 97.5], axis=0).T, "k", label=r"$\hat{x}_{95\%}(t)$")
plot(x_t, "r", label="$x(t)$")
legend()

subplot(122)
hist(trace[lam], 30, label=r"$\hat{\lambda}$", alpha=0.5)
axvline(λ, color="r", label=r"$\lambda$", alpha=0.5)
legend();
```

+++ {"button": false, "new_sheet": false, "run_control": {"read_only": false}, "slideshow": {"slide_type": "subslide"}}

A model can fit the data precisely and still be wrong; we need to use _posterior predictive checks_ to assess if, under our fit model, the data our likely.

In other words, we 
- assume the model is correct
- simulate new observations
- check that the new observations fit with the original data

```{code-cell} ipython3
---
button: false
new_sheet: false
run_control:
  read_only: false
---
# generate trace from posterior
ppc_trace = pm.sample_posterior_predictive(trace, model=model)

# plot with data
figure(figsize=(10, 3))
plot(percentile(ppc_trace["zh"], [2.5, 97.5], axis=0).T, "k", label=r"$z_{95\% PP}(t)$")
plot(z_t, "r", label="$z(t)$")
legend()
```

+++ {"button": false, "new_sheet": false, "run_control": {"read_only": false}}

Note that 

- inference also estimates the initial conditions
- the observed data $z(t)$ lies fully within the 95% interval of the PPC.
- there are many other ways of evaluating fit

+++ {"button": false, "new_sheet": false, "run_control": {"read_only": false}, "slideshow": {"slide_type": "slide"}}

### Toy model 2

As the next model, let's use a 2D deterministic oscillator, 
\begin{align}
\dot{x} &= \tau (x - x^3/3 + y) \\
\dot{y} &= \frac{1}{\tau} (a - x)
\end{align}

with noisy observation $z(t) = m x + (1 - m) y + N(0, 0.05)$.

```{code-cell} ipython3
---
button: false
new_sheet: false
run_control:
  read_only: false
---
N, τ, a, m, σ2 = 200, 3.0, 1.05, 0.2, 1e-1
xs, ys = [0.0], [1.0]
for i in range(N):
    x, y = xs[-1], ys[-1]
    dx = τ * (x - x**3.0 / 3.0 + y)
    dy = (1.0 / τ) * (a - x)
    xs.append(x + dt * dx + sqrt(dt) * σ2 * randn())
    ys.append(y + dt * dy + sqrt(dt) * σ2 * randn())
xs, ys = array(xs), array(ys)
zs = m * xs + (1 - m) * ys + randn(xs.size) * 0.1

figure(figsize=(10, 2))
plot(xs, label="$x(t)$")
plot(ys, label="$y(t)$")
plot(zs, label="$z(t)$")
legend()
```

+++ {"button": false, "new_sheet": false, "run_control": {"read_only": false}, "slideshow": {"slide_type": "subslide"}}

Now, estimate the hidden states $x(t)$ and $y(t)$, as well as parameters $\tau$, $a$ and $m$.

As before, we rewrite our SDE as a function returned drift & diffusion coefficients:

```{code-cell} ipython3
---
button: false
new_sheet: false
run_control:
  read_only: false
---
def osc_sde(xy, τ, a):
    x, y = xy[:, 0], xy[:, 1]
    dx = τ * (x - x**3.0 / 3.0 + y)
    dy = (1.0 / τ) * (a - x)
    dxy = tt.stack([dx, dy], axis=0).T
    return dxy, σ2
```

+++ {"button": false, "new_sheet": false, "run_control": {"read_only": false}}

As before, the Euler-Maruyama discretization of the SDE is written as a prediction of the state at step $i+1$ based on the state at step $i$.

+++ {"button": false, "new_sheet": false, "run_control": {"read_only": false}, "slideshow": {"slide_type": "subslide"}}

We can now write our statistical model as before, with uninformative priors on $\tau$, $a$ and $m$:

```{code-cell} ipython3
---
button: false
new_sheet: false
run_control:
  read_only: false
---
xys = c_[xs, ys]

with pm.Model() as model:
    τh = pm.Uniform("τh", lower=0.1, upper=5.0)
    ah = pm.Uniform("ah", lower=0.5, upper=1.5)
    mh = pm.Uniform("mh", lower=0.0, upper=1.0)
    xyh = EulerMaruyama("xyh", dt, osc_sde, (τh, ah), shape=xys.shape, testval=xys)
    zh = pm.Normal("zh", mu=mh * xyh[:, 0] + (1 - mh) * xyh[:, 1], sigma=0.1, observed=zs)
```

```{code-cell} ipython3
---
button: false
new_sheet: false
run_control:
  read_only: false
---
with model:
    trace = pm.sample(2000, tune=1000)
```

+++ {"button": false, "new_sheet": false, "run_control": {"read_only": false}, "slideshow": {"slide_type": "subslide"}}

Again, the result is a set of samples from the posterior, including our parameters of interest but also the hidden states

```{code-cell} ipython3
---
button: false
new_sheet: false
run_control:
  read_only: false
---
figure(figsize=(10, 6))
subplot(211)
plot(percentile(trace[xyh][..., 0], [2.5, 97.5], axis=0).T, "k", label=r"$\hat{x}_{95\%}(t)$")
plot(xs, "r", label="$x(t)$")
legend(loc=0)
subplot(234), hist(trace["τh"]), axvline(τ), xlim([1.0, 4.0]), title("τ")
subplot(235), hist(trace["ah"]), axvline(a), xlim([0, 2.0]), title("a")
subplot(236), hist(trace["mh"]), axvline(m), xlim([0, 1]), title("m")
tight_layout()
```

+++ {"button": false, "new_sheet": false, "run_control": {"read_only": false}, "slideshow": {"slide_type": "subslide"}}

Again, we can perform a posterior predictive check, that our data are likely given the fit model

```{code-cell} ipython3
---
button: false
new_sheet: false
run_control:
  read_only: false
---
# generate trace from posterior
ppc_trace = pm.sample_posterior_predictive(trace, model=model)

# plot with data
figure(figsize=(10, 3))
plot(percentile(ppc_trace["zh"], [2.5, 97.5], axis=0).T, "k", label=r"$z_{95\% PP}(t)$")
plot(zs, "r", label="$z(t)$")
legend()
```

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w
```
