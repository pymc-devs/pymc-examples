---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# General API quickstart

```{code-cell} ipython3
import warnings

import aesara.tensor as at
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

warnings.simplefilter(action="ignore", category=FutureWarning)
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")
print(f"Running on PyMC v{pm.__version__}")
print(f"Running on ArviZ v{az.__version__}")
```

## 1. Model creation

Models in PyMC are centered around the `Model` class. It has references to all random variables (RVs) and computes the model logp and its gradients. Usually, you would instantiate it as part of a `with` context:

```{code-cell} ipython3
with pm.Model() as model:
    # Model definition
    pass
```

We discuss RVs further below but let's create a simple model to explore the `Model` class.

```{code-cell} ipython3
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=1, observed=np.random.randn(100))
```

```{code-cell} ipython3
model.basic_RVs
```

```{code-cell} ipython3
model.free_RVs
```

```{code-cell} ipython3
model.observed_RVs
```

```{code-cell} ipython3
model.logp({"mu": 0})
```

It's worth highlighting the design choice we made with `logp`.  As you can see above, `logp` is being called with arguments, so it's a method of the model instance. More precisely, it puts together a function based on the current state of the model -- or on the state given as argument to `logp` (see example below).

For diverse reasons, we assume that a `Model` instance isn't static. If you need to use `logp` in an inner loop and it needs to be static, simply use something like `logp = model.logp`. Here is an example below -- note the caching effect and the speed up:

```{code-cell} ipython3
%timeit model.logp({mu: 0.1})
logp = model.logp
%timeit logp({mu: 0.1})
```

## 2. Probability Distributions

Every probabilistic program consists of observed and unobserved Random Variables (RVs). Observed RVs are defined via likelihood distributions, while unobserved RVs are defined via prior distributions. In PyMC, probability distributions are available from the main module space:

```{code-cell} ipython3
help(pm.Normal)
```

In the PyMC module, the structure for probability distributions looks like this:

[pymc.distributions](../api/distributions.rst)
- [continuous](../api/distributions/continuous.rst)
- [discrete](../api/distributions/discrete.rst)
- [timeseries](../api/distributions/timeseries.rst)
- [mixture](../api/distributions/mixture.rst)

```{code-cell} ipython3
dir(pm.distributions.mixture)
```

### Unobserved Random Variables

+++

Every unobserved RV has the following calling signature: name (str), parameter keyword arguments. Thus, a normal prior can be defined in a model context like this:

```{code-cell} ipython3
with pm.Model():
    x = pm.Normal("x", mu=0, sigma=1)
```

As with the model, we can evaluate its logp:

```{code-cell} ipython3
x.logp({"x": 0})
```

### Observed Random Variables

+++

Observed RVs are defined just like unobserved RVs but require data to be passed into the `observed` keyword argument:

```{code-cell} ipython3
with pm.Model():
    obs = pm.Normal("x", mu=0, sigma=1, observed=np.random.randn(100))
```

`observed` supports lists, `numpy.ndarray`, `aesara` and `pandas` data structures.

+++

### Deterministic transforms

+++

PyMC allows you to freely do algebra with RVs in all kinds of ways:

```{code-cell} ipython3
with pm.Model():
    x = pm.Normal("x", mu=0, sigma=1)
    y = pm.Gamma("y", alpha=1, beta=1)
    plus_2 = x + 2
    summed = x + y
    squared = x**2
    sined = pm.math.sin(x)
```

While these transformations work seamlessly, their results are not stored automatically. Thus, if you want to keep track of a transformed variable, you have to use `pm.Deterministic`:

```{code-cell} ipython3
with pm.Model():
    x = pm.Normal("x", mu=0, sigma=1)
    plus_2 = pm.Deterministic("x plus 2", x + 2)
```

Note that `plus_2` can be used in the identical way to above, we only tell PyMC to keep track of this RV for us.

+++

### Automatic transforms of bounded RVs

In order to sample models more efficiently, PyMC automatically transforms bounded RVs to be unbounded.

```{code-cell} ipython3
with pm.Model() as model:
    x = pm.Uniform("x", lower=0, upper=1)
```

When we look at the RVs of the model, we would expect to find `x` there, however:

```{code-cell} ipython3
model.free_RVs
```

`x_interval__` represents `x` transformed to accept parameter values between -inf and +inf. In the case of an upper and a lower bound, a `LogOdd`s transform is applied. Sampling in this transformed space makes it easier for the sampler. PyMC also keeps track of the non-transformed, bounded parameters. These are common determinstics (see above):

```{code-cell} ipython3
model.deterministics
```

When displaying results, PyMC will usually hide transformed parameters. You can pass the `include_transformed=True` parameter to many functions to see the transformed parameters that are used for sampling.

You can also turn transforms off:

```{code-cell} ipython3
with pm.Model() as model:
    x = pm.Uniform("x", lower=0, upper=1, transform=None)

print(model.free_RVs)
```

Or specify different transformation other than the default:

```{code-cell} ipython3
import pymc.distributions.transforms as tr

with pm.Model() as model:
    # use the default log transformation
    x1 = pm.Gamma("x1", alpha=1, beta=1)
    # specify a different transformation
    x2 = pm.Gamma("x2", alpha=1, beta=1, transform=tr.log_exp_m1)

print("The default transformation of x1 is: " + x1.transformation.name)
print("The user specified transformation of x2 is: " + x2.transformation.name)
```

### Transformed distributions and changes of variables
PyMC does not provide explicit functionality to transform one distribution to another. Instead, a dedicated distribution is usually created in consideration of optimising performance. However, users can still create transformed distribution by passing the inverse transformation to `transform` kwarg. Take the classical textbook example of LogNormal: $log(y) \sim \text{Normal}(\mu, \sigma)$

```{code-cell} ipython3
class Exp(tr.ElemwiseTransform):
    name = "exp"

    def backward(self, x):
        return at.log(x)

    def forward(self, x):
        return at.exp(x)

    def jacobian_det(self, x):
        return -at.log(x)


with pm.Model() as model:
    x1 = pm.Normal("x1", 0.0, 1.0, transform=Exp())
    x2 = pm.Lognormal("x2", 0.0, 1.0)

lognorm1 = model.named_vars["x1_exp__"]
lognorm2 = model.named_vars["x2"]

_, ax = plt.subplots(1, 1, figsize=(5, 3))
x = np.linspace(0.0, 10.0, 100)
ax.plot(
    x,
    np.exp(lognorm1.distribution.logp(x).eval()),
    "--",
    alpha=0.5,
    label="log(y) ~ Normal(0, 1)",
)
ax.plot(
    x,
    np.exp(lognorm2.distribution.logp(x).eval()),
    alpha=0.5,
    label="y ~ Lognormal(0, 1)",
)
plt.legend();
```

Notice from above that the named variable `x1_exp__` in the `model` is Lognormal distributed.  
Using similar approach, we can create ordered RVs following some distribution. For example, we can combine the `ordered` transformation and `logodds` transformation using `Chain` to create a 2D RV that satisfy $x_1, x_2 \sim \text{Uniform}(0, 1) \space and \space x_1< x_2$

```{code-cell} ipython3
Order = tr.Ordered()
Logodd = tr.LogOdds()
chain_tran = tr.Chain([Logodd, Order])

with pm.Model() as m0:
    x = pm.Uniform("x", 0.0, 1.0, shape=2, transform=chain_tran, testval=[0.1, 0.9])
    trace = pm.sample(5000, tune=1000, progressbar=False, return_inferencedata=False)
```

```{code-cell} ipython3
_, ax = plt.subplots(1, 2, figsize=(10, 5))
for ivar, varname in enumerate(trace.varnames):
    ax[ivar].scatter(trace[varname][:, 0], trace[varname][:, 1], alpha=0.01)
    ax[ivar].set_xlabel(varname + "[0]")
    ax[ivar].set_ylabel(varname + "[1]")
    ax[ivar].set_title(varname)
plt.tight_layout()
```

### Lists of RVs / higher-dimensional RVs

Above we have seen how to create scalar RVs. In many models, you want multiple RVs. There is a tendency (mainly inherited from PyMC 2.x) to create list of RVs, like this:

```{code-cell} ipython3
with pm.Model():
    # bad:
    x = [pm.Normal(f"x_{i}", mu=0, sigma=1) for i in range(10)]
```

However, even though this works it is quite slow and not recommended. Instead, use the `shape` kwarg:

```{code-cell} ipython3
with pm.Model() as model:
    # good:
    x = pm.Normal("x", mu=0, sigma=1, shape=10)
```

`x` is now a random vector of length 10. We can index into it or do linear algebra operations on it:

```{code-cell} ipython3
with model:
    y = x[0] * x[1]  # full indexing is supported
    x.dot(x.T)  # Linear algebra is supported
```

### Initialization with test_values

While PyMC tries to automatically initialize models it is sometimes helpful to define initial values for RVs. This can be done via the `testval` kwarg:

```{code-cell} ipython3
with pm.Model():
    x = pm.Normal("x", mu=0, sigma=1, shape=5)

x.tag.test_value
```

```{code-cell} ipython3
with pm.Model():
    x = pm.Normal("x", mu=0, sigma=1, shape=5, testval=np.random.randn(5))

x.tag.test_value
```

This technique is quite useful to identify problems with model specification or initialization.

+++

## 3. Inference

Once we have defined our model, we have to perform inference to approximate the posterior distribution. PyMC supports two broad classes of inference: sampling and variational inference.

### 3.1 Sampling

The main entry point to MCMC sampling algorithms is via the `pm.sample(return_inferencedata=False)` function. By default, this function tries to auto-assign the right sampler(s) and auto-initialize if you don't pass anything.

With PyMC version >=3.9 the `` kwarg makes the `sample` function return an `arviz.InferenceData` object instead of a `MultiTrace`. `InferenceData` has many advantages, compared to a `MultiTrace`: For example it can be saved/loaded from a file, and can also carry additional (meta)data such as date/version, or posterior predictive distributions. Take a look at the [ArviZ Quickstart](https://arviz-devs.github.io/arviz/getting_started/Introduction.html) to learn more.

```{code-cell} ipython3
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=1, observed=np.random.randn(100))

    idata = pm.sample(2000, tune=1500)
```

As you can see, on a continuous model, PyMC assigns the NUTS sampler, which is very efficient even for complex models. PyMC also runs tuning to find good starting parameters for the sampler. Here we draw 2000 samples from the posterior in each chain and allow the sampler to adjust its parameters in an additional 1500 iterations.
If not set via the `cores` kwarg, the number of chains is determined from the number of available CPU cores.

```{code-cell} ipython3
idata.posterior.dims
```

The tuning samples are discarded by default. With `discard_tuned_samples=False` they can be kept and end up in a special property of the `InferenceData` object.

You can also run multiple chains in parallel using the `chains` and `cores` kwargs:

```{code-cell} ipython3
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=1, observed=np.random.randn(100))

    idata = pm.sample(cores=4, chains=6)
```

```{code-cell} ipython3
idata.posterior["mu"].shape
```

```{code-cell} ipython3
# get values of a single chain
idata.posterior["mu"].sel(chain=1).shape
```

PyMC, offers a variety of other samplers, found in `pm.step_methods`.

```{code-cell} ipython3
list(filter(lambda x: x[0].isupper(), dir(pm.step_methods)))
```

Commonly used step-methods besides NUTS are `Metropolis` and `Slice`. **For almost all continuous models, `NUTS` should be preferred.** There are hard-to-sample models for which `NUTS` will be very slow causing many users to use `Metropolis` instead. This practice, however, is rarely successful. NUTS is fast on simple models but can be slow if the model is very complex or it is badly initialized. In the case of a complex model that is hard for NUTS, Metropolis, while faster, will have a very low effective sample size or not converge properly at all. A better approach is to instead try to improve initialization of NUTS, or reparameterize the model.

For completeness, other sampling methods can be passed to sample:

```{code-cell} ipython3
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=1, observed=np.random.randn(100))

    step = pm.Metropolis()
    trace = pm.sample(1000, step=step)
```

You can also assign variables to different step methods.

```{code-cell} ipython3
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    sd = pm.HalfNormal("sd", sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=sd, observed=np.random.randn(100))

    step1 = pm.Metropolis(vars=[mu])
    step2 = pm.Slice(vars=[sd])
    idata = pm.sample(10000, step=[step1, step2], cores=4)
```

### 3.2 Analyze sampling results

The most common used plot to analyze sampling results is the so-called trace-plot:

```{code-cell} ipython3
az.plot_trace(idata);
```

Another common metric to look at is R-hat, also known as the Gelman-Rubin statistic:

```{code-cell} ipython3
az.summary(idata)
```

These are also part of the `forestplot`:

```{code-cell} ipython3
az.plot_forest(idata, r_hat=True);
```

Finally, for a plot of the posterior that is inspired by the book [Doing Bayesian Data Analysis](http://www.indiana.edu/~kruschke/DoingBayesianDataAnalysis/), you can use the:

```{code-cell} ipython3
az.plot_posterior(idata);
```

For high-dimensional models it becomes cumbersome to look at all parameter's traces. When using `NUTS` we can look at the energy plot to assess problems of convergence:

```{code-cell} ipython3
with pm.Model() as model:
    x = pm.Normal("x", mu=0, sigma=1, shape=100)
    idata = pm.sample(cores=4)

az.plot_energy(idata);
```

For more information on sampler stats and the energy plot, see [here](sampler-stats.ipynb). For more information on identifying sampling problems and what to do about them, see [here](Diagnosing_biased_Inference_with_Divergences.ipynb).

+++

### 3.3 Variational inference

PyMC supports various Variational Inference techniques. While these methods are much faster, they are often also less accurate and can lead to biased inference. The main entry point is `pymc.fit()`.

```{code-cell} ipython3
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    sd = pm.HalfNormal("sd", sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=sd, observed=np.random.randn(100))

    approx = pm.fit()
```

The returned `Approximation` object has various capabilities, like drawing samples from the approximated posterior, which we can analyse like a regular sampling run:

```{code-cell} ipython3
approx.sample(500)
```

The `variational` submodule offers a lot of flexibility in which VI to use and follows an object oriented design. For example, full-rank ADVI estimates a full covariance matrix:

```{code-cell} ipython3
mu = pm.floatX([0.0, 0.0])
cov = pm.floatX([[1, 0.5], [0.5, 1.0]])
with pm.Model() as model:
    pm.MvNormal("x", mu=mu, cov=cov, shape=2)
    approx = pm.fit(method="fullrank_advi")
```

An equivalent expression using the object-oriented interface is:

```{code-cell} ipython3
with pm.Model() as model:
    pm.MvNormal("x", mu=mu, cov=cov, shape=2)
    approx = pm.FullRankADVI().fit()
```

```{code-cell} ipython3
plt.figure()
trace = approx.sample(10000)
az.plot_kde(trace["x"][:, 0], trace["x"][:, 1]);
```

Stein Variational Gradient Descent (SVGD) uses particles to estimate the posterior:

```{code-cell} ipython3
w = pm.floatX([0.2, 0.8])
mu = pm.floatX([-0.3, 0.5])
sd = pm.floatX([0.1, 0.1])
with pm.Model() as model:
    pm.NormalMixture("x", w=w, mu=mu, sigma=sd)
    approx = pm.fit(method=pm.SVGD(n_particles=200, jitter=1.0))
```

```{code-cell} ipython3
plt.figure()
trace = approx.sample(10000)
az.plot_dist(trace["x"]);
```

For more information on variational inference, see [these examples](http://pymc-devs.github.io/pymc/examples.html#variational-inference).

+++

## 4. Posterior Predictive Sampling

The `sample_posterior_predictive()` function performs prediction on hold-out data and posterior predictive checks.

```{code-cell} ipython3
data = np.random.randn(100)
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    sd = pm.HalfNormal("sd", sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=sd, observed=data)

    idata = pm.sample()
```

```{code-cell} ipython3
with model:
    post_pred = pm.sample_posterior_predictive(idata.posterior)
# add posterior predictive to the InferenceData
az.concat(idata, pm.to_inference_data(posterior_predictive=post_pred), inplace=True)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
az.plot_ppc(idata, ax=ax)
ax.axvline(data.mean(), ls="--", color="r", label="True mean")
ax.legend(fontsize=10);
```

## 4.1 Predicting on hold-out data

In many cases you want to predict on unseen / hold-out data. This is especially relevant in Probabilistic Machine Learning and Bayesian Deep Learning. We recently improved the API in this regard with the `pm.Data` container. It is a wrapper around a `aesara.shared` variable whose values can be changed later. Otherwise they can be passed into PyMC just like any other numpy array or tensor.

This distinction is significant since internally all models in PyMC are giant symbolic expressions. When you pass data directly into a model, you are giving Aesara permission to treat this data as a constant and optimize it away as it sees fit. If you need to change this data later you might not have a way to point at it in the symbolic expression. Using `aesara.shared` offers a way to point to a place in that symbolic expression, and change what is there.

```{code-cell} ipython3
x = np.random.randn(100)
y = x > 0

with pm.Model() as model:
    # create shared variables that can be changed later on
    x_shared = pm.Data("x_obs", x)
    y_shared = pm.Data("y_obs", y)

    coeff = pm.Normal("x", mu=0, sigma=1)
    logistic = pm.math.sigmoid(coeff * x_shared)
    pm.Bernoulli("obs", p=logistic, observed=y_shared)
    idata = pm.sample()
```

Now assume we want to predict on unseen data. For this we have to change the values of `x_shared` and `y_shared`. Theoretically we don't need to set `y_shared` as we want to predict it but it has to match the shape of `x_shared`.

```{code-cell} ipython3
with model:
    # change the value and shape of the data
    pm.set_data(
        {
            "x_obs": [-1, 0, 1.0],
            # use dummy values with the same shape:
            "y_obs": [0, 0, 0],
        }
    )

    post_pred = pm.sample_posterior_predictive(idata.posterior)
```

```{code-cell} ipython3
post_pred["obs"].mean(axis=0)
```

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w
```
