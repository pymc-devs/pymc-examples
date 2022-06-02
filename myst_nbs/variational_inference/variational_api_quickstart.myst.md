---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: pymc
  language: python
  name: pymc
---

# Variational API quickstart

:tags: variational inference

The variational inference (VI) API is focused on approximating posterior distributions for Bayesian models. Common use cases to which this module can be applied include:

* Sampling from model posterior and computing arbitrary expressions
* Conduct Monte Carlo approximation of expectation, variance, and other statistics
* Remove symbolic dependence on PyMC random nodes and evaluate expressions (using `eval`)
* Provide a bridge to arbitrary Theano code

Sounds good, doesn't it?

The module provides an interface to a variety of inference methods, so you are free to choose what is most appropriate for the problem.

```{code-cell} ipython3
%matplotlib inline
import aesara
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

np.random.seed(42)
pm.set_at_rng(42)
```

## Basic setup

We do not need complex models to play with the VI API; let's begin with a simple mixture model:

```{code-cell} ipython3
w = pm.floatX([0.2, 0.8])
mu = pm.floatX([-0.3, 0.5])
sd = pm.floatX([0.1, 0.1])

with pm.Model() as model:
    x = pm.NormalMixture("x", w=w, mu=mu, sigma=sd)
    x2 = x**2
    sin_x = pm.math.sin(x)
```

We can't compute analytical expectations for this model. However, we can obtain an approximation using Markov chain Monte Carlo methods; let's use NUTS first. 

To allow samples of the expressions to be saved, we need to wrap them in `Deterministic` objects:

```{code-cell} ipython3
with model:
    pm.Deterministic("x2", x2)
    pm.Deterministic("sin_x", sin_x)
```

```{code-cell} ipython3
with model:
    trace = pm.sample(50000)
```

```{code-cell} ipython3
az.plot_trace(trace);
```

Above are traces for $x^2$ and $sin(x)$. We can see there is clear multi-modality in this model. One drawback, is that you need to know in advance what exactly you want to see in trace and wrap it with `Deterministic`.

The VI API takes an alternate approach: You obtain inference from model, then calculate expressions based on this model afterwards. 

Let's use the same model:

```{code-cell} ipython3
with pm.Model() as model:

    x = pm.NormalMixture("x", w=w, mu=mu, sigma=sd)
    x2 = x**2
    sin_x = pm.math.sin(x)
```

Here we will use automatic differentiation variational inference (ADVI).

```{code-cell} ipython3
with model:
    mean_field = pm.fit(method="advi")
```

```{code-cell} ipython3
az.plot_posterior(mean_field.sample(1000), color="LightSeaGreen");
```

Notice that ADVI has failed to approximate the multimodal distribution, since it uses a Gaussian distribution that has a single mode.

## Checking convergence

```{code-cell} ipython3
help(pm.callbacks.CheckParametersConvergence)
```

Let's use the default arguments for `CheckParametersConvergence` as they seem to be reasonable.

```{code-cell} ipython3
from pymc.variational.callbacks import CheckParametersConvergence

with model:
    mean_field = pm.fit(method="advi", callbacks=[CheckParametersConvergence()])
```

We can access inference history via `.hist` attribute.

```{code-cell} ipython3
plt.plot(mean_field.hist);
```

This is not a good convergence plot, despite the fact that we ran many iterations. The reason is that the mean of the ADVI approximation is close to zero, and therefore taking the relative difference (the default method) is unstable for checking convergence.

```{code-cell} ipython3
with model:
    mean_field = pm.fit(
        method="advi", callbacks=[pm.callbacks.CheckParametersConvergence(diff="absolute")]
    )
```

```{code-cell} ipython3
plt.plot(mean_field.hist);
```

That's much better! We've reached convergence after less than 5000 iterations.

+++

## Tracking parameters

+++

Another useful callback allows users to track parameters. It allows for the tracking of arbitrary statistics during inference, though it can be memory-hungry. Using the `fit` function, we do not have direct access to the approximation before inference. However, tracking parameters requires access to the approximation. We can get around this constraint by using the object-oriented (OO) API for inference.

```{code-cell} ipython3
with model:
    advi = pm.ADVI()
```

```{code-cell} ipython3
advi.approx
```

Different approximations have different hyperparameters. In mean-field ADVI, we have $\rho$ and $\mu$ (inspired by [Bayes by BackProp](https://arxiv.org/abs/1505.05424)).

```{code-cell} ipython3
advi.approx.shared_params
```

There are convenient shortcuts to relevant statistics associated with the approximation. This can be useful, for example, when specifying a mass matrix for NUTS sampling:

```{code-cell} ipython3
advi.approx.mean.eval(), advi.approx.std.eval()
```

We can roll these statistics into the `Tracker` callback.

```{code-cell} ipython3
tracker = pm.callbacks.Tracker(
    mean=advi.approx.mean.eval,  # callable that returns mean
    std=advi.approx.std.eval,  # callable that returns std
)
```

Now, calling `advi.fit` will record the mean and standard deviation of the approximation as it runs.

```{code-cell} ipython3
approx = advi.fit(20000, callbacks=[tracker])
```

We can now plot both the evidence lower bound and parameter traces:

```{code-cell} ipython3
fig = plt.figure(figsize=(16, 9))
mu_ax = fig.add_subplot(221)
std_ax = fig.add_subplot(222)
hist_ax = fig.add_subplot(212)
mu_ax.plot(tracker["mean"])
mu_ax.set_title("Mean track")
std_ax.plot(tracker["std"])
std_ax.set_title("Std track")
hist_ax.plot(advi.hist)
hist_ax.set_title("Negative ELBO track");
```

Notice that there are convergence issues with the mean, and that lack of convergence does not seem to change the ELBO trajectory significantly. As we are using the OO API, we can run the approximation longer until convergence is achieved.

```{code-cell} ipython3
advi.refine(100000)
```

Let's take a look:

```{code-cell} ipython3
fig = plt.figure(figsize=(16, 9))
mu_ax = fig.add_subplot(221)
std_ax = fig.add_subplot(222)
hist_ax = fig.add_subplot(212)
mu_ax.plot(tracker["mean"])
mu_ax.set_title("Mean track")
std_ax.plot(tracker["std"])
std_ax.set_title("Std track")
hist_ax.plot(advi.hist)
hist_ax.set_title("Negative ELBO track");
```

We still see evidence for lack of convergence, as the mean has devolved into a random walk. This could be the result of choosing a poor algorithm for inference. At any rate, it is unstable and can produce very different results even using different random seeds.

Let's compare results with the NUTS output:

```{code-cell} ipython3
import seaborn as sns

ax = sns.kdeplot(trace.posterior["x"].values.flatten(), label="NUTS")
sns.kdeplot(approx.sample(10000).posterior["x"].values.flatten(), label="ADVI");
```

Again, we see that ADVI is not able to cope with multimodality; we can instead use SVGD, which generates an approximation based on a large number of particles.

```{code-cell} ipython3
with model:
    svgd_approx = pm.fit(
        300,
        method="svgd",
        inf_kwargs=dict(n_particles=1000),
        obj_optimizer=pm.sgd(learning_rate=0.01),
    )
```

```{code-cell} ipython3
ax = sns.kdeplot(trace.posterior["x"].values.flatten(), label="NUTS")
sns.kdeplot(approx.sample(10000).posterior["x"].values.flatten(), label="ADVI")
sns.kdeplot(svgd_approx.sample(2000).posterior["x"].values.flatten(), label="SVGD");
```

That did the trick, as we now have a multimodal approximation using SVGD. 

With this, it is possible to calculate arbitrary functions of the parameters with this variational approximation. For example we can calculate $x^2$ and $sin(x)$, as with the NUTS model.

```{code-cell} ipython3
# recall x ~ NormalMixture
a = x**2
b = pm.math.sin(x)
```

To evaluate these expressions with the approximation, we need `approx.sample_node`.

```{code-cell} ipython3
help(svgd_approx.sample_node)
```

```{code-cell} ipython3
a_sample = svgd_approx.sample_node(a)
a_sample.eval()
```

```{code-cell} ipython3
aesara.dprint(a)
```

```{code-cell} ipython3
a_sample.eval()
```

```{code-cell} ipython3
a_sample.eval()
```

Every call yields a different value from the same theano node. This is because it is **stochastic**. 

By applying replacements, we are now free of the dependence on the PyMC model; instead, we now depend on the approximation. Changing it will change the distribution for stochastic nodes:

```{code-cell} ipython3
sns.kdeplot(np.array([a_sample.eval() for _ in range(2000)]))
plt.title("$x^2$ distribution");
```

There is a more convenient way to get lots of samples at once: `sample_node`

```{code-cell} ipython3
a_samples = svgd_approx.sample_node(a, size=1000)
```

```{code-cell} ipython3
sns.kdeplot(a_samples.eval())
plt.title("$x^2$ distribution");
```

The `sample_node` function includes an additional dimension, so taking expectations or calculating variance is specified by `axis=0`.

```{code-cell} ipython3
a_samples.var(0).eval()  # variance
```

```{code-cell} ipython3
a_samples.mean(0).eval()  # mean
```

A symbolic sample size can also be specified:

```{code-cell} ipython3
i = theano.tensor.iscalar("i")
i.tag.test_value = 1
a_samples_i = svgd_approx.sample_node(a, size=i)
```

```{code-cell} ipython3
a_samples_i.eval({i: 100}).shape
```

```{code-cell} ipython3
a_samples_i.eval({i: 10000}).shape
```

Unfortunately the size must be a scalar value.

+++

### Converting a Trace to an Approximation

We can convert a MCMC trace into an Approximation. It will have the same API as approximations above with same `sample_node` methods:

```{code-cell} ipython3
trace_approx = pm.Empirical(trace, model=model)
trace_approx
```

We can then draw samples from the `Emipirical` object:

```{code-cell} ipython3
az.plot_posterior(trace_approx.sample(10000));
```

## Multilabel logistic regression

Let's illustrate the use of `Tracker` with the famous Iris dataset. We'll attempy multi-label classification and compute the expected accuracy score as a diagnostic.

```{code-cell} ipython3
import pandas as pd
import theano.tensor as tt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

![](http://5047-presscdn.pagely.netdna-cdn.com/wp-content/uploads/2015/04/iris_petal_sepal.png)

+++

A relatively simple model will be sufficient here because the classes are roughly linearly separable; we are going to fit multinomial logistic regression.

```{code-cell} ipython3
Xt = theano.shared(X_train)
yt = theano.shared(y_train)

with pm.Model() as iris_model:

    # Coefficients for features
    β = pm.Normal("β", 0, sigma=1e2, shape=(4, 3))
    # Transoform to unit interval
    a = pm.Flat("a", shape=(3,))
    p = tt.nnet.softmax(Xt.dot(β) + a)

    observed = pm.Categorical("obs", p=p, observed=yt)
```

### Applying replacements in practice
PyMC models have symbolic inputs for latent variables. To evaluate an espression that requires knowledge of latent variables, one needs to provide fixed values. We can use values approximated by VI for this purpose. The function `sample_node` removes the symbolic dependenices. 

`sample_node` will use the whole distribution at each step, so we will use it here. We can apply more replacements in single function call using the `more_replacements` keyword argument in both replacement functions.

> **HINT:** You can use `more_replacements` argument when calling `fit` too:
>   * `pm.fit(more_replacements={full_data: minibatch_data})`
>   * `inference.fit(more_replacements={full_data: minibatch_data})`

```{code-cell} ipython3
with iris_model:

    # We'll use SVGD
    inference = pm.SVGD(n_particles=500, jitter=1)

    # Local reference to approximation
    approx = inference.approx

    # Here we need `more_replacements` to change train_set to test_set
    test_probs = approx.sample_node(p, more_replacements={Xt: X_test}, size=100)

    # For train set no more replacements needed
    train_probs = approx.sample_node(p)
```

By applying the code above, we now have 100 sampled probabilities (default number for `sample_node` is `None`) for each observation.

+++

Next we create symbolic expressions for sampled accuracy scores:

```{code-cell} ipython3
test_ok = tt.eq(test_probs.argmax(-1), y_test)
train_ok = tt.eq(train_probs.argmax(-1), y_train)
test_accuracy = test_ok.mean(-1)
train_accuracy = train_ok.mean(-1)
```

Tracker expects callables so we can pass `.eval` method of theano node that is function itself. 

Calls to this function are cached so they can be reused.

```{code-cell} ipython3
eval_tracker = pm.callbacks.Tracker(
    test_accuracy=test_accuracy.eval, train_accuracy=train_accuracy.eval
)
```

```{code-cell} ipython3
inference.fit(100, callbacks=[eval_tracker]);
```

```{code-cell} ipython3
_, ax = plt.subplots(1, 1)
df = pd.DataFrame(eval_tracker["test_accuracy"]).T.melt()
sns.lineplot(x="variable", y="value", data=df, color="red", ax=ax)
ax.plot(eval_tracker["train_accuracy"], color="blue")
ax.set_xlabel("epoch")
plt.legend(["test_accuracy", "train_accuracy"])
plt.title("Training Progress");
```

Training does not seem to be working here. Let's use a different optimizer and boost the learning rate.

```{code-cell} ipython3
inference.fit(400, obj_optimizer=pm.adamax(learning_rate=0.1), callbacks=[eval_tracker]);
```

```{code-cell} ipython3
_, ax = plt.subplots(1, 1)
df = pd.DataFrame(np.asarray(eval_tracker["test_accuracy"])).T.melt()
sns.lineplot(x="variable", y="value", data=df, color="red", ax=ax)
ax.plot(eval_tracker["train_accuracy"], color="blue")
ax.set_xlabel("epoch")
plt.legend(["test_accuracy", "train_accuracy"])
plt.title("Training Progress");
```

This is much better! 

So, `Tracker` allows us to monitor our approximation and choose good training schedule.

+++

## Minibatches
When dealing with large datasets, using minibatch training can drastically speed up and improve approximation performance. Large datasets impose a hefty cost on the computation of gradients. 

There is a nice API in pymc to handle these cases, which is available through the `pm.Minibatch` class. The minibatch is just a highly specialized Theano tensor:

```{code-cell} ipython3
issubclass(pm.Minibatch, theano.tensor.TensorVariable)
```

To demonstrate, let's simulate a large quantity of data:

```{code-cell} ipython3
# Raw values
data = np.random.rand(40000, 100)
# Scaled values
data *= np.random.randint(1, 10, size=(100,))
# Shifted values
data += np.random.rand(100) * 10
```

For comparison, let's fit a model without minibatch processing:

```{code-cell} ipython3
with pm.Model() as model:
    mu = pm.Flat("mu", shape=(100,))
    sd = pm.HalfNormal("sd", shape=(100,))
    lik = pm.Normal("lik", mu, sd, observed=data)
```

Just for fun, let's create a custom special purpose callback to halt slow optimization. Here we define a callback that causes a hard stop when approximation runs too slowly:

```{code-cell} ipython3
def stop_after_10(approx, loss_history, i):
    if (i > 0) and (i % 10) == 0:
        raise StopIteration("I was slow, sorry")
```

```{code-cell} ipython3
with model:
    advifit = pm.fit(callbacks=[stop_after_10])
```

Inference is too slow, taking several seconds per iteration; fitting the approximation would have taken hours!

Now let's use minibatches. At every iteration, we will draw 500 random values:

> Remember to set `total_size` in observed

**total_size** is an important parameter that allows pymc to infer the right way of rescaling densities. If it is not set, you are likely to get completely wrong results. For more information please refer to the comprehensive documentation of `pm.Minibatch`.

```{code-cell} ipython3
X = pm.Minibatch(data, batch_size=500)

with pm.Model() as model:

    mu = pm.Flat("mu", shape=(100,))
    sd = pm.HalfNormal("sd", shape=(100,))
    likelihood = pm.Normal("likelihood", mu, sd, observed=X, total_size=data.shape)
```

```{code-cell} ipython3
with model:
    advifit = pm.fit()
```

```{code-cell} ipython3
plt.plot(advifit.hist);
```

Minibatch inference is dramatically faster. Multidimensional minibatches may be needed for some corner cases where you do matrix factorization or model is very wide.

Here is the docstring for `Minibatch` to illustrate how it can be customized.

```{code-cell} ipython3
print(pm.Minibatch.__doc__)
```

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w
```
