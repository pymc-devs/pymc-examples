---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3.9.7 ('pymc3_stable')
  language: python
  name: python3
---

# Sample callback

This notebook demonstrates the usage of the callback attribute in `pm.sample`. A callback is a function which gets called for every sample from the trace of a chain. The function is called with the trace and the current draw as arguments and will contain all samples for a single trace.

The sampling process can be interrupted by throwing a `KeyboardInterrupt` from inside the callback.

use-cases for this callback include:

 - Stopping sampling when a number of effective samples is reached
 - Stopping sampling when there are too many divergences
 - Logging metrics to external tools (such as TensorBoard)
 
We'll start with defining a simple model

```{code-cell} ipython3
import numpy as np
import pymc3 as pm

X = np.array([1, 2, 3, 4, 5])
y = X * 2 + np.random.randn(len(X))
with pm.Model() as model:

    intercept = pm.Normal("intercept", 0, 10)
    slope = pm.Normal("slope", 0, 10)

    mean = intercept + slope * X
    error = pm.HalfCauchy("error", 1)
    obs = pm.Normal("obs", mean, error, observed=y)
```

We can then for example add a callback that stops sampling whenever 100 samples are made, regardless of the number of draws set in the `pm.sample`

```{code-cell} ipython3
def my_callback(trace, draw):
    if len(trace) >= 100:
        raise KeyboardInterrupt()


with model:
    trace = pm.sample(tune=0, draws=500, callback=my_callback, chains=1)

print(len(trace))
```

Something to note though, is that the trace we get passed in the callback only correspond to a single chain. That means that if we want to do calculations over multiple chains at once, we'll need a bit of machinery to make this possible.

```{code-cell} ipython3
def my_callback(trace, draw):
    if len(trace) % 100 == 0:
        print(len(trace))


with model:
    trace = pm.sample(tune=0, draws=500, callback=my_callback, chains=2, cores=2)
```

We can use the `draw.chain` attribute to figure out which chain the current draw and trace belong to. Combined with some kind of convergence statistic like r_hat we can stop when we have converged, regardless of the amount of specified draws.

```{code-cell} ipython3
import arviz as az


class MyCallback:
    def __init__(self, every=1000, max_rhat=1.05):
        self.every = every
        self.max_rhat = max_rhat
        self.traces = {}

    def __call__(self, trace, draw):
        if draw.tuning:
            return

        self.traces[draw.chain] = trace
        if len(trace) % self.every == 0:
            multitrace = pm.backends.base.MultiTrace(list(self.traces.values()))
            if pm.stats.rhat(multitrace).to_array().max() < self.max_rhat:
                raise KeyboardInterrupt


with model:
    trace = pm.sample(tune=1000, draws=100000, callback=MyCallback(), chains=2, cores=2)
```

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w
```
