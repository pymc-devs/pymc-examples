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

# Empirical Approximation overview

:::
:tags: variational inference
:category: intermediate
:::

For most models we use sampling MCMC algorithms like Metropolis or NUTS. In PyMC3 we got used to store traces of MCMC samples and then do analysis using them. There is a similar concept for the variational inference submodule in PyMC3: *Empirical*. This type of approximation stores particles for the SVGD sampler. There is no difference between independent SVGD particles and MCMC samples. *Empirical* acts as a bridge between MCMC sampling output and full-fledged VI utils like `apply_replacements` or `sample_node`. For the interface description, see [variational_api_quickstart](variational_api_quickstart.ipynb). Here we will just focus on `Emprical` and give an overview of specific things for the *Empirical* approximation

```{code-cell} ipython3
import aesara
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from pandas import DataFrame

print(f"Running on PyMC v{pm.__version__}")
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")
np.random.seed(42)
pm.set_at_rng(42)
```

## Multimodal density
Let's recall the problem from [variational_api_quickstart](variational_api_quickstart.ipynb) where we first got a NUTS trace

```{code-cell} ipython3
w = pm.floatX([0.2, 0.8])
mu = pm.floatX([-0.3, 0.5])
sd = pm.floatX([0.1, 0.1])

with pm.Model() as model:
    x = pm.NormalMixture("x", w=w, mu=mu, sigma=sd)
    # Empirical approx does not support inference data
    trace = pm.sample(50000, return_inferencedata=False)
    idata = pm.to_inference_data(trace)
```

```{code-cell} ipython3
az.plot_trace(idata);
```

Great. First having a trace we can create `Empirical` approx

```{code-cell} ipython3
print(pm.Empirical.__doc__)
```

```{code-cell} ipython3
with model:
    approx = pm.Empirical(trace)
```

```{code-cell} ipython3
approx
```

This type of approximation has it's own underlying storage for samples that is `aesara.shared` itself

```{code-cell} ipython3
approx.histogram
```

```{code-cell} ipython3
approx.histogram.get_value()[:10]
```

```{code-cell} ipython3
approx.histogram.get_value().shape
```

It has exactly the same number of samples that you had in trace before. In our particular case it is 50k.  Another thing to notice is that if you have multitrace with **more than one chain** you'll get much **more samples** stored at once. We flatten all the trace for creating `Empirical`.

This *histogram* is about *how* we store samples. The structure is pretty simple: `(n_samples, n_dim)` The order of these variables is stored internally in the class and in most cases will not be needed for end user

```{code-cell} ipython3
approx.ordering
```

Sampling from posterior is done uniformly with replacements. Call `approx.sample(1000)` and you'll get again the trace but the order is not determined. There is no way now to reconstruct the underlying trace again with `approx.sample`.

```{code-cell} ipython3
new_trace = approx.sample(50000)
```

```{code-cell} ipython3
%timeit new_trace = approx.sample(50000)
```

After sampling function is compiled sampling bacomes really fast

```{code-cell} ipython3
az.plot_trace(new_trace);
```

You see there is no order any more but reconstructed density is the same.

## 2d density

```{code-cell} ipython3
mu = pm.floatX([0.0, 0.0])
cov = pm.floatX([[1, 0.5], [0.5, 1.0]])
with pm.Model() as model:
    pm.MvNormal("x", mu=mu, cov=cov, shape=2)
    trace = pm.sample(1000, return_inferencedata=False)
    idata = pm.to_inference_data(trace)
```

```{code-cell} ipython3
with model:
    approx = pm.Empirical(trace)
```

```{code-cell} ipython3
az.plot_trace(approx.sample(10000));
```

```{code-cell} ipython3
az.plot_pair(data=approx.sample(10000))
plt.show()
```

Previously we had a `trace_cov` function

```{code-cell} ipython3
with model:
    print(pm.trace_cov(trace))
```

Now we can estimate the same covariance using `Empirical`

```{code-cell} ipython3
print(approx.cov)
```

That's a tensor itself

```{code-cell} ipython3
print(approx.cov.eval())
```

Estimations are very close and differ due to precision error. We can get the mean in the same way

```{code-cell} ipython3
print(approx.mean.eval())
```

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w
```
