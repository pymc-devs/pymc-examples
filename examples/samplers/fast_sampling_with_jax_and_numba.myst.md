---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: pymc5recent
  language: python
  name: pymc5recent
---

(faster_sampling_notebook)=

# Faster Sampling with JAX and Numba

:::{post} July 11, 2023
:tags: hierarchical model, JAX, numba, scaling
:category: reference, intermediate
:author: Thomas Wiecki
:::

+++

PyMC can compile its models to various execution backends through PyTensor, including:
* C
* JAX
* Numba

By default, PyMC is using the C backend which then gets called by the Python-based samplers.

However, by compiling to other backends, we can use samplers written in other languages than Python that call the PyMC model without any Python-overhead.

For the JAX backend there is the NumPyro and BlackJAX NUTS sampler available. To use these samplers, you have to install `numpyro` and `blackjax`. Both of them are available through conda/mamba: `mamba install -c conda-forge numpyro blackjax`.

For the Numba backend, there is the [Nutpie sampler](https://github.com/pymc-devs/nutpie) writte in Rust. To use this sampler you need `nutpie` installed: `mamba install -c conda-forge nutpie`. 

```{code-cell} ipython3
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

rng = np.random.default_rng(seed=42)
print(f"Running on PyMC v{pm.__version__}")
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")
```

We will use a simple probabilistic PCA model as our example.

```{code-cell} ipython3
def build_toy_dataset(N, D, K, sigma=1):
    x_train = np.zeros((D, N))
    w = rng.normal(
        0.0,
        2.0,
        size=(D, K),
    )
    z = rng.normal(0.0, 1.0, size=(K, N))
    mean = np.dot(w, z)
    for d in range(D):
        for n in range(N):
            x_train[d, n] = rng.normal(mean[d, n], sigma)

    print("True principal axes:")
    print(w)
    return x_train


N = 5000  # number of data points
D = 2  # data dimensionality
K = 1  # latent dimensionality

data = build_toy_dataset(N, D, K)
```

```{code-cell} ipython3
plt.scatter(data[0, :], data[1, :], color="blue", alpha=0.1)
plt.axis([-10, 10, -10, 10])
plt.title("Simulated data set")
```

```{code-cell} ipython3
with pm.Model() as PPCA:
    w = pm.Normal("w", mu=0, sigma=2, shape=[D, K], transform=pm.distributions.transforms.Ordered())
    z = pm.Normal("z", mu=0, sigma=1, shape=[N, K])
    x = pm.Normal("x", mu=w.dot(z.T), sigma=1, shape=[D, N], observed=data)
```

## Sampling using Python NUTS sampler

```{code-cell} ipython3
%%time
with PPCA:
    idata_pymc = pm.sample()
```

## Sampling using NumPyro JAX NUTS sampler

```{code-cell} ipython3
%%time
with PPCA:
    idata_numpyro = pm.sample(nuts_sampler="numpyro", progressbar=False)
```

## Sampling using BlackJAX NUTS sampler

```{code-cell} ipython3
%%time
with PPCA:
    idata_blackjax = pm.sample(nuts_sampler="blackjax")
```

## Sampling using Nutpie Rust NUTS sampler

```{code-cell} ipython3
%%time
with PPCA:
    idata_nutpie = pm.sample(nuts_sampler="nutpie")
```

## Authors
Authored by Thomas Wiecki in July 2023

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,arviz,pymc,numpyro,blackjax,nutpie
```

:::{include} ../page_footer.md
:::
