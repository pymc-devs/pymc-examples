---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: default
  language: python
  name: python3
---

(faster_sampling_notebook)=

# Faster Sampling with JAX and Numba

:::{post} July 11, 2023
:tags: hierarchical model, JAX, numba, scaling
:category: reference, intermediate
:author: Thomas Wiecki
:::

+++

PyMC offers multiple sampling backends that can dramatically improve performance depending on your model size and requirements. Each backend has distinct advantages and is optimized for different use cases.

### PyMC's Built-in Sampler

```python
pm.sample()
```

The default PyMC sampler uses a Python-based NUTS implementation that provides maximum compatibility with all PyMC features. This sampler is required when working with models that contain discrete variables, as it's the only option that supports non-gradient based samplers like Slice and Metropolis. While this sampler can compile the underlying model to different backends (C, Numba, or JAX) using PyTensor's compilation system via the `compile_kwargs` parameter, it maintains Python overhead that can limit performance for large models.

### Nutpie Sampler

```python
pm.sample(nuts_sampler="nutpie", nuts_sampler_kwargs={"backend": "numba"})
pm.sample(nuts_sampler="nutpie", nuts_sampler_kwargs={"backend": "jax"})
pm.sample(nuts_sampler="nutpie", nuts_sampler_kwargs={"backend": "jax", "gradient_backend": "pytensor"})
```

Nutpie is PyMC's cutting-edge performance sampler. Written in Rust, it eliminates Python overhead and provides exceptional performance for continuous models. The Numba backend typically offers the highest performance for most use cases, while the JAX backend excels with very large models and provides GPU acceleration capabilities. Nutpie is particularly well-suited for production workflows where sampling speed is critical.

### NumPyro Sampler

```python
pm.sample(nuts_sampler="numpyro", nuts_sampler_kwargs={"chain_method": "parallel"})
# GPU-accelerated
pm.sample(nuts_sampler="numpyro", nuts_sampler_kwargs={"chain_method": "vectorized"})
```

NumPyro provides a mature JAX-based sampling implementation that integrates seamlessly with the broader JAX ecosystem. This sampler benefits from years of development within the JAX community and provides reliable performance characteristics, with excellent GPU support for accelerated computation.

### BlackJAX Sampler

```python
pm.sample(nuts_sampler="blackjax")
```

BlackJAX offers another JAX-based sampling implementation focused on flexibility and research applications. While it provides similar capabilities to NumPyro, it's less commonly used in production environments. BlackJAX can be valuable for experimental workflows or when specific JAX-based features are required.

+++

## Performance Guidelines

Understanding when to use each sampler depends on several key factors including model size, variable types, and computational requirements.

For **small models**, NumPyro typically provides the best balance of performance and reliability. The compilation overhead is minimal, and its mature JAX implementation handles these models efficiently. **Large models** generally perform best with Nutpie's Numba backend for consistent CPU performance or Nutpie's JAX backend when GPU acceleration is needed or memory efficiency is critical.

Models containing **discrete variables** must use PyMC's built-in sampler, as it's the only implementation that supports compatible (*i.e.*, non-gradient based) sampling algorithms. For purely continuous models, all sampling backends are available, making performance the primary consideration.

**Numba** excels at CPU optimization and provides consistent performance across different model types. It's particularly effective for models with complex mathematical operations that benefit from just-in-time compilation. **JAX** offers superior performance for very large models and provides natural GPU acceleration, making it ideal when computational resources are a limiting factor. The **C** backend serves as a reliable fallback option with broad compatibility but typically offers lower performance than the alternatives.

```{code-cell} ipython3
import platform

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

if platform.system() == "linux":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)

rng = np.random.default_rng(seed=42)
print(f"Running on PyMC v{pm.__version__}")
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")
```

We'll demonstrate the performance differences using a Probabilistic Principal Component Analysis (PPCA) model.

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

## Performance Comparison

Now let's compare the performance of different sampling backends on our PPCA model. We'll measure both compilation time and sampling speed.

### 1. PyMC Default Sampler (Python NUTS)

```{code-cell} ipython3
%%time
with PPCA:
    idata_pymc = pm.sample(progressbar=False)
```

### 2. Nutpie with Numba Backend

```{code-cell} ipython3
%%time
with PPCA:
    idata_nutpie_numba = pm.sample(
        nuts_sampler="nutpie", nuts_sampler_kwargs={"backend": "numba"}, progressbar=False
    )
```

### 3. Nutpie with JAX Backend

```{code-cell} ipython3
%%time
with PPCA:
    idata_nutpie_jax = pm.sample(
        nuts_sampler="nutpie", nuts_sampler_kwargs={"backend": "jax"}, progressbar=False
    )
```

### 4. NumPyro Sampler

```{code-cell} ipython3
%%time
with PPCA:
    idata_numpyro = pm.sample(nuts_sampler="numpyro", progressbar=False)
```

## Installation Requirements

To use the various sampling backends, you need to install the corresponding packages. Nutpie is the recommended high-performance option and can be installed with pip or conda/mamba (e.g. `conda install nutpie`). For JAX-based workflows, NumPyro provides mature functionality and is installed with the `numpyro` package. BlackJAX offers an alternative JAX implementation and is available in the `blackjax` package.

+++

## Special Cases and Advanced Usage

### Using PyMC's Built-in Sampler with Different Backends

In certain scenarios, you may need to use PyMC's Python-based sampler while still benefiting from faster computational backends. This situation commonly arises when working with models that contain discrete variables, which require PyMC's specialized sampling algorithms. Even in these cases, you can significantly improve performance by compiling the model's computational graph to more efficient backends.

The following examples demonstrate how to use PyMC's built-in sampler with different compilation targets. The `fast_run` mode uses optimized C compilation, which provides good performance while maintaining full compatibility. The `numba` mode offers the only way to access Numba's just-in-time compilation benefits when using PyMC's sampler. The `jax` mode enables JAX compilation, though for JAX workflows, Nutpie or NumPyro typically provide better performance.

```{code-cell} ipython3
with PPCA:
    idata_c = pm.sample(nuts_sampler="pymc", compile_kwargs={"mode": "fast_run"}, progressbar=False)

# with PPCA:
#     idata_pymc_numba = pm.sample(nuts_sampler="pymc", compile_kwargs={"mode": "numba"}, progressbar=False)

# with PPCA:
#     idata_pymc_jax = pm.sample(nuts_sampler="pymc", compile_kwargs={"mode": "jax"}, progressbar=False)
```

The above examples are commented out to avoid redundant sampling in this demonstration notebook. In practice, you would uncomment and run the configuration that matches your model's requirements. These compilation modes allow you to access faster computational backends even when you must use PyMC's Python-based sampler for compatibility reasons.

+++

### Models with Discrete Variables

When working with models that contain discrete variables, you have no choice but to use PyMC's built-in sampler. This is because discrete variables require specialized sampling algorithms like Slice sampling or Metropolis-Hastings that are only available in PyMC's Python implementation. The example below demonstrates a typical scenario where this constraint applies.

```{code-cell} ipython3
with pm.Model() as discrete_model:
    cluster = pm.Categorical("cluster", p=[0.3, 0.7], shape=100)
    mu = pm.Normal("mu", 0, 1, shape=2)
    sigma = pm.HalfNormal("sigma", 1, shape=2)
    obs = pm.Normal("obs", mu=mu[cluster], sigma=sigma[cluster], observed=rng.normal(0, 1, 100))

    trace_discrete = pm.sample(progressbar=False)
```

## Authors

- Originally authored by Thomas Wiecki in July 2023  
- Substantially updated and expanded by Chris Fonnesbeck in May 2025

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,arviz,pymc,numpyro,blackjax,nutpie
```

:::{include} ../page_footer.md
:::
