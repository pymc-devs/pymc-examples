---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(GP-MeansAndCovs)=
# Mean and Covariance Functions

:::{post} Mar 22, 2022
:tags: gaussian process
:category: intermediate, reference
:author: Bill Engels, Oriol Abril Pla
:::

```{code-cell} ipython3
---
papermill:
  duration: 5.306978
  end_time: '2020-12-22T18:36:31.587812'
  exception: false
  start_time: '2020-12-22T18:36:26.280834'
  status: completed
---
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt

%config InlineBackend.figure_format = "retina"
```

```{code-cell} ipython3
---
papermill:
  duration: 0.047175
  end_time: '2020-12-22T18:36:31.674100'
  exception: false
  start_time: '2020-12-22T18:36:31.626925'
  status: completed
---
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
plt.rcParams["figure.figsize"] = (10, 4)
```

+++ {"papermill": {"duration": 0.037844, "end_time": "2020-12-22T18:36:31.751886", "exception": false, "start_time": "2020-12-22T18:36:31.714042", "status": "completed"}}

A large set of mean and covariance functions are available in PyMC.  It is relatively easy to define custom mean and covariance functions.  Since PyMC uses PyTensor, their gradients do not need to be defined by the user.  

## Mean functions

The following mean functions are available in PyMC.

- {class}`pymc.gp.mean.Zero`
- {class}`pymc.gp.mean.Constant`
- {class}`pymc.gp.mean.Linear`

All follow a similar usage pattern.  First, the mean function is specified.  Then it can be evaluated over some inputs.  The first two mean functions are very simple.  Regardless of the inputs, `gp.mean.Zero` returns a vector of zeros with the same length as the number of input values.

### Zero

```{code-cell} ipython3
---
papermill:
  duration: 1.075408
  end_time: '2020-12-22T18:36:32.865469'
  exception: false
  start_time: '2020-12-22T18:36:31.790061'
  status: completed
---
zero_func = pm.gp.mean.Zero()

X = np.linspace(0, 1, 5)[:, None]
print(zero_func(X).eval())
```

+++ {"papermill": {"duration": 0.040891, "end_time": "2020-12-22T18:36:32.947028", "exception": false, "start_time": "2020-12-22T18:36:32.906137", "status": "completed"}}

The default mean functions for all GP implementations in PyMC is `Zero`.

### Constant

`gp.mean.Constant` returns a vector whose value is provided.

```{code-cell} ipython3
---
papermill:
  duration: 2.12553
  end_time: '2020-12-22T18:36:35.113789'
  exception: false
  start_time: '2020-12-22T18:36:32.988259'
  status: completed
---
const_func = pm.gp.mean.Constant(25.2)

print(const_func(X).eval())
```

+++ {"papermill": {"duration": 0.039627, "end_time": "2020-12-22T18:36:35.195057", "exception": false, "start_time": "2020-12-22T18:36:35.155430", "status": "completed"}}

As long as the shape matches the input it will receive, `gp.mean.Constant` can also accept a PyTensor tensor or vector of PyMC random variables.

```{code-cell} ipython3
---
papermill:
  duration: 1.408839
  end_time: '2020-12-22T18:36:36.644770'
  exception: false
  start_time: '2020-12-22T18:36:35.235931'
  status: completed
---
const_func_vec = pm.gp.mean.Constant(pt.ones(5))

print(const_func_vec(X).eval())
```

+++ {"papermill": {"duration": 0.04127, "end_time": "2020-12-22T18:36:36.726017", "exception": false, "start_time": "2020-12-22T18:36:36.684747", "status": "completed"}}

### Linear

`gp.mean.Linear` is a takes as input a matrix of coefficients and a vector of intercepts (or a slope and scalar intercept in one dimension).

```{code-cell} ipython3
---
papermill:
  duration: 0.073879
  end_time: '2020-12-22T18:36:36.839351'
  exception: false
  start_time: '2020-12-22T18:36:36.765472'
  status: completed
---
beta = rng.normal(size=3)
b = 0.0

lin_func = pm.gp.mean.Linear(coeffs=beta, intercept=b)

X = rng.normal(size=(5, 3))
print(lin_func(X).eval())
```

+++ {"papermill": {"duration": 0.03931, "end_time": "2020-12-22T18:36:36.918672", "exception": false, "start_time": "2020-12-22T18:36:36.879362", "status": "completed"}}

## Defining a custom mean function

To define a custom mean function, subclass `gp.mean.Mean`, and provide `__call__` and `__init__` methods.  For example, the code for the `Constant` mean function is

```python
import theano.tensor as tt

class Constant(pm.gp.mean.Mean):
    
    def __init__(self, c=0):
        Mean.__init__(self)
        self.c = c 

    def __call__(self, X): 
        return tt.alloc(1.0, X.shape[0]) * self.c

```

Remember that PyTensor must be used instead of NumPy.

+++ {"papermill": {"duration": 0.039306, "end_time": "2020-12-22T18:36:36.998649", "exception": false, "start_time": "2020-12-22T18:36:36.959343", "status": "completed"}}

## Covariance functions

PyMC contains a much larger suite of {mod}`built-in covariance functions <pymc.gp.cov>`.  The following shows functions drawn from a GP prior with a given covariance function, and demonstrates how composite covariance functions can be constructed with Python operators in a straightforward manner.  Our goal was for our API to follow kernel algebra (see Ch.4 of {cite:t}`rasmussen2003gaussian`) as closely as possible.  See the main documentation page for an overview on their usage in PyMC.

+++ {"papermill": {"duration": 0.039789, "end_time": "2020-12-22T18:36:37.078199", "exception": false, "start_time": "2020-12-22T18:36:37.038410", "status": "completed"}}

### Exponentiated Quadratic

$$
k(x, x') = \mathrm{exp}\left[ -\frac{(x - x')^2}{2 \ell^2} \right]
$$

```{code-cell} ipython3
---
papermill:
  duration: 7.505078
  end_time: '2020-12-22T18:36:44.626679'
  exception: false
  start_time: '2020-12-22T18:36:37.121601'
  status: completed
---
lengthscale = 0.2
eta = 2.0
cov = eta**2 * pm.gp.cov.ExpQuad(1, lengthscale)
# Add white noise to stabilise
cov += pm.gp.cov.WhiteNoise(1e-6)

X = np.linspace(0, 2, 200)[:, None]
K = cov(X).eval()

plt.plot(
    X,
    pm.draw(
        pm.MvNormal.dist(mu=np.zeros(len(K)), cov=K, shape=K.shape[0]), draws=3, random_seed=rng
    ).T,
)
plt.title("Samples from the GP prior")
plt.ylabel("y")
plt.xlabel("X");
```

+++ {"papermill": {"duration": 0.042546, "end_time": "2020-12-22T18:36:44.712169", "exception": false, "start_time": "2020-12-22T18:36:44.669623", "status": "completed"}}

### Two (and higher) Dimensional Inputs

#### Both dimensions active

It is easy to define kernels with higher dimensional inputs.  Notice that the ```ls``` (lengthscale) parameter is an array of length 2.  Lists of PyMC random variables can be used for automatic relevance determination (ARD).

```{code-cell} ipython3
---
papermill:
  duration: 3.19044
  end_time: '2020-12-22T18:36:47.946218'
  exception: false
  start_time: '2020-12-22T18:36:44.755778'
  status: completed
---
x1 = np.linspace(0, 1, 10)
x2 = np.arange(1, 4)
# Cartesian product
X2 = np.dstack(np.meshgrid(x1, x2)).reshape(-1, 2)

ls = np.array([0.2, 1.0])
cov = pm.gp.cov.ExpQuad(input_dim=2, ls=ls)

m = plt.imshow(cov(X2).eval(), cmap="inferno", interpolation="none")
plt.colorbar(m);
```

+++ {"papermill": {"duration": 0.043142, "end_time": "2020-12-22T18:36:48.032797", "exception": false, "start_time": "2020-12-22T18:36:47.989655", "status": "completed"}}

#### One dimension active

```{code-cell} ipython3
---
papermill:
  duration: 0.673374
  end_time: '2020-12-22T18:36:48.749451'
  exception: false
  start_time: '2020-12-22T18:36:48.076077'
  status: completed
---
ls = 0.2
cov = pm.gp.cov.ExpQuad(input_dim=2, ls=ls, active_dims=[0])

m = plt.imshow(cov(X2).eval(), cmap="inferno", interpolation="none")
plt.colorbar(m);
```

+++ {"papermill": {"duration": 0.045376, "end_time": "2020-12-22T18:36:48.840086", "exception": false, "start_time": "2020-12-22T18:36:48.794710", "status": "completed"}}

#### Product of covariances over different dimensions

Note that this is equivalent to using a two dimensional `ExpQuad` with separate lengthscale parameters for each dimension.

```{code-cell} ipython3
---
papermill:
  duration: 1.600894
  end_time: '2020-12-22T18:36:50.486049'
  exception: false
  start_time: '2020-12-22T18:36:48.885155'
  status: completed
---
ls1 = 0.2
ls2 = 1.0
cov1 = pm.gp.cov.ExpQuad(2, ls1, active_dims=[0])
cov2 = pm.gp.cov.ExpQuad(2, ls2, active_dims=[1])
cov = cov1 * cov2

m = plt.imshow(cov(X2).eval(), cmap="inferno", interpolation="none")
plt.colorbar(m);
```

+++ {"papermill": {"duration": 0.046821, "end_time": "2020-12-22T18:36:50.579012", "exception": false, "start_time": "2020-12-22T18:36:50.532191", "status": "completed"}}

### White Noise

$$
k(x, x') = \sigma^2 \mathrm{I}_{xx}
$$

```{code-cell} ipython3
---
papermill:
  duration: 0.99526
  end_time: '2020-12-22T18:36:51.620630'
  exception: false
  start_time: '2020-12-22T18:36:50.625370'
  status: completed
---
sigma = 2.0
cov = pm.gp.cov.WhiteNoise(sigma)

X = np.linspace(0, 2, 200)[:, None]
K = cov(X).eval()

plt.plot(
    X,
    pm.draw(pm.MvNormal.dist(mu=np.zeros(len(K)), cov=K, shape=len(K)), draws=3, random_seed=rng).T,
)
plt.title("Samples from the GP prior")
plt.ylabel("y")
plt.xlabel("X");
```

+++ {"papermill": {"duration": 0.05125, "end_time": "2020-12-22T18:36:51.723154", "exception": false, "start_time": "2020-12-22T18:36:51.671904", "status": "completed"}}

### Constant

$$
k(x, x') = c
$$

```{code-cell} ipython3
---
papermill:
  duration: 1.931356
  end_time: '2020-12-22T18:36:53.705539'
  exception: false
  start_time: '2020-12-22T18:36:51.774183'
  status: completed
---
c = 2.0
cov = pm.gp.cov.Constant(c)

X = np.linspace(0, 2, 200)[:, None]
K = cov(X).eval()

plt.plot(
    X,
    pm.draw(pm.MvNormal.dist(mu=np.zeros(len(K)), cov=K, shape=len(K)), draws=3, random_seed=rng).T,
)
plt.title("Samples from the GP prior")
plt.ylabel("y")
plt.xlabel("X");
```

+++ {"papermill": {"duration": 0.051694, "end_time": "2020-12-22T18:36:53.810105", "exception": false, "start_time": "2020-12-22T18:36:53.758411", "status": "completed"}}

### Rational Quadratic

$$
k(x, x') = \left(1 + \frac{(x - x')^2}{2\alpha\ell^2} \right)^{-\alpha}
$$

```{code-cell} ipython3
---
papermill:
  duration: 2.381363
  end_time: '2020-12-22T18:36:56.245016'
  exception: false
  start_time: '2020-12-22T18:36:53.863653'
  status: completed
---
alpha = 0.1
ls = 0.2
tau = 2.0
cov = tau * pm.gp.cov.RatQuad(1, ls, alpha)

X = np.linspace(0, 2, 200)[:, None]
K = cov(X).eval()

plt.plot(
    X,
    pm.draw(pm.MvNormal.dist(mu=np.zeros(len(K)), cov=K, shape=len(K)), draws=3, random_seed=rng).T,
)
plt.title("Samples from the GP prior")
plt.ylabel("y")
plt.xlabel("X");
```

+++ {"papermill": {"duration": 0.055808, "end_time": "2020-12-22T18:36:56.357806", "exception": false, "start_time": "2020-12-22T18:36:56.301998", "status": "completed"}}

### Exponential

$$
k(x, x') = \mathrm{exp}\left[ -\frac{||x - x'||}{2\ell^2} \right]
$$

```{code-cell} ipython3
---
papermill:
  duration: 1.343198
  end_time: '2020-12-22T18:36:57.756310'
  exception: false
  start_time: '2020-12-22T18:36:56.413112'
  status: completed
---
inverse_lengthscale = 5
cov = pm.gp.cov.Exponential(1, ls_inv=inverse_lengthscale)

X = np.linspace(0, 2, 200)[:, None]
K = cov(X).eval()

plt.plot(
    X,
    pm.draw(pm.MvNormal.dist(mu=np.zeros(len(K)), cov=K, shape=len(K)), draws=3, random_seed=rng).T,
)
plt.title("Samples from the GP prior")
plt.ylabel("y")
plt.xlabel("X");
```

+++ {"papermill": {"duration": 0.058891, "end_time": "2020-12-22T18:36:57.874371", "exception": false, "start_time": "2020-12-22T18:36:57.815480", "status": "completed"}}

### Matern 5/2

$$
k(x, x') = \left(1 + \frac{\sqrt{5(x - x')^2}}{\ell} +
            \frac{5(x-x')^2}{3\ell^2}\right)
            \mathrm{exp}\left[ - \frac{\sqrt{5(x - x')^2}}{\ell} \right]
$$

```{code-cell} ipython3
---
papermill:
  duration: 2.417182
  end_time: '2020-12-22T18:37:00.350538'
  exception: false
  start_time: '2020-12-22T18:36:57.933356'
  status: completed
---
ls = 0.2
tau = 2.0
cov = tau * pm.gp.cov.Matern52(1, ls)

X = np.linspace(0, 2, 200)[:, None]
K = cov(X).eval()

plt.plot(
    X,
    pm.draw(pm.MvNormal.dist(mu=np.zeros(len(K)), cov=K, shape=len(K)), draws=3, random_seed=rng).T,
)
plt.title("Samples from the GP prior")
plt.ylabel("y")
plt.xlabel("X");
```

+++ {"papermill": {"duration": 0.061663, "end_time": "2020-12-22T18:37:00.473343", "exception": false, "start_time": "2020-12-22T18:37:00.411680", "status": "completed"}}

### Matern 3/2

$$
k(x, x') = \left(1 + \frac{\sqrt{3(x - x')^2}}{\ell}\right)
           \mathrm{exp}\left[ - \frac{\sqrt{3(x - x')^2}}{\ell} \right]
$$

```{code-cell} ipython3
---
papermill:
  duration: 0.494084
  end_time: '2020-12-22T18:37:01.028428'
  exception: false
  start_time: '2020-12-22T18:37:00.534344'
  status: completed
---
ls = 0.2
tau = 2.0
cov = tau * pm.gp.cov.Matern32(1, ls)

X = np.linspace(0, 2, 200)[:, None]
K = cov(X).eval()

plt.plot(
    X,
    pm.draw(pm.MvNormal.dist(mu=np.zeros(len(K)), cov=K, shape=len(K)), draws=3, random_seed=rng).T,
)
plt.title("Samples from the GP prior")
plt.ylabel("y")
plt.xlabel("X");
```

+++ {"papermill": {"duration": 0.064186, "end_time": "2020-12-22T18:37:01.159126", "exception": false, "start_time": "2020-12-22T18:37:01.094940", "status": "completed"}}

### Matern 1/2

$$k(x, x') = \mathrm{exp}\left[ -\frac{(x - x')^2}{\ell} \right]$$

```{code-cell} ipython3
---
papermill:
  duration: 0.477568
  end_time: '2020-12-22T18:37:01.701402'
  exception: false
  start_time: '2020-12-22T18:37:01.223834'
  status: completed
---
ls = 0.2
tau = 2.0
cov = tau * pm.gp.cov.Matern12(1, ls)

X = np.linspace(0, 2, 200)[:, None]
K = cov(X).eval()

plt.plot(
    X,
    pm.draw(pm.MvNormal.dist(mu=np.zeros(len(K)), cov=K, shape=len(K)), draws=3, random_seed=rng).T,
)
plt.title("Samples from the GP prior")
plt.ylabel("y")
plt.xlabel("X");
```

+++ {"papermill": {"duration": 0.068504, "end_time": "2020-12-22T18:37:01.837835", "exception": false, "start_time": "2020-12-22T18:37:01.769331", "status": "completed"}}

### Cosine

$$
k(x, x') = \mathrm{cos}\left( 2 \pi \frac{||x - x'||}{ \ell^2} \right)
$$

```{code-cell} ipython3
---
papermill:
  duration: 1.457975
  end_time: '2020-12-22T18:37:03.365039'
  exception: false
  start_time: '2020-12-22T18:37:01.907064'
  status: completed
---
period = 0.5
cov = pm.gp.cov.Cosine(1, period)
# Add white noise to stabilise
cov += pm.gp.cov.WhiteNoise(1e-4)

X = np.linspace(0, 2, 200)[:, None]
K = cov(X).eval()

plt.plot(
    X,
    pm.draw(pm.MvNormal.dist(mu=np.zeros(len(K)), cov=K, shape=len(K)), draws=3, random_seed=rng).T,
)
plt.title("Samples from the GP prior")
plt.ylabel("y")
plt.xlabel("X");
```

+++ {"papermill": {"duration": 0.077444, "end_time": "2020-12-22T18:37:03.548722", "exception": false, "start_time": "2020-12-22T18:37:03.471278", "status": "completed"}}

### Linear

$$
k(x, x') = (x - c)(x' - c)
$$

```{code-cell} ipython3
---
papermill:
  duration: 1.524742
  end_time: '2020-12-22T18:37:05.145867'
  exception: false
  start_time: '2020-12-22T18:37:03.621125'
  status: completed
---
c = 1.0
tau = 2.0
cov = tau * pm.gp.cov.Linear(1, c)
# Add white noise to stabilise
cov += pm.gp.cov.WhiteNoise(1e-6)

X = np.linspace(0, 2, 200)[:, None]
K = cov(X).eval()

plt.plot(
    X,
    pm.draw(pm.MvNormal.dist(mu=np.zeros(len(K)), cov=K, shape=len(K)), draws=3, random_seed=rng).T,
)
plt.title("Samples from the GP prior")
plt.ylabel("y")
plt.xlabel("X");
```

+++ {"papermill": {"duration": 0.073236, "end_time": "2020-12-22T18:37:05.293217", "exception": false, "start_time": "2020-12-22T18:37:05.219981", "status": "completed"}}

### Polynomial

$$
k(x, x') = [(x - c)(x' - c) + \mathrm{offset}]^{d}
$$

```{code-cell} ipython3
---
papermill:
  duration: 1.371418
  end_time: '2020-12-22T18:37:06.738888'
  exception: false
  start_time: '2020-12-22T18:37:05.367470'
  status: completed
---
c = 1.0
d = 3
offset = 1.0
tau = 0.1
cov = tau * pm.gp.cov.Polynomial(1, c=c, d=d, offset=offset)
# Add white noise to stabilise
cov += pm.gp.cov.WhiteNoise(1e-6)

X = np.linspace(0, 2, 200)[:, None]
K = cov(X).eval()

plt.plot(
    X,
    pm.draw(pm.MvNormal.dist(mu=np.zeros(len(K)), cov=K, shape=len(K)), draws=3, random_seed=rng).T,
)
plt.title("Samples from the GP prior")
plt.ylabel("y")
plt.xlabel("X");
```

+++ {"papermill": {"duration": 0.07702, "end_time": "2020-12-22T18:37:06.892733", "exception": false, "start_time": "2020-12-22T18:37:06.815713", "status": "completed"}}

### Multiplication with a precomputed covariance matrix

A covariance function ```cov``` can be multiplied with numpy matrix, ```K_cos```, as long as the shapes are appropriate.

```{code-cell} ipython3
---
papermill:
  duration: 1.546032
  end_time: '2020-12-22T18:37:08.514887'
  exception: false
  start_time: '2020-12-22T18:37:06.968855'
  status: completed
---
# first evaluate a covariance function into a matrix
period = 0.2
cov_cos = pm.gp.cov.Cosine(1, period)
K_cos = cov_cos(X).eval()

# now multiply it with a covariance *function*
cov = pm.gp.cov.Matern32(1, 0.5) * K_cos

X = np.linspace(0, 2, 200)[:, None]
K = cov(X).eval()

plt.plot(
    X,
    pm.draw(pm.MvNormal.dist(mu=np.zeros(len(K)), cov=K, shape=len(K)), draws=3, random_seed=rng).T,
)
plt.title("Samples from the GP prior")
plt.ylabel("y")
plt.xlabel("X");
```

+++ {"papermill": {"duration": 0.078461, "end_time": "2020-12-22T18:37:08.672218", "exception": false, "start_time": "2020-12-22T18:37:08.593757", "status": "completed"}}

### Applying an arbitrary warping function on the inputs

If $k(x, x')$ is a valid covariance function, then so is $k(w(x), w(x'))$.

The first argument of the warping function must be the input ```X```.  The remaining arguments can be anything else, including random variables.

```{code-cell} ipython3
---
papermill:
  duration: 6.061177
  end_time: '2020-12-22T18:37:14.812998'
  exception: false
  start_time: '2020-12-22T18:37:08.751821'
  status: completed
---
def warp_func(x, a, b, c):
    return 1.0 + x + (a * pt.tanh(b * (x - c)))


a = 1.0
b = 5.0
c = 1.0

cov_exp = pm.gp.cov.ExpQuad(1, 0.2)
cov = pm.gp.cov.WarpedInput(1, warp_func=warp_func, args=(a, b, c), cov_func=cov_exp)
# Add white noise to stabilise
cov += pm.gp.cov.WhiteNoise(1e-6)

X = np.linspace(0, 2, 400)[:, None]
wf = warp_func(X.flatten(), a, b, c).eval()

plt.plot(X, wf)
plt.xlabel("X")
plt.ylabel("warp_func(X)")
plt.title("The warping function used")

K = cov(X).eval()
plt.plot(
    X,
    pm.draw(pm.MvNormal.dist(mu=np.zeros(len(K)), cov=K, shape=len(K)), draws=3, random_seed=rng).T,
)
plt.title("Samples from the GP prior")
plt.ylabel("y")
plt.xlabel("X");
```

+++ {"papermill": {"duration": 0.085228, "end_time": "2020-12-22T18:37:14.983640", "exception": false, "start_time": "2020-12-22T18:37:14.898412", "status": "completed"}}

### Constructing `Periodic` using `WarpedInput`

The `WarpedInput` kernel can be used to create the `Periodic` covariance.  This covariance models functions that are periodic, but are not an exact sine wave (like the `Cosine` kernel is).

The periodic kernel is given by

$$
k(x, x') = \exp\left( -\frac{2 \sin^{2}(\pi |x - x'|\frac{1}{T})}{\ell^2}     \right)
$$

Where T is the period, and $\ell$ is the lengthscale.  It can be derived by warping the input of an `ExpQuad` kernel with the function $\mathbf{u}(x) = (\sin(2\pi x \frac{1}{T})\,, \cos(2 \pi x \frac{1}{T}))$.  Here we use the `WarpedInput` kernel to construct it.

The input `X`, which is defined at the top of this page, is 2 "seconds" long.  We use a period of $0.5$, which means that functions
drawn from this GP prior will repeat 4 times over 2 seconds.

```{code-cell} ipython3
---
papermill:
  duration: 3.628528
  end_time: '2020-12-22T18:37:18.698932'
  exception: false
  start_time: '2020-12-22T18:37:15.070404'
  status: completed
---
def mapping(x, T):
    c = 2.0 * np.pi * (1.0 / T)
    u = pt.concatenate((pt.sin(c * x), pt.cos(c * x)), 1)
    return u


T = 0.6
ls = 0.4
# note that the input of the covariance function taking
#    the inputs is 2 dimensional
cov_exp = pm.gp.cov.ExpQuad(2, ls)
cov = pm.gp.cov.WarpedInput(1, cov_func=cov_exp, warp_func=mapping, args=(T,))
# Add white noise to stabilise
cov += pm.gp.cov.WhiteNoise(1e-6)

K = cov(X).eval()
plt.plot(
    X,
    pm.draw(pm.MvNormal.dist(mu=np.zeros(len(K)), cov=K, shape=len(K)), draws=3, random_seed=rng).T,
)
plt.title("Samples from the GP prior")
plt.ylabel("y")
plt.xlabel("X");
```

+++ {"papermill": {"duration": 0.089186, "end_time": "2020-12-22T18:37:18.877629", "exception": false, "start_time": "2020-12-22T18:37:18.788443", "status": "completed"}}

### Periodic

There is no need to construct the periodic covariance this way every time.  A more efficient implementation of this covariance function is built in.

```{code-cell} ipython3
---
papermill:
  duration: 2.454314
  end_time: '2020-12-22T18:37:21.420790'
  exception: false
  start_time: '2020-12-22T18:37:18.966476'
  status: completed
---
period = 0.6
ls = 0.4
cov = pm.gp.cov.Periodic(1, period=period, ls=ls)
# Add white noise to stabilise
cov += pm.gp.cov.WhiteNoise(1e-6)

K = cov(X).eval()
plt.plot(
    X,
    pm.draw(pm.MvNormal.dist(mu=np.zeros(len(K)), cov=K, shape=len(K)), draws=3, random_seed=rng).T,
)
for p in np.arange(0, 2, period):
    plt.axvline(p, color="black")
plt.axhline(0, color="black")
plt.title("Samples from the GP prior")
plt.ylabel("y")
plt.xlabel("X");
```

+++ {"papermill": {"duration": 0.090578, "end_time": "2020-12-22T18:37:21.604122", "exception": false, "start_time": "2020-12-22T18:37:21.513544", "status": "completed"}}

### Circular

Circular kernel is similar to Periodic one but has an additional nuisance parameter $\tau$

In {cite:t}`padonou2015polar`, the Weinland function is used to solve the problem and ensures positive definite kernel on the circular domain (and not only).

$$
W_c(t) = \left(1 + \tau \frac{t}{c}\right)\left(1-\frac{t}{c}\right)_+^\tau
$$
where $c$ is maximum value for $t$ and $\tau\ge 4$ is some positive number 

The kernel itself for geodesic distance (arc length) on a circle looks like

$$
k_g(x, y) = W_\pi(\text{dist}_{\mathit{geo}}(x, y))
$$

Briefly, you can think

* $t$ is time, it runs from $0$ to $24$ and then goes back to $0$
* $c$ is maximum distance between any timestamps, here it would be $12$
* $\tau$ controls for correlation strength, larger $\tau$ leads to less smooth functions

```{code-cell} ipython3
---
papermill:
  duration: 4.35163
  end_time: '2020-12-22T18:37:26.047326'
  exception: false
  start_time: '2020-12-22T18:37:21.695696'
  status: completed
---
period = 0.6
tau = 4
cov = pm.gp.cov.Circular(1, period=period, tau=tau)

K = cov(X).eval()
plt.plot(
    X,
    pm.draw(pm.MvNormal.dist(mu=np.zeros(len(K)), cov=K, shape=len(K)), draws=3, random_seed=rng).T,
)
for p in np.arange(0, 2, period):
    plt.axvline(p, color="black")
plt.axhline(0, color="black")
plt.title("Samples from the GP prior")
plt.ylabel("y")
plt.xlabel("X");
```

+++ {"papermill": {"duration": 0.094257, "end_time": "2020-12-22T18:37:26.237410", "exception": false, "start_time": "2020-12-22T18:37:26.143153", "status": "completed"}}

We can see the effect of $\tau$, it adds more non-smooth patterns

```{code-cell} ipython3
---
papermill:
  duration: 0.613972
  end_time: '2020-12-22T18:37:26.946669'
  exception: false
  start_time: '2020-12-22T18:37:26.332697'
  status: completed
---
period = 0.6
tau = 40
cov = pm.gp.cov.Circular(1, period=period, tau=tau)

K = cov(X).eval()
plt.plot(
    X,
    pm.draw(pm.MvNormal.dist(mu=np.zeros(len(K)), cov=K, shape=len(K)), draws=3, random_seed=rng).T,
)
for p in np.arange(0, 2, period):
    plt.axvline(p, color="black")
plt.axhline(0, color="black")
plt.title("Samples from the GP prior")
plt.ylabel("y")
plt.xlabel("X");
```

+++ {"papermill": {"duration": 0.099739, "end_time": "2020-12-22T18:37:27.146953", "exception": false, "start_time": "2020-12-22T18:37:27.047214", "status": "completed"}}

### Gibbs

The Gibbs covariance function applies a positive definite warping function to the lengthscale.  Similarly to ```WarpedInput```, the lengthscale warping function can be specified with parameters that are either fixed or random variables.

```{code-cell} ipython3
---
papermill:
  duration: 4.779819
  end_time: '2020-12-22T18:37:32.026714'
  exception: false
  start_time: '2020-12-22T18:37:27.246895'
  status: completed
---
def tanh_func(x, ls1, ls2, w, x0):
    """
    ls1: left saturation value
    ls2: right saturation value
    w:   transition width
    x0:  transition location.
    """
    return (ls1 + ls2) / 2.0 - (ls1 - ls2) / 2.0 * pt.tanh((x - x0) / w)


ls1 = 0.05
ls2 = 0.6
w = 0.3
x0 = 1.0
cov = pm.gp.cov.Gibbs(1, tanh_func, args=(ls1, ls2, w, x0))
# Add white noise to stabilise
cov += pm.gp.cov.WhiteNoise(1e-6)

wf = tanh_func(X, ls1, ls2, w, x0).eval()
plt.plot(X, wf)
plt.ylabel("lengthscale")
plt.xlabel("X")
plt.title("Lengthscale as a function of X")

K = cov(X).eval()
plt.plot(
    X,
    pm.draw(pm.MvNormal.dist(mu=np.zeros(len(K)), cov=K, shape=len(K)), draws=3, random_seed=rng).T,
)
plt.title("Samples from the GP prior")
plt.ylabel("y")
plt.xlabel("X");
```

+++ {"papermill": {"duration": 0.106362, "end_time": "2020-12-22T18:37:32.238582", "exception": false, "start_time": "2020-12-22T18:37:32.132220", "status": "completed"}}

### Scaled Covariance

One can construct a new kernel or covariance function by multiplying some base kernel by a nonnegative function $\phi(x)$,

$$
k_{\mathrm{scaled}}(x, x') = \phi(x) k_{\mathrm{base}}(x, x') \phi(x') \,.
$$

This is useful for specifying covariance functions whose amplitude changes across the domain.

```{code-cell} ipython3
---
papermill:
  duration: 6.455011
  end_time: '2020-12-22T18:37:38.798884'
  exception: false
  start_time: '2020-12-22T18:37:32.343873'
  status: completed
---
def logistic(x, a, x0, c, d):
    # a is the slope, x0 is the location
    return d * pm.math.invlogit(a * (x - x0)) + c


a = 2.0
x0 = 5.0
c = 0.1
d = 2.0

cov_base = pm.gp.cov.ExpQuad(1, 0.2)
cov = pm.gp.cov.ScaledCov(1, scaling_func=logistic, args=(a, x0, c, d), cov_func=cov_base)
# Add white noise to stabilise
cov += pm.gp.cov.WhiteNoise(1e-5)

X = np.linspace(0, 10, 400)[:, None]
lfunc = logistic(X.flatten(), a, b, c, d).eval()

plt.plot(X, lfunc)
plt.xlabel("X")
plt.ylabel(r"$\phi(x)$")
plt.title("The scaling function")

K = cov(X).eval()
plt.plot(
    X,
    pm.draw(pm.MvNormal.dist(mu=np.zeros(len(K)), cov=K, shape=len(K)), draws=3, random_seed=rng).T,
)
plt.title("Samples from the GP prior")
plt.ylabel("y")
plt.xlabel("X");
```

+++ {"papermill": {"duration": 0.109017, "end_time": "2020-12-22T18:37:39.017681", "exception": false, "start_time": "2020-12-22T18:37:38.908664", "status": "completed"}}

### Constructing a Changepoint kernel using `ScaledCov`

The `ScaledCov` kernel can be used to create the `Changepoint` covariance.  This covariance models 
a process that gradually transitions from one type of behavior to another.

The changepoint kernel is given by

$$
k(x, x') = \phi(x)k_{1}(x, x')\phi(x)  + (1 - \phi(x))k_{2}(x, x')(1 - \phi(x'))
$$

where $\phi(x)$ is the logistic function.

```{code-cell} ipython3
---
papermill:
  duration: 2.436655
  end_time: '2020-12-22T18:37:41.563496'
  exception: false
  start_time: '2020-12-22T18:37:39.126841'
  status: completed
---
def logistic(x, a, x0):
    # a is the slope, x0 is the location
    return pm.math.invlogit(a * (x - x0))


a = 2.0
x0 = 5.0

cov1 = pm.gp.cov.ScaledCov(
    1, scaling_func=logistic, args=(-a, x0), cov_func=pm.gp.cov.ExpQuad(1, 0.2)
)
cov2 = pm.gp.cov.ScaledCov(
    1, scaling_func=logistic, args=(a, x0), cov_func=pm.gp.cov.Cosine(1, 0.5)
)
cov = cov1 + cov2
# Add white noise to stabilise
cov += pm.gp.cov.WhiteNoise(1e-5)

X = np.linspace(0, 10, 400)
plt.fill_between(
    X,
    np.zeros(400),
    logistic(X, -a, x0).eval(),
    label="ExpQuad region",
    color="slateblue",
    alpha=0.4,
)
plt.fill_between(
    X, np.zeros(400), logistic(X, a, x0).eval(), label="Cosine region", color="firebrick", alpha=0.4
)
plt.legend()
plt.xlabel("X")
plt.ylabel(r"$\phi(x)$")
plt.title("The two scaling functions")

K = cov(X[:, None]).eval()
plt.plot(
    X,
    pm.draw(pm.MvNormal.dist(mu=np.zeros(len(K)), cov=K, shape=len(K)), draws=3, random_seed=rng).T,
)
plt.title("Samples from the GP prior")
plt.ylabel("y")
plt.xlabel("X");
```

+++ {"papermill": {"duration": 0.123091, "end_time": "2020-12-22T18:37:41.801550", "exception": false, "start_time": "2020-12-22T18:37:41.678459", "status": "completed"}}

### Combination of two or more Covariance functions

You can combine different covariance functions to model complex data.

In particular, you can perform the following operations on any covaraince functions:

- Add other covariance function with equal or broadcastable dimensions with first covariance function
- Multiply with a scalar or a covariance function with equal or broadcastable dimensions with first covariance function
- Exponentiate with a scalar.

+++ {"papermill": {"duration": 0.114783, "end_time": "2020-12-22T18:37:42.043753", "exception": false, "start_time": "2020-12-22T18:37:41.928970", "status": "completed"}}

#### Addition

```{code-cell} ipython3
---
papermill:
  duration: 0.565388
  end_time: '2020-12-22T18:37:42.722540'
  exception: false
  start_time: '2020-12-22T18:37:42.157152'
  status: completed
---
ls_1 = 0.1
tau_1 = 2.0
ls_2 = 0.5
tau_2 = 1.0
cov_1 = tau_1 * pm.gp.cov.ExpQuad(1, ls=ls_1)
cov_2 = tau_2 * pm.gp.cov.ExpQuad(1, ls=ls_2)

cov = cov_1 + cov_2
# Add white noise to stabilise
cov += pm.gp.cov.WhiteNoise(1e-6)

X = np.linspace(0, 2, 200)[:, None]
K = cov(X).eval()

plt.plot(
    X,
    pm.draw(pm.MvNormal.dist(mu=np.zeros(len(K)), cov=K, shape=len(K)), draws=3, random_seed=rng).T,
)
plt.title("Samples from the GP prior")
plt.ylabel("y")
plt.xlabel("X");
```

+++ {"papermill": {"duration": 0.11646, "end_time": "2020-12-22T18:37:42.956319", "exception": false, "start_time": "2020-12-22T18:37:42.839859", "status": "completed"}}

#### Multiplication

```{code-cell} ipython3
---
papermill:
  duration: 0.554047
  end_time: '2020-12-22T18:37:43.627013'
  exception: false
  start_time: '2020-12-22T18:37:43.072966'
  status: completed
---
ls_1 = 0.1
tau_1 = 2.0
ls_2 = 0.5
tau_2 = 1.0
cov_1 = tau_1 * pm.gp.cov.ExpQuad(1, ls=ls_1)
cov_2 = tau_2 * pm.gp.cov.ExpQuad(1, ls=ls_2)

cov = cov_1 * cov_2
# Add white noise to stabilise
cov += pm.gp.cov.WhiteNoise(1e-6)

X = np.linspace(0, 2, 200)[:, None]
K = cov(X).eval()

plt.plot(
    X,
    pm.draw(pm.MvNormal.dist(mu=np.zeros(len(K)), cov=K, shape=len(K)), draws=3, random_seed=rng).T,
)
plt.title("Samples from the GP prior")
plt.ylabel("y")
plt.xlabel("X");
```

+++ {"papermill": {"duration": 0.125568, "end_time": "2020-12-22T18:37:43.873379", "exception": false, "start_time": "2020-12-22T18:37:43.747811", "status": "completed"}}

#### Exponentiation

```{code-cell} ipython3
---
papermill:
  duration: 0.525416
  end_time: '2020-12-22T18:37:44.521691'
  exception: false
  start_time: '2020-12-22T18:37:43.996275'
  status: completed
---
ls_1 = 0.1
tau_1 = 2.0
power = 2
cov_1 = tau_1 * pm.gp.cov.ExpQuad(1, ls=ls_1)

cov = cov_1**power
# Add white noise to stabilise
cov += pm.gp.cov.WhiteNoise(1e-6)

X = np.linspace(0, 2, 200)[:, None]
K = cov(X).eval()

plt.plot(
    X,
    pm.draw(pm.MvNormal.dist(mu=np.zeros(len(K)), cov=K, shape=len(K)), draws=3, random_seed=rng).T,
)
plt.title("Samples from the GP prior")
plt.ylabel("y")
plt.xlabel("X");
```

+++ {"papermill": {"duration": 0.124028, "end_time": "2020-12-22T18:37:44.770709", "exception": false, "start_time": "2020-12-22T18:37:44.646681", "status": "completed"}}

### Defining a custom covariance function

Covariance function objects in PyMC need to implement the `__init__`, `diag`, and `full` methods, and subclass `gp.cov.Covariance`.  `diag` returns only the diagonal of the covariance matrix, and `full` returns the full covariance matrix.  The `full` method has two inputs `X` and `Xs`.  `full(X)` returns the square covariance matrix, and `full(X, Xs)` returns the cross-covariances between the two sets of inputs.

For example, here is the implementation of the `WhiteNoise` covariance function:

```python
class WhiteNoise(pm.gp.cov.Covariance):
    def __init__(self, sigma):
        super(WhiteNoise, self).__init__(1, None)
        self.sigma = sigma

    def diag(self, X):
        return tt.alloc(tt.square(self.sigma), X.shape[0])

    def full(self, X, Xs=None):
        if Xs is None:
            return tt.diag(self.diag(X))
        else:
            return tt.alloc(0.0, X.shape[0], Xs.shape[0])
```

If we have forgotten an important covariance or mean function, please feel free to submit a pull request!

+++

## References

:::{bibliography}
:filter: docname in docnames
:::

+++

## Authors
* Authored by Bill Engels
* Updated to v4 by Oriol Abril Pla in Nov 2022 ([pymc-examples#301](https://github.com/pymc-devs/pymc-examples/pull/301))
* Updated to v5 by Juan Orduz in Nov 2023 ([pymc-examples#593](https://github.com/pymc-devs/pymc-examples/pull/593))

+++

## Watermark

```{code-cell} ipython3
---
papermill:
  duration: 0.212109
  end_time: '2020-12-22T18:37:55.023502'
  exception: false
  start_time: '2020-12-22T18:37:54.811393'
  status: completed
---
%load_ext watermark
%watermark -n -u -v -iv -w -p xarray
```

:::{include} ../page_footer.md
:::

```{code-cell} ipython3

```
