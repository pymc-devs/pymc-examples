---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python PyMC3 (Dev)
  language: python
  name: pymc3-dev-py38
---

# Multilevel Gravity Survey with MLDA

+++

### The MLDA sampler
This notebook is designed to demonstrate the Multi-Level Delayed Acceptance MCMC algorithm (MLDA) proposed in Dodwell (2019), as implemented within PyMC3. If you are using MLDA for the first time, we recommend first running the `MLDA_simple_linear_regression.ipynb` notebook in the same folder.

The MLDA sampler can be more efficient than other MCMC samplers when dealing with computationally intensive problems where we have access not only to the desired (fine) posterior distribution but also to a set of approximate (coarse) posteriors of decreasing accuracy and decreasing computational cost. In simple terms, we can use multiple chains on different coarseness levels and coarser chains' samples are used as proposals for the finer chains. This has been shown to improve the effective sample size of the finest chain and this allows us to reduce the number of expensive fine-chain likelihood evaluations. 

The notebook initially defines the necessary classes that describe the model. These classes use scipy to do the numerical solve in the forward model. It then instantiates models in two levels (with different granularities) and generates data for inference. Finally, the model classes are passed to two pymc3 models using Theano Ops and inference is done using three different MCMC methods (including MLDA). Some summary results and comparison plots are shown at the end to demonstrate the results. The use of Theano Ops is common when users want to use external code to calculate their likelihood (e.g. some fast PDE solver) and this example is designed to serve as a starting point for users to employ MLDA in their own problems.

Please note that the MLDA sampler is new in PyMC3. The user should be extra critical about the results and report any problems as issues in the pymc3's github repository.

The notebook results shown below were generated on a MacBook Pro with a 2.6 GHz 6-core Intel Core i7, 32 GB DDR4 and macOS 10.15.4.

### Gravity Surveying
In this notebook, we solve a 2-dimensional gravity surveying problem, adapted from the 1D problem presented in Hansen (2010). 

Our aim is to recover a two-dimensional mass distribution $f(\vec{t})$ at a known depth $d$ below the surface from measurements $g(\vec{s})$ of the vertical component of the gravitational field at the surface. The contribution to $g(\vec{s})$ from infinitesimally small areas of the subsurface mass distribution are given by:

\begin{equation}
    dg = \frac{\sin \theta}{r^2} f(\vec{t}) \: d\vec{t}
\end{equation}
where $\theta$ is the angle between the vertical plane and a straight line between two points $f(\vec{t})$ and $g(\vec{s})$, and $r = | \vec{s} - \vec{t} |$ is the Eucledian distance between the points. We exploit that $\sin \theta = \frac{d}{r}$, so that

\begin{equation}
    \frac{\sin \theta}{r^2} f(\vec{t}) \: d\vec{t} = \frac{d}{r^3} f(\vec{t}) \: d\vec{t} = \frac{d}{ | \vec{s} - \vec{t} |^3} f(\vec{t}) \: d\vec{t}
\end{equation}

This yields the integral equation,

\begin{equation}
    g(\vec{s}) = \iint_T \frac{d}{ | \vec{s} - \vec{t} |^3} f(\vec{t}) \: d\vec{t}
\end{equation}

where $T = [0,1]^2$ is the domain of the function $f(\vec{t})$. This constitutes our forward model.

We solve this integral numerically using midpoint quadrature. For simplicity, we use the same number of quadrature points along each axis, so that in discrete form our forward model becomes

\begin{equation}
    g(\vec{s}_i) = \sum_{j=1}^{m} \omega_j \frac{d}{ | \vec{s}_i - \vec{t}_j |^3} \hat{f}(\vec{t}_j), \quad i = 1, \dots, n, \quad j = 1, \dots, m
\end{equation}

where $\omega_j = \frac{1}{m}$ are quadrature weights, $\hat{f}(\vec{t}_j)$ is the approximate subsurface mass at quadrature points $j = 1, \dots, m$, and  $g(\vec{s}_i)$ is surface measurements at collocation points $i = 1, \dots, n$. Hence when $n > m$, we are dealing with an overdetermined problem and vice versa. 

This results in a linear system $\mathbf{Ax = b}$, where
\begin{equation}
    a_{ij} = \omega_j \frac{d}{ | \vec{s}_i - \vec{t}_j |^3}, \quad x_j = \hat{f}(\vec{t}_j), \quad b_i = g(\vec{s}_i).
\end{equation}
In this particular problem, the matrix $\mathbf{A}$ has a very high condition number, leading to an ill-posed inverse problem, which entails numerical instability and spurious, often oscillatory, solutions for noisy right hand sides. These types of problems are traditionally solved by way of some manner of *regularisation*, but they can be handled in a natural and elegant fashion in the context of a Bayesian inverse problem.

### Mass Distribution as a Gaussian Random Process
We model the unknown mass distribution as a Gaussian Random Process with a Matern 5/2 covariance kernel (Rasmussen and Williams, 2006):
\begin{equation}
    C_{5/2}(\vec{t}, \vec{t}') = \sigma^2 \left( 1 + \frac{\sqrt{5} | \vec{t}-\vec{t}' | }{l} + \frac{5 | \vec{t}-\vec{t}' |^2}{3l^2} \right) \exp \left( - \frac{\sqrt{5} | \vec{t}-\vec{t}' | }{l} \right)
\end{equation}
where $l$ is the covariance length scale and $\sigma^2$ is the variance.

### Comparison
Within this notebook, a simple MLDA sampler is compared to a Metropolis and a DEMetropolisZ sampler. The example demonstrates that MLDA is more efficient than the other samplers when measured by the Effective Samples per Second they can generate from the posterior. 

### References

Dodwell, Tim & Ketelsen, Chris & Scheichl, Robert & Teckentrup, Aretha. (2019). Multilevel Markov Chain Monte Carlo. SIAM Review. 61. 509-545. https://doi.org/10.1137/19M126966X

Per Christian Hansen. *Discrete Inverse Problems: Insight and Algorithms*. Society for Industrial and Applied Mathematics, January 2010.

Carl Edward Rasmussen and Christopher K. I. Williams. *Gaussian processes for machine learning*. Adaptive computation and machine learning. 2006.

+++

## Import modules

```{code-cell} ipython3
import os as os
import warnings

os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Set environment variable

import sys as sys
import time as time

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import theano
import theano.tensor as tt

from numpy.linalg import inv
from scipy.interpolate import RectBivariateSpline
from scipy.linalg import eigh
from scipy.spatial import distance_matrix
```

```{code-cell} ipython3
warnings.simplefilter(action="ignore", category=FutureWarning)
```

```{code-cell} ipython3
RANDOM_SEED = 123446
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")
```

```{code-cell} ipython3
# Checking versions
print(f"Theano version: {theano.__version__}")
print(f"PyMC3 version: {pm.__version__}")
```

## Define Matern52 kernel for modelling Gaussian Random Field
This is utility code which is necessary for defining the model later - you are free to ignore it or place it in an external file

```{code-cell} ipython3
class SquaredExponential:
    def __init__(self, coords, mkl, lamb):
        """
        This class sets up a random process
        on a grid and generates
        a realisation of the process, given
        parameters or a random vector.
        """

        # Internalise the grid and set number of vertices.
        self.coords = coords
        self.n_points = self.coords.shape[0]
        self.eigenvalues = None
        self.eigenvectors = None
        self.parameters = None
        self.random_field = None

        # Set some random field parameters.
        self.mkl = mkl
        self.lamb = lamb

        self.assemble_covariance_matrix()

    def assemble_covariance_matrix(self):
        """
        Create a snazzy distance-matrix for rapid
        computation of the covariance matrix.
        """
        dist = distance_matrix(self.coords, self.coords)

        # Compute the covariance between all
        # points in the space.
        self.cov = np.exp(-0.5 * dist**2 / self.lamb**2)

    def plot_covariance_matrix(self):
        """
        Plot the covariance matrix.
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(self.cov, cmap="binary")
        plt.colorbar()
        plt.show()

    def compute_eigenpairs(self):
        """
        Find eigenvalues and eigenvectors using Arnoldi iteration.
        """
        eigvals, eigvecs = eigh(self.cov, eigvals=(self.n_points - self.mkl, self.n_points - 1))

        order = np.flip(np.argsort(eigvals))
        self.eigenvalues = eigvals[order]
        self.eigenvectors = eigvecs[:, order]

    def generate(self, parameters=None):
        """
        Generate a random field, see
        Scarth, C., Adhikari, S., Cabral, P. H.,
        Silva, G. H. C., & Prado, A. P. do. (2019).
        Random field simulation over curved surfaces:
        Applications to computational structural mechanics.
        Computer Methods in Applied Mechanics and Engineering,
        345, 283–301. https://doi.org/10.1016/j.cma.2018.10.026
        """

        if parameters is None:
            self.parameters = np.random.normal(size=self.mkl)
        else:
            self.parameters = np.array(parameters).flatten()

        self.random_field = np.linalg.multi_dot(
            (self.eigenvectors, np.sqrt(np.diag(self.eigenvalues)), self.parameters)
        )

    def plot(self, lognormal=True):
        """
        Plot the random field.
        """

        if lognormal:
            random_field = self.random_field
            contour_levels = np.linspace(min(random_field), max(random_field), 20)
        else:
            random_field = np.exp(self.random_field)
            contour_levels = np.linspace(min(random_field), max(random_field), 20)

        plt.figure(figsize=(12, 10))
        plt.tricontourf(
            self.coords[:, 0],
            self.coords[:, 1],
            random_field,
            levels=contour_levels,
            cmap="plasma",
        )
        plt.colorbar()
        plt.show()


class Matern52(SquaredExponential):
    def assemble_covariance_matrix(self):
        """
        This class inherits from RandomProcess and creates a Matern 5/2 covariance matrix.
        """

        # Compute scaled distances.
        dist = np.sqrt(5) * distance_matrix(self.coords, self.coords) / self.lamb

        # Set up Matern 5/2 covariance matrix.
        self.cov = (1 + dist + dist**2 / 3) * np.exp(-dist)
```

## Define the Gravity model and generate data
This is a bit lengthy due to the model used in this case, it contains class definitions and also instantiation of class objects and data generation.

```{code-cell} ipython3
# Set the model parameters.
depth = 0.1
n_quad = 64
n_data = 64

# noise level
noise_level = 0.02

# Set random process parameters.
lamb = 0.1
mkl = 14

# Set the quadrature degree for each model level (coarsest first)
n_quadrature = [16, 64]
```

```{code-cell} ipython3
class Gravity:
    """
    Gravity is a class that implements a simple gravity surveying problem,
    as described in Hansen, P. C. (2010). Discrete Inverse Problems: Insight and Algorithms.
    Society for Industrial and Applied Mathematics.
    It uses midpoint quadrature to evaluate a Fredholm integral of the first kind.
    """

    def __init__(self, f_function, depth, n_quad, n_data):
        # Set the function describing the distribution of subsurface density.
        self.f_function = f_function

        # Set the depth of the density (distance to the surface measurements).
        self.depth = depth

        # Set the quadrature degree along one dimension.
        self.n_quad = n_quad

        # Set the number of data points along one dimension
        self.n_data = n_data

        # Set the quadrature points.
        x = np.linspace(0, 1, self.n_quad + 1)
        self.tx = (x[1:] + x[:-1]) / 2
        y = np.linspace(0, 1, self.n_quad + 1)
        self.ty = (y[1:] + y[:-1]) / 2
        TX, TY = np.meshgrid(self.tx, self.ty)

        # Set the measurement points.
        self.sx = np.linspace(0, 1, self.n_data)
        self.sy = np.linspace(0, 1, self.n_data)
        SX, SY = np.meshgrid(self.sx, self.sy)

        # Create coordinate vectors.
        self.T_coords = np.c_[TX.ravel(), TY.ravel(), np.zeros(self.n_quad**2)]
        self.S_coords = np.c_[SX.ravel(), SY.ravel(), self.depth * np.ones(self.n_data**2)]

        # Set the quadrature weights.
        self.w = 1 / self.n_quad**2

        # Compute a distance matrix
        dist = distance_matrix(self.S_coords, self.T_coords)

        # Create the Fremholm kernel.
        self.K = self.w * self.depth / dist**3

        # Evaluate the density function on the quadrature points.
        self.f = self.f_function(TX, TY).flatten()

        # Compute the surface density (noiseless measurements)
        self.g = np.dot(self.K, self.f)

    def plot_model(self):
        # Plot the density and the signal.
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].set_title("Density")
        f = axes[0].imshow(
            self.f.reshape(self.n_quad, self.n_quad),
            extent=(0, 1, 0, 1),
            origin="lower",
            cmap="plasma",
        )
        fig.colorbar(f, ax=axes[0])
        axes[1].set_title("Signal")
        g = axes[1].imshow(
            self.g.reshape(self.n_data, self.n_data),
            extent=(0, 1, 0, 1),
            origin="lower",
            cmap="plasma",
        )
        fig.colorbar(g, ax=axes[1])
        plt.show()

    def plot_kernel(self):
        # Plot the kernel.
        plt.figure(figsize=(8, 6))
        plt.imshow(self.K, cmap="plasma")
        plt.colorbar()
        plt.show()
```

```{code-cell} ipython3
# This is a function describing the subsurface density.
def f(TX, TY):
    f = np.sin(np.pi * TX) + np.sin(3 * np.pi * TY) + TY + 1
    f = f / f.max()
    return f
```

```{code-cell} ipython3
# Initialise a model
model_true = Gravity(f, depth, n_quad, n_data)
```

```{code-cell} ipython3
model_true.plot_model()
```

```{code-cell} ipython3
# Add noise to the data.
np.random.seed(123)
noise = np.random.normal(0, noise_level, n_data**2)
data = model_true.g + noise

# Plot the density and the signal.
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].set_title("Noiseless Signal")
g = axes[0].imshow(
    model_true.g.reshape(n_data, n_data),
    extent=(0, 1, 0, 1),
    origin="lower",
    cmap="plasma",
)
fig.colorbar(g, ax=axes[0])
axes[1].set_title("Noisy Signal")
d = axes[1].imshow(data.reshape(n_data, n_data), extent=(0, 1, 0, 1), origin="lower", cmap="plasma")
fig.colorbar(d, ax=axes[1])
plt.show()
```

```{code-cell} ipython3
class Gravity_Forward(Gravity):
    """
    Gravity forward is a class that implements the gravity problem,
    but computation of signal and density is delayed to the "solve"
    method, since it relied on a Gaussian Random Field to model
    the (unknown) density.
    """

    def __init__(self, depth, n_quad, n_data):
        # Set the depth of the density (distance to the surface measurements).
        self.depth = depth

        # Set the quadrature degree along one axis.
        self.n_quad = n_quad

        # Set the number of data points along one axis.
        self.n_data = n_data

        # Set the quadrature points.
        x = np.linspace(0, 1, self.n_quad + 1)
        self.tx = (x[1:] + x[:-1]) / 2
        y = np.linspace(0, 1, self.n_quad + 1)
        self.ty = (y[1:] + y[:-1]) / 2
        TX, TY = np.meshgrid(self.tx, self.ty)

        # Set the measurement points.
        self.sx = np.linspace(0, 1, self.n_data)
        self.sy = np.linspace(0, 1, self.n_data)
        SX, SY = np.meshgrid(self.sx, self.sy)

        # Create coordinate vectors.
        self.T_coords = np.c_[TX.ravel(), TY.ravel(), np.zeros(self.n_quad**2)]
        self.S_coords = np.c_[SX.ravel(), SY.ravel(), self.depth * np.ones(self.n_data**2)]

        # Set the quadrature weights.
        self.w = 1 / self.n_quad**2

        # Compute a distance matrix
        dist = distance_matrix(self.S_coords, self.T_coords)

        # Create the Fremholm kernel.
        self.K = self.w * self.depth / dist**3

    def set_random_process(self, random_process, lamb, mkl):
        # Set the number of KL modes.
        self.mkl = mkl

        # Initialise a random process on the quadrature points.
        # and compute the eigenpairs of the covariance matrix,
        self.random_process = random_process(self.T_coords, self.mkl, lamb)
        self.random_process.compute_eigenpairs()

    def solve(self, parameters):
        # Internalise the Random Field parameters
        self.parameters = parameters

        # Create a realisation of the random process, given the parameters.
        self.random_process.generate(self.parameters)
        mean = 0.0
        stdev = 1.0

        # Set the density.
        self.f = mean + stdev * self.random_process.random_field

        # Compute the signal.
        self.g = np.dot(self.K, self.f)

    def get_data(self):
        # Get the data vector.
        return self.g
```

```{code-cell} ipython3
# We project the eigenmodes of the fine model to the quadrature points
# of the coarse model using linear interpolation.
def project_eigenmodes(model_coarse, model_fine):
    model_coarse.random_process.eigenvalues = model_fine.random_process.eigenvalues
    for i in range(model_coarse.mkl):
        interpolator = RectBivariateSpline(
            model_fine.tx,
            model_fine.ty,
            model_fine.random_process.eigenvectors[:, i].reshape(
                model_fine.n_quad, model_fine.n_quad
            ),
        )
        model_coarse.random_process.eigenvectors[:, i] = interpolator(
            model_coarse.tx, model_coarse.ty
        ).ravel()
```

```{code-cell} ipython3
# Initialise the models, according the quadrature degree.
my_models = []
for i, n_quad in enumerate(n_quadrature):
    my_models.append(Gravity_Forward(depth, n_quad, n_data))
    my_models[i].set_random_process(Matern52, lamb, mkl)

# Project the eigenmodes of the fine model to the coarse model.
for m in my_models[:-1]:
    project_eigenmodes(m, my_models[-1])
```

## Solve and plot models to demonstrate coarse/fine difference

```{code-cell} ipython3
# Plot the same random realisation for each level, and the corresponding signal,
# to validate that the levels are equivalents.
for i, m in enumerate(my_models):
    print(f"Level {i}:")
    np.random.seed(2)
    m.solve(np.random.normal(size=mkl))
    m.plot_model()
```

```{code-cell} ipython3
plt.title(f"Largest {mkl} KL eigenvalues of GP prior")
plt.plot(my_models[-1].random_process.eigenvalues)
plt.show()
```

## Compare computation cost of coarse and fine model solve
The bigger the difference in time, the more MLDA has potential to increase efficiency

```{code-cell} ipython3
%%timeit
my_models[0].solve(np.random.normal(size=mkl))
```

```{code-cell} ipython3
%%timeit
my_models[-1].solve(np.random.normal(size=mkl))
```

## Set MCMC parameters for inference

```{code-cell} ipython3
# Number of draws from the distribution
ndraws = 15000

# Number of burn-in samples
nburn = 10000

# MLDA and Metropolis tuning parameters
tune = True
tune_interval = 100  # Set high to prevent tuning.
discard_tuning = True

# Number of independent chains.
nchains = 3

# Subsampling rate for MLDA
nsub = 5

# Set prior parameters for multivariate Gaussian prior distribution.
mu_prior = np.zeros(mkl)
cov_prior = np.eye(mkl)

# Set the sigma for inference.
sigma = 1.0

# Sampling seed
sampling_seed = RANDOM_SEED
```

## Define a Theano Op for the likelihood
This creates the theano op needed to pass the above model to the PyMC3 sampler

```{code-cell} ipython3
def my_loglik(my_model, theta, data, sigma):
    """
    This returns the log-likelihood of my_model given theta,
    datapoints, the observed data and sigma. It uses the
    model_wrapper function to do a model solve.
    """
    my_model.solve(theta)
    output = my_model.get_data()
    return -(0.5 / sigma**2) * np.sum((output - data) ** 2)


class LogLike(tt.Op):
    """
    Theano Op that wraps the log-likelihood computation, necessary to
    pass "black-box" code into pymc3.
    Based on the work in:
    https://docs.pymc.io/notebooks/blackbox_external_likelihood.html
    https://docs.pymc.io/Advanced_usage_of_Theano_in_PyMC3.html
    """

    # Specify what type of object will be passed and returned to the Op when it is
    # called. In our case we will be passing it a vector of values (the parameters
    # that define our model and a model object) and returning a single "scalar"
    # value (the log-likelihood)
    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, my_model, loglike, data, sigma):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        my_model:
            A Model object (defined in model.py) that contains the parameters
            and functions of out model.
        loglike:
            The log-likelihood function we've defined, in this example it is
            my_loglik.
        data:
            The "observed" data that our log-likelihood function takes in. These
            are the true data generated by the finest model in this example.
        x:
            The dependent variable (aka 'x') that our model requires. This is
            the datapoints in this example.
        sigma:
            The noise standard deviation that our function requires.
        """
        # add inputs as class attributes
        self.my_model = my_model
        self.likelihood = loglike
        self.data = data
        self.sigma = sigma

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(self.my_model, theta, self.data, self.sigma)

        outputs[0][0] = np.array(logl)  # output the log-likelihood
```

```{code-cell} ipython3
# create Theano Ops to wrap likelihoods of all model levels and store them in list
logl = []
for i, m_i in enumerate(my_models):
    logl.append(LogLike(m_i, my_loglik, data, sigma))
```

## Create coarse model in PyMC3

```{code-cell} ipython3
# Set up models in pymc3 for each level - excluding finest model level
coarse_models = []
for j in range(len(my_models) - 1):
    with pm.Model() as model:
        # Multivariate normal prior.
        theta = pm.MvNormal("theta", mu=mu_prior, cov=cov_prior, shape=mkl)

        # Use the Potential class to evaluate likelihood
        pm.Potential("likelihood", logl[j](theta))

    coarse_models.append(model)
```

## Create fine model and perform inference
Note that we sample using all three methods and that we use the MAP as the starting point for sampling

```{code-cell} ipython3
# Set up finest model and perform inference with PyMC3, using the MLDA algorithm
# and passing the coarse_models list created above.
method_names = []
traces = []
runtimes = []

with pm.Model() as model:
    # Multivariate normal prior.
    theta = pm.MvNormal("theta", mu=mu_prior, cov=cov_prior, shape=mkl)

    # Use the Potential class to evaluate likelihood
    pm.Potential("likelihood", logl[-1](theta))

    # Find the MAP estimate which is used as the starting point for sampling
    MAP = pm.find_MAP()

    # Initialise a Metropolis, DEMetropolisZ and MLDA step method objects (passing the subsampling rate and
    # coarse models list for the latter)
    step_metropolis = pm.Metropolis(tune=tune, tune_interval=tune_interval)
    step_demetropolisz = pm.DEMetropolisZ(tune_interval=tune_interval)
    step_mlda = pm.MLDA(
        coarse_models=coarse_models, subsampling_rates=nsub, base_tune_interval=tune_interval
    )

    # Inference!
    # Metropolis
    t_start = time.time()
    method_names.append("Metropolis")
    traces.append(
        pm.sample(
            draws=ndraws,
            step=step_metropolis,
            chains=nchains,
            tune=nburn,
            discard_tuned_samples=discard_tuning,
            random_seed=sampling_seed,
            start=MAP,
            cores=1,
            mp_ctx="forkserver",
        )
    )
    runtimes.append(time.time() - t_start)

    # DEMetropolisZ
    t_start = time.time()
    method_names.append("DEMetropolisZ")
    traces.append(
        pm.sample(
            draws=ndraws,
            step=step_demetropolisz,
            chains=nchains,
            tune=nburn,
            discard_tuned_samples=discard_tuning,
            random_seed=sampling_seed,
            start=MAP,
            cores=1,
            mp_ctx="forkserver",
        )
    )
    runtimes.append(time.time() - t_start)

    # MLDA
    t_start = time.time()
    method_names.append("MLDA")
    traces.append(
        pm.sample(
            draws=ndraws,
            step=step_mlda,
            chains=nchains,
            tune=nburn,
            discard_tuned_samples=discard_tuning,
            random_seed=sampling_seed,
            start=MAP,
            cores=1,
            mp_ctx="forkserver",
        )
    )
    runtimes.append(time.time() - t_start)
```

## Get post-sampling stats and diagnostics

+++

#### Print MAP estimate and pymc3 sampling summary

```{code-cell} ipython3
with model:
    print(
        f"\nDetailed summaries and plots:\nMAP estimate: {MAP['theta']}. Not used as starting point."
    )
    for i, trace in enumerate(traces):
        print(f"\n{method_names[i]} Sampler:\n")
        display(az.summary(trace))
```

#### Show ESS and ESS/sec for all samplers

```{code-cell} ipython3
acc = []
ess = []
ess_n = []
performances = []

# Get some more statistics.
with model:
    for i, trace in enumerate(traces):
        acc.append(trace.get_sampler_stats("accepted").mean())
        ess.append(np.array(az.ess(trace).to_array()))
        ess_n.append(ess[i] / len(trace) / trace.nchains)
        performances.append(ess[i] / runtimes[i])
        print(
            f"\n{method_names[i]} Sampler: {len(trace)} drawn samples in each of "
            f"{trace.nchains} chains."
            f"\nRuntime: {runtimes[i]} seconds"
            f"\nAcceptance rate: {acc[i]}"
            f"\nESS list: {np.round(ess[i][0], 3)}"
            f"\nNormalised ESS list: {np.round(ess_n[i][0], 3)}"
            f"\nESS/sec: {np.round(performances[i][0], 3)}"
        )

    # Plot the effective sample size (ESS) and relative ESS (ES/sec) of each of the sampling strategies.
    colors = ["firebrick", "darkgoldenrod", "darkcyan", "olivedrab"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].set_title("ESS")
    for i, e in enumerate(ess):
        axes[0].bar(
            [j + i * 0.2 for j in range(mkl)],
            e.ravel(),
            width=0.2,
            color=colors[i],
            label=method_names[i],
        )
    axes[0].set_xticks([i + 0.3 for i in range(mkl)])
    axes[0].set_xticklabels([f"theta_{i}" for i in range(mkl)])
    axes[0].legend()

    axes[1].set_title("ES/sec")
    for i, p in enumerate(performances):
        axes[1].bar(
            [j + i * 0.2 for j in range(mkl)],
            p.ravel(),
            width=0.2,
            color=colors[i],
            label=method_names[i],
        )
    axes[1].set_xticks([i + 0.3 for i in range(mkl)])
    axes[1].set_xticklabels([f"theta_{i}" for i in range(mkl)])
    axes[1].legend()
    plt.show()
```

#### Plot distributions and trace.
Vertical grey lines represent the MAP estimate of each parameter.

```{code-cell} ipython3
with model:
    lines = (("theta", {}, MAP["theta"].tolist()),)
    for i, trace in enumerate(traces):
        az.plot_trace(trace, lines=lines)

        # Ugly hack to get some titles in.
        x_offset = -0.1 * ndraws
        y_offset = trace.get_values("theta").max() + 0.25 * (
            trace.get_values("theta").max() - trace.get_values("theta").min()
        )
        plt.text(x_offset, y_offset, "{} Sampler".format(method_names[i]))
```

#### Plot true and recovered densities
This is useful for verification, i.e. to compare the true model density and signal to the estimated ones from the samplers.

```{code-cell} ipython3
print("True Model")
model_true.plot_model()
with model:
    print("MAP estimate:")
    my_models[-1].solve(MAP["theta"])
    my_models[-1].plot_model()
    for i, t in enumerate(traces):
        print(f"Recovered by: {method_names[i]}")
        my_models[-1].solve(az.summary(t)["mean"].values)
        my_models[-1].plot_model()
```

```{code-cell} ipython3
# Show trace of lowest energy mode for Metropolis sampler
plt.figure(figsize=(8, 3))
plt.plot(traces[0]["theta"][:5000, -1])
plt.show()
```

```{code-cell} ipython3
# Show trace of lowest energy mode for MLDA sampler
plt.figure(figsize=(8, 3))
plt.plot(traces[2]["theta"][:5000:, -1])
plt.show()
```

```{code-cell} ipython3
# Make sure samplers have converged
assert all(az.rhat(traces[0]) < 1.03)
assert all(az.rhat(traces[1]) < 1.03)
assert all(az.rhat(traces[2]) < 1.03)
```

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w
```
