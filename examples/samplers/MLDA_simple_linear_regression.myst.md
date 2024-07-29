---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python (PyMC Dev)
  language: python
  name: pymc-dev
---

### The MLDA sampler
This notebook is a good starting point to understand the basic usage of the Multi-Level Delayed Acceptance MCMC algorithm (MLDA) proposed in [1], as implemented within PyMC.

It uses a simple linear regression model (and a toy coarse model counterpart) to show the basic workflow when using MLDA. The model is similar to the one used in {ref}`pymc:glm_linear`.

The MLDA sampler is designed to deal with computationally intensive problems where we have access not only to the desired (fine) posterior distribution but also to a set of approximate (coarse) posteriors of decreasing accuracy and decreasing computational cost (we need at least one of those). Its main idea is that coarser chains' samples are used as proposals for the finer chains. A coarse chain runs for a fixed number of iterations and the last sample is used as a proposal for the finer chain. This has been shown to improve the effective sample size of the finest chain and this allows us to reduce the number of expensive fine-chain likelihood evaluations. 

The PyMC implementation supports:
- Any number of levels
- Two types of bottom-level samplers (Metropolis, DEMetropolisZ)
- Various tuning parameters for the bottom-level samplers
- Separate subsampling rates for each level
- A choice between blocked and compound sampling for bottom-level Metropolis. 
- An adaptive error model to correct bias between coarse and fine models 
- A variance reduction technique that utilizes samples from all chains to reduce the variance of an estimated quantity of interest.

For more details about the MLDA sampler and the way it should be used and parameterised, the user can refer to the docstrings in the code and to the other example notebooks which deal with more complex problem settings and more advanced MLDA features.

Please note that the MLDA sampler is new in PyMC. The user should be extra critical about the results and report any problems as issues in the PyMC's github repository.

[1] Dodwell, Tim & Ketelsen, Chris & Scheichl, Robert & Teckentrup, Aretha. (2019). Multilevel Markov Chain Monte Carlo. SIAM Review. 61. 509-545. https://doi.org/10.1137/19M126966X

+++

### Work flow

MLDA is used in a similar way as most step method in PyMC. It has the special requirement that the user need to provide at least one coarse model to allow it to work.

The basic flow to use MLDA consists of four steps, which we demonstrate here using a simple linear regression model with a toy coarse model counterpart.

+++

##### Step 1: Generate some data

Here, we generate a vector `x` of 200 points equally spaced between 0.0 and 1.0. Then we project those onto a straight line with intercept 1.0 and slope 2.0, adding some random noise, resulting in a vector `y`. The goal is to infer the intercept and slope from `x` and `y`, i.e. a very simple linear regression problem.

```{code-cell} ipython3
# Import libraries
import time as time
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
```

```{code-cell} ipython3
az.style.use("arviz-darkgrid")
```

```{code-cell} ipython3
warnings.filterwarnings("ignore")
```

```{code-cell} ipython3
# Generate data
RANDOM_SEED = 915623497
np.random.seed(RANDOM_SEED)

true_intercept = 1
true_slope = 2
sigma = 1

size = 200
x = np.linspace(0, 1, size)
y = true_intercept + true_slope * x + np.random.normal(0, sigma**2, size)
```

```{code-cell} ipython3
# Plot the data
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, xlabel="x", ylabel="y", title="Generated data and underlying model")
ax.plot(x, y, "x", label="sampled data")
ax.plot(x, true_intercept + true_slope * x, label="true regression line", lw=2.0)
plt.legend(loc=0);
```

##### Step 2: Define the fine model

In this step we use the PyMC model definition language to define the priors and the likelihood. We choose non-informative Normal priors for both intercept and slope and a Normal likelihood, where we feed in `x` and `y`.

```{code-cell} ipython3
# Constructing the fine model
with pm.Model() as fine_model:
    # Define priors
    intercept = pm.Normal("intercept", 0, sigma=20)
    slope = pm.Normal("slope", 0, sigma=20)

    # Define likelihood
    likelihood = pm.Normal("y", mu=intercept + slope * x, sigma=sigma, observed=y)
```

##### Step 3: Define a coarse model
    
Here, we define a toy coarse model where coarseness is introduced by using fewer data in the likelihood compared to the fine model, i.e. we only use every 2nd data point from the original data set.

```{code-cell} ipython3
# Thinning the data set
x_coarse = x[::2]
y_coarse = y[::2]
```

```{code-cell} ipython3
# Constructing the coarse model
with pm.Model() as coarse_model:
    # Define priors
    intercept = pm.Normal("intercept", 0, sigma=20)
    slope = pm.Normal("slope", 0, sigma=20)

    # Define likelihood
    likelihood = pm.Normal("y", mu=intercept + slope * x_coarse, sigma=sigma, observed=y_coarse)
```

##### Step 4: Draw MCMC samples from the posterior using MLDA 

We feed `coarse_model` to the MLDA instance and we also set `subsampling_rate` to 10. The subsampling rate is the number of samples drawn in the coarse chain to construct a proposal for the fine chain. In this case, MLDA draws 10 samples in the coarse chain and uses the last one as a proposal for the fine chain. This is accepted or rejected by the fine chain and then control goes back to the coarse chain which generates another 10 samples, etc. Note that `pm.MLDA` has many other tuning arguments which can be found in the documentation. 

Next, we use the universal `pm.sample` method, passing the MLDA instance to it. This runs MLDA and returns a `trace`, containing all MCMC samples and various by-products. Here, we also run standard Metropolis and DEMetropolisZ samplers for comparison, which return separate traces. We time the runs to compare later.

Finally, PyMC provides various functions to visualise the trace and print summary statistics (two of them are shown below).

```{code-cell} ipython3
with fine_model:
    # Initialise step methods
    step = pm.MLDA(coarse_models=[coarse_model], subsampling_rates=[10])
    step_2 = pm.Metropolis()
    step_3 = pm.DEMetropolisZ()

    # Sample using MLDA
    t_start = time.time()
    trace = pm.sample(draws=6000, chains=4, tune=2000, step=step, random_seed=RANDOM_SEED)
    runtime = time.time() - t_start

    # Sample using Metropolis
    t_start = time.time()
    trace_2 = pm.sample(draws=6000, chains=4, tune=2000, step=step_2, random_seed=RANDOM_SEED)
    runtime_2 = time.time() - t_start

    # Sample using DEMetropolisZ
    t_start = time.time()
    trace_3 = pm.sample(draws=6000, chains=4, tune=2000, step=step_3, random_seed=RANDOM_SEED)
    runtime_3 = time.time() - t_start
```

```{code-cell} ipython3
# Trace plots
az.plot_trace(trace)
az.plot_trace(trace_2)
az.plot_trace(trace_3)
```

```{code-cell} ipython3
# Summary statistics for MLDA
az.summary(trace)
```

```{code-cell} ipython3
# Summary statistics for Metropolis
az.summary(trace_2)
```

```{code-cell} ipython3
# Summary statistics for DEMetropolisZ
az.summary(trace_3)
```

```{code-cell} ipython3
# Make sure samplers have converged
assert all(az.rhat(trace) < 1.03)
assert all(az.rhat(trace_2) < 1.03)
assert all(az.rhat(trace_3) < 1.03)
```

```{code-cell} ipython3
# Display runtimes
print(f"Runtimes: MLDA: {runtime}, Metropolis: {runtime_2}, DEMetropolisZ: {runtime_3}")
```

##### Comments

**Performance:**

You can see from the summary statistics above that MLDA's ESS is ~13x higher than Metropolis and ~2.5x higher than DEMetropolisZ. The runtime of MLDA is ~3.5x larger than either Metropolis or DEMetropolisZ. Therefore in this toy example MLDA is almost an overkill (especially compared to DEMetropolisZ). For more complex problems, where the difference in computational cost between the coarse and fine models/likelihoods is orders of magnitude, MLDA is expected to outperform the other two samplers, as long as the coarse model is reasonably close to the fine one. This case is often encountered in inverse problems in engineering, ecology, imaging, etc where a forward model can be defined with varying coarseness in space and/or time (e.g. subsurface water flow, predator prey models, etc). For an example of this, please see the `MLDA_gravity_surveying.ipynb notebook` in the same folder.

**Subsampling rate:**

The MLDA sampler is based on the assumption that the coarse proposal samples (i.e. the samples proposed from the coarse chain to the fine one) are independent (or almost independent) from each other. In order to generate independent samples, it is necessary to run the coarse chain for an adequate number of iterations to get rid of autocorrelation. Therefore, the higher the autocorrelation in the coarse chain, the more iterations are needed and the larger the subsampling rate should be. 

Values larger than the minimum for beating autocorreletion can further improve the proposal (as the distribution is explored better and the proposal are imptoved), and thus ESS. But at the same time more steps cost more computationally. Users are encouraged to do test runs with different subsampling rates to understand which gives the best ESS/sec. 

Note that in cases where you have more than one coarse model/level, MLDA allows you to choose a different subsampling rate for each coarse level (as a list of integers when you instantiate the stepper).

```{code-cell} ipython3
# Show packages' and Python's versions
%load_ext watermark
%watermark -n -u -v -iv -w
```
