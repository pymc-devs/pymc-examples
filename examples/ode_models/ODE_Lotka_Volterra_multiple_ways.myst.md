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

(ODE_Lotka_Volterra_fit_multiple_ways)= 
# ODE Lotka-Volterra With Bayesian Inference in Multiple Ways

:::{post} January 16, 2023
:tags: ODE, PyTensor, gradient-free inference
:category: intermediate, how-to
:author: Greg Brunkhorst
:::

```{code-cell} ipython3
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt

from numba import njit
from pymc.ode import DifferentialEquation
from pytensor.compile.ops import as_op
from scipy.integrate import odeint
from scipy.optimize import least_squares

print(f"Running on PyMC v{pm.__version__}")
```

```{code-cell} ipython3
%load_ext watermark
az.style.use("arviz-darkgrid")
rng = np.random.default_rng(1234)
```

## Purpose
The purpose of this notebook is to demonstrate how to perform Bayesian inference on a system of ordinary differential equations (ODEs), both with and without gradients.  The accuracy and efficiency of different samplers are compared.

We will first present the Lotka-Volterra predator-prey ODE model and example data.  Next, we will solve the ODE using `scipy.odeint` and (non-Bayesian) least squares optimization.  Next, we perform Bayesian inference in PyMC using non-gradient-based samplers.  Finally, we use gradient-based samplers and compare results.    

### Key Conclusions
Based on the experiments in this notebook, the most simple and efficient method for performing Bayesian inference on the Lotka-Volterra equations was to specify the ODE system in Scipy, wrap the function as a Pytensor op, and use a Differential Evolution Metropolis (DEMetropolis) sampler in PyMC.  

+++

## Background
### Motivation
Ordinary differential equation models (ODEs) are used in a variety of science and engineering domains to model the time evolution of physical variables.  A natural choice to estimate the values and uncertainty of model parameters given experimental data is Bayesian inference.  However, ODEs can be challenging to specify and solve in the Bayesian setting, therefore, this notebook steps through multiple methods for solving an ODE inference problem using PyMC. The Lotka-Volterra model used in this example has often been used for benchmarking Bayesian inference methods (e.g., in this Stan [case study](https://mc-stan.org/users/documentation/case-studies/lotka-volterra-predator-prey.html), and in Chapter 16 of *Statistical Rethinking* {cite:p}`mcelreath2018statistical`.

+++

### Lotka-Volterra Predator-Prey Model
The Lotka-Volterra model describes the interaction between a predator and prey species. This ODE given by:

$$
\begin{aligned}
\frac{d x}{dt} &=\alpha x -\beta xy \\ 
\frac{d y}{dt} &=-\gamma y + \delta xy
\end{aligned}
$$

+++

The state vector $X(t)=[x(t),y(t)]$ comprises the densities of the prey and the predator species respectively.  Parameters $\boldsymbol{\theta}=[\alpha,\beta,\gamma,\delta, x(0),y(0)]$ are the unknowns that we wish to infer from experimental observations.  $x(0), y(0)$ are the initial values of the states needed to solve the ODE, and $\alpha,\beta,\gamma$, and $\delta$ are unknown model parameters which represent the following:  
* $\alpha$ is the growing rate of prey when there's no predator.
* $\beta$ is the dying rate of prey due to predation.
* $\gamma$ is the dying rate of predator when there is no prey.
* $\delta$ is the growing rate of predator in the presence of prey.    

+++

### The Hudson's Bay Company data
The Lotka-Volterra predator prey model has been used to successfully explain the dynamics of natural populations of predators and prey, such as the lynx and snowshoe hare data of the Hudson's Bay Company. Since the dataset is small, we will hand-enter the values.

```{code-cell} ipython3
# fmt: off
data = pd.DataFrame(dict(
    year = np.arange(1900., 1921., 1),
    lynx = np.array([4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4,
                8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6]),
    hare = np.array([30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4, 
                 27.1, 40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7])))
data.head()
# fmt: on
```

```{code-cell} ipython3
# plot data function for reuse later
def plot_data(ax, lw=2, title="Hudson's Bay Company Data"):
    ax.plot(data.year, data.lynx, color="b", lw=lw, marker="o", markersize=12, label="Lynx (Data)")
    ax.plot(data.year, data.hare, color="g", lw=lw, marker="+", markersize=14, label="Hare (Data)")
    ax.legend(fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xlim([1900, 1920])
    ax.set_ylim(0)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Pelts (Thousands)", fontsize=14)
    ax.set_xticks(data.year.astype(int))
    ax.set_xticklabels(ax.get_xticks(), rotation=45)
    ax.set_title(title, fontsize=16)
    return ax
```

```{code-cell} ipython3
_, ax = plt.subplots(figsize=(12, 4))
plot_data(ax);
```

### Problem Statement
The purpose of this analysis is to estimate, with uncertainty, the parameters for the Lotka-Volterra model for the Hudson's Bay Company data from 1900 to 1920.  

+++

## Scipy `odeint`

+++

Here, we make a Python function that represents the right-hand-side of the ODE equations with the call signature needed for the `odeint` function.  Note that Scipy's `solve_ivp` could also be used, but the older `odeint` function was faster in speed tests and is therefore used in this notebook.  

```{code-cell} ipython3
# define the right hand side of the ODE equations in the Scipy odeint signature
from numba import njit


@njit
def rhs(X, t, theta):
    # unpack parameters
    x, y = X
    alpha, beta, gamma, delta, xt0, yt0 = theta
    # equations
    dx_dt = alpha * x - beta * x * y
    dy_dt = -gamma * y + delta * x * y
    return [dx_dt, dy_dt]
```

To get a feel for the model and make sure the equations are working correctly, let's run the model once with reasonable values for $\theta$ and plot the results.  

```{code-cell} ipython3
# plot model function
def plot_model(
    ax,
    x_y,
    time=np.arange(1900, 1921, 0.01),
    alpha=1,
    lw=3,
    title="Hudson's Bay Company Data and\nExample Model Run",
):
    ax.plot(time, x_y[:, 1], color="b", alpha=alpha, lw=lw, label="Lynx (Model)")
    ax.plot(time, x_y[:, 0], color="g", alpha=alpha, lw=lw, label="Hare (Model)")
    ax.legend(fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(title, fontsize=16)
    return ax
```

```{code-cell} ipython3
# note theta = alpha, beta, gamma, delta, xt0, yt0
theta = np.array([0.52, 0.026, 0.84, 0.026, 34.0, 5.9])
time = np.arange(1900, 1921, 0.01)

# call Scipy's odeint function
x_y = odeint(func=rhs, y0=theta[-2:], t=time, args=(theta,))

# plot
_, ax = plt.subplots(figsize=(12, 4))
plot_data(ax, lw=0)
plot_model(ax, x_y);
```

Looks like the `odeint` function is working as expected.  

+++

## Least Squares Solution

+++

Now, we can solve the ODE using least squares.  Make a function that calculates the residual error.  

```{code-cell} ipython3
# function that calculates residuals based on a given theta
def ode_model_resid(theta):
    return (
        data[["hare", "lynx"]] - odeint(func=rhs, y0=theta[-2:], t=data.year, args=(theta,))
    ).values.flatten()
```

Feed the residual error function to the Scipy `least_squares` solver.   

```{code-cell} ipython3
# calculate least squares using the Scipy solver
results = least_squares(ode_model_resid, x0=theta)

# put the results in a dataframe for presentation and convenience
df = pd.DataFrame()
parameter_names = ["alpha", "beta", "gamma", "delta", "h0", "l0"]
df["Parameter"] = parameter_names
df["Least Squares Solution"] = results.x
df.round(2)
```

Plot  

```{code-cell} ipython3
time = np.arange(1900, 1921, 0.01)
theta = results.x
x_y = odeint(func=rhs, y0=theta[-2:], t=time, args=(theta,))
fig, ax = plt.subplots(figsize=(12, 4))
plot_data(ax, lw=0)
plot_model(ax, x_y, title="Least Squares Solution");
```

Looks right.  If we didn't care about uncertainty, then we would be done.  But we do care about uncertainty, so let's move on to Bayesian inference.  

+++

## PyMC Model Specification for Gradient-Free Bayesian Inference

+++

Like other Numpy or Scipy-based functions, the `scipy.integrate.odeint` function cannot be used directly in a PyMC model because PyMC needs to know the variable input and output types to compile.  Therefore, we use a Pytensor wrapper to give the variable types to PyMC.  Then the function can be used in PyMC in conjunction with gradient-free samplers.   

+++

### Convert Python Function to a Pytensor Operator using @as_op decorator
We tell PyMC the input variable types and the output variable types using the `@as_op` decorator.  `odeint` returns Numpy arrays, but we tell PyMC that they are Pytensor double float tensors for this purpose.  

```{code-cell} ipython3
# decorator with input and output types a Pytensor double float tensors
@as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
def pytensor_forward_model_matrix(theta):
    return odeint(func=rhs, y0=theta[-2:], t=data.year, args=(theta,))
```

### PyMC Model

+++

Now, we can specify the PyMC model using the ode solver!  For priors, we will use the results from the least squares calculation (`results.x`) to assign priors that start in the right range.  These are empirically derived weakly informative priors.  We also make them positive-only for this problem.      

We will use a normal likelihood on untransformed data (i.e., not log transformed) to best fit the peaks of the data. 

```{code-cell} ipython3
theta = results.x  # least squares solution used to inform the priors
with pm.Model() as model:
    # Priors
    alpha = pm.TruncatedNormal("alpha", mu=theta[0], sigma=0.1, lower=0, initval=theta[0])
    beta = pm.TruncatedNormal("beta", mu=theta[1], sigma=0.01, lower=0, initval=theta[1])
    gamma = pm.TruncatedNormal("gamma", mu=theta[2], sigma=0.1, lower=0, initval=theta[2])
    delta = pm.TruncatedNormal("delta", mu=theta[3], sigma=0.01, lower=0, initval=theta[3])
    xt0 = pm.TruncatedNormal("xto", mu=theta[4], sigma=1, lower=0, initval=theta[4])
    yt0 = pm.TruncatedNormal("yto", mu=theta[5], sigma=1, lower=0, initval=theta[5])
    sigma = pm.HalfNormal("sigma", 10)

    # Ode solution function
    ode_solution = pytensor_forward_model_matrix(
        pm.math.stack([alpha, beta, gamma, delta, xt0, yt0])
    )

    # Likelihood
    pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=data[["hare", "lynx"]].values)
```

```{code-cell} ipython3
pm.model_to_graphviz(model=model)
```

### Plotting Functions
A couple of plotting functions that we will reuse below.  

```{code-cell} ipython3
def plot_model_trace(ax, trace_df, row_idx, lw=1, alpha=0.2):
    cols = ["alpha", "beta", "gamma", "delta", "xto", "yto"]
    row = trace_df.iloc[row_idx, :][cols].values

    # alpha, beta, gamma, delta, Xt0, Yt0
    time = np.arange(1900, 1921, 0.01)
    theta = row
    x_y = odeint(func=rhs, y0=theta[-2:], t=time, args=(theta,))
    plot_model(ax, x_y, time=time, lw=lw, alpha=alpha);
```

```{code-cell} ipython3
def plot_inference(
    ax,
    trace,
    num_samples=25,
    title="Hudson's Bay Company Data and\nInference Model Runs",
    plot_model_kwargs=dict(lw=1, alpha=0.2),
):
    trace_df = az.extract(trace, num_samples=num_samples).to_dataframe()
    plot_data(ax, lw=0)
    for row_idx in range(num_samples):
        plot_model_trace(ax, trace_df, row_idx, **plot_model_kwargs)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(title, fontsize=16)
```


## Gradient-Free Sampler Options
Having good gradient free samplers can open up the models that can be fit within PyMC.  There are five options for gradient-free samplers in PyMC that are applicable to this problem: 
* `Slice` - the default gradient-free sampler
* `DEMetropolisZ` - a differential evolution Metropolis sampler that uses the past to inform sampling jumps
* `DEMetropolis` - a differential evolution Metropolis sampler
* `Metropolis` - the vanilla Metropolis sampler
* `SMC` - Sequential Monte Carlo  

Let's give them a shot.

A few notes on running these inferences.  For each sampler, the number of tuning steps and draws have been reduced to run the inference in a reasonable amount of time (on the order of minutes).  This is not a sufficient number of draws to get a good inferences, in some cases, but it works for demonstration purposes.  In addition, multicore processing was not working for the Pytensor op function on all machines, so inference is performed on one core.         

+++

### Slice Sampler

```{code-cell} ipython3
# Variable list to give to the sample step parameter
vars_list = list(model.values_to_rvs.keys())[:-1]
```

```{code-cell} ipython3
# Specify the sampler
sampler = "Slice Sampler"
tune = draws = 2000

# Inference!
with model:
    trace_slice = pm.sample(step=[pm.Slice(vars_list)], tune=tune, draws=draws)
trace = trace_slice
az.summary(trace)
```

```{code-cell} ipython3
az.plot_trace(trace, kind="rank_bars")
plt.suptitle(f"Trace Plot {sampler}");
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(12, 4))
plot_inference(ax, trace, title=f"Data and Inference Model Runs\n{sampler} Sampler");
```

**Notes:**  
The Slice sampler was slow and resulted in a low effective sample size.  Despite this, the results are starting to look reasonable!  

+++

### DE MetropolisZ Sampler

```{code-cell} ipython3
sampler = "DEMetropolisZ"
tune = draws = 5000
with model:
    trace_DEMZ = pm.sample(step=[pm.DEMetropolisZ(vars_list)], tune=tune, draws=draws)
trace = trace_DEMZ
az.summary(trace)
```

```{code-cell} ipython3
az.plot_trace(trace, kind="rank_bars")
plt.suptitle(f"Trace Plot {sampler}");
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(12, 4))
plot_inference(ax, trace, title=f"Data and Inference\n{sampler} Sampler")
```

**Notes:**  
DEMetropolisZ sampled much quicker than the Slice sampler and therefore had a higher ESS per minute spent sampling.  The parameter estimates are similar.  A "final" inference would still need to beef up the number of samples.  

+++

### DEMetropolis Sampler

+++

In these experiments, DEMetropolis sampler was not accepting `tune` and requiring `chains` to be at least 8. We set draws at 5000, lower number like 3000 produce bad mixing.

```{code-cell} ipython3
sampler = "DEMetropolis"
chains = 8
draws = 6000
with model:
    trace_DEM = pm.sample(step=[pm.DEMetropolis(vars_list)], draws=draws, chains=chains)
trace = trace_DEM
az.summary(trace)
```

```{code-cell} ipython3
az.plot_trace(trace, kind="rank_bars")
plt.suptitle(f"Trace Plot {sampler}");
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(12, 4))
plot_inference(ax, trace, title=f"Data and Inference Model Runs\n{sampler} Sampler");
```

**Notes:**  
KDEs looks too wiggly, but ESS is high R-hat is good and rank_plots also look good

+++

### Metropolis Sampler

```{code-cell} ipython3
sampler = "Metropolis"
tune = draws = 5000
with model:
    trace_M = pm.sample(step=[pm.Metropolis(vars_list)], tune=tune, draws=draws)
trace = trace_M
az.summary(trace)
```

```{code-cell} ipython3
az.plot_trace(trace, kind="rank_bars")
plt.suptitle(f"Trace Plot {sampler}");
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(12, 4))
plot_inference(ax, trace, title=f"Data and Inference Model Runs\n{sampler} Sampler");
```

**Notes:**  
The old-school Metropolis sampler is less reliable and slower than the DEMetroplis samplers.  Not recommended.   

+++

### SMC Sampler

+++

The Sequential Monte Carlo (SMC) sampler can be used to sample a regular Bayesian model or to run model without a likelihood (Aproximate Bayesian Computation). Let's try first with a regular model,

+++

#### SMC with a Likelihood Function

```{code-cell} ipython3
sampler = "SMC with Likelihood"
draws = 2000
with model:
    trace_SMC_like = pm.sample_smc(draws)
trace = trace_SMC_like
az.summary(trace)
```

```{code-cell} ipython3
trace.sample_stats._t_sampling
```

```{code-cell} ipython3
az.plot_trace(trace, kind="rank_bars")
plt.suptitle(f"Trace Plot {sampler}");
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(12, 4))
plot_inference(ax, trace, title=f"Data and Inference Model Runs\n{sampler} Sampler");
```

**Notes:**  
At this number of samples and tuning scheme, the SMC algorithm results in wider uncertainty bounds compared with the other samplers.  

+++

#### SMC Using `pm.Simulator` Epsilon=1

+++

As outlined in the SMC tutorial on PyMC.io, the SMC sampler can be used for Aproximate Bayesian Computation, i.e. we can use a `pm.Simulator` instead of a explicit likelihood.  Here is a rewrite of the PyMC - odeint model for SMC-ABC.

The simulator function needs to have the correct signature (e.g., accept an rng argument first).  

```{code-cell} ipython3
# simulator function based on the signature rng, parameters, size.
def simulator_forward_model(rng, alpha, beta, gamma, delta, xt0, yt0, sigma, size=None):
    theta = alpha, beta, gamma, delta, xt0, yt0
    mu = odeint(func=rhs, y0=theta[-2:], t=data.year, args=(theta,))
    return rng.normal(mu, sigma)
```

Here is the model with the simulator function. Instead of a explicit likelihood function, the simulator uses distance metric (defaults to `gaussian`) between the simulated and observed values. When using a simulator we also need to specify epsilon, that is a tolerance value for the discrepancy between simulated and observed values. If epsilon is too low, SMC will not be able to move away from the initial values or a few values. We can easily see this with `az.plot_trace`. If epsilon is too high, the posterior will virtually be the prior. So

```{code-cell} ipython3
with pm.Model() as model:
    # Specify prior distributions for model parameters
    alpha = pm.TruncatedNormal("alpha", mu=theta[0], sigma=0.1, lower=0, initval=theta[0])
    beta = pm.TruncatedNormal("beta", mu=theta[1], sigma=0.01, lower=0, initval=theta[1])
    gamma = pm.TruncatedNormal("gamma", mu=theta[2], sigma=0.1, lower=0, initval=theta[2])
    delta = pm.TruncatedNormal("delta", mu=theta[3], sigma=0.01, lower=0, initval=theta[3])
    xt0 = pm.TruncatedNormal("xto", mu=theta[4], sigma=1, lower=0, initval=theta[4])
    yt0 = pm.TruncatedNormal("yto", mu=theta[5], sigma=1, lower=0, initval=theta[5])
    sigma = pm.HalfNormal("sigma", 10)

    # ode_solution
    pm.Simulator(
        "Y_obs",
        simulator_forward_model,
        params=(alpha, beta, gamma, delta, xt0, yt0, sigma),
        epsilon=1,
        observed=data[["hare", "lynx"]].values,
    )
```

Inference.  Note the `progressbar` was throwing an error so it is turned off.  

```{code-cell} ipython3
sampler = "SMC_epsilon=1"
draws = 2000
with model:
    trace_SMC_e1 = pm.sample_smc(draws=draws, progressbar=False)
trace = trace_SMC_e1
az.summary(trace)
```

```{code-cell} ipython3
az.plot_trace(trace, kind="rank_bars")
plt.suptitle(f"Trace Plot {sampler}");
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(12, 4))
plot_inference(ax, trace, title=f"Data and Inference Model Runs\n{sampler} Sampler");
```

**Notes:**  
We can see that if epsilon is too low `plot_trace` will clearly show it.

+++

#### SMC with Epsilon = 10

```{code-cell} ipython3
with pm.Model() as model:
    # Specify prior distributions for model parameters
    alpha = pm.TruncatedNormal("alpha", mu=theta[0], sigma=0.1, lower=0, initval=theta[0])
    beta = pm.TruncatedNormal("beta", mu=theta[1], sigma=0.01, lower=0, initval=theta[1])
    gamma = pm.TruncatedNormal("gamma", mu=theta[2], sigma=0.1, lower=0, initval=theta[2])
    delta = pm.TruncatedNormal("delta", mu=theta[3], sigma=0.01, lower=0, initval=theta[3])
    xt0 = pm.TruncatedNormal("xto", mu=theta[4], sigma=1, lower=0, initval=theta[4])
    yt0 = pm.TruncatedNormal("yto", mu=theta[5], sigma=1, lower=0, initval=theta[5])
    sigma = pm.HalfNormal("sigma", 10)

    # ode_solution
    pm.Simulator(
        "Y_obs",
        simulator_forward_model,
        params=(alpha, beta, gamma, delta, xt0, yt0, sigma),
        epsilon=10,
        observed=data[["hare", "lynx"]].values,
    )
```

```{code-cell} ipython3
sampler = "SMC epsilon=10"
draws = 2000
with model:
    trace_SMC_e10 = pm.sample_smc(draws=draws)
trace = trace_SMC_e10
az.summary(trace)
```

```{code-cell} ipython3
az.plot_trace(trace, kind="rank_bars")
plt.suptitle(f"Trace Plot {sampler}");
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(12, 4))
plot_inference(ax, trace, title=f"Data and Inference Model Runs\n{sampler} Sampler");
```

**Notes:**  
Now that we set a larger value for epsilon we can see that the SMC sampler (plus simulator) provides good results. Choosing a value for epsilon will always involve some trial and error. So, what to do in practice? As epsilon is the scale of the distance function. If you don't have any idea of how much error do you expected to get between simulated and observed values then a rule of thumb for picking an initial guess for epsilon is to use a number smaller than the standard deviation of the observed data, how much smaller maybe one order of magnitude or so.

+++

### Posterior Correlations
As an aside, it is worth pointing out that the posterior parameter space is a difficult geometry for sampling.  

```{code-cell} ipython3
az.plot_pair(trace_DEM, figsize=(8, 6), scatter_kwargs=dict(alpha=0.01), marginals=True)
plt.suptitle("Pair Plot Showing Posterior Correlations", size=18);
```

The major observation here is that the posterior shape is pretty difficult for a sampler to handle, with positive correlations, negative correlations, crecent-shapes, and large variations in scale.  This contributes to the slow sampling (in addition to the computational overhead in solving the ODE thousands of times).  This is also fun to look at for understanding how the model parameters impact each other.       

+++

## Bayesian Inference with Gradients

+++

NUTS, the PyMC default sampler can only be used if gradients are supplied to the sampler.  In this section, we will solve the system of ODEs within PyMC in two different ways that supply the sampler with gradients.  The first is the built-in `pymc.ode.DifferentialEquation` solver, and the second is to forward simulate using `pytensor.scan`, which allows looping.  Note that there may be other better and faster ways to perform Bayesian inference with ODEs using gradients, such as the [sunode](https://sunode.readthedocs.io/en/latest/index.html) project, and [diffrax](https://www.pymc-labs.io/blog-posts/jax-functions-in-pymc-3-quick-examples/), which relies on JAX.

+++

### PyMC ODE Module

+++

`Pymc.ode` uses `scipy.odeint` under the hood to estimate a solution and then estimate the gradient through finite differences. 

The `pymc.ode` API is similar to `scipy.odeint`.  The right-hand-side equations are put in a function and written as if `y` and `p` are vectors, as follows.  (Even when your model has one state and/or one parameter, you should explicitly write `y[0]` and/or `p[0]`.)

```{code-cell} ipython3
def rhs_pymcode(y, t, p):
    dX_dt = p[0] * y[0] - p[1] * y[0] * y[1]
    dY_dt = -p[2] * y[1] + p[3] * y[0] * y[1]
    return [dX_dt, dY_dt]
```

`DifferentialEquation` takes as arguments:

* `func`: A function specifying the differential equation (i.e. $f(\mathbf{y},t,\mathbf{p})$),
* `times`: An array of times at which data was observed,
* `n_states`: The dimension of $f(\mathbf{y},t,\mathbf{p})$ (number of output parameters),
* `n_theta`: The dimension of $\mathbf{p}$ (number of input parameters),
* `t0`: Optional time to which the initial condition belongs,  

as follows:

```{code-cell} ipython3
ode_model = DifferentialEquation(
    func=rhs_pymcode, times=data.year.values, n_states=2, n_theta=4, t0=data.year.values[0]
)
```

Once the ODE is specified, we can use it in our PyMC model.

+++

#### Inference with NUTS
`pymc.ode` is quite slow, so for demonstration purposes, we will only draw a few samples.  

```{code-cell} ipython3
with pm.Model() as model:
    # Priors
    alpha = pm.TruncatedNormal("alpha", mu=theta[0], sigma=0.1, lower=0, initval=theta[0])
    beta = pm.TruncatedNormal("beta", mu=theta[1], sigma=0.01, lower=0, initval=theta[1])
    gamma = pm.TruncatedNormal("gamma", mu=theta[2], sigma=0.1, lower=0, initval=theta[2])
    delta = pm.TruncatedNormal("delta", mu=theta[3], sigma=0.01, lower=0, initval=theta[3])
    xt0 = pm.TruncatedNormal("xto", mu=theta[4], sigma=1, lower=0, initval=theta[4])
    yt0 = pm.TruncatedNormal("yto", mu=theta[5], sigma=1, lower=0, initval=theta[5])
    sigma = pm.HalfNormal("sigma", 10)

    # ode_solution
    ode_solution = ode_model(y0=[xt0, yt0], theta=[alpha, beta, gamma, delta])

    # Likelihood
    pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=data[["hare", "lynx"]].values)
```

```{code-cell} ipython3
sampler = "NUTS PyMC ODE"
tune = draws = 15
with model:
    trace_pymc_ode = pm.sample(tune=tune, draws=draws)
```

```{code-cell} ipython3
trace = trace_pymc_ode
az.summary(trace)
```

```{code-cell} ipython3
az.plot_trace(trace, kind="rank_bars")
plt.suptitle(f"Trace Plot {sampler}");
```

```{code-cell} ipython3
_, ax = plt.subplots(figsize=(12, 4))
plot_inference(ax, trace, title=f"Data and Inference Model Runs\n{sampler} Sampler");
```

**Notes:**  
NUTS is starting to find to the correct posterior, but would need a whole lot more time to make a good inference.   

+++

### Simulate with Pytensor Scan

+++

Finally, we can write the system of ODEs as a forward simulation solver within PyMC.  The way to write for-loops in PyMC is with `pytensor.scan.`  Gradients are then supplied to the sampler via autodifferentiation.    

First, we should test that the time steps are sufficiently small to get a reasonable estimate.  

+++

#### Check Time Steps

+++

Create a function that accepts different numbers of time steps for testing.  The function also demonstrates how `pytensor.scan` is used.  

```{code-cell} ipython3
# Lotka-Volterra forward simulation model using scan
def lv_scan_simulation_model(theta, steps_year=100, years=21):
    # variables to control time steps
    n_steps = years * steps_year
    dt = 1 / steps_year

    # PyMC model
    with pm.Model() as model:
        # Priors (these are static for testing)
        alpha = theta[0]
        beta = theta[1]
        gamma = theta[2]
        delta = theta[3]
        xt0 = theta[4]
        yt0 = theta[5]

        # Lotka-Volterra calculation function
        ## Similar to the right-hand-side functions used earlier
        ## but with dt applied to the equations
        def ode_update_function(x, y, alpha, beta, gamma, delta):
            x_new = x + (alpha * x - beta * x * y) * dt
            y_new = y + (-gamma * y + delta * x * y) * dt
            return x_new, y_new

        # Pytensor scan looping function
        ## The function argument names are not intuitive in this context!
        result, updates = pytensor.scan(
            fn=ode_update_function,  # function
            outputs_info=[xt0, yt0],  # initial conditions
            non_sequences=[alpha, beta, gamma, delta],  # parameters
            n_steps=n_steps,  # number of loops
        )

        # Put the results together and track the result
        pm.Deterministic("result", pm.math.stack([result[0], result[1]], axis=1))

    return model
```

Run the simulation for various time steps and plot the results.   

```{code-cell} ipython3
_, ax = plt.subplots(figsize=(12, 4))

steps_years = [12, 100, 1000, 10000]
for steps_year in steps_years:
    time = np.arange(1900, 1921, 1 / steps_year)
    model = lv_scan_simulation_model(theta, steps_year=steps_year)
    with model:
        prior = pm.sample_prior_predictive(1)
    ax.plot(time, prior.prior.result[0][0].values, label=str(steps_year) + " steps/year")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax.set_title("Lotka-Volterra Forward Simulation Model with different step sizes");
```

Notice how the lower resolution simulations are less accurate over time.  Based on this check, 100 time steps per year is sufficiently accurate.  12 steps per year has too much "numerical diffusion" over 20 years of simulation.    

+++

#### Inference Using NUTs

+++

Now that we are OK with 100 time steps per year, we write the model with indexing to align the data with the results.  

```{code-cell} ipython3
def lv_scan_inference_model(theta, steps_year=100, years=21):
    # variables to control time steps
    n_steps = years * steps_year
    dt = 1 / steps_year

    # variables to control indexing to get annual values
    segment = [True] + [False] * (steps_year - 1)
    boolist_idxs = []
    for _ in range(years):
        boolist_idxs += segment

    # PyMC model
    with pm.Model() as model:
        # Priors
        alpha = pm.TruncatedNormal("alpha", mu=theta[0], sigma=0.1, lower=0, initval=theta[0])
        beta = pm.TruncatedNormal("beta", mu=theta[1], sigma=0.01, lower=0, initval=theta[1])
        gamma = pm.TruncatedNormal("gamma", mu=theta[2], sigma=0.1, lower=0, initval=theta[2])
        delta = pm.TruncatedNormal("delta", mu=theta[3], sigma=0.01, lower=0, initval=theta[3])
        xt0 = pm.TruncatedNormal("xto", mu=theta[4], sigma=1, lower=0, initval=theta[4])
        yt0 = pm.TruncatedNormal("yto", mu=theta[5], sigma=1, lower=0, initval=theta[5])
        sigma = pm.HalfNormal("sigma", 10)

        # Lotka-Volterra calculation function
        def ode_update_function(x, y, alpha, beta, gamma, delta):
            x_new = x + (alpha * x - beta * x * y) * dt
            y_new = y + (-gamma * y + delta * x * y) * dt
            return x_new, y_new

        # Pytensor scan is a looping function
        result, updates = pytensor.scan(
            fn=ode_update_function,  # function
            outputs_info=[xt0, yt0],  # initial conditions
            non_sequences=[alpha, beta, gamma, delta],  # parameters
            n_steps=n_steps,
        )  # number of loops

        # Put the results together
        final_result = pm.math.stack([result[0], result[1]], axis=1)
        # Filter the results down to annual values
        annual_value = final_result[np.array(boolist_idxs), :]

        # Likelihood function
        pm.Normal("Y_obs", mu=annual_value, sigma=sigma, observed=data[["hare", "lynx"]].values)
    return model
```

This is also quite slow, so we will just pull a few samples for demonstration purposes.  

```{code-cell} ipython3
steps_year = 100
model = lv_scan_inference_model(theta, steps_year=steps_year)
sampler = "NUTS Pytensor Scan"
tune = draws = 50
with model:
    trace_scan = pm.sample(tune=tune, draws=draws)
```

```{code-cell} ipython3
trace = trace_scan
az.summary(trace)
```

```{code-cell} ipython3
az.plot_trace(trace, kind="rank_bars")
plt.suptitle(f"Trace Plot {sampler}");
```

```{code-cell} ipython3
time = np.arange(1900, 1921, 0.01)
odeint(func=rhs, y0=theta[-2:], t=time, args=(theta,)).shape
```

```{code-cell} ipython3
_, ax = plt.subplots(figsize=(12, 4))
plot_inference(ax, trace, title=f"Data and Inference Model Runs\n{sampler} Sampler");
```

**Notes:**  
The sampler is faster than the `pymc.ode` implementation, but still slower than scipy `odeint` combined with gradient-free inference methods. 

+++

## Summary

+++

Let's compare inference results among these different methods.  Recall that, in order to run this notebook in a reasonable amount of time, we have an insufficient number of samples for many inference methods.  For a fair comparison, we would need to bump up the number of samples and run the notebook for longer.  Regardless, let's take a look.     

```{code-cell} ipython3
# Make lists with variable for looping
var_names = [str(s).split("_")[0] for s in list(model.values_to_rvs.keys())[:-1]]
# Make lists with model results and model names for plotting
inference_results = [
    trace_slice,
    trace_DEMZ,
    trace_DEM,
    trace_M,
    trace_SMC_like,
    trace_SMC_e1,
    trace_SMC_e10,
    trace_pymc_ode,
    trace_scan,
]
model_names = [
    "Slice Sampler",
    "DEMetropolisZ",
    "DEMetropolis",
    "Metropolis",
    "SMC with Likelihood",
    "SMC e=1",
    "SMC e=10",
    "PyMC ODE NUTs",
    "Pytensor Scan NUTs",
]

# Loop through variable names
for var_name in var_names:
    axes = az.plot_forest(
        inference_results,
        model_names=model_names,
        var_names=var_name,
        kind="forestplot",
        legend=False,
        combined=True,
        figsize=(7, 3),
    )
    axes[0].set_title(f"Marginal Probability: {var_name}")
    # Clean up ytick labels
    ylabels = axes[0].get_yticklabels()
    new_ylabels = []
    for label in ylabels:
        txt = label.get_text()
        txt = txt.replace(": " + var_name, "")
        label.set_text(txt)
        new_ylabels.append(label)
    axes[0].set_yticklabels(new_ylabels)

    plt.show();
```

**Notes:**  
If we ran the samplers for long enough to get good inferences, we would expect them to converge on the same posterior probability distributions. This is not necessarily true for Aproximate Bayssian Computation, unless we first ensure that the approximation too the likelihood is good enough. For instance SMCe=1 is providing a wrong result, we have been warning that this was most likely the case when we use `plot_trace` as a diagnostic. For SMC e=10, we see that posterior mean agrees with the other samplers, but the posterior is wider. This is expected with ABC methods. A smaller value of epsilon, maybe 5, should provide a posterior closer to the true one.

+++

### Key Conclusions
We performed Bayesian inference on a system of ODEs in 4 main ways: 
* Scipy `odeint` wrapped in a Pytensor `op` and sampled with non-gradient-based samplers (comparing 5 different samplers).  
* Scipy `odeint` wrapped in a `pm.Simulator` function and sampled with a non-likelihood-based sequential Monte Carlo (SMC) sampler.  
* PyMC `ode.DifferentialEquation` sampled with NUTs.  
* Forward simulation using `pytensor.scan` and sampled with NUTs.  

The "winner" for this problem was the Scipy `odeint` solver with a differential evolution (DE) Metropolis sampler and SMC (for a model with a Likelihood) provide good results with SMC being somewhat slower (but also better diagnostics). The improved efficiency of the NUTS sampler did not make up for the inefficiency in using the slow ODE solvers with gradients.  Both DEMetropolis and SMC enable the simplest workflow for a scientist with a working numeric model and the desire to perform Bayesian inference. Just wrapping the numeric model in a Pytensor op and plugging it into a PyMC model can get you a long way!

+++

## Authors
Organized and rewritten by [Greg Brunkhorst](https://github.com/gbrunkhorst)  from multiple legacy PyMC.io example notebooks by Sanmitra Ghosh, Demetri Pananos, and the PyMC Team ({ref}`ABC_introduction`).

Osvaldo Martin added some clarification about SMC-ABC and  minor fixes in Mar, 2023

+++

## References

:::{bibliography}
:filter: docname in docnames
:::

+++

## Watermark

```{code-cell} ipython3
%watermark -n -u -v -iv -w
```

:::{include} ../page_footer.md
:::
