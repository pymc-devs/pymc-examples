---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# MLDA sampler: Introduction and resources

This notebook contains an introduction to the Multi-Level Delayed Acceptance MCMC algorithm (MLDA) proposed in [1]. It explains the main idea behind the method, gives an overview of the problems it is good for and points to specific notebooks with examples of how to use it within PyMC. 

[1] Dodwell, Tim & Ketelsen, Chris & Scheichl, Robert & Teckentrup, Aretha. (2019). Multilevel Markov Chain Monte Carlo. SIAM Review. 61. 509-545. https://doi.org/10.1137/19M126966X

+++

### The MLDA sampler

The MLDA sampler is designed to deal with computationally intensive problems where we have access not only to the desired (fine) posterior distribution but also to a set of approximate (coarse) posteriors of decreasing accuracy and decreasing computational cost (we need at least one of those). 

Its main idea is to run multiple chains, where each chain samples from a different version of the posterior (the lower the level, the coarser the posterior). Each chain generates a number of samples and the last of them is passed as a proposal to the chain one level up. The latter accepts or rejects the sample and then gives back control to the first chain. This strategy is applied recursively so that each chain uses the chain below as a source of proposed samples. 

MLDA improves the effective sample size of the finest chain compared to standard samplers (e.g. Metropolis) and this allows us to reduce the number of expensive fine-chain likelihood evaluations while still exploring the posterior adequately. Note that the bottom level sampler is a standard MCMC sampler like Metropolis or DEMetropolisZ.

+++

### Problems it is good for

In many real-world problems, we use models to represent spatially or temporally varying quantities. We often have the ability to modify the spatial and/or temporal granularity of the models. For example, when representing a high-resolution image, we can use a coarse 64x64 grid but we can also use a much finer 512x512 grid, which is more accurate and more computationally demanding when working with it. In those cases it is often possible to apply multilevel modeling to infer unknown quantities in the model. In multilevel modeling, a hierarchy of models of increasing accuracy/resolution and increasing computational cost are used together to perform inference more efficiently than doing inference using the finest model only. 

Example applications include inverse problems for physical, natural or other systems, e.g. subsurface fluid transportation, predator-prey models in ecology, impedance imaging, ultrasound imaging, emission tomography, flow field of ocean circulation. 

In many of those inverse problems, evaluating the Bayesian likelihood requires solving a partial differential equation (PDE) numerically. Doing this on a fine-resolution model can be orders of magnitude slower than doing it on a coarse-resolution model. This is the ideal scenario for multilevel modeling and MLDA as MLDA allows us to get away with only calculating a fraction of the expensive fine-resolution likelihoods in exchange for many cheap coarse-resolution likelihoods.

+++

### PyMC implementation

MLDA is one of the MCMC inference methods available in PyMC. You can instantiate an MLDA sampler using the `pm.MLDA(coarse_models=...)`, where you need to pass at least one coarse model within a list.

The PyMC implementation of MLDA supports any number of levels, tuning parameterization for the bottom-level sampler, separate subsampling rates for each level, choice between blocked and compound sampling for the bottom-level sampler, two types of bottom-level samplers (Metropolis, DEMetropolisZ), adaptive error correction and variance reduction.

For more details about the MLDA sampler and the way it should be used and parameterised, the user can refer to the docstrings in the code and to the other example notebooks (links below) which deal with more complex problem settings and more advanced MLDA features.

Please note that the MLDA sampler is new in PyMC. The user should be extra critical about the results and report any problems as issues in the PyMC's github repository.

+++

### Notebooks with example code


[Simple linear regression](./MLDA_simple_linear_regression.ipynb): This notebook demonstrates the workflow for using MLDA within PyMC. It employes a very simple toy model.

[Gravity surveying](./MLDA_gravity_surveying.ipynb): In this notebook, we use MLDA to solve a 2-dimensional gravity surveying inverse problem. Evaluating the likelihood requires solving a PDE, which we do using [scipy](https://www.scipy.org/). We also compare the performance of MLDA with other PyMC samplers (Metropolis, DEMetropolisZ).

[Variance reduction 1](./MLDA_variance_reduction_linear_regression.ipynb) and [Variance reduction 2](https://github.com/alan-turing-institute/pymc/blob/mlda_all_notebooks/docs/source/notebooks/MLDA_variance_reduction_groundwater.ipynb) (external link): Those two notebooks demonstrate the variance reduction feature in a linear regression model and a groundwater flow model. This feature allows the user to define a quantity of interest that they need to estimate using the MCMC samples. It then collects those quantities of interest, as well as differences of these quantities between levels, during MLDA sampling. The collected quentities can then be used to produce an estimate which has lower variance than a standard estimate that uses samples from the fine chain only. The first notebook does not have external dependencies, while the second one requires FEniCS. Note that the second notebook is outside the core PyMC repository because FEniCS is not a PyMC dependency.

[Adaptive error model](https://github.com/alan-turing-institute/pymc/blob/mlda_all_notebooks/docs/source/notebooks/MLDA_adaptive_error_model.ipynb) (external link): In this notebook we use MLDA to tackle another inverse problem; groundwarer flow modeling. The aim is to infer the posterior distribution of model parameters (hydraulic conductivity) given data (measurements of hydraulic head). In this example we make use of PyTensor Ops in order to define a "black box" likelihood, i.e. a likelihood that uses external code. Specifically, our likelihood uses the [FEniCS](https://fenicsproject.org/) library to solve a PDE. This is a common scenario, as PDEs of this type are slow to solve with scipy or other standard libraries. Note that this notebook is outside the core PyMC repository because FEniCS is not a PyMC dependency. We employ the adaptive error model (AEM) feature and compare the performance of basic MLDA with AEM-enhanced MLDA. The idea of Adaptive Error Model (AEM) is to estimate the mean and variance of the forward-model error between adjacent levels, i.e. estimate the bias of the coarse forward model compared to the fine forward model, and use those estimates to correct the coarse model. Using the technique should improve ESS/sec on the fine level.

[Benchmarks and tuning](https://github.com/alan-turing-institute/pymc/blob/mlda_all_notebooks/docs/source/notebooks/MLDA_benchmarks_tuning.ipynb) (external link): In this notebook we benchmark MLDA against other samplers using different parameterizations of the groundwater flow model. We also give some advice on tuning MLDA. Note that this notebook is outside the core PyMC repository because FEniCS is not a PyMC dependency.

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w
```
