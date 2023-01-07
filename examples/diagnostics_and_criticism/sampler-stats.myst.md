---
jupytext:
  notebook_metadata_filter: substitutions
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(sampler_stats)=
# Sampler Statistics

:::{post} May 31, 2022
:tags: diagnostics
:category: beginner
:author: Meenal Jhajharia, Christian Luhmann
:::

```{code-cell} ipython3
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

%matplotlib inline

print(f"Running on PyMC v{pm.__version__}")
```

```{code-cell} ipython3
az.style.use("arviz-darkgrid")
plt.rcParams["figure.constrained_layout.use"] = False
```

When checking for convergence or when debugging a badly behaving sampler, it is often helpful to take a closer look at what the sampler is doing. For this purpose some samplers export statistics for each generated sample.

As a minimal example we sample from a standard normal distribution:

```{code-cell} ipython3
model = pm.Model()
with model:
    mu1 = pm.Normal("mu1", mu=0, sigma=1, shape=10)
```

```{code-cell} ipython3
with model:
    step = pm.NUTS()
    idata = pm.sample(2000, tune=1000, init=None, step=step, chains=4)
```

- `Note`: NUTS provides the following statistics (these are internal statistics that the sampler uses, you don't need to do anything with them when using PyMC, to learn more about them, {class}`pymc.NUTS`.

```{code-cell} ipython3
idata.sample_stats
```

The sample statistics variables are defined as follows:

- `process_time_diff`: The time it took to draw the sample, as defined by the python standard library time.process_time. This counts all the CPU time, including worker processes in BLAS and OpenMP.

- `step_size`: The current integration step size.

- `diverging`: (boolean) Indicates the presence of leapfrog transitions with large energy deviation from starting and subsequent termination of the trajectory. “large” is defined as `max_energy_error` going over a threshold.

- `lp`: The joint log posterior density for the model (up to an additive constant).

- `energy`: The value of the Hamiltonian energy for the accepted proposal (up to an additive constant).

- `energy_error`: The difference in the Hamiltonian energy between the initial point and the accepted proposal.

- `perf_counter_diff`: The time it took to draw the sample, as defined by the python standard library time.perf_counter (wall time).

- `perf_counter_start`: The value of time.perf_counter at the beginning of the computation of the draw.

- `n_steps`: The number of leapfrog steps computed. It is related to `tree_depth` with `n_steps <= 2^tree_dept`.

- `max_energy_error`: The maximum absolute difference in Hamiltonian energy between the initial point and all possible samples in the proposed tree.

- `acceptance_rate`: The average acceptance probabilities of all possible samples in the proposed tree.

- `step_size_bar`: The current best known step-size. After the tuning samples, the step size is set to this value. This should converge during tuning.

- `tree_depth`: The number of tree doublings in the balanced binary tree.

+++

Some points to `Note`:
- Some of the sample statistics used by NUTS are renamed when converting to `InferenceData` to follow {ref}`ArviZ's naming convention <arviz:schema>`, while some are specific to PyMC3 and keep their internal PyMC3 name in the resulting InferenceData object.
- `InferenceData` also stores additional info like the date, versions used, sampling time and tuning steps as attributes.

```{code-cell} ipython3
idata.sample_stats["tree_depth"].plot(col="chain", ls="none", marker=".", alpha=0.3);
```

```{code-cell} ipython3
az.plot_posterior(
    idata, group="sample_stats", var_names="acceptance_rate", hdi_prob="hide", kind="hist"
);
```

We check if there are any divergences, if yes, how many?

```{code-cell} ipython3
idata.sample_stats["diverging"].sum()
```

In this case no divergences are found. If there are any, check [this notebook](https://github.com/pymc-devs/pymc-examples/blob/main/examples/diagnostics_and_criticism/Diagnosing_biased_Inference_with_Divergences.ipynb) for  information on handling divergences.

+++

It is often useful to compare the overall distribution of the
energy levels with the change of energy between successive samples.
Ideally, they should be very similar:

```{code-cell} ipython3
az.plot_energy(idata, figsize=(6, 4));
```

If the overall distribution of energy levels has longer tails, the efficiency of the sampler will deteriorate quickly.

+++

## Multiple samplers

If multiple samplers are used for the same model (e.g. for continuous and discrete variables), the exported values are merged or stacked along a new axis.

```{code-cell} ipython3
coords = {"step": ["BinaryMetropolis", "Metropolis"], "obs": ["mu1"]}
dims = {"accept": ["step"]}

with pm.Model(coords=coords) as model:
    mu1 = pm.Bernoulli("mu1", p=0.8)
    mu2 = pm.Normal("mu2", mu=0, sigma=1, dims="obs")
```

```{code-cell} ipython3
with model:
    step1 = pm.BinaryMetropolis([mu1])
    step2 = pm.Metropolis([mu2])
    idata = pm.sample(
        10000,
        init=None,
        step=[step1, step2],
        chains=4,
        tune=1000,
        idata_kwargs={"dims": dims, "coords": coords},
    )
```

```{code-cell} ipython3
list(idata.sample_stats.data_vars)
```

Both samplers export `accept`, so we get one acceptance probability for each sampler:

```{code-cell} ipython3
az.plot_posterior(
    idata,
    group="sample_stats",
    var_names="accept",
    hdi_prob="hide",
    kind="hist",
);
```

We notice that `accept` sometimes takes really high values (jumps from regions of low probability to regions of much higher probability).

```{code-cell} ipython3
# Range of accept values
idata.sample_stats["accept"].max("draw") - idata.sample_stats["accept"].min("draw")
```

```{code-cell} ipython3
# We can try plotting the density and view the high density intervals to understand the variable better
az.plot_density(
    idata,
    group="sample_stats",
    var_names="accept",
    point_estimate="mean",
);
```

## Authors
* Updated by Meenal Jhajharia in April 2021 ([pymc-examples#95](https://github.com/pymc-devs/pymc-examples/pull/95))
* Updated to v4 by Christian Luhmann in May 2022 ([pymc-examples#338](https://github.com/pymc-devs/pymc-examples/pull/338))

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w
```

:::{include} ../page_footer.md
:::
