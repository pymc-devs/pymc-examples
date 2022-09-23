---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: pymc_env
  language: python
  name: pymc_env
---

(difference_in_differences)=
# Difference in differences

:::{post} Sept, 2022
:tags: counterfactuals, causal inference, time series, regression, posterior predictive, difference in differences, quasi experiments
:category: beginner
:author: Benjamin T. Vincent
:::

+++

## XXX
Some introduction here

```{code-cell} ipython3
import aesara.tensor as at
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import xarray as xr

from scipy.stats import norm
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
```

## Difference in Differences

### Causal DAG

TODO

+++

![](DAG_difference_in_differences.png)

Without `Group` then `Time` is a confounder of the `Treatment` $\rightarrow$ `Outcome` relationship

+++

### Define the difference in differences model

$$
\mu = \beta_{c} 
      + (\beta_{\Delta} \cdot \mathrm{group})
      + (t \cdot \mathrm{trend})
      + (\Delta \cdot \mathrm{post} \cdot \mathrm{group})
$$

where:
* $g$ is a dummary variable for control ($g=0$) or treatment ($g=1$) group
* $\beta_c$ is the intercept for the control group
* $\beta_{\Delta}$ is a deflection of the treatment group intercept from the control group intercept
* $\mathrm{trend}$ is the slope, and a core assumption of the model is that the slopes are identical for both groups
* $\mathrm{post}$ is an indicator variable for pre or post-treatment
* $\Delta$ is the causal impact of the treatment

```{code-cell} ipython3
def outcome(t, control_intercept, treat_intercept_delta, Δ, group):
    mu = (
        control_intercept
        + (treat_intercept_delta * group)
        + (t * trend)
        + (Δ * (t > intervention_time) * group)
    )
    return mu
```

### Visualise the difference in differences model

```{code-cell} ipython3
# true parameters
control_intercept = 1
treat_intercept_delta = 0.25
trend = 1
Δ = 1
intervention_time = 0.5
```

```{code-cell} ipython3
:tags: [hide-input]

t = np.linspace(-0.5, 1.5, 1000)

# with plt.xkcd():
fig, ax = plt.subplots()
ax.plot(
    t,
    outcome(t, control_intercept, treat_intercept_delta, Δ=0, group=1),
    color="blue",
    label="counterfactual",
    ls=":",
)
ax.plot(
    t,
    outcome(t, control_intercept, treat_intercept_delta, Δ, group=1),
    color="blue",
    label="treatment group",
)
ax.plot(
    t,
    outcome(t, control_intercept, treat_intercept_delta, Δ, group=0),
    color="C1",
    label="control group",
)
ax.axvline(x=intervention_time, ls="-", color="r", label="treatment time", lw=3)
t = np.array([0, 1])
ax.plot(t, outcome(t, control_intercept, treat_intercept_delta, Δ, group=1), "o", color="blue")
ax.plot(t, outcome(t, control_intercept, treat_intercept_delta, Δ=0, group=0), "o", color="C1")
ax.set(
    xlabel="time",
    ylabel="metric",
    xticks=[0, 1],
    xticklabels=["pre", "post"],
    title="Difference in Differences",
)
ax.legend();
```

This is a good point to mention some of the other aspects of the difference in differences approach:
* You want the control and treatment groups to be as similar to each other as possible (i.e. $\beta_\Delta \rightarrow 0$). A large difference in the pre-treatment measures suggests that the groups are _not_ that similar. This opens up the possibility that there are unobserved confounders that could provide an alternative explanation to any treatment effect detected.
* The _parallel trends assumption_ is a major one! if we draw lines through the pre and post treatment outcomes for each group individually then we get 2 slopes. If the slopes are very different, then the _only_ way we can attribute that to a treatment effect is to assume that the slopes (change in groups over time) are identical. Ideally we would be able to measure at multiple time points, but this then turns the approach into more of an interrupted time series design. Without that, you have to carefully examine and justify if the parallel trends assumption is reasonable to make in your case.

+++

## Generate a synthetic dataset

```{code-cell} ipython3

```

## Bayesian difference in differences

+++

### PyMC model

```{code-cell} ipython3

```

### Inference

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

## References

:::{bibliography}
:filter: docname in docnames
:::

+++

## Authors
- Authored by [Benjamin T. Vincent](https://github.com/drbenvincent) in Sept 2022.

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p aesara,aeppl,xarray
```

:::{include} ../page_footer.md
:::
