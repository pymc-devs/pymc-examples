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

(mediation_analysis)=
# Bayesian mediation analysis

:::{post} February, 2022
:tags: mediation, path analysis, regression 
:category: beginner
:author: Benjamin T. Vincent
:::

This notebook covers Bayesian [mediation analysis](https://en.wikipedia.org/wiki/Mediation_(statistics) ). This is useful when we want to explore possible mediating pathways between a predictor and an outcome variable.

It is important to note that the approach to mediation analysis has evolved over time. This notebook was heavily influenced by the approach of {cite:t}`hayes2017introduction` as set out in his textbook "Introduction to Mediation, Moderation and Conditional Process Analysis."

Readers should be aware that mediation analysis is commonly confused with moderation analysis for which we have a separate example ({ref}`moderation_analysis`).

```{code-cell} ipython3
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import seaborn as sns
import xarray as xr

from arviz_base.labels import MapLabeller
from pandas import DataFrame
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'
plt.rcParams.update({"font.size": 14})
seed = sum(map(ord, "mediation_analysis"))
rng = np.random.default_rng(seed)
```

## The mediation model

The simple mediation model is very simple where $m$ is a linear function of $x$, and $y$ is a linear function of $x$ and $m$:

$$
m_i \sim \mathrm{Normal}(i_M + a \cdot x_i, \sigma_M)
$$

$$
y_i \sim \mathrm{Normal}(i_Y + c' \cdot x_i + b \cdot m_i, \sigma_Y)
$$

where $i$ indexes each observation (row in the dataset), and $i_M$ and $i_Y$ are intercept parameters. Note that $x_i$, $m_i$, and $y_i$ are observed data.

![](mediation.png)

Using definitions from {cite:t}`hayes2017introduction`, we can define a few effects of interest:
- **Direct effect:** is given by $c'$. Two cases that differ by one unit on $x$ but are equal on $m$ are estimated to differ by $c'$ units on $y$.
- **Indirect effect:** is given by $a \cdot b$. Two cases which differ by one unit of $x$ are estimated to differ by $a \cdot b$ units on $y$ as a result of the effect of $x \rightarrow m$ and $m \rightarrow y$.
- **Total effect:** is $c = c' + a \cdot b$ which is simply the sum of the direct and indirect effects. This could be understood as: two cases that differ by one unit on $x$ are estimated to differ by $a \cdot b$ units on $y$ due to the indirect pathway $x \rightarrow m \rightarrow y$, and by $c'$ units due to the direct pathway $x \rightarrow y$. The total effect could also be estimated by evaluating the alternative model $y_i \sim \mathrm{Normal}(i_{Y*} + c \cdot x_i, \sigma_{Y*})$.

+++

## Define the PyMC model

We write the mediation model as a function that takes $x$, and optionally the observed $m$ and $y$, and returns a PyMC model. The same function generates the simulated data and fits the model.

```{code-cell} ipython3
def mediation_model(x, m=None, y=None):
    with pm.Model(coords={"obs_id": np.arange(len(x))}) as model:
        x_data = pm.Data("x", x, dims="obs_id")

        # intercept priors
        im = pm.Normal("im", mu=0, sigma=10)
        iy = pm.Normal("iy", mu=0, sigma=10)

        # slope priors
        a = pm.Normal("a", mu=0, sigma=10)
        b = pm.Normal("b", mu=0, sigma=10)
        cprime = pm.Normal("cprime", mu=0, sigma=10)

        # noise priors
        σm = pm.HalfCauchy("σm", 1)
        σy = pm.HalfCauchy("σy", 1)

        # likelihood
        mediator = pm.Normal("m", mu=im + a * x_data, sigma=σm, observed=m, dims="obs_id")
        pm.Normal("y", mu=iy + b * mediator + cprime * x_data, sigma=σy, observed=y, dims="obs_id")

        # calculate quantities of interest
        indirect_effect = pm.Deterministic("indirect effect", a * b)
        total_effect = pm.Deterministic("total effect", a * b + cprime)

    return model
```

## Generate simulated data

We fix the parameters to their true values with `pm.do` and take a single draw of $m$ and $y$.

```{code-cell} ipython3
N = 75
true_params = {"a": 0.5, "b": 0.6, "cprime": 0.3, "im": 2.0, "iy": 0.0, "σm": 0.5, "σy": 0.5}
x = rng.normal(loc=0, scale=1, size=N)

true_model = pm.do(model=mediation_model(x), vars_to_interventions=true_params)
prior_draw = pm.sample_prior_predictive(draws=1, model=true_model, random_seed=rng)
m = prior_draw.prior["m"].isel(chain=0, draw=0).to_numpy()
y = prior_draw.prior["y"].isel(chain=0, draw=0).to_numpy()

true_direct = true_params["cprime"]
true_indirect = true_params["a"] * true_params["b"]
print(f"True direct effect = {true_direct}")
print(f"True indirect effect = {true_indirect}")
print(f"True total effect = {true_direct + true_indirect}")

sns.pairplot(DataFrame({"x": x, "m": m, "y": y}));
```

## Conduct inference

We build the model from the simulated data and sample from its posterior.

```{code-cell} ipython3
model = mediation_model(x, m, y)
pm.model_to_graphviz(model)
```

```{code-cell} ipython3
with model:
    result = pm.sample(random_seed=rng)
```

Visualise the trace to check for convergence.

```{code-cell} ipython3
az.plot_trace_dist(result, figure_kwargs={"tight_layout": True, "sharex": "col"});
```

We have good chain mixing and the posteriors for each chain look very similar, so no problems in that regard.

+++

## Visualise the important parameters

First we will use a pair plot to look at joint posterior distributions.

```{code-cell} ipython3
az.plot_pair(
    result,
    var_names=["a", "b", "cprime", "indirect effect", "total effect"],
    figure_kwargs={"figsize": (12, 12)},
);
```

## Interpreting the results
We can take a closer look at the indirect, total, and direct effects:

```{code-cell} ipython3
pc = az.plot_dist(
    result,
    var_names=["cprime", "indirect effect", "total effect"],
    ci_prob=0.95,
    labeller=MapLabeller(var_name_map={"cprime": "direct effect"}),
    figure_kwargs={"figsize": (14, 4)},
)
az.add_lines(pc, 0);
```

- The posterior mean **direct effect** is 0.19, meaning that for every 1 unit of increase in $x$, $y$ increases by 0.19 due to the direct effect $x \rightarrow y$.
- The posterior mean **indirect effect** is 0.36, meaning that for every 1 unit of increase in $x$, $y$ increases by 0.36 through the pathway $x \rightarrow m \rightarrow y$. The probability that the indirect effect is zero is infinitesimal.
- The posterior mean **total effect** is 0.55, meaning that for every 1 unit of increase in $x$, $y$ increases by 0.55 through both the direct and indirect pathways.

+++

## Double check with total effect only model
Above, we stated that the total effect could also be estimated by evaluating the alternative model $y_i \sim \mathrm{Normal}(i_{Y*} + c \cdot x_i, \sigma_{Y*})$. Here we will check this by comparing the posterior distribution for $c'$ in the mediation model, and the posterior distribution for $c$ in this alternative model.

```{code-cell} ipython3
with pm.Model(coords={"obs_id": np.arange(len(x))}) as total_effect_model:
    _x = pm.Data("_x", x, dims="obs_id")
    iy = pm.Normal("iy", mu=0, sigma=1)
    c = pm.Normal("c", mu=0, sigma=1)
    σy = pm.HalfCauchy("σy", 1)
    μy = iy + c * _x
    pm.Normal("yy", mu=μy, sigma=σy, observed=y, dims="obs_id")
```

```{code-cell} ipython3
with total_effect_model:
    total_effect_result = pm.sample(random_seed=rng)
```

```{code-cell} ipython3
total_effects = {
    "mediation model": xr.DataTree.from_dict(
        {"posterior": result.posterior.to_dataset()[["total effect"]]}
    ),
    "total effect model": xr.DataTree.from_dict(
        {"posterior": total_effect_result.posterior.to_dataset()[["c"]].rename(c="total effect")}
    ),
}
pc = az.plot_dist(
    total_effects,
    var_names=["total effect"],
    visuals={"credible_interval": False, "point_estimate": False, "point_estimate_text": False},
    figure_kwargs={"figsize": (14, 4)},
)
pc.add_legend("model");
```

As we can see, the posterior distributions over the total effect are near-identical for the mediation model and the total effect model.

+++

## Parameter estimation versus hypothesis testing
This notebook has focused on the approach of Bayesian parameter estimation. For many situations this is entirely sufficient, and more information can be found in {cite:t}`yuan2009bayesian`. It will tell us, amongst other things, what our posterior beliefs are in the direct effects, indirect effects, and total effects. And we can use those posterior beliefs to conduct posterior predictive checks to visually check how well the model accounts for the data.

However, depending upon the use case it may be preferable to test hypotheses about the presence or absence of an indirect effect ($x \rightarrow m \rightarrow y$) for example. In this case, it may be more appropriate to take a more explicit hypothesis testing approach to see examine the relative credibility of the mediation model as compared to a simple direct effect model (i.e. $y_i = \mathrm{Normal}(i_{Y*} + c \cdot x_i, \sigma_{Y*})$). Readers are referred to {cite:t}`nuijten2015default` for a hypothesis testing approach to Bayesian mediation models and to {cite:t}`kruschke2011bayesian` for more information on parameter estimation versus hypothesis testing.

+++

## Summary
As stated at the outset, the procedures used in mediation analysis have evolved over time. So there are plenty of people who are not necessarily up to speed with modern best practice. The approach in this notebook sticks to that outlined by {cite:t}`hayes2017introduction`, but it is relevant to be aware of some of this history to avoid confusion - which is particularly important if defending your approach in peer review.

+++

## Authors
- Authored by Benjamin T. Vincent in August 2021
- Updated by Benjamin T. Vincent in February 2022
- Updated by [Jesse Grabowski](https://github.com/jessegrabowski) in October 2026 to run on PyMC v6 and ArviZ v1

+++

## References
:::{bibliography}
:filter: docname in docnames
:::

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor
```

:::{include} ../page_footer.md
:::
