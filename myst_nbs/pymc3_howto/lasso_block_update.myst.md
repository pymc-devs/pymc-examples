---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python PyMC (Dev)
  language: python
  name: pymc-dev-py38
---

# Lasso regression with block updating

Sometimes, it is very useful to update a set of parameters together. For example, variables that are highly correlated are often good to update together. In PyMC 3 block updating is simple, as example will demonstrate.

Here we have a LASSO regression model where the two coefficients are strongly correlated. Normally, we would define the coefficient parameters as a single random variable, but here we define them separately to show how to do block updates.

First we generate some fake data.

```{code-cell} ipython3
%matplotlib inline

import arviz as az
import numpy as np
import pymc as pm

from matplotlib import pylab

d = np.random.normal(size=(3, 30))
d1 = d[0] + 4
d2 = d[1] + 4
yd = 0.2 * d1 + 0.3 * d2 + d[2]
```

Then define the random variables.

```{code-cell} ipython3
lam = 3

with pm.Model() as model:
    s = pm.Exponential("s", 1)
    tau = pm.Uniform("tau", 0, 1000)
    b = lam * tau
    m1 = pm.Laplace("m1", 0, b)
    m2 = pm.Laplace("m2", 0, b)

    p = d1 * m1 + d2 * m2

    y = pm.Normal("y", mu=p, sigma=s, observed=yd)
```

For most samplers, including Metropolis and HamiltonianMC, simply pass a list of variables to sample as a block. This works with both scalar and array parameters.

```{code-cell} ipython3
with model:
    start = pm.find_MAP()

    step1 = pm.Metropolis([m1, m2])

    step2 = pm.Slice([s, tau])

    idata = pm.sample(10000, [step1, step2], start=start)
```

```{code-cell} ipython3
az.plot_trace(idata);
```

```{code-cell} ipython3
pylab.hexbin(idata.posterior["m1"], idata.posterior["m2"], gridsize=50)
pylab.axis("off");
```

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w
```
