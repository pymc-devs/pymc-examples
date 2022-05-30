---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# How to debug a model

There are various levels on which to debug a model. One of the simplest is to just print out the values that different variables are taking on.

Because `PyMC` uses `Aesara` expressions to build the model, and not functions, there is no way to place a `print` statement into a likelihood function. Instead, you can use the `Aesara` `Print` operatator. For more information, see:  aesara Print operator for this before: http://deeplearning.net/software/aesara/tutorial/debug_faq.html#how-do-i-print-an-intermediate-value-in-a-function.

Let's build a simple model with just two parameters:

```{code-cell} ipython3
%matplotlib inline

import aesara.tensor as at
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

x = np.random.randn(100)

with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    sd = pm.Normal("sd", mu=0, sigma=1)

    obs = pm.Normal("obs", mu=mu, sigma=sd, observed=x)
    step = pm.Metropolis()
    trace = pm.sample(5000, step)
```

```{code-cell} ipython3
trace["mu"]
```

Hm, looks like something has gone wrong, but what? Let's look at the values getting proposed using the `Print` operator:

```{code-cell} ipython3
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    sd = pm.Normal("sd", mu=0, sigma=1)

    mu_print = at.printing.Print("mu")(mu)
    sd_print = at.printing.Print("sd")(sd)

    obs = pm.Normal("obs", mu=mu_print, sigma=sd_print, observed=x)
    step = pm.Metropolis()
    trace = pm.sample(
        3, step, tune=0, chains=1, progressbar=False
    )  # Make sure not to draw too many samples
```

In the code above, we set the `tune=0, chains=1, progressbar=False` in the `pm.sample`, this is done so that the output is cleaner.

Looks like `sd` is always `0` which will cause the logp to go to `-inf`. Of course, we should not have used a prior that has negative mass for `sd` but instead something like a `HalfNormal`.

We can also redirect the output to a string buffer and access the proposed values later on (thanks to [Lindley Lentati](https://github.com/LindleyLentati) for providing this example):

```{code-cell} ipython3
import sys

from io import StringIO

x = np.random.randn(100)

old_stdout = sys.stdout
mystdout = sys.stdout = StringIO()

with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    sd = pm.Normal("sd", mu=0, sigma=1)

    mu_print = at.printing.Print("mu")(mu)
    sd_print = at.printing.Print("sd")(sd)

    obs = pm.Normal("obs", mu=mu_print, sigma=sd_print, observed=x)
    step = pm.Metropolis()
    trace = pm.sample(
        5, step, tune=0, chains=1, progressbar=False
    )  # Make sure not to draw too many samples

sys.stdout = old_stdout

output = mystdout.getvalue().split("\n")
mulines = [s for s in output if "mu" in s]

muvals = [line.split()[-1] for line in mulines]
plt.plot(np.arange(0, len(muvals)), muvals)
plt.xlabel("proposal iteration")
plt.ylabel("mu value");
```

```{code-cell} ipython3
trace["mu"]
```

Notice that for each iteration, 3 values were printed and recorded. The printed values are the original value (last sample), the proposed value and the accepted value. Plus the starting value in the very beginning, we recorded in total `1+3*5=16` value above.

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w
```
