---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(howto_debugging)=
# How to debug a model

:::{post} August 2, 2022
:tags: debugging, aesara
:category: beginner
:author: Thomas Wiecki, Igor Kuvychko
:::

+++

## Introduction
There are various levels on which to debug a model. One of the simplest is to just print out the values that different variables are taking on.

Because `PyMC` uses `Aesara` expressions to build the model, and not functions, there is no way to place a `print` statement into a likelihood function. Instead, you can use the `aesara.printing.Print` class to print intermediate values.

```{code-cell} ipython3
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
```

```{code-cell} ipython3
%matplotlib inline
%config InlineBackend.figure_format = "retina"

RANDOM_SEED = 8927
```

### How to print intermediate values of `Aesara` functions
Since `Aesara` functions are compiled to C, you have to use `aesara.printing.Print` class to print intermediate values (imported  below as `Print`). Python `print` function will not work. Below is a simple example of using `Print`. For more information, see {ref}`Debugging Aesara <aesara:debug_faq>`.

```{code-cell} ipython3
import aesara.tensor as at

from aesara import function
from aesara.printing import Print
```

```{code-cell} ipython3
x = at.dvector("x")
y = at.dvector("y")
func = function([x, y], 1 / (x - y))
func([1, 2, 3], [1, 0, -1])
```

To see what causes the `inf` value in the output, we can print intermediate values of $(x-y)$ using `Print`. `Print` class simply passes along its caller but prints out its value along a user-define message:

```{code-cell} ipython3
z_with_print = Print("x - y = ")(x - y)
func_with_print = function([x, y], 1 / z_with_print)
func_with_print([1, 2, 3], [1, 0, -1])
```

`Print` reveals the root cause: $(x-y)$ takes a zero value when $x=1, y=1$, causing the `inf` output.

+++

### How to capture `Print` output for further analysis

When we expect many rows of output from `Print`, it can be desirable to redirect the output to a string buffer and access the values later on (thanks to **Lindley Lentati** for inspiring this example). Here is a toy example using Python `print` function:

```{code-cell} ipython3
import sys

from io import StringIO

old_stdout = sys.stdout
mystdout = sys.stdout = StringIO()

for i in range(5):
    print(f"Test values: {i}")

output = mystdout.getvalue().split("\n")
sys.stdout = old_stdout  # setting sys.stdout back
output
```

### Troubleshooting a toy PyMC model

```{code-cell} ipython3
rng = np.random.default_rng(RANDOM_SEED)
x = rng.normal(size=100)

with pm.Model() as model:
    # priors
    mu = pm.Normal("mu", mu=0, sigma=1)
    sd = pm.Normal("sd", mu=0, sigma=1)

    # setting out printing for mu and sd
    mu_print = Print("mu")(mu)
    sd_print = Print("sd")(sd)

    # likelihood
    obs = pm.Normal("obs", mu=mu_print, sigma=sd_print, observed=x)
```

```{code-cell} ipython3
pm.model_to_graphviz(model)
```

```{code-cell} ipython3
with model:
    step = pm.Metropolis()
    trace = pm.sample(5, step, tune=0, chains=1, progressbar=False, random_seed=RANDOM_SEED)
```

Exception handling of PyMC v4 has improved, so now SamplingError exception prints out the intermediate values of `mu` and `sd` which led to likelihood of `-inf`. However, this technique of printing intermediate values with `aeasara.printing.Print` can be valuable in more complicated cases.

+++

### Bringing it all together

```{code-cell} ipython3
rng = np.random.default_rng(RANDOM_SEED)
y = rng.normal(loc=5, size=20)

old_stdout = sys.stdout
mystdout = sys.stdout = StringIO()

with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=10)
    a = pm.Normal("a", mu=0, sigma=10, initval=0.1)
    b = pm.Normal("b", mu=0, sigma=10, initval=0.1)
    sd_print = Print("Delta")(a / b)
    obs = pm.Normal("obs", mu=mu, sigma=sd_print, observed=y)

    # limiting number of samples and chains to simplify output
    trace = pm.sample(draws=10, tune=0, chains=1, progressbar=False, random_seed=RANDOM_SEED)

output = mystdout.getvalue()
sys.stdout = old_stdout  # setting sys.stdout back
```

```{code-cell} ipython3
output
```

Raw output is a bit messy and requires some cleanup and formatting to convert to `numpy.ndarray`. In the example below regex is used to clean up the output, and then it is evaluated with `eval` to give a list of floats. Code below also works with higher-dimensional outputs (in case you want to experiment with different models).

```{code-cell} ipython3
import re

# output cleanup and conversion to numpy array
# this is code accepts more complicated inputs
pattern = re.compile("Delta __str__ = ")
output = re.sub(pattern, " ", output)
pattern = re.compile("\\s+")
output = re.sub(pattern, ",", output)
pattern = re.compile(r"\[,")
output = re.sub(pattern, "[", output)
output += "]"
output = "[" + output[1:]
output = eval(output)
output = np.array(output)
```

```{code-cell} ipython3
output
```

Notice that we requested 5 draws, but got 34 sets of $a/b$ values. The reason is that for each iteration, all proposed values are printed (not just the accepted values). Negative values are clearly problematic.

```{code-cell} ipython3
output.shape
```

## Authors

* Authored by Thomas Wiecki in July, 2016
* Updated by Igor Kuvychko in August, 2022 ([pymc#406] (https://github.com/pymc-devs/pymc-examples/pull/406))

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p aesara,xarray
```

:::{include} ../page_footer.md
:::
