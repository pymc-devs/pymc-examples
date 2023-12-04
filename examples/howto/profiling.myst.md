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

(profiling)=
# Profiling
Sometimes computing the likelihood is not as fast as we would like. Theano provides handy profiling tools which are wrapped in PyMC3 by {func}`model.profile <pymc.model.core.Model.profile>`. This function returns a `ProfileStats` object conveying information about the underlying Theano operations. Here we'll profile the likelihood and gradient for the stochastic volatility example.

First we build the model.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import pymc as pm

print(f"Running on PyMC3 v{pm.__version__}")
```

```{code-cell} ipython3
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
```

```{code-cell} ipython3
# Load the data
returns = pd.read_csv(pm.get_data("SP500.csv"), index_col=0, parse_dates=True)
```

```{code-cell} ipython3
# Stochastic volatility example
with pm.Model() as model:
    sigma = pm.Exponential("sigma", 1.0 / 0.02, initval=0.1)
    nu = pm.Exponential("nu", 1.0 / 10)
    s = pm.GaussianRandomWalk("s", sigma**-2, shape=returns.shape[0])
    r = pm.StudentT("r", nu, lam=np.exp(-2 * s), observed=returns["change"])
```

Then we call the `profile` function and summarize its return values.

```{code-cell} ipython3
# Profiling of the logp call
model.profile(model.logp()).summary()
```

```{code-cell} ipython3
# Profiling of the gradient call dlogp/dx
model.profile(pm.gradient(model.logp(), vars=None)).summary()
```

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w
```

```{code-cell} ipython3

```
