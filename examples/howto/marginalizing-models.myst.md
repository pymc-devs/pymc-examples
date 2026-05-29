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
myst:
  substitutions:
    extra_dependencies: pymc-extras
---

(marginalizing-models)=
# Automatic marginalization of discrete variables

:::{post} Jan 20, 2024
:tags: mixture model
:category: intermediate, how-to
:author: Rob Zinkov
:::

PyMC is very amendable to sampling models with discrete latent variables. But if you insist on using the NUTS sampler exclusively, you will need to get rid of your discrete variables somehow. The best way to do this is by marginalizing them out, as then you benefit from Rao-Blackwell's theorem and get a lower variance estimate of your parameters.

Formally the argument goes like this, samplers can be understood as approximating the expectation $\mathbb{E}_{p(x, z)}[f(x, z)]$ for some function $f$ with respect to a distribution $p(x, z)$. By [law of total expectation](https://en.wikipedia.org/wiki/Law_of_total_expectation) we know that

$$ \mathbb{E}_{p(x, z)}[f(x, z)] =  \mathbb{E}_{p(z)}\left[\mathbb{E}_{p(x \mid z)}\left[f(x, z)\right]\right] $$

Letting $g(z) = \mathbb{E}_{p(x \mid z)}\left[f(x, z)\right]$, we know by [law of total variance](https://en.wikipedia.org/wiki/Law_of_total_variance) that

$$ \mathbb{V}_{p(x, z)}[f(x, z)] = \mathbb{V}_{p(z)}[g(z)] + \mathbb{E}_{p(z)}\left[\mathbb{V}_{p(x \mid z)}\left[f(x, z)\right]\right] $$

Because the expectation is over a variance it must always be positive, and thus we know

$$ \mathbb{V}_{p(x, z)}[f(x, z)] \geq \mathbb{V}_{p(z)}[g(z)] $$

Intuitively, marginalizing variables in your model lets you use $g$ instead of $f$. This lower variance manifests most directly in lower Monte-Carlo standard error (mcse), and indirectly in a generally higher effective sample size (ESS).

Unfortunately, the computation to do this is often tedious and unintuitive. Luckily, `pymc_extras` supports a way to do this work automatically!

```{code-cell} ipython3
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
```

:::{include} ../extra_installs.md
:::

```{code-cell} ipython3
import pymc_extras as pmx
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'  # high resolution figures
az.style.use("arviz-variat")
rng = np.random.default_rng(32)
```

As a motivating example, consider a gaussian mixture model

+++

## Gaussian Mixture model

+++

There are two ways to specify the same model. One where the choice of mixture is explicit.

```{code-cell} ipython3
mu = pt.as_tensor([-2.0, 2.0])

with pm.Model() as explicit_mixture:
    idx = pm.Bernoulli("idx", 0.7)
    y = pm.Normal("y", mu=mu[idx], sigma=1.0)
```

```{code-cell} ipython3
plt.hist(pm.draw(y, draws=2000, random_seed=rng), bins=30, rwidth=0.9);
```

The other way is where we use the built-in {class}`NormalMixture <pymc.NormalMixture>` distribution. Here the mixture assignment is not an explicit variable in our model — the distribution already integrates it out for us. In the first model the assignment `idx` is an explicit latent variable, which we will instead marginalize out ourselves using {func}`pmx.marginalize <pymc_extras.marginalize>`.

```{code-cell} ipython3
with pm.Model() as prebuilt_mixture:
    y = pm.NormalMixture("y", w=[0.3, 0.7], mu=[-2, 2])
```

```{code-cell} ipython3
plt.hist(pm.draw(y, draws=2000, random_seed=rng), bins=30, rwidth=0.9);
```

```{code-cell} ipython3
with prebuilt_mixture:
    idata = pm.sample(draws=2000, chains=4, random_seed=rng)

az.summary(idata)
```

```{code-cell} ipython3
with explicit_mixture:
    idata = pm.sample(draws=2000, chains=4, random_seed=rng)

az.summary(idata)
```

We can immediately see that the marginalized model has a higher ESS. Let's now marginalize out the choice and see what it changes in our model.

```{code-cell} ipython3
explicit_mixture_marginalized = pmx.marginalize(explicit_mixture, ["idx"])

with explicit_mixture_marginalized:
    idata = pm.sample(draws=2000, chains=4, random_seed=rng)

az.summary(idata)
```

As we can see, the `idx` variable is gone now. We also were able to use the NUTS sampler, and the ESS has improved.

But `pymc_extras` still knows about the discrete variables that were marginalized out. We can obtain samples from the conditional posterior of `idx` using {func}`recover_marginals <pymc_extras.recover_marginals>`, and we can get the full conditional model using {func}`conditional <pymc_extras.conditional>`.

```{code-cell} ipython3
idata = pmx.recover_marginals(idata, model=explicit_mixture_marginalized, random_seed=rng)
```

```{code-cell} ipython3
az.summary(idata, var_names=["y", "idx"])
```

This `idx` variable lets us recover the mixture assignment variable after running the NUTS sampler! We can split out the samples of `y` by reading off the mixture label from the associated `idx` for each sample.

```{code-cell} ipython3
# fmt: off
post = idata.posterior.dataset
plt.hist(
    post.where(post.idx == 0).y.values.reshape(-1),
    bins=30,
    rwidth=0.9,
    alpha=0.75,
    label='idx = 0',
)
plt.hist(
    post.where(post.idx == 1).y.values.reshape(-1),
    bins=30,
    rwidth=0.9,
    alpha=0.75,
    label='idx = 1'
)
# fmt: on
plt.legend();
```

One important thing to notice is that this discrete variable has a lower ESS, and particularly so for the tail. This means `idx` might not be estimated well particularly for the tails. For a cleaner estimate, we can compute the log-probabilities of `idx` directly using {func}`conditional <pymc_extras.conditional>`, which returns a model where the marginalized variable has its conditional posterior as its distribution. The benefits of working with log-probabilities will be explored further in the next example.

+++

## Coal mining model

+++

The same methods work for the {ref}`Coal mining <pymc:pymc_overview#case-study-2-coal-mining-disasters>` switchpoint model as well. The coal mining dataset records the number of coal mining disasters in the UK between 1851 and 1962. The time series dataset captures a time when mining safety regulations are being introduced, we try to estimate when this occurred using a discrete `switchpoint` variable.

```{code-cell} ipython3
# fmt: off
disaster_data = pd.Series(
    [4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
    3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
    2, 2, 3, 4, 2, 1, 3, np.nan, 2, 1, 1, 1, 1, 3, 0, 0,
    1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
    0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
    3, 3, 1, np.nan, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
    0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]
)

# fmt: on
years = np.arange(1851, 1962)

with pm.Model() as disaster_model:
    switchpoint = pm.DiscreteUniform("switchpoint", lower=years.min(), upper=years.max())
    early_rate = pm.Exponential("early_rate", 1.0)
    late_rate = pm.Exponential("late_rate", 1.0)
    rate = pm.math.switch(switchpoint >= years, early_rate, late_rate)
    disasters = pm.Poisson("disasters", rate, observed=disaster_data)
```

We will sample the model both before and after we marginalize out the `switchpoint` variable

```{code-cell} ipython3
with disaster_model:
    before_marg = pm.sample(chains=2, random_seed=rng)

disaster_model_marginalized = pmx.marginalize(disaster_model, ["switchpoint"])

with disaster_model_marginalized:
    after_marg = pm.sample(chains=2, random_seed=rng)
```

```{code-cell} ipython3
az.summary(before_marg, var_names=["~disasters"], filter_vars="like")
```

```{code-cell} ipython3
az.summary(after_marg, var_names=["~disasters"], filter_vars="like")
```

As before, the ESS improved massively

+++

Finally, let us recover the `switchpoint` variable

```{code-cell} ipython3
after_marg = pmx.recover_marginals(after_marg, model=disaster_model_marginalized)
```

```{code-cell} ipython3
az.summary(after_marg, var_names=["~disasters"], filter_vars="like")
```

While `recover_marginals` draws samples of the marginalized variable, the probabilities associated with each value often offer a cleaner estimate. Particularly for lower probability values. This is best illustrated by comparing the histogram of the sampled values with the plot of the log-probabilities.

To compute conditional log-probabilities, we use {func}`conditional <pymc_extras.conditional>` to get a model where `switchpoint` is a free RV with its conditional posterior distribution, then use `compile_logp` to evaluate it at each domain value.

```{code-cell} ipython3
post = after_marg.posterior.dataset.switchpoint.values.reshape(-1)
bins = np.arange(post.min(), post.max())
plt.hist(post, bins, rwidth=0.9);
```

```{code-cell} ipython3
from pymc.model.transform.conditioning import remove_value_transforms

# `conditional` returns a model where `switchpoint` is a free RV with its
# conditional posterior distribution. We strip the value transforms so the
# logp inputs are on the natural scale and match the posterior variable names.
cond_model = pmx.conditional(
    remove_value_transforms(disaster_model_marginalized), "switchpoint"
)
logp_fn = cond_model.compile_logp(vars=[cond_model["switchpoint"]])

# The switchpoint conditional depends on these variables (`disasters_unobserved`
# is the imputed missing data, which enters the conditional log-probability).
rv_names = ["early_rate", "late_rate", "disasters_unobserved"]
post = after_marg.posterior.dataset[rv_names].stack(sample=("chain", "draw"))
draws = [{v: post[v].values[..., s] for v in rv_names} for s in range(post.sizes["sample"])]

# For each posterior draw, evaluate log P(switchpoint=year | rest) at every
# candidate year, then average over draws.
lp_switchpoint = np.mean(
    [[logp_fn(draw | {"switchpoint": yr}) for yr in years] for draw in draws],
    axis=0,
)

x_max = years[lp_switchpoint.argmax()]
plt.scatter(years, lp_switchpoint)
plt.axvline(x=x_max, c="orange")
plt.xlabel(r"$\mathrm{year}$")
plt.ylabel(r"$\log p(\mathrm{switchpoint}=\mathrm{year})$");
```

By plotting a histogram of sampled values instead of working with the conditional log-probabilities directly, we are left with noisier and more incomplete exploration of the underlying discrete distribution. The `conditional()` function returns a standard PyMC model, so you can use any of PyMC's tools (e.g., `compile_logp`, `sample_posterior_predictive`) to analyze the conditional posterior.

+++

## Authors
* Authored by [Rob Zinkov](https://zinkov.com) in January, 2024

+++

## References

:::{bibliography}
:filter: docname in docnames 
:::

* [STAN manual section on marginalization](https://mc-stan.org/docs/stan-users-guide/latent-discrete.html)

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,xarray
```

:::{include} ../page_footer.md
:::
