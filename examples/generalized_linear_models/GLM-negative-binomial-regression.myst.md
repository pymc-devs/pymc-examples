---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3.9.12 ('pymc-dev-py39')
  language: python
  name: python3
---

(GLM-negative-binomial-regression)=
# GLM: Negative Binomial Regression

:::{post} June, 2022
:tags: negative binomial regression, generalized linear model, 
:category: beginner
:author: Ian Ozsvald, Abhipsha Das, Benjamin Vincent
:::

```{code-cell} ipython3
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

from scipy import stats
```

```{code-cell} ipython3
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

%config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")
```

This notebook closely follows the GLM Poisson regression example by [Jonathan Sedar](https://github.com/jonsedar) (which is in turn inspired by [a project by Ian Osvald](http://ianozsvald.com/2016/05/07/statistically-solving-sneezes-and-sniffles-a-work-in-progress-report-at-pydatalondon-2016/)) except the data here is negative binomially distributed instead of Poisson distributed.

Negative binomial regression is used to model count data for which the variance is higher than the mean. The [negative binomial distribution](https://en.wikipedia.org/wiki/Negative_binomial_distribution) can be thought of as a Poisson distribution whose rate parameter is gamma distributed, so that rate parameter can be adjusted to account for the increased variance.

+++

### Generate Data

As in the Poisson regression example, we assume that sneezing occurs at some baseline rate, and that consuming alcohol, not taking antihistamines, or doing both, increase its frequency.

#### Poisson Data

First, let's look at some Poisson distributed data from the Poisson regression example.

```{code-cell} ipython3
# Mean Poisson values
theta_noalcohol_meds = 1  # no alcohol, took an antihist
theta_alcohol_meds = 3  # alcohol, took an antihist
theta_noalcohol_nomeds = 6  # no alcohol, no antihist
theta_alcohol_nomeds = 36  # alcohol, no antihist

# Create samples
q = 1000
df_pois = pd.DataFrame(
    {
        "nsneeze": np.concatenate(
            (
                rng.poisson(theta_noalcohol_meds, q),
                rng.poisson(theta_alcohol_meds, q),
                rng.poisson(theta_noalcohol_nomeds, q),
                rng.poisson(theta_alcohol_nomeds, q),
            )
        ),
        "alcohol": np.concatenate(
            (
                np.repeat(False, q),
                np.repeat(True, q),
                np.repeat(False, q),
                np.repeat(True, q),
            )
        ),
        "nomeds": np.concatenate(
            (
                np.repeat(False, q),
                np.repeat(False, q),
                np.repeat(True, q),
                np.repeat(True, q),
            )
        ),
    }
)
```

```{code-cell} ipython3
df_pois.groupby(["nomeds", "alcohol"])["nsneeze"].agg(["mean", "var"])
```

Since the mean and variance of a Poisson distributed random variable are equal, the sample means and variances are very close.

#### Negative Binomial Data

Now, suppose every subject in the dataset had the flu, increasing the variance of their sneezing (and causing an unfortunate few to sneeze over 70 times a day). If the mean number of sneezes stays the same but variance increases, the data might follow a negative binomial distribution.

```{code-cell} ipython3
# Gamma shape parameter
alpha = 10


def get_nb_vals(mu, alpha, size):
    """Generate negative binomially distributed samples by
    drawing a sample from a gamma distribution with mean `mu` and
    shape parameter `alpha', then drawing from a Poisson
    distribution whose rate parameter is given by the sampled
    gamma variable.

    """

    g = stats.gamma.rvs(alpha, scale=mu / alpha, size=size)
    return stats.poisson.rvs(g)


# Create samples
n = 1000
df = pd.DataFrame(
    {
        "nsneeze": np.concatenate(
            (
                get_nb_vals(theta_noalcohol_meds, alpha, n),
                get_nb_vals(theta_alcohol_meds, alpha, n),
                get_nb_vals(theta_noalcohol_nomeds, alpha, n),
                get_nb_vals(theta_alcohol_nomeds, alpha, n),
            )
        ),
        "alcohol": np.concatenate(
            (
                np.repeat(False, n),
                np.repeat(True, n),
                np.repeat(False, n),
                np.repeat(True, n),
            )
        ),
        "nomeds": np.concatenate(
            (
                np.repeat(False, n),
                np.repeat(False, n),
                np.repeat(True, n),
                np.repeat(True, n),
            )
        ),
    }
)
df
```

```{code-cell} ipython3
df.groupby(["nomeds", "alcohol"])["nsneeze"].agg(["mean", "var"])
```

As in the Poisson regression example, we see that drinking alcohol and/or not taking antihistamines increase the sneezing rate to varying degrees. Unlike in that example, for each combination of `alcohol` and `nomeds`, the variance of `nsneeze` is higher than the mean. This suggests that a Poisson distribution would be a poor fit for the data since the mean and variance of a Poisson distribution are equal.

+++

### Visualize the Data

```{code-cell} ipython3
g = sns.catplot(x="nsneeze", row="nomeds", col="alcohol", data=df, kind="count", aspect=1.5)

# Make x-axis ticklabels less crowded
ax = g.axes[1, 0]
labels = range(len(ax.get_xticklabels(which="both")))
ax.set_xticks(labels[::5])
ax.set_xticklabels(labels[::5]);
```

## Negative Binomial Regression

+++

### Create GLM Model

```{code-cell} ipython3
COORDS = {"regressor": ["nomeds", "alcohol", "nomeds:alcohol"], "obs_idx": df.index}

with pm.Model(coords=COORDS) as m_sneeze_inter:
    a = pm.Normal("intercept", mu=0, sigma=5)
    b = pm.Normal("slopes", mu=0, sigma=1, dims="regressor")
    alpha = pm.Exponential("alpha", 0.5)

    M = pm.ConstantData("nomeds", df.nomeds.to_numpy(), dims="obs_idx")
    A = pm.ConstantData("alcohol", df.alcohol.to_numpy(), dims="obs_idx")
    S = pm.ConstantData("nsneeze", df.nsneeze.to_numpy(), dims="obs_idx")

    λ = pm.math.exp(a + b[0] * M + b[1] * A + b[2] * M * A)

    y = pm.NegativeBinomial("y", mu=λ, alpha=alpha, observed=S, dims="obs_idx")

    idata = pm.sample()
```

### View Results

```{code-cell} ipython3
az.plot_trace(idata, compact=False);
```

```{code-cell} ipython3
# Transform coefficients to recover parameter values
az.summary(np.exp(idata.posterior), kind="stats", var_names=["intercept", "slopes"])
```

```{code-cell} ipython3
az.summary(idata.posterior, kind="stats", var_names="alpha")
```

The mean values are close to the values we specified when generating the data:
- The base rate is a constant 1.
- Drinking alcohol triples the base rate.
- Not taking antihistamines increases the base rate by 6 times.
- Drinking alcohol and not taking antihistamines doubles the rate that would be expected if their rates were independent. If they were independent, then doing both would increase the base rate by 3\*6=18 times, but instead the base rate is increased by 3\*6\*2=36 times.

Finally, the mean of `nsneeze_alpha` is also quite close to its actual value of 10.

+++

See also, [`bambi's` negative binomial example](https://bambinos.github.io/bambi/master/notebooks/negative_binomial.html) for further reference.

+++

## Authors
- Created by [Ian Ozsvald](https://github.com/ianozsvald)
- Updated by [Abhipsha Das](https://github.com/chiral-carbon) in August 2021
- Updated by [Benjamin Vincent](https://github.com/drbenvincent) to PyMC v4 in June 2022

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,aeppl,xarray
```

:::{include} ../page_footer.md
:::
