---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: pymc_examples_new
  language: python
  name: pymc_examples_new
---

# Discrete Choice Models

```{code-cell} ipython3
import arviz as az
import numpy as np  # For vectorized math operations
import pandas as pd  # For file input/output
import pymc as pm
import pytensor.tensor as pt

from matplotlib import pyplot as plt
```

In this example we'll examine the technique of discrete choice modelling using a data set from the R `mlogit` package. However we'll pursue a Bayesian approach to estimating the model rather than the MLE methodology used reported in their vigenette. The data set shows household choices over offeres of heating systems in California.  The observations consist of single-family houses in California that were newly built and had central air-conditioning. Five types of systems are considered to have been possible:

 - gas central (gc),
 - gas room (gr),
 - electric central (ec),
 - electric room (er),
 - heat pump (hp).

The data set reports the installation `ic.alt` and operating costs `oc.alt` each household was faced with for each of the five alternatives with some broad demographic information about the household and crucially the choice `depvar`. This is what one choice scenario over the five alternative looks like in the data.

```{code-cell} ipython3
wide_heating_df = pd.read_csv("../data/heating_data_r.csv")
wide_heating_df[wide_heating_df["idcase"] == 1]
```

The core idea of these kinds of models is to conceive of this as a choice over options with attached latent utility. The utility ascribed to each option is viewed as a linear combination of the attributes for each option, which drives the probability of choosing amongst each option. For each $j$ in all the alternatives $Alt$ which is assumed to take a Gumbel distribution. 
$$ \mathbf{U} \sim Gumbel $$

$$ \begin{pmatrix}
u_{gc}   \\
u_{gr}   \\
u_{ec}   \\
u_{gr}   \\
u_{hp}   \\
\end{pmatrix} =  \begin{pmatrix}
gc_{ic} & gc_{oc}  \\
gr_{ic} & gr_{oc}  \\
ec_{ic} & ec_{oc}  \\
gr_{ic} & gr_{oc}  \\
hp_{ic} & hp_{oc}  \\
\end{pmatrix} \begin{pmatrix}
\beta_{ic}   \\
\beta_{oc}   \\
\end{pmatrix}  $$

This assumption proves to be mathematically convenient because the difference between two Gumbel distributions can be modelled as a logistic function, meaning we can model a contrast difference among multiple alternatives with the softmax function: 

$$ \text{softmax}(u)_{j} = \frac{\exp(u_{j})}{\sum_{q=1}^{J}\exp(u_{q})} $$

The model then assumes that decision maker chooses the option that maximises their subjective utility, the individual utility functions can be richly parameterised. The model is identified just when the utility measures of the alternatives are benchmarked against the fixed utility of the "outside good." The last quantity is fixed at 0. 

$$\begin{pmatrix}
u_{gc}   \\
u_{gr}   \\
u_{ec}   \\
u_{gr}   \\
0   \\
\end{pmatrix}
$$

With these constraints applied we can build out conditional random utility model and it's hierarchical variants. Like nearly all subjects in statistics the precise vocabulary for the model specification is overloaded. The conditional logit parameters $\beta$
may be fixed at the level of the individual, but can vary across individuals too. 

+++

### Digression on Data Formats

Discrete choice models are often estimated using a long-data format where each choice scenario is represented with a row per alternative ID and a binary flag denoting the chosen option in each scenario. This data format is recommended for estimating these kinds of models in `stan` and in `pylogit`. The reason for doing this is that once the columns `installation_costs` and `operating_costs` have been pivoted in this fashion it's easier to include them in matrix calculations. 


```{code-cell} ipython3
long_heating_df = pd.read_csv("../data/long_heating_data.csv")
long_heating_df[long_heating_df["idcase"] == 1]
```

## The Basic Model

We will show here how to incorporate the random utility specifications in PyMC. 

```{code-cell} ipython3
N = wide_heating_df.shape[0]
observed = pd.Categorical(wide_heating_df["depvar"]).codes
coords = {
    "alts_probs": ["ec", "er", "gc", "gr", "hp"],
    "obs": range(N),
}

with pm.Model(coords=coords) as model:
    beta_ic = pm.Normal("beta_ic", 0, 1)
    beta_oc = pm.Normal("beta_oc", 0, 1)

    ## Construct Utility matrix and Pivot
    u0 = beta_ic * wide_heating_df["ic.ec"] + beta_oc * wide_heating_df["oc.ec"]
    u1 = beta_ic * wide_heating_df["ic.er"] + beta_oc * wide_heating_df["oc.er"]
    u2 = beta_ic * wide_heating_df["ic.gc"] + beta_oc * wide_heating_df["oc.gc"]
    u3 = beta_ic * wide_heating_df["ic.gr"] + beta_oc * wide_heating_df["oc.gr"]
    u4 = np.zeros(N)  # Outside Good
    s = pm.Deterministic("u", pm.math.stack([u0, u1, u2, u3, u4]).T)

    ## Apply Softmax Transform
    p_ = pm.Deterministic("p", pm.math.softmax(s, axis=1), dims=("obs", "alts_probs"))

    ## Likelihood
    choice_obs = pm.Categorical("y_cat", p=p_, observed=observed, dims="obs")

    idata_m1 = pm.sample_prior_predictive()
    idata_m1.extend(
        pm.sample(nuts_sampler="numpyro", idata_kwargs={"log_likelihood": True}, random_seed=101)
    )
    idata_m1.extend(pm.sample_posterior_predictive(idata_m1))

pm.model_to_graphviz(model)
```

```{code-cell} ipython3
idata_m1
```

```{code-cell} ipython3
summaries = az.summary(idata_m1, var_names=["beta_ic", "beta_oc"])
summaries
```

In the `mlogit` vignette they report how the above model specification leads to inadequate parameter estimates. They note for instance that while the utility scale itself is hard to interpret the value of the ratio of the coefficients is often meaningful because:
$$ U = \beta_{oc}oc + \beta_{ic}ic $$
$$ dU = \beta_{ic} dic + \beta_{oc} doc = 0 \Rightarrow 
-\frac{dic}{doc}\mid_{dU=0}=\frac{\beta_{oc}}{\beta_{ic}}$$

Our parameter estimates differ from the reported estimates, but we agree the model is inadequate. We will show a number of Bayesian model checks to demonstrate this fact, but the main call out is that the parameter values ought to be negative. To interpret the beta coefficient as the increase in utility as a function of a one unit increase in terms of price, so it's strange that an increase in price would increase the utility of generated by the installation even marginally as here. Although we might imagine that some kind of quality assurance comes with price which drives satisfaction.

```{code-cell} ipython3
## marginal rate of substitution for a reduction in installation costs
summaries["mean"]["beta_oc"] / summaries["mean"]["beta_ic"]
```

which suggests that there is almost twice the value accorded to the a unit reduction in recurring operating costs over the one-off installation costs. Whether this is remotely plausible is almost beside the point since the model does not even closely capture the data generating process. 

```{code-cell} ipython3
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
ax = axs[0]
counts = wide_heating_df.groupby("depvar")["idcase"].count()
predicted_shares = az.extract(idata_m1, var_names=["p"]).mean(axis=2).mean(axis=0)
ax.bar(range(5), counts / counts.sum(), label="Observed Shares", alpha=0.3)
ax.bar(range(5), predicted_shares, label="Predicted Shares", alpha=0.3)
ax.legend()
ax.set_title("Observed V Predicted Shares")
az.plot_ppc(idata_m1, ax=axs[1])
```

We can see here that the model is fairly inadequate, and fails quite dramatically to recapture the posterior predictive distribution. 

+++

## Improved Model: Adding Alternative Specific Intercepts

```{code-cell} ipython3
N = wide_heating_df.shape[0]
observed = pd.Categorical(wide_heating_df["depvar"]).codes
# observed = pd.get_dummies(wide_heating_df['depvar'])
# observed = long_heating_df['choice']

coords = {
    "alts_intercepts": ["ec", "er", "gc", "gr"],
    "alts_probs": ["ec", "er", "gc", "gr", "hp"],
    "obs": range(N),
}
with pm.Model(coords=coords) as model:
    beta_ic = pm.Normal("beta_ic", 0, 1)
    beta_oc = pm.Normal("beta_oc", 0, 1)
    alphas = pm.Normal("alpha", 0, 1, dims="alts_intercepts")

    u0 = alphas[0] + beta_ic * wide_heating_df["ic.ec"] + beta_oc * wide_heating_df["oc.ec"]
    u1 = alphas[1] + beta_ic * wide_heating_df["ic.er"] + beta_oc * wide_heating_df["oc.er"]
    u2 = alphas[2] + beta_ic * wide_heating_df["ic.gc"] + beta_oc * wide_heating_df["oc.gc"]
    u3 = alphas[3] + beta_ic * wide_heating_df["ic.gr"] + beta_oc * wide_heating_df["oc.gr"]
    u4 = np.zeros(N)  # pivot
    s = pm.Deterministic("u", pm.math.stack([u0, u1, u2, u3, u4]).T)

    p_ = pm.Deterministic("p", pm.math.softmax(s, axis=1), dims=("obs", "alts_probs"))
    choice_obs = pm.Categorical("y_cat", p=p_, observed=observed, dims="obs")
    # choice_obs = pm.Multinomial('y_cat', n=1, p=p_, observed=observed)

    idata_m2 = pm.sample_prior_predictive()
    idata_m2.extend(
        pm.sample(nuts_sampler="numpyro", idata_kwargs={"log_likelihood": True}, random_seed=103)
    )
    idata_m2.extend(pm.sample_posterior_predictive(idata_m2))


pm.model_to_graphviz(model)
```

```{code-cell} ipython3
az.summary(idata_m2, var_names=["beta_ic", "beta_oc", "alpha"])
```

```{code-cell} ipython3
-0.003 / 0.001
```

```{code-cell} ipython3
az.plot_ppc(idata_m2)
```

```{code-cell} ipython3
coords = {
    "alts_intercepts": ["ec", "er", "gc", "gr"],
    "alts_probs": ["ec", "er", "gc", "gr", "hp"],
    "obs": range(N),
}
with pm.Model(coords=coords) as model:
    beta_ic = pm.Normal("beta_ic", 0, 1)
    beta_oc = pm.Normal("beta_oc", 0, 1)
    beta_income = pm.Normal("beta_income", 0, 1, dims="alts_intercepts")
    alphas = pm.Normal("alpha", 0, 1, dims="alts_intercepts")

    u0 = (
        alphas[0]
        + beta_ic * wide_heating_df["ic.ec"]
        + beta_oc * wide_heating_df["oc.ec"]
        + beta_income[0] * wide_heating_df["income"]
    )
    u1 = (
        alphas[1]
        + beta_ic * wide_heating_df["ic.er"]
        + beta_oc * wide_heating_df["oc.er"]
        + beta_income[1] * wide_heating_df["income"]
    )
    u2 = (
        alphas[2]
        + beta_ic * wide_heating_df["ic.gc"]
        + beta_oc * wide_heating_df["oc.gc"]
        + beta_income[2] * wide_heating_df["income"]
    )
    u3 = (
        alphas[3]
        + beta_ic * wide_heating_df["ic.gr"]
        + beta_oc * wide_heating_df["oc.gr"]
        + beta_income[3] * wide_heating_df["income"]
    )
    u4 = np.zeros(N)  # pivot
    s = pm.Deterministic("u", pm.math.stack([u0, u1, u2, u3, u4]).T)

    p_ = pm.Deterministic("p", pm.math.softmax(s, axis=1), dims=("obs", "alts_probs"))
    choice_obs = pm.Categorical("y_cat", p=p_, observed=observed, dims="obs")
    # choice_obs = pm.Multinomial('y_cat', n=1, p=p_, observed=observed)

    idata_m3 = pm.sample_prior_predictive()
    idata_m3.extend(pm.sample(nuts_sampler="numpyro", idata_kwargs={"log_likelihood": True}))
    idata_m3.extend(pm.sample_posterior_predictive(idata_m3))


pm.model_to_graphviz(model)
```

```{code-cell} ipython3
az.plot_ppc(idata_m3)
```

```{code-cell} ipython3
az.summary(idata_m3, var_names=["beta_income", "beta_ic", "beta_oc", "alpha"])
```

```{code-cell} ipython3
az.compare({"m1": idata_m1, "m2": idata_m2, "m3": idata_m3}, "loo")
```

```{code-cell} ipython3
idata_m3
```

```{code-cell} ipython3
az.extract(idata_m3, var_names=["p"]).mean(axis=2).mean(axis=0)
```

```{code-cell} ipython3

```
