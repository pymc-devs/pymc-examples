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

(template_notebook)=
# Discrete Choice and Random Utility Models

:::{post} June, 2023
:tags: categorical regression, generalized linear model, discrete choice 
:category: advance, reference
:author: Nathaniel Forde
:::

```{code-cell} ipython3
import arviz as az
import numpy as np  # For vectorized math operations
import pandas as pd  # For file input/output
import pymc as pm
import pytensor.tensor as pt

from matplotlib import pyplot as plt
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'  # high resolution figures
az.style.use("arviz-darkgrid")
rng = np.random.default_rng(42)
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

Our parameter estimates differ from the reported estimates, but we agree the model is inadequate. We will show a number of Bayesian model checks to demonstrate this fact, but the main call out is that the parameter values ought to be negative. To interpret the beta coefficient as the increase in utility as a function of a one unit increase in terms of price, so it's strange that an increase in price would increase the utility of generated by the installation even marginally as here. Although we might imagine that some kind of quality assurance comes with price which drives satisfaction. Below we'll see how we can incorporate prior knowledge to adjust for this kind of interpretation. 

We can calculate the marginal rate of substitution as follows:

```{code-cell} ipython3
## marginal rate of substitution for a reduction in installation costs
summaries["mean"]["beta_oc"] / summaries["mean"]["beta_ic"]
```

But being good Bayesians we actually want to calculate the posterior distribution for this statistic.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(20, 4))

ax.hist(
    az.extract(idata_m1["posterior"]["beta_oc"] / idata_m1["posterior"]["beta_ic"])["x"],
    bins=30,
    ec="black",
)
ax.set_title("Uncertainty in Marginal Rate of Substitution \n Operating Costs / Installation Costs")
```

which suggests that there is almost twice the value accorded to the a unit reduction in recurring operating costs over the one-off installation costs. Whether this is remotely plausible is almost beside the point since the model does not even closely capture the data generating process. 

```{code-cell} ipython3
fig, axs = plt.subplots(1, 2, figsize=(20, 5))
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

We can address some of the issues with the prior model specification by adding intercept terms for each of the unique alternatives `gr, gc, ec, er`. 

```{code-cell} ipython3
N = wide_heating_df.shape[0]
observed = pd.Categorical(wide_heating_df["depvar"]).codes

coords = {
    "alts_intercepts": ["ec", "er", "gc", "gr"],
    "alts_probs": ["ec", "er", "gc", "gr", "hp"],
    "obs": range(N),
}
with pm.Model(coords=coords) as model_2:
    beta_ic = pm.Normal("beta_ic", 0, 1)
    beta_oc = pm.Normal("beta_oc", 0, 1)
    alphas = pm.Normal("alpha", 0, 1, dims="alts_intercepts")

    ## Construct Utility matrix and Pivot using an intercept per alternative
    u0 = alphas[0] + beta_ic * wide_heating_df["ic.ec"] + beta_oc * wide_heating_df["oc.ec"]
    u1 = alphas[1] + beta_ic * wide_heating_df["ic.er"] + beta_oc * wide_heating_df["oc.er"]
    u2 = alphas[2] + beta_ic * wide_heating_df["ic.gc"] + beta_oc * wide_heating_df["oc.gc"]
    u3 = alphas[3] + beta_ic * wide_heating_df["ic.gr"] + beta_oc * wide_heating_df["oc.gr"]
    u4 = np.zeros(N)  # Outside Good
    s = pm.Deterministic("u", pm.math.stack([u0, u1, u2, u3, u4]).T)

    ## Apply Softmax Transform
    p_ = pm.Deterministic("p", pm.math.softmax(s, axis=1), dims=("obs", "alts_probs"))

    ## Likelihood
    choice_obs = pm.Categorical("y_cat", p=p_, observed=observed, dims="obs")

    idata_m2 = pm.sample_prior_predictive()
    idata_m2.extend(
        pm.sample(nuts_sampler="numpyro", idata_kwargs={"log_likelihood": True}, random_seed=103)
    )
    idata_m2.extend(pm.sample_posterior_predictive(idata_m2))


pm.model_to_graphviz(model_2)
```

```{code-cell} ipython3
az.summary(idata_m2, var_names=["beta_ic", "beta_oc", "alpha"])
```

We can see now how this model performs much better in capturing aspects of the data generating process. 

```{code-cell} ipython3
fig, axs = plt.subplots(1, 2, figsize=(20, 5))
ax = axs[0]
counts = wide_heating_df.groupby("depvar")["idcase"].count()
predicted_shares = az.extract(idata_m2, var_names=["p"]).mean(axis=2).mean(axis=0)
ci_lb = np.quantile(az.extract(idata_m2, var_names=["p"]).mean(axis=2), 0.025, axis=0)
ci_ub = np.quantile(az.extract(idata_m2, var_names=["p"]).mean(axis=2), 0.975, axis=0)
ax.scatter(ci_lb, ["ec", "er", "gc", "gr", "hp"], color="k", s=2)
ax.scatter(ci_ub, ["ec", "er", "gc", "gr", "hp"], color="k", s=2)
ax.scatter(
    counts / counts.sum(), ["ec", "er", "gc", "gr", "hp"], label="Observed Shares", color="red"
)
ax.hlines(
    ["ec", "er", "gc", "gr", "hp"], ci_lb, ci_ub, label="Predicted 95% Interval", color="black"
)
ax.legend()
ax.set_title("Observed V Predicted Shares")
az.plot_ppc(idata_m2, ax=axs[1])
axs[1].set_title("Posterior Predictive Checks")
ax.set_xlabel("Shares")
ax.set_ylabel("Heating System");
```

This model represents a substantial improvement. It's worth pausing to consider how

+++

## Experimental Model: Adding Correlation Structure

We might think that there is a correlation among the alternative goods that we should capture too. We can capture those effects in so far as they exist by placing a multvariate normal prior on the intercepts, (or alternatively the beta parameters). In addition we add information about how the effect of income influences the utility accorded to each alternative. 

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
    chol, corr, stds = pm.LKJCholeskyCov(
        "chol", n=4, eta=2.0, sd_dist=pm.Exponential.dist(1.0, shape=4)
    )
    alphas = pm.MvNormal("alpha", mu=0, chol=chol)

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

    idata_m3 = pm.sample_prior_predictive()
    idata_m3.extend(
        pm.sample(nuts_sampler="numpyro", idata_kwargs={"log_likelihood": True}, random_seed=100)
    )
    idata_m3.extend(pm.sample_posterior_predictive(idata_m3))


pm.model_to_graphviz(model)
```

Plotting the model fit we see a similar story.The model predictive performance is not drastically improved and we have added some complexity to the model.

```{code-cell} ipython3
fig, axs = plt.subplots(1, 2, figsize=(20, 5))
ax = axs[0]
counts = wide_heating_df.groupby("depvar")["idcase"].count()
predicted_shares = az.extract(idata_m3, var_names=["p"]).mean(axis=2).mean(axis=0)
ci_lb = np.quantile(az.extract(idata_m3, var_names=["p"]).mean(axis=2), 0.025, axis=0)
ci_ub = np.quantile(az.extract(idata_m3, var_names=["p"]).mean(axis=2), 0.975, axis=0)
ax.scatter(ci_lb, ["ec", "er", "gc", "gr", "hp"], color="k", s=2)
ax.scatter(ci_ub, ["ec", "er", "gc", "gr", "hp"], color="k", s=2)
ax.scatter(
    counts / counts.sum(), ["ec", "er", "gc", "gr", "hp"], label="Observed Shares", color="red"
)
ax.hlines(
    ["ec", "er", "gc", "gr", "hp"], ci_lb, ci_ub, label="Predicted 95% Interval", color="black"
)
ax.legend()
ax.set_title("Observed V Predicted Shares")
az.plot_ppc(idata_m3, ax=axs[1])
axs[1].set_title("Posterior Predictive Checks")
ax.set_xlabel("Shares")
ax.set_ylabel("Heating System");
```

However, that complexity can be informative. 

```{code-cell} ipython3
az.summary(idata_m3, var_names=["beta_income", "beta_ic", "beta_oc", "alpha", "chol_corr"])
```

```{code-cell} ipython3
ax = az.plot_posterior(
    idata_m3,
    var_names="chol_corr",
    hdi_prob="hide",
    point_estimate="mean",
    grid=(4, 4),
    kind="hist",
    ec="black",
    figsize=(20, 12),
)
```

### Compare Models

We'll now evaluate all three model fits on their predictive performance. 

```{code-cell} ipython3
compare = az.compare({"m1": idata_m1, "m2": idata_m2, "m3": idata_m3})
compare
```

```{code-cell} ipython3
az.plot_compare(compare)
```

## Choosing Crackers over Repeated Choices

Moving to another example, we see a choice scenario where the same individual has been repeatedly polled on their choice of crackers among alternatives. This affords us the opportunity to evaluate the preferences of individuals by adding in coefficients for individuals for each product. 

```{code-cell} ipython3
c_df = pd.read_csv("../data/cracker_choice_short.csv")
## Focus on smaller subset of the decision makers. Need to use scan due bracket nesting level error
c_df = c_df[c_df["personId"] < 50]
c_df
```

```{code-cell} ipython3
N = c_df.shape[0]
observed = pd.Categorical(c_df["choice"]).codes
uniques = c_df["personId"].unique()
coords = {
    "alts_intercepts": ["sunshine", "keebler", "nabisco"],
    "alts_probs": ["sunshine", "keebler", "nabisco", "private"],
    "individuals": uniques,
    "obs": range(N),
}
with pm.Model(coords=coords) as model_4:
    beta_feat = pm.Normal("beta_feat", 0, 0.1)
    beta_disp = pm.Normal("beta_disp", 0, 0.1)
    ## Stronger Prior on Price to ensure an increase in price negatively impacts utility
    beta_price = pm.TruncatedNormal("beta_price", 0, 1, upper=0, lower=-10)
    alphas = pm.Normal("alpha", 0, 1, dims="alts_intercepts")
    beta_individual = pm.Normal("beta_individual", 0, 0.05, dims=("alts_intercepts", "individuals"))

    person_choice_scenario = []
    for id, indx in zip(uniques, range(len(uniques))):
        ## Construct Utility matrix and Pivot using an intercept per alternative
        n = c_df[c_df["personId"] == id].shape[0]
        u0 = (
            (alphas[0] + beta_individual[0, indx])
            + beta_disp * c_df[c_df["personId"] == id]["disp.sunshine"]
            + beta_feat * c_df[c_df["personId"] == id]["feat.sunshine"]
            + beta_price * c_df[c_df["personId"] == id]["price.sunshine"]
        )
        u1 = (
            (alphas[1] + beta_individual[1, indx])
            + beta_disp * c_df[c_df["personId"] == id]["disp.keebler"]
            + beta_feat * c_df[c_df["personId"] == id]["feat.keebler"]
            + beta_price * c_df[c_df["personId"] == id]["price.keebler"]
        )
        u2 = (
            (alphas[2] + beta_individual[2, indx])
            + beta_disp * c_df[c_df["personId"] == id]["disp.nabisco"]
            + beta_feat * c_df[c_df["personId"] == id]["feat.nabisco"]
            + beta_price * c_df[c_df["personId"] == id]["price.nabisco"]
        )
        u3 = np.zeros(n)  # Outside Good
        s = pm.math.stack([u0, u1, u2, u3]).T
        person_choice_scenario.append(s)

    s = pm.Deterministic("stacked", pt.concatenate(person_choice_scenario))

    ## Apply Softmax Transform
    p_ = pm.Deterministic("p", pm.math.softmax(s, axis=1), dims=("obs", "alts_probs"))

    ## Likelihood
    choice_obs = pm.Categorical("y_cat", p=p_, observed=observed, dims="obs")

    idata_m4 = pm.sample_prior_predictive()
    idata_m4.extend(
        pm.sample(nuts_sampler="numpyro", idata_kwargs={"log_likelihood": True}, random_seed=103)
    )
    idata_m4.extend(pm.sample_posterior_predictive(idata_m4))


pm.model_to_graphviz(model_4)
```

```{code-cell} ipython3
az.summary(idata_m4, var_names=["beta_disp", "beta_feat", "beta_price", "alpha", "beta_individual"])
```

Note here that we have explicitly set a negative prior on price and recovered a parameter specification more in line with the basic theory of rational choice. The effect of price should have a negative impact on utility. The flexibility of priors here is key for incorporating theoretical knowledge about the process involved in choice.  

```{code-cell} ipython3
fig, axs = plt.subplots(1, 2, figsize=(20, 5))
ax = axs[0]
counts = c_df.groupby("choice")["choiceId"].count()
predicted_shares = az.extract(idata_m4, var_names=["p"]).mean(axis=2).mean(axis=0)
ci_lb = np.quantile(az.extract(idata_m4, var_names=["p"]).mean(axis=2), 0.025, axis=0)
ci_ub = np.quantile(az.extract(idata_m4, var_names=["p"]).mean(axis=2), 0.975, axis=0)
mean = np.mean(az.extract(idata_m4, var_names=["p"]).mean(axis=2), axis=0)
ax.scatter(ci_lb, ["sunshine", "keebler", "nabisco", "private"], color="k", s=2)
ax.scatter(ci_ub, ["sunshine", "keebler", "nabisco", "private"], color="k", s=2)
ax.scatter(
    counts / counts.sum(),
    ["sunshine", "keebler", "nabisco", "private"],
    label="Observed Shares",
    color="red",
)
ax.scatter(
    mean, ["sunshine", "keebler", "nabisco", "private"], label="Predicted Mean", color="purple"
)
ax.hlines(
    ["sunshine", "keebler", "nabisco", "private"],
    ci_lb,
    ci_ub,
    label="Predicted 95% Interval",
    color="black",
)
ax.legend()
ax.set_title("Observed V Predicted Shares")
az.plot_ppc(idata_m4, ax=axs[1])
axs[1].set_title("Posterior Predictive Checks")
ax.set_xlabel("Shares")
ax.set_ylabel("Crackers");
```

We can now also recover the differences among individuals estimated by the model for particular cracker choices. 

```{code-cell} ipython3
idata_m4
```

```{code-cell} ipython3
fig, axs = plt.subplots(1, 3, figsize=(20, 8))
axs = axs.flatten()
axs[0].bar(range(49), az.extract(idata_m4, var_names=["beta_individual"])[0, :, :].mean(axis=1))
axs[1].bar(range(49), az.extract(idata_m4, var_names=["beta_individual"])[1, :, :].mean(axis=1))
axs[2].bar(range(49), az.extract(idata_m4, var_names=["beta_individual"])[2, :, :].mean(axis=1))
axs[0].set_title("Individual Modifications of the Sunshine Intercept ")
axs[1].set_title("Individual Modifications of the Keebler Intercept ")
axs[2].set_title("Individual Modifications of the Nabisco Intercept ")
axs[1].set_xlabel("Individual ID")
axs[0].set_xlabel("Individual ID")
axs[2].set_xlabel("Individual ID")
axs[0].set_ylabel("Individual Beta Parameter");
```

We can see here the flexibility and richly parameterised possibilities for modelling individual choice of discrete options. These techniques are useful in a wide variety of domains from microeconomics, to marketing and product development. 

+++

## Authors
- Authored by [Nathaniel Forde](https://nathanielf.github.io/) in June 2023 

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

+++
