---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: pymc_causal
  language: python
  name: pymc_causal
---

(cfa_sem_notebook)=
# Confirmatory Factor Analysis and Structural Equation Models

:::{post} September, 2024
:tags: cfa, sem, regression, 
:category: intermediate, reference
:author: Nathaniel Forde
:::

+++

In the psychometrics literature the data is often derived from a strategically constructed survey aimed at a particular target phenomena. Some intuited, but not yet measured, concept that arguably plays a role in human action, motivation or sentiment. The relative “fuzziness” of the subject matter in psychometrics has had a catalyzing effect on the methodological rigour sought in the science. Survey designs are agonized over for correct tone and rhythm of sentence structure. Measurement scales are doubly checked for reliability and correctness. The literature is consulted and questions are refined. Analysis steps are justified and tested under a wealth of modelling routines. 

Model architectures are defined and refined to better express the hypothesized structures in the data-generating process. We will see how such due diligence leads to powerful and expressive models that grant us tractability on thorny questions of human affect.

```{code-cell} ipython3
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'  # high resolution figures
az.style.use("arviz-darkgrid")
rng = np.random.default_rng(42)
```

### Latent Constructs and Measurement

Our data is borrowed from work by Boris Mayer and Andrew Ellis found [here](https://methodenlehre.github.io/SGSCLM-R-course/cfa-and-sem-with-lavaan.html#structural-equation-modelling-sem). They demonstrate CFA and SEM modelling with lavaan. We’ll load up their data. We have survey responses from ~300 individuals who have answered questions regarding their upbringing, self-efficacy and reported life-satisfaction. The hypothetical dependency structure in this life-satisfaction data-set posits a moderated relationship between scores related to life-satisfaction, parental and family support and self-efficacy. It is not a trivial task to be able to design a survey that can elicit answers plausibly mapped to each of these “factors” or themes, never mind finding a model of their relationship that can inform us as to the relative of impact of each on life-satisfaction outcomes.

First we'll pull out the data and examine some summary statistics.


```{code-cell} ipython3
df = pd.read_csv("../data/sem_data.csv")
df.head()
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(20, 7))
drivers = [c for c in df.columns if not c in ["region", "gender", "age", "ID"]]
corr_df = df[drivers].corr()
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(corr_df, annot=True, cmap="Blues", ax=ax, center=0, mask=mask)
ax.set_title("Sample Correlations between indicator Metrics")
fig, ax = plt.subplots(figsize=(20, 7))
sns.heatmap(df[drivers].cov(), annot=True, cmap="Blues", ax=ax, center=0, mask=mask)
ax.set_title("Sample Covariances between indicator Metrics");
```

Next we'll plot the pairplot to visualise the nature of the correlations

```{code-cell} ipython3
ax = sns.pairplot(df[drivers], kind="reg", corner=True, diag_kind="kde")
plt.suptitle("Pair Plot of Indicator Metrics with Regression Fits", fontsize=30);
```

## Measurement Models

+++

A measurement model is a key component within the more general structural equation model. A measurement model specifies the relationships between observed variables (typically survey items or indicators) and their underlying latent constructs (often referred to as factors or latent variables). We start our presentation of SEM models more generally by focusing on a type of measurement model with it's own history - the confirmatory factor model (CFA) which specifies a particular structure to the relationships between our indicator variables and the latent factors. It is this structure which is up for confirmation in our modelling. 

We'll start by fitting a "simple" CFA model in `PyMC` to demonstrate how the pieces fit together, we'll then expand our focus. Here we ignore the majority of our indicator variables and focus on the idea that there are two latent constructs: (1) Social Self-efficacy and (2) Life Satisfaction. 

We're aiming to articulate a mathematical structure where our indicator variables $y_{ij}$ are determined by a latent factor $\text{Ksi}_{j}$ through an estimated factor loading $\lambda_{ij}$.  Functionally we have a set of equations with error terms $\psi_i$

$$ y_{1} = \tau_{1}  + \lambda_{11}\text{Ksi}_{1} + \psi_{1}  \\ 
y_{2} = \tau_{2}  + \lambda_{21}\text{Ksi}_{1} + \psi_{2} \\
 ... \\
y_{n} = \tau_{n}  + \lambda_{n2}\text{Ksi}_{2} + \psi_{3} 
$$ 

The goal is to articulate the relationship between the different factors in terms of the covariances between these latent terms and estimate the relationships each latent factor has with the manifest indicator variables. At a high level, we're saying the joint distribution can be represented through conditionalisation in the following schema

$$p(x_{i}.....x_{n} | \text{Ksi}, \Psi, \tau, \Lambda) \sim Normal(\tau + \Lambda\cdot \text{Ksi}, \Psi) $$

We will show how to build these structures into our model below

```{code-cell} ipython3
# Set up coordinates for appropriate indexing of latent factors
coords = {
    "obs": list(range(len(df))),
    "indicators": ["se_social_p1", "se_social_p2", "se_social_p3", "ls_p1", "ls_p2", "ls_p3"],
    "indicators_1": ["se_social_p1", "se_social_p2", "se_social_p3"],
    "indicators_2": ["ls_p1", "ls_p2", "ls_p3"],
    "latent": ["SE_SOC", "LS"],
}


obs_idx = list(range(len(df)))
with pm.Model(coords=coords) as model:

    # Set up Factor Loadings
    lambdas_ = pm.Normal("lambdas_1", 1, 10, dims=("indicators_1"))
    # Force a fixed scale on the factor loadings for factor 1
    lambdas_1 = pm.Deterministic(
        "lambdas1", pt.set_subtensor(lambdas_[0], 1), dims=("indicators_1")
    )
    lambdas_ = pm.Normal("lambdas_2", 1, 10, dims=("indicators_2"))
    # Force a fixed scale on the factor loadings for factor 2
    lambdas_2 = pm.Deterministic(
        "lambdas2", pt.set_subtensor(lambdas_[0], 1), dims=("indicators_2")
    )

    # Specify covariance structure between latent factors
    kappa = 0
    sd_dist = pm.Exponential.dist(1.0, shape=2)
    chol, _, _ = pm.LKJCholeskyCov("chol_cov", n=2, eta=2, sd_dist=sd_dist, compute_corr=True)
    ksi = pm.MvNormal("ksi", kappa, chol=chol, dims=("obs", "latent"))

    # Construct Pseudo Observation matrix based on Factor Loadings
    tau = pm.Normal("tau", 3, 10, dims="indicators")
    m1 = tau[0] + ksi[obs_idx, 0] * lambdas_1[0]
    m2 = tau[1] + ksi[obs_idx, 0] * lambdas_1[1]
    m3 = tau[2] + ksi[obs_idx, 0] * lambdas_1[2]
    m4 = tau[3] + ksi[obs_idx, 1] * lambdas_2[0]
    m5 = tau[4] + ksi[obs_idx, 1] * lambdas_2[1]
    m6 = tau[5] + ksi[obs_idx, 1] * lambdas_2[2]

    mu = pm.Deterministic("mu", pm.math.stack([m1, m2, m3, m4, m5, m6]).T)

    ## Error Terms
    Psi = pm.InverseGamma("Psi", 5, 10, dims="indicators")

    # Likelihood
    _ = pm.Normal(
        "likelihood",
        mu,
        Psi,
        observed=df[
            ["se_social_p1", "se_social_p2", "se_social_p3", "ls_p1", "ls_p2", "ls_p3"]
        ].values,
    )

    idata = pm.sample(
        nuts_sampler="numpyro", target_accept=0.95, idata_kwargs={"log_likelihood": True}
    )
    idata.extend(pm.sample_posterior_predictive(idata))

pm.model_to_graphviz(model)
```

### Meausurement Model Structure

We can now see how the covariance structure among the latent constructs is integral piece of the overarching model design which is fed forward into our pseudo regression components and weighted with the respective lambda terms. 

```{code-cell} ipython3
az.summary(idata, var_names=["lambdas1", "lambdas2"])
```

These factor loadings are generally important to interpret in terms of construct validity. Because we've specified one of the indicator variables to be fixed at 1, the other indicators which load on that factor should have a loading coefficient in broadly the same scale as the fixed point indicator that defines the construct scale. We're looking for consistency among the loadings to assess whether the indicators are reliable measures of the construct.

```{code-cell} ipython3
idata
```

Let's plot the trace diagnostics to validate the sampler has converged well to the posterior distribution. 

```{code-cell} ipython3
az.plot_trace(idata, var_names=["lambdas1", "lambdas2", "tau", "Psi", "ksi"]);
```

### Sampling the Latent Constructs

One thing to highlight in particular about the Bayesian manner of fitting CFA and SEM models is that we now have access to the posterior distribution of the latent quantities. These samples can offer insight into particular individuals in our survey that is harder to glean from the multivariate presentation of the manifest variables. 

```{code-cell} ipython3
fig, axs = plt.subplots(1, 2, figsize=(20, 9))
axs = axs.flatten()
ax1 = axs[0]
ax2 = axs[1]
az.plot_forest(
    idata,
    var_names=["ksi"],
    combined=True,
    ax=ax1,
    colors="forestgreen",
    coords={"latent": ["SE_SOC"]},
)
az.plot_forest(
    idata, var_names=["ksi"], combined=True, ax=ax2, colors="slateblue", coords={"latent": ["LS"]}
)
ax1.set_yticklabels([])
ax1.set_xlabel("SE_SOCIAL")
ax2.set_yticklabels([])
ax2.set_xlabel("LS")
ax1.axvline(-2, color="red")
ax2.axvline(-2, color="red")
ax1.set_title("Individual Social Self Efficacy \n On Latent Factor SE_SOCIAL")
ax2.set_title("Individual Life Satisfaction Metric \n On Latent Factor LS")
plt.show();
```

In this way we can identify and zero-in on individuals that appear to be outliers on one or more of the latent constructs.

+++

### Posterior Predictive Checks

As in more traditional Bayesian modelling, a core component of model evaluation is the assessment of the posterior predictive distribution i.e. the implied outcome distribution. Here too we can pull out draws against each of the indicator variables to assess for coherence and adequacy. 

```{code-cell} ipython3
def make_ppc(
    idata,
    samples=100,
    drivers=["se_social_p1", "se_social_p2", "se_social_p3", "ls_p1", "ls_p2", "ls_p3"],
    dims=(2, 3),
):
    fig, axs = plt.subplots(dims[0], dims[1], figsize=(20, 10))
    axs = axs.flatten()
    for i in range(len(drivers)):
        for j in range(samples):
            temp = az.extract(idata["posterior_predictive"].sel({"likelihood_dim_3": i}))[
                "likelihood"
            ].values[:, j]
            temp = pd.DataFrame(temp, columns=["likelihood"])
            if j == 0:
                axs[i].hist(df[drivers[i]], alpha=0.3, ec="black", bins=20, label="Observed Scores")
                axs[i].hist(
                    temp["likelihood"], color="purple", alpha=0.1, bins=20, label="Predicted Scores"
                )
            else:
                axs[i].hist(df[drivers[i]], alpha=0.3, ec="black", bins=20)
                axs[i].hist(temp["likelihood"], color="purple", alpha=0.1, bins=20)
            axs[i].set_title(f"Posterior Predictive Checks {drivers[i]}")
            axs[i].legend()
    plt.tight_layout()
    plt.show()


make_ppc(idata)
del idata
```

### Intermediate Cross-Loading Model

The idea of a measurment is maybe a little opaque when we only see models that fit well. Instead we want to briefly show how a in-apt measurement model gets reflected in the estimated parameters for the factor loadings. Here we specify a measurement model which attempts to couple the `se_social` and `sup_parents` indicators and bundle them into the same factor. 

```{code-cell} ipython3
coords = {
    "obs": list(range(len(df))),
    "indicators": [
        "se_social_p1",
        "se_social_p2",
        "se_social_p3",
        "sup_parents_p1",
        "sup_parents_p2",
        "sup_parents_p3",
        "ls_p1",
        "ls_p2",
        "ls_p3",
    ],
    ## Attempt Cross-Loading of two metric types on one factor
    "indicators_1": [
        "se_social_p1",
        "se_social_p2",
        "se_social_p3",
        "sup_parents_p1",
        "sup_parents_p2",
        "sup_parents_p3",
    ],
    "indicators_2": ["ls_p1", "ls_p2", "ls_p3"],
    "latent": ["SE_SOC", "LS"],
}


obs_idx = list(range(len(df)))
with pm.Model(coords=coords) as model:

    Psi = pm.InverseGamma("Psi", 5, 10, dims="indicators")
    lambdas_ = pm.Normal("lambdas_1", 1, 10, dims=("indicators_1"))
    # Force a fixed scale on the factor loadings for factor 1
    lambdas_1 = pm.Deterministic(
        "lambdas1", pt.set_subtensor(lambdas_[0], 1), dims=("indicators_1")
    )
    lambdas_ = pm.Normal("lambdas_2", 1, 10, dims=("indicators_2"))
    # Force a fixed scale on the factor loadings for factor 2
    lambdas_2 = pm.Deterministic(
        "lambdas2", pt.set_subtensor(lambdas_[0], 1), dims=("indicators_2")
    )
    tau = pm.Normal("tau", 3, 10, dims="indicators")
    # Specify covariance structure between latent factors
    kappa = 0
    sd_dist = pm.Exponential.dist(1.0, shape=2)
    chol, _, _ = pm.LKJCholeskyCov("chol_cov", n=2, eta=2, sd_dist=sd_dist, compute_corr=True)
    ksi = pm.MvNormal("ksi", kappa, chol=chol, dims=("obs", "latent"))

    # Construct Observation matrix
    m1 = tau[0] + ksi[obs_idx, 0] * lambdas_1[0]
    m2 = tau[1] + ksi[obs_idx, 0] * lambdas_1[1]
    m3 = tau[2] + ksi[obs_idx, 0] * lambdas_1[2]
    m4 = tau[3] + ksi[obs_idx, 0] * lambdas_1[3]
    m5 = tau[4] + ksi[obs_idx, 0] * lambdas_1[4]
    m6 = tau[5] + ksi[obs_idx, 0] * lambdas_1[5]
    m7 = tau[3] + ksi[obs_idx, 1] * lambdas_2[0]
    m8 = tau[4] + ksi[obs_idx, 1] * lambdas_2[1]
    m9 = tau[5] + ksi[obs_idx, 1] * lambdas_2[2]

    mu = pm.Deterministic("mu", pm.math.stack([m1, m2, m3, m4, m5, m6, m7, m8, m9]).T)
    _ = pm.Normal(
        "likelihood",
        mu,
        Psi,
        observed=df[
            [
                "se_social_p1",
                "se_social_p2",
                "se_social_p3",
                "sup_parents_p1",
                "sup_parents_p2",
                "sup_parents_p3",
                "ls_p1",
                "ls_p2",
                "ls_p3",
            ]
        ].values,
    )

    idata = pm.sample(
        draws=10000,
        nuts_sampler="numpyro",
        target_accept=0.99,
        idata_kwargs={"log_likelihood": True},
        random_seed=114,
    )
    idata.extend(pm.sample_posterior_predictive(idata))

pm.model_to_graphviz(model)
```

Again our model samples well but the parameter estimates suggest that there is some inconsistency between the scale on which we're trying to force both sets of metrics. 

```{code-cell} ipython3
az.summary(idata, var_names=["lambdas1", "lambdas2"])
```

```{code-cell} ipython3
fig, axs = plt.subplots(1, 2, figsize=(20, 9))
axs = axs.flatten()
az.plot_energy(idata, ax=axs[0])
az.plot_forest(idata, var_names=["lambdas1"], combined=True, ax=axs[1]);
```

This hints at a variety of measurement model misspecification and should force us back to the drawing board. 

+++

## Full Measurement Model

```{code-cell} ipython3
drivers = [
    "se_acad_p1",
    "se_acad_p2",
    "se_acad_p3",
    "se_social_p1",
    "se_social_p2",
    "se_social_p3",
    "sup_friends_p1",
    "sup_friends_p2",
    "sup_friends_p3",
    "sup_parents_p1",
    "sup_parents_p2",
    "sup_parents_p3",
    "ls_p1",
    "ls_p2",
    "ls_p3",
]

coords = {
    "obs": list(range(len(df))),
    "indicators": drivers,
    "indicators_1": ["se_acad_p1", "se_acad_p2", "se_acad_p3"],
    "indicators_2": ["se_social_p1", "se_social_p2", "se_social_p3"],
    "indicators_3": ["sup_friends_p1", "sup_friends_p2", "sup_friends_p3"],
    "indicators_4": ["sup_parents_p1", "sup_parents_p2", "sup_parents_p3"],
    "indicators_5": ["ls_p1", "ls_p2", "ls_p3"],
    "latent": ["SE_ACAD", "SE_SOCIAL", "SUP_F", "SUP_P", "LS"],
    "latent1": ["SE_ACAD", "SE_SOCIAL", "SUP_F", "SUP_P", "LS"],
}

obs_idx = list(range(len(df)))
with pm.Model(coords=coords) as model:

    Psi = pm.InverseGamma("Psi", 5, 10, dims="indicators")
    lambdas_ = pm.Normal("lambdas_1", 1, 10, dims=("indicators_1"))
    lambdas_1 = pm.Deterministic(
        "lambdas1", pt.set_subtensor(lambdas_[0], 1), dims=("indicators_1")
    )
    lambdas_ = pm.Normal("lambdas_2", 1, 10, dims=("indicators_2"))
    lambdas_2 = pm.Deterministic(
        "lambdas2", pt.set_subtensor(lambdas_[0], 1), dims=("indicators_2")
    )
    lambdas_ = pm.Normal("lambdas_3", 1, 10, dims=("indicators_3"))
    lambdas_3 = pm.Deterministic(
        "lambdas3", pt.set_subtensor(lambdas_[0], 1), dims=("indicators_3")
    )
    lambdas_ = pm.Normal("lambdas_4", 1, 10, dims=("indicators_4"))
    lambdas_4 = pm.Deterministic(
        "lambdas4", pt.set_subtensor(lambdas_[0], 1), dims=("indicators_4")
    )
    lambdas_ = pm.Normal("lambdas_5", 1, 10, dims=("indicators_5"))
    lambdas_5 = pm.Deterministic(
        "lambdas5", pt.set_subtensor(lambdas_[0], 1), dims=("indicators_5")
    )
    tau = pm.Normal("tau", 3, 10, dims="indicators")
    kappa = 0
    sd_dist = pm.Exponential.dist(1.0, shape=5)
    chol, _, _ = pm.LKJCholeskyCov("chol_cov", n=5, eta=2, sd_dist=sd_dist, compute_corr=True)
    cov = pm.Deterministic("cov", chol.dot(chol.T), dims=("latent", "latent1"))
    ksi = pm.MvNormal("ksi", kappa, chol=chol, dims=("obs", "latent"))

    m0 = tau[0] + ksi[obs_idx, 0] * lambdas_1[0]
    m1 = tau[1] + ksi[obs_idx, 0] * lambdas_1[1]
    m2 = tau[2] + ksi[obs_idx, 0] * lambdas_1[2]
    m3 = tau[3] + ksi[obs_idx, 1] * lambdas_2[0]
    m4 = tau[4] + ksi[obs_idx, 1] * lambdas_2[1]
    m5 = tau[5] + ksi[obs_idx, 1] * lambdas_2[2]
    m6 = tau[6] + ksi[obs_idx, 2] * lambdas_3[0]
    m7 = tau[7] + ksi[obs_idx, 2] * lambdas_3[1]
    m8 = tau[8] + ksi[obs_idx, 2] * lambdas_3[2]
    m9 = tau[9] + ksi[obs_idx, 3] * lambdas_4[0]
    m10 = tau[10] + ksi[obs_idx, 3] * lambdas_4[1]
    m11 = tau[11] + ksi[obs_idx, 3] * lambdas_4[2]
    m12 = tau[12] + ksi[obs_idx, 4] * lambdas_5[0]
    m13 = tau[13] + ksi[obs_idx, 4] * lambdas_5[1]
    m14 = tau[14] + ksi[obs_idx, 4] * lambdas_5[2]

    mu = pm.Deterministic(
        "mu", pm.math.stack([m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14]).T
    )
    _ = pm.Normal("likelihood", mu, Psi, observed=df[drivers].values)

    idata_mm = pm.sample(
        draws=10000,
        nuts_sampler="numpyro",
        target_accept=0.98,
        tune=1000,
        idata_kwargs={"log_likelihood": True},
        random_seed=100,
    )
    idata_mm.extend(pm.sample_posterior_predictive(idata_mm))
```

```{code-cell} ipython3
fig, axs = plt.subplots(1, 2, figsize=(20, 9))
axs = axs.flatten()
az.plot_energy(idata_mm, ax=axs[0])
az.plot_forest(idata_mm, var_names=["lambdas1", "lambdas2", "lambdas3"], combined=True, ax=axs[1]);
```

```{code-cell} ipython3
def get_posterior_resids(idata, samples=100, metric="cov"):
    resids = []
    for i in range(100):
        if metric == "cov":
            model_cov = pd.DataFrame(
                az.extract(idata["posterior_predictive"])["likelihood"][:, :, i]
            ).cov()
            obs_cov = df[drivers].cov()
        else:
            model_cov = pd.DataFrame(
                az.extract(idata["posterior_predictive"])["likelihood"][:, :, i]
            ).corr()
            obs_cov = df[drivers].corr()
        model_cov.index = obs_cov.index
        model_cov.columns = obs_cov.columns
        residuals = model_cov - obs_cov
        resids.append(residuals.values.flatten())

    residuals_posterior = pd.DataFrame(pd.DataFrame(resids).mean().values.reshape(15, 15))
    residuals_posterior.index = obs_cov.index
    residuals_posterior.columns = obs_cov.index
    return residuals_posterior


residuals_posterior_cov = get_posterior_resids(idata_mm, 2500)
residuals_posterior_corr = get_posterior_resids(idata_mm, 2500, metric="corr")
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(20, 7))
mask = np.triu(np.ones_like(residuals_posterior_corr, dtype=bool))
ax = sns.heatmap(residuals_posterior_corr, annot=True, cmap="bwr", mask=mask)
ax.set_title("Residuals between Model Implied and Sample Correlations", fontsize=25);
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(20, 7))
ax = sns.heatmap(residuals_posterior_cov, annot=True, cmap="bwr", mask=mask)
ax.set_title("Residuals between Model Implied and Sample Covariances", fontsize=25);
```

```{code-cell} ipython3
make_ppc(idata_mm, 100, drivers=residuals_posterior_cov.columns, dims=(5, 3));
```

## Bayesian Structural Equation Models

```{code-cell} ipython3
drivers = [
    "se_acad_p1",
    "se_acad_p2",
    "se_acad_p3",
    "se_social_p1",
    "se_social_p2",
    "se_social_p3",
    "sup_friends_p1",
    "sup_friends_p2",
    "sup_friends_p3",
    "sup_parents_p1",
    "sup_parents_p2",
    "sup_parents_p3",
    "ls_p1",
    "ls_p2",
    "ls_p3",
]


def make_indirect_sem(priors):

    coords = {
        "obs": list(range(len(df))),
        "indicators": drivers,
        "indicators_1": ["se_acad_p1", "se_acad_p2", "se_acad_p3"],
        "indicators_2": ["se_social_p1", "se_social_p2", "se_social_p3"],
        "indicators_3": ["sup_friends_p1", "sup_friends_p2", "sup_friends_p3"],
        "indicators_4": ["sup_parents_p1", "sup_parents_p2", "sup_parents_p3"],
        "indicators_5": ["ls_p1", "ls_p2", "ls_p3"],
        "latent": ["SUP_F", "SUP_P"],
        "latent1": ["SUP_F", "SUP_P"],
        "latent_regression": ["SUP_F->SE_ACAD", "SUP_P->SE_ACAD", "SUP_F->SE_SOC", "SUP_P->SE_SOC"],
        "regression": ["SE_ACAD", "SE_SOCIAL", "SUP_F", "SUP_P"],
    }

    obs_idx = list(range(len(df)))
    with pm.Model(coords=coords) as model:

        Psi = pm.InverseGamma("Psi", 5, 10, dims="indicators")
        lambdas_ = pm.Normal(
            "lambdas_1", priors["lambda"][0], priors["lambda"][1], dims=("indicators_1")
        )
        lambdas_1 = pm.Deterministic(
            "lambdas1", pt.set_subtensor(lambdas_[0], 1), dims=("indicators_1")
        )
        lambdas_ = pm.Normal(
            "lambdas_2", priors["lambda"][0], priors["lambda"][1], dims=("indicators_2")
        )
        lambdas_2 = pm.Deterministic(
            "lambdas2", pt.set_subtensor(lambdas_[0], 1), dims=("indicators_2")
        )
        lambdas_ = pm.Normal(
            "lambdas_3", priors["lambda"][0], priors["lambda"][1], dims=("indicators_3")
        )
        lambdas_3 = pm.Deterministic(
            "lambdas3", pt.set_subtensor(lambdas_[0], 1), dims=("indicators_3")
        )
        lambdas_ = pm.Normal(
            "lambdas_4", priors["lambda"][0], priors["lambda"][1], dims=("indicators_4")
        )
        lambdas_4 = pm.Deterministic(
            "lambdas4", pt.set_subtensor(lambdas_[0], 1), dims=("indicators_4")
        )
        lambdas_ = pm.Normal(
            "lambdas_5", priors["lambda"][0], priors["lambda"][1], dims=("indicators_5")
        )
        lambdas_5 = pm.Deterministic(
            "lambdas5", pt.set_subtensor(lambdas_[0], 1), dims=("indicators_5")
        )
        tau = pm.Normal("tau", 3, 10, dims="indicators")
        kappa = 0
        sd_dist = pm.Exponential.dist(1.0, shape=2)
        chol, _, _ = pm.LKJCholeskyCov(
            "chol_cov", n=2, eta=priors["eta"], sd_dist=sd_dist, compute_corr=True
        )
        cov = pm.Deterministic("cov", chol.dot(chol.T), dims=("latent", "latent1"))
        ksi = pm.MvNormal("ksi", kappa, chol=chol, dims=("obs", "latent"))

        # Regression Components
        beta_r = pm.Normal("beta_r", 0, priors["beta_r"], dims="latent_regression")
        beta_r2 = pm.Normal("beta_r2", 0, priors["beta_r2"], dims="regression")
        resid_chol, _, _ = pm.LKJCholeskyCov(
            "resid_chol", n=2, eta=priors["eta"], sd_dist=sd_dist, compute_corr=True
        )
        _ = pm.Deterministic("resid_cov", chol.dot(chol.T))
        sigmas_resid = pm.MvNormal("sigmas_resid", kappa, chol=resid_chol)

        # SE_ACAD ~ SUP_FRIENDS + SUP_PARENTS
        regression_se_acad = pm.Normal(
            "regr_se_acad",
            beta_r[0] * ksi[obs_idx, 0] + beta_r[1] * ksi[obs_idx, 1],
            sigmas_resid[0],
        )
        # SE_SOCIAL ~ SUP_FRIENDS + SUP_PARENTS

        regression_se_social = pm.Normal(
            "regr_se_social",
            beta_r[2] * ksi[obs_idx, 0] + beta_r[3] * ksi[obs_idx, 1],
            sigmas_resid[1],
        )

        # LS ~ SE_ACAD + SE_SOCIAL + SUP_FRIEND + SUP_PARENTS
        regression = pm.Normal(
            "regr",
            beta_r2[0] * regression_se_acad
            + beta_r2[1] * regression_se_social
            + beta_r2[2] * ksi[obs_idx, 0]
            + beta_r2[3] * ksi[obs_idx, 1],
            1,
        )

        m0 = tau[0] + regression_se_acad * lambdas_1[0]
        m1 = tau[1] + regression_se_acad * lambdas_1[1]
        m2 = tau[2] + regression_se_acad * lambdas_1[2]
        m3 = tau[3] + regression_se_social * lambdas_2[0]
        m4 = tau[4] + regression_se_social * lambdas_2[1]
        m5 = tau[5] + regression_se_social * lambdas_2[2]
        m6 = tau[6] + ksi[obs_idx, 0] * lambdas_3[0]
        m7 = tau[7] + ksi[obs_idx, 0] * lambdas_3[1]
        m8 = tau[8] + ksi[obs_idx, 0] * lambdas_3[2]
        m9 = tau[9] + ksi[obs_idx, 1] * lambdas_4[0]
        m10 = tau[10] + ksi[obs_idx, 1] * lambdas_4[1]
        m11 = tau[11] + ksi[obs_idx, 1] * lambdas_4[2]
        m12 = tau[12] + regression * lambdas_5[0]
        m13 = tau[13] + regression * lambdas_5[1]
        m14 = tau[14] + regression * lambdas_5[2]

        mu = pm.Deterministic(
            "mu", pm.math.stack([m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14]).T
        )
        _ = pm.Normal("likelihood", mu, Psi, observed=df[drivers].values)

        idata = pm.sample(
            10_000,
            chains=4,
            nuts_sampler="numpyro",
            target_accept=0.99,
            tune=2000,
            idata_kwargs={"log_likelihood": True},
            random_seed=110,
        )
        idata.extend(pm.sample_posterior_predictive(idata))

        return model, idata


model_sem0, idata_sem0 = make_indirect_sem(
    priors={"eta": 2, "lambda": [1, 1], "beta_r": 0.1, "beta_r2": 0.1}
)
model_sem1, idata_sem1 = make_indirect_sem(
    priors={"eta": 2, "lambda": [1, 1], "beta_r": 0.2, "beta_r2": 0.2}
)
model_sem2, idata_sem2 = make_indirect_sem(
    priors={"eta": 2, "lambda": [1, 1], "beta_r": 0.5, "beta_r2": 0.5}
)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(20, 15))
az.plot_forest(
    [idata_sem0, idata_sem1, idata_sem2],
    model_names=["SEM0", "SEM1", "SEM2"],
    var_names=["lambdas1", "lambdas2", "lambdas3", "lambdas4", "lambdas5", "beta_r", "beta_r2"],
    combined=True,
    ax=ax,
);
```

```{code-cell} ipython3
residuals_posterior_cov = get_posterior_resids(idata_sem0, 2500)
residuals_posterior_corr = get_posterior_resids(idata_sem0, 2500, metric="corr")

fig, ax = plt.subplots(figsize=(20, 7))
mask = np.triu(np.ones_like(residuals_posterior_corr, dtype=bool))
ax = sns.heatmap(residuals_posterior_corr, annot=True, cmap="bwr", center=0, mask=mask)
ax.set_title("Residuals between Model Implied and Sample Correlations", fontsize=25);
```

```{code-cell} ipython3
make_ppc(idata_sem0, 100, drivers=drivers, dims=(3, 5))
```

```{code-cell} ipython3
def calculate_effects(idata, var="SUP_P"):
    summary_df = az.summary(idata, var_names=["beta_r", "beta_r2"])
    # Indirect Paths
    ## VAR -> SE_SOC ->LS
    indirect_parent_soc = (
        summary_df.loc[f"beta_r[{var}->SE_SOC]"]["mean"]
        * summary_df.loc["beta_r2[SE_SOCIAL]"]["mean"]
    )

    ## VAR -> SE_SOC ->LS
    indirect_parent_acad = (
        summary_df.loc[f"beta_r[{var}->SE_ACAD]"]["mean"]
        * summary_df.loc["beta_r2[SE_ACAD]"]["mean"]
    )

    ## Total Indirect Effects
    total_indirect = indirect_parent_soc + indirect_parent_acad

    ## Total Effects
    total_effect = total_indirect + summary_df.loc[f"beta_r2[{var}]"]["mean"]

    return pd.DataFrame(
        [[indirect_parent_soc, indirect_parent_acad, total_indirect, total_effect]],
        columns=[
            f"{var} -> SE_SOC ->LS",
            f"{var} -> SE_ACAD ->LS",
            f"Total Indirect Effects {var}",
            f"Total Effects {var}",
        ],
    )
```

```{code-cell} ipython3
summary_p = pd.concat(
    [calculate_effects(idata_sem0), calculate_effects(idata_sem1), calculate_effects(idata_sem2)]
)

summary_p.index = ["SEM0", "SEM1", "SEM2"]
summary_p
```

```{code-cell} ipython3
summary_f = pd.concat(
    [
        calculate_effects(idata_sem0, "SUP_F"),
        calculate_effects(idata_sem1, "SUP_F"),
        calculate_effects(idata_sem2, "SUP_F"),
    ]
)

summary_f.index = ["SEM0", "SEM1", "SEM2"]
summary_f
```

## Authors
- Authored by [Nathaniel Forde](https://nathanielf.github.io/posts/post-with-code/CFA_AND_SEM/CFA_AND_SEM.html) in September 2024 

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
