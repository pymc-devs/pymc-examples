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

(bayes_np_causal)=
# Bayesian Non-parametric Causal Inference

:::{post} January, 2024
:tags: bart, propensity scores, dirichlet process regression  
:category: advanced, reference
:author: Nathaniel Forde
:::

```{code-cell} ipython3
import arviz as az
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pymc_bart as pmb
import pytensor.tensor as pt
import statsmodels.api as sm
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'  # high resolution figures
az.style.use("arviz-darkgrid")
rng = np.random.default_rng(42)
```

## Causal Inference and Propensity Scores

You always run the risk of being wrong. 

Your appetite for that risk is likely proportional to the strength of the claim you are making - less concerned with _casual_ assertions than __causal__ claims!  Each causal statement is an inferential gamble underwritten by our faith in (occasionally arcane) methodology. The conversational stakes are high - you risk your brittle credibility with false claims. There are few claims stronger than the assertion of a causal relationship, few claims more contestable. A naive world model, rich with tenuous connections and non-sequiter implications will expose you as an idiotic charlatan overly impressed by conspiracy theory. We don't want that for you. 

In this notebook we will explain and motivate the usage of propensity scores in analysis of causal inference questions. We will avoid the impression of magic - our focus will be on the manner in which we (a) estimate propensity scores and (b) use them in the analysis of causal questions. We will see how they help avoid risks of selection bias in causal inference and where they can go wrong. This method should be comfortable for the Bayeisan analyst who is familiar with weighting and re-weighting their claims with with information in the form priors. Propensity score weighting is just another opportunity to enrich your model with knowledge about the world.

We will illustrate these patterns using two data sets: (i) the NHEFS data used througout Miguel Hernan's _Causal Inference: What If_ book and a second patient focused data set used throughout _Bayesian Nonparametrics for Causal Inference and Missing Data_ by Daniels, Linero and Roy. Throughout we will contrast the use of non-parametric BART models with simpler regression models for the estimation of propensity scores and causal effects.

+++

:::{admonition} Note on Propensity Score Matching
:class: tip

Propensity scores are often synonymous with the technique of propensity score matching. We will not be covering this topic. It is a natural extension of propensity score modelling but to our mind introduces complexity through the requirements around matching algorithms and information loss with reduced sample size. 
:::

+++

## NHEFS Data

This data set from the National Health and Nutrition Examination survey records details of weight, activity and smoking habits of around 1500 individuals over two periods. The first period established a baseline and a follow-up period 10 years later. We will analyse whether the individual (`trt` == 1) quit smoking before the follow up visit. Each individuals' `outcome` represents a relative weight gain/loss comparing the two periods. 

```{code-cell} ipython3
nhefs_df = pd.read_csv("../data/nhefs.csv")
nhefs_df.head()
```

We might wonder who is represented in the survey responses and to what degree the measured differences in this survey corresspond the patterns in the wider population? If we look at the overall differences in outcomes:

```{code-cell} ipython3
raw_diff = nhefs_df.groupby("trt")[["outcome"]].mean()
print("Treatment Diff:", raw_diff["outcome"].iloc[0] - raw_diff["outcome"].iloc[1])
raw_diff
```

We see that there is some overall differences between the two groups, but splitting this out further we might worry that the differences are due to how the groups are imbalanced across the different covariate profiles in the treatment and control groups

```{code-cell} ipython3
strata_df = (
    nhefs_df.groupby(
        [
            "trt",
            "sex",
            "race",
            "active_1",
            "active_2",
            "education_2",
        ]
    )[["outcome"]]
    .agg(["count", "mean"])
    .rename({"age": "count"}, axis=1)
)

global_avg = nhefs_df["outcome"].mean()
strata_df["global_avg"] = global_avg
strata_df["diff"] = strata_df[("outcome", "mean")] - strata_df["global_avg"]
strata_df.reset_index(inplace=True)
strata_df.columns = [" ".join(col).strip() for col in strata_df.columns.values]
strata_df.style.background_gradient(axis=0)
```

We then take the average of the stratum specific averages to see a sharper distinction emerge. 

```{code-cell} ipython3
strata_expected_df = strata_df.groupby("trt")[["outcome count", "outcome mean", "diff"]].agg(
    {"outcome count": ["sum"], "outcome mean": "mean", "diff": "mean"}
)
print(
    "Treatment Diff:",
    strata_expected_df[("outcome mean", "mean")].iloc[0]
    - strata_expected_df[("outcome mean", "mean")].iloc[1],
)
strata_expected_df
```

This kind of exercise suggests that the manner in which our sample was constructed i.e. some aspect of the data generating process pulls some strata of the population away from adopting the treatment group. Their propensity for being treated is negatively keyed, so contaminates any causal inference claims. We should be legitimately concerned that failure to account for this kind of bias risks incorrect conclusions about (a) the direction and (b) the degree of effect that quitting has on weight.

+++

### Prepare Modelling Data

+++

We now simply prepare the data for modelling in a specific format, removing the `outcome` and `trt` from the covariate data X.

```{code-cell} ipython3
X = nhefs_df.copy()
y = nhefs_df["outcome"]
t = nhefs_df["trt"]
X = X.drop(["trt", "outcome"], axis=1)
X.head()
```

## Propensity Score Model

+++

In this first step we define a model building function to capture the probability of treatment i.e. our propensity score for each individual. 

We specify two types of model which are to  be assessed. One which relies entirely on the Logistic regression and another which uses BART to model the relationships between and the covariates and the outcome. The BART model has the benefit of using a tree-based algorithm to explore the interaction effects among the various strata in our sample data. 

Having a flexible model like BART is key to understanding what we are doing when we undertake inverse propensity weighting (IVPw) adjustments. The thought is that any given strata in our dataset will be described by a set of covariates. Types of individual will be represented by these covariate profiles - the attribute vector $X$. The share of observations within our data which are picked out by any given covariate profile represents a bias towards that type of individual. If our treatment status is such that individuals will more or less actively select themselves into the status, then a naive comparisons of differences between treatment groups and control groups will be misleading to the degree that we have over-represented types of individual (covariate profiles) in the population.

Randomisation solves this. But we can't always randomise.

What happens when we randomise? Randomisation of treatment status aims to ensure that we have a balance of covariate profiles across both groups. Additionally randomisation guarantees independence of the potential outcomes with respect to the treatment assignment mechanism. This helps avoid the selection-bias just discussed. Propensity scores are useful because they can help emulate _as-if_ random assignment of treatment status in the sample data through a specific transformation of the observed data. 

First we model the individual propensity scores as a function of the indivifua covariate profiles:

```{code-cell} ipython3
def make_propensity_model(X, t, bart=True, probit=True, samples=1000):
    coords = {"coeffs": list(X.columns), "obs": range(len(X))}
    with pm.Model(coords=coords) as model_ps:
        if bart:
            mu = pmb.BART("mu", X, t)
            if probit:
                p = pm.Deterministic("p", pm.math.invprobit(mu))
            else:
                p = pm.Deterministic("p", pm.math.invlogit(mu))
        else:
            b = pm.Normal("b", mu=0, sigma=1, dims="coeffs")
            mu = pm.math.dot(X, b)
            p = pm.Deterministic("p", pm.math.invlogit(mu))

        t_pred = pm.Bernoulli("t_pred", p=p, observed=t, dims="obs")

        idata = pm.sample_prior_predictive()
        idata.extend(pm.sample(samples, idata_kwargs={"log_likelihood": True}))
        idata.extend(pm.sample_posterior_predictive(idata))
    return model_ps, idata


m_ps_logit, idata_logit = make_propensity_model(X, t, bart=False, samples=1000)
```

```{code-cell} ipython3
m_ps_probit, idata_probit = make_propensity_model(X, t, bart=True, probit=False, samples=4000)
```

### Using Propensity Scores: Weights and Pseudo Populations

Once we have fitted these models we can compare how they attribute the propensity to treatment (in our case the propensity of quitting) to each and every such measured individual. One thing to note is how this sample seems to suggest a greater uncertainty of attributed score for the BART model. We have used the inverse probit link function when fitting our data. 

```{code-cell} ipython3
az.plot_forest(
    [idata_logit, idata_probit],
    var_names=["p"],
    coords={"p_dim_0": range(20)},
    figsize=(10, 13),
    combined=True,
    kind="ridgeplot",
    model_names=["Logistic Regression", "BART"],
    r_hat=True,
    ridgeplot_alpha=0.4,
);
```

These propensity scores can be pulled out and examined alongside the other covariates. But it's probably worth pausing here to explain how and why propensity scores are useful for accounting for selection bias. 

Firstly, and somewhat superficially, the propensity score is a dimension reduction technique. We take a complex covariate profile $X_{i}$ and reduce it to a scaler $p^{i}_{T}(X)$. It is tool for thinking about the potential outcomes of an individual under different treatment regimes. In a policy evaluation context it can help partial out the degree of incentives for policy adoption strata of the population. 

The pivotal idea is that we cannot license causal claims unless (i) the treatment assignment is independent of the covariate profiles i.e $T     \perp\!\!\!\perp X$  and (ii) the outcomes $Y(0)$, and $Y(1)$ and similarly conditionally independent of the treatement $T | X$. If these conditions hold, then we say that $T$ is __strongly ignorable__ given $X$. This also occasionally noted as the __unconfoundedness__ assumption. It is a theorem that if $T$ is strongly ignorable given $X$, then (ii) holds given $p_{T}(X)$. So valid statistical inference proceeds in a lower dimensional space using the propensity score as a proxy for the higher dimensional data. There is a great discussion of the details in Aronow and Miller's _Foundations of Agnostic Statistics_.

We are, as with all causal inference methods, making the assumption of __strong ignorability__. But given this assumption that we are measuring the right covariate profile to induce strong ignorability, then propensity scores can be used thoughtfully to underwrite causal claims. With observational data we cannot re-run the assignment mechanism but we can estimate it, and transform our data to proportionally weight the data summaries within each group so that the analysis is less effected by the over-representation of different strata in each group. This is what we hope to use the propensity scores to achieve. 


```{code-cell} ipython3
ps_logit = idata_logit["posterior"]["p"].mean(dim=("chain", "draw")).round(2)
ps_logit
```

```{code-cell} ipython3
ps_probit = idata_probit["posterior"]["p"].mean(dim=("chain", "draw")).round(2)
ps_probit
```

Here we plot the distribution of propensity scores under each model and show how the inverse of the propensity score weights would apply to the observed data points.

```{code-cell} ipython3
fig, axs = plt.subplots(2, 2, figsize=(20, 10))
axs = axs.flatten()

colors = {1: "blue", 0: "red"}
axs[0].hist(ps_logit, ec="black", color="slateblue", bins=20)
axs[1].hist(ps_probit, ec="black", color="skyblue", bins=20)
axs[0].set_xlim(0, 1)
axs[1].set_xlim(0, 1)
axs[0].set_title("Propensity Scores under Logistic Regression")
axs[1].set_title("Propensity Scores under Non-Parametric BART model \n with probit transform")
axs[2].scatter(
    X["age"], y, color=t.map(colors), s=(1 / ps_logit.values) * 10, ec="black", alpha=0.4
)
axs[2].set_xlabel("Age")
axs[3].set_xlabel("Age")
axs[3].set_ylabel("y")
axs[2].set_ylabel("y")
axs[2].set_title("y~Age \n Size by IP Weights")
axs[3].set_title("y~Age \n Size by IP Weights")
axs[3].scatter(
    X["age"], y, color=t.map(colors), s=(1 / ps_probit.values) * 10, ec="black", alpha=0.4
)
red_patch = mpatches.Patch(color="red", label="Control")
blue_patch = mpatches.Patch(color="blue", label="Treated")
axs[2].legend(handles=[red_patch, blue_patch])
axs[3].legend(handles=[red_patch, blue_patch]);
```

These weighting schemes can now be incorporated into various models of statistical summaries so as to "correct" the representation of covariate profiles across both groups. If an individual's propensity score is such that they are are highly likely to receive the treatment status e.g .95 then we want to downweight their importance if they occur in the treatment and upweight their importance if they appear in the control group. This makes sense because their high propensity score implies that similar individuals are already heavily present in the treatment group, but less likely to occur in the control group. Hence our corrective strategy re-weights their contribution to the summary statistics across each group. 

+++

### Robust and Doubly Robust Propensity Scores

We've been keen to stress that IVPw are a corrective. An opportunity for the causal analyst to put their finger on the scale and adjust the representative shares accorded to individuals in the treatment and control groups. As such there are no universal correctives, and naturally a variety of alternatives have arisen to fill gaps where simple propensity score weighting fails. We will see below a number of alternative weighting schemes. 

The main distinction to call out is between the raw propensity score weights and the doubly-robust theory of propensity score weights. 

Doubly robust methods are so named as they represent a compromise estimtator for causal effect that combines (i) a treatment assignment model (like propensity scores) and (ii) a more direct response outcome model. The method combines these two estimators in a way to generate a statistically unbiased estimate of the treatment effect. They work well because the way they are combined requires that only one of the models needs to be well-specified. 

+++

## Estimated Expected Causal Effect (ATE)

The next code block builds a set of functions to pull out an extract a sample from our posterior distribution of propensity scores and use this propensity score to reweight the observed outcome variable across our treatment and control groups to re-calculate the average treatment effect (ATE). It reweights our data using the inverse probability weighting scheme and then plots three views (1) the raw propensity scores across groups (2) the raw outcome distribution and (3) the re-weighted outcome distribution. 


```{code-cell} ipython3
def plot_weights(bins, top0, top1, ylim, ax):
    ax.axhline(0, c="gray", linewidth=1)
    ax.set_ylim(ylim)
    bars0 = ax.bar(bins[:-1] + 0.025, top0, width=0.04, facecolor="red", alpha=0.4)
    bars1 = ax.bar(bins[:-1] + 0.025, -top1, width=0.04, facecolor="blue", alpha=0.4)

    for bars in (bars0, bars1):
        for bar in bars:
            bar.set_edgecolor("black")

    for x, y in zip(bins, top0):
        ax.text(x + 0.025, y + 10, str(y), ha="center", va="bottom")

    for x, y in zip(bins, top1):
        ax.text(x + 0.025, -y - 10, str(y), ha="center", va="top")


def make_plot(
    X,
    idata,
    lower_bins=np.arange(1, 30, 1),
    ylims=[
        (-100, 370),
        (
            -40,
            100,
        ),
        (-50, 110),
    ],
    text_pos=(20, 80),
    ps=None,
    method="robust",
):
    X = X.copy()
    if ps is None:
        n_list = list(range(1000))
        ## Choose random ps score from posterior
        choice = np.random.choice(n_list, 1)[0]
        X["ps"] = idata["posterior"]["p"].stack(z=("chain", "draw"))[:, choice].values
    else:
        X["ps"] = ps
    X["trt"] = t
    propensity0 = X[X["trt"] == 0]["ps"]
    propensity1 = X[X["trt"] == 1]["ps"]
    if method == "robust":
        p_of_t = X["trt"].mean()
        X["i_ps"] = np.where(t, (p_of_t / X["ps"]), (1 - p_of_t) / (1 - X["ps"]))
        n_ntrt = X[X["trt"] == 0].shape[0]
        n_trt = X[X["trt"] == 1].shape[0]
    elif method == "raw":
        X["ps"] = np.where(X["trt"], X["ps"], 1 - X["ps"])
        X["i_ps"] = 1 / X["ps"]
        n_ntrt = n_trt = len(X)
    else:
        m0 = sm.OLS(y[t == 0], X[t == 0]).fit()
        m1 = sm.OLS(y[t == 1], X[t == 1]).fit()
        m0_pred = m0.predict(X)
        m1_pred = m0.predict(X)
        X["trt"] = t
        X["y"] = y
        p_of_t = X["trt"].mean()
        X["i_ps"] = np.where(t, (p_of_t / X["ps"]), (1 - p_of_t) / (1 - X["ps"]))

        weighted_outcome0 = (1 - X["trt"]) * (X["y"] - m0_pred) / (1 - X["ps"]) + m0_pred
        weighted_outcome1 = X["trt"] * (X["y"] - m1_pred) / X["ps"] + m1_pred
    X["outcome"] = y

    bins = np.arange(0.025, 0.85, 0.05)
    top0, _ = np.histogram(propensity0, bins=bins)
    top1, _ = np.histogram(propensity1, bins=bins)

    fig, axs = plt.subplots(3, 1, figsize=(15, 15))
    axs = axs.flatten()

    plot_weights(bins, top0, top1, ylims[0], axs[0])
    axs[0].text(0.05, 230, "Control = 0")
    axs[0].text(0.05, -90, "Treatment = 1")

    axs[0].set_ylabel("No. Patients", fontsize=14)
    axs[0].set_xlabel("Estimated Propensity Score", fontsize=14)
    axs[0].set_title(
        "Inferred Propensity Scores and IP Weighted Outcome \n by Treatment and Control",
        fontsize=20,
    )

    bins = lower_bins

    outcome_trt = X[X["trt"] == 1]["outcome"]
    outcome_ntrt = X[X["trt"] == 0]["outcome"]
    i_propensity0 = X[X["trt"] == 0]["i_ps"]
    i_propensity1 = X[X["trt"] == 1]["i_ps"]

    axs[2].set_ylabel("No. Patients", fontsize=14)
    if method in ["raw", "robust"]:
        weighted_outcome1 = outcome_trt * i_propensity1
        weighted_outcome0 = outcome_ntrt * i_propensity0
        top0, _ = np.histogram(weighted_outcome0, bins=bins)
        top1, _ = np.histogram(weighted_outcome1, bins=bins)
        plot_weights(bins, top0, top1, ylims[1], axs[2])
        axs[2].set_xlabel("Estimated IP Weighted Outcome \n Shifted", fontsize=14)
        axs[2].text(text_pos[0], text_pos[1], f"Control: E(Y) = {weighted_outcome0.sum() / n_ntrt}")
        axs[2].text(
            text_pos[0], text_pos[1] - 20, f"Treatment: E(Y) = {weighted_outcome1.sum() / n_trt}"
        )
        axs[2].text(
            text_pos[0],
            text_pos[1] - 40,
            f"tau: E(Y(1) - Y(0)) = {weighted_outcome0.sum() / n_ntrt - weighted_outcome1.sum() / n_trt}",
        )
    else:
        top0, _ = np.histogram(weighted_outcome0, bins=bins)
        top1, _ = np.histogram(weighted_outcome1, bins=bins)
        plot_weights(bins, top0, top1, ylims[1], axs[2])
        trt = np.mean(X["trt"] * (X["y"] - m1_pred) / X["ps"] + m1_pred)
        ntrt = np.mean((1 - X["trt"]) * (X["y"] - m0_pred) / (1 - X["ps"]) + m0_pred)
        axs[2].set_xlabel("Estimated IP Weighted Outcome \n Shifted", fontsize=14)
        axs[2].text(text_pos[0], text_pos[1], f"Control: E(Y) = {ntrt}")
        axs[2].text(text_pos[0], text_pos[1] - 20, f"Treatment: E(Y) = {trt}")
        axs[2].text(
            text_pos[0],
            text_pos[1] - 40,
            f"tau: E(Y(1) - Y(0)) = {ntrt - trt}",
        )

    top0, _ = np.histogram(outcome_ntrt, bins=bins)
    top1, _ = np.histogram(outcome_trt, bins=bins)
    plot_weights(bins, top0, top1, ylims[2], axs[1])
    axs[1].set_ylabel("No. Patients", fontsize=14)
    axs[1].set_xlabel("Raw Outcome Measure", fontsize=14)
    axs[1].text(text_pos[0], text_pos[1], f"Control: E(Y) = {outcome_ntrt.mean()}")
    axs[1].text(text_pos[0], text_pos[1] - 20, f"Treatment: E(Y) = {outcome_trt.mean()}")
    axs[1].text(
        text_pos[0],
        text_pos[1] - 40,
        f"tau: E(Y(1) - Y(0)) = {outcome_trt.mean() - outcome_ntrt.mean()}",
    )
```

## The Logit Propensity Model

We plot the outcome and re-weighted outcome distribution using the robust propensity score estimation method.

```{code-cell} ipython3
make_plot(X, idata_logit, method="robust")
```

Next, and because we are Bayesians - we pull out and evaluate the posterior distribution of the ATE basd on the sampled propensity scores. We've seen a point estimate for the ATE above, but it's often more important in the causal inference context to understand the uncertainty in the estimate. 

```{code-cell} ipython3
def get_ate(X, t, y, i, idata, method="doubly_robust"):
    X = X.copy()
    X["ps"] = idata["posterior"]["p"].stack(z=("chain", "draw"))[:, i].values
    if method == "robust":
        X["trt"] = t
        p_of_t = X["trt"].mean()
        X["i_ps"] = np.where(t, (p_of_t / X["ps"]), (1 - p_of_t) / (1 - X["ps"]))
        i_propensity0 = X[X["trt"] == 0]["i_ps"]
        i_propensity1 = X[X["trt"] == 1]["i_ps"]
    elif method == "raw":
        X["trt"] = t
        X["ps"] = np.where(X["trt"], X["ps"], 1 - X["ps"])
        X["i_ps"] = 1 / X["ps"]
        i_propensity0 = X[X["trt"] == 0]["i_ps"]
        i_propensity1 = X[X["trt"] == 1]["i_ps"]
    else:
        ## model estimates of Doubly robust method
        m0 = sm.OLS(y[t == 0], X[t == 0]).fit()
        m1 = sm.OLS(y[t == 1], X[t == 1]).fit()
        m0_pred = m0.predict(X)
        m1_pred = m0.predict(X)
        X["trt"] = t
        X["y"] = y
    X["outcome"] = y
    outcome_trt = X[X["trt"] == 1]["outcome"]
    outcome_ntrt = X[X["trt"] == 0]["outcome"]
    if method == "robust":
        weighted_outcome_ntrt = i_propensity0 * outcome_ntrt
        weighted_outcome_trt = i_propensity1 * outcome_trt
        ntrt = weighted_outcome_ntrt.sum() / len(X[X["trt"] == 0])
        trt = weighted_outcome_trt.sum() / len(X[X["trt"] == 1])
    elif method == "raw":
        weighted_outcome_ntrt = i_propensity0 * outcome_ntrt
        weighted_outcome_trt = i_propensity1 * outcome_trt
        ntrt = weighted_outcome_ntrt.sum() / len(X)
        trt = weighted_outcome_trt.sum() / len(X)
    else:
        ## Compromise step between assignment model and outcome model
        ## Model mispecification forces the terms towards zero cancelling
        ## out misspecification.
        trt = np.mean(X["trt"] * (X["y"] - m1_pred) / X["ps"] + m1_pred)
        ntrt = np.mean((1 - X["trt"]) * (X["y"] - m0_pred) / (1 - X["ps"]) + m0_pred)
    ate = ntrt - trt
    return [ate, trt, ntrt]


qs = range(4000)
ate_dist = [get_ate(X, t, y, q, idata_logit, method="robust") for q in qs]

ate_dist_df_logit = pd.DataFrame(ate_dist, columns=["ATE", "E(Y(1))", "E(Y(0))"])
ate_dist_df_logit.head()
```

Next we plot the posterior distribution of the ATE. 

```{code-cell} ipython3
def plot_ate(ate_dist_df, xy=(-4.5, 250)):
    fig, axs = plt.subplots(1, 2, figsize=(20, 7))
    axs = axs.flatten()
    axs[0].hist(
        ate_dist_df["E(Y(1))"], bins=30, ec="black", color="blue", label="E(Y(1))", alpha=0.5
    )
    axs[0].hist(
        ate_dist_df["E(Y(0))"], bins=30, ec="black", color="red", label="E(Y(0))", alpha=0.7
    )
    axs[1].hist(ate_dist_df["ATE"], bins=30, ec="black", color="slateblue", label="ATE", alpha=0.6)
    ate = np.round(ate_dist_df["ATE"].mean(), 2)
    axs[1].axvline(ate, label="E(ATE)", linestyle="--", color="black")
    axs[1].annotate(f"E(ATE): {ate}", xy, fontsize=20, fontweight="bold")
    axs[1].set_title(f"Average Treatment Effect \n E(ATE): {ate}", fontsize=20)
    axs[0].set_title("E(Y) Distributions for Treated and Control", fontsize=20)
    axs[1].legend()
    axs[0].legend()


plot_ate(ate_dist_df_logit)
```

Note how this estimate of the treatment effect quite different than what we got taking the simple difference of averages across groups. 

+++

## The BART Propensity Model

Next we'll apply the doubly robust estimator to the propensity distribution achieved using the BART non-parametric model.

```{code-cell} ipython3
make_plot(X, idata_probit, method="doubly_robust", ylims=[(-150, 370), (-220, 150), (-50, 120)])
```

```{code-cell} ipython3
ate_dist_probit = [get_ate(X, t, y, q, idata_probit, method="doubly_robust") for q in qs]
ate_dist_df_probit = pd.DataFrame(ate_dist_probit, columns=["ATE", "E(Y(1))", "E(Y(0))"])
ate_dist_df_probit.head()
```

```{code-cell} ipython3
plot_ate(ate_dist_df_probit)
```

Note the tighter variance of the measures using the doubly robust method. 

+++

### Considerations Choosing between models

It is one thing to evalute change in average over the population, but we might want to allow for the idea of effect heterogenity across the population and as such the BART model is generally better at ensuring accurate predictions acros the deepter strata of our data. But the flexibility of machine learning models for prediction tasks do not guarantee that the propensity scores attributed across the sample are well calibrated to recover the true-treatment effects when used in causal effect estimation. We have to be careful in how we use the flexibility of non-parametric models in the causal context. 

First observe the hetereogenous accuracy induced by the BART model across increasingly narrow strata of our sample. 

```{code-cell} ipython3
fig, axs = plt.subplots(4, 2, figsize=(20, 25))
axs = axs.flatten()
az.plot_ppc(idata_logit, ax=axs[0])
az.plot_ppc(idata_probit, ax=axs[1])
idx1 = list((X[X["race"] == 1].index).values)
idx0 = list((X[X["race"] == 0].index).values)
az.plot_ppc(idata_logit, ax=axs[2], coords={"obs": idx1})
az.plot_ppc(idata_probit, ax=axs[3], coords={"obs": idx0})
idx1 = list((X[(X["race"] == 1) & (X["sex"] == 1)].index).values)
idx0 = list((X[(X["race"] == 0) & (X["sex"] == 1)].index).values)
az.plot_ppc(idata_logit, ax=axs[4], coords={"obs": idx1})
az.plot_ppc(idata_probit, ax=axs[5], coords={"obs": idx0})
idx1 = list((X[(X["race"] == 1) & (X["sex"] == 1) & (X["active_1"] == 1)].index).values)
idx0 = list((X[(X["race"] == 0) & (X["sex"] == 1) & (X["active_1"] == 1)].index).values)
az.plot_ppc(idata_logit, ax=axs[6], coords={"obs": idx1})
az.plot_ppc(idata_probit, ax=axs[7], coords={"obs": idx0})
axs[0].set_title("Overall PPC - Logit")
axs[1].set_title("Overall PPC - BART")
axs[2].set_title("Race Specific PPC - Logit")
axs[3].set_title("Race Specific PPC - BART")
axs[4].set_title("Race/Gender Specific PPC - Logit")
axs[5].set_title("Race/Gender Specific PPC - BART")
axs[6].set_title("Race/Gender/Active Specific PPC - Logit")
axs[7].set_title("Race/Gender/Active Specific PPC - BART")
plt.suptitle("Posterior Predictive Checks - Heterogenous Effects", fontsize=20);
```

Observations like this go along way to motivating the work in the Double ML causal inference paradigm. The model used to capture the outcome distribution or the propensity score distribution ought to be sensetive to variation across extremities of the data. We can see above that the predictive power of the simpler logistic regression model deterioriates as we progress down the partitions of the data. We will see an example below where this flexibility becomes a problem. 

+++

### Regression with Propensity Scores

Another perhaps more direct method of causal inference is to just use regression directly. Angrist and Pischke suggest that the familiar properties of regression make it more desirable, but concede that there is a role for propensity and that the methods can be combined by the cautious analyst. Here we'll show how we can combine the propensity score in a regression context to derive estimates of treatment effects. 

```{code-cell} ipython3
def make_prop_reg_model(X, t, y, idata_ps, covariates=None, samples=1000):
    ps = idata_ps["posterior"]["p"].mean(dim=("chain", "draw")).values
    X_temp = pd.DataFrame({"ps": ps, "trt": t, "trt*ps": t * ps})
    if covariates is None:
        X = X_temp
    else:
        X = pd.concat([X_temp, X[covariates]], axis=1)
    coords = {"coeffs": list(X.columns), "obs": range(len(X))}
    with pm.Model(coords=coords) as model_ps_reg:
        sigma = pm.HalfNormal("sigma", 1)
        b = pm.Normal("b", mu=0, sigma=1, dims="coeffs")
        X = pm.MutableData("X", X)
        mu = pm.math.dot(X, b)
        t_pred = pm.Normal("pred", mu, sigma, observed=y, dims="obs")

        idata = pm.sample_prior_predictive()
        idata.extend(pm.sample(samples, idata_kwargs={"log_likelihood": True}))
        idata.extend(pm.sample_posterior_predictive(idata))
    return model_ps_reg, idata


model_ps_reg, idata_ps_reg = make_prop_reg_model(X, t, y, idata_logit)
```

Fitting the regression model using the propensity as a dimensional reduction technique seems to work well here. We recover substantially the same treatment effect estimate as above. 

```{code-cell} ipython3
az.summary(idata_ps_reg)
```

```{code-cell} ipython3
model_ps_reg_bart, idata_ps_reg_bart = make_prop_reg_model(X, t, y, idata_probit)
```

```{code-cell} ipython3
az.summary(idata_ps_reg_bart)
```

### Causal Inference as Regression Imputation

Above we read-off the causal effect estimate as the coefficient on the treatment variable in our regression model. An arguably more direct approach uses the fitted regression models to impute the distribution of potential outcomes under different treatment regimes. In this way we have yet another perspective on causal inference 

```{code-cell} ipython3
X_mod = X.copy()
X_mod["ps"] = ps = idata_probit["posterior"]["p"].mean(dim=("chain", "draw")).values
X_mod["trt"] = 1
X_mod["trt*ps"] = X_mod["ps"] * X_mod["trt"]
with model_ps_reg_bart:
    # update values of predictors:
    pm.set_data({"X": X_mod[["ps", "trt", "trt*ps"]]})
    idata_trt = pm.sample_posterior_predictive(idata_ps_reg_bart)

idata_trt
```

```{code-cell} ipython3
X_mod = X.copy()
X_mod["ps"] = ps = idata_probit["posterior"]["p"].mean(dim=("chain", "draw")).values
X_mod["trt"] = 0
X_mod["trt*ps"] = X_mod["ps"] * X_mod["trt"]
with model_ps_reg_bart:
    # update values of predictors:
    pm.set_data({"X": X_mod[["ps", "trt", "trt*ps"]]})
    idata_ntrt = pm.sample_posterior_predictive(idata_ps_reg_bart)

idata_ntrt
```

```{code-cell} ipython3
idata_trt["posterior_predictive"]["pred"].mean()
```

```{code-cell} ipython3
idata_ntrt["posterior_predictive"]["pred"].mean()
```

```{code-cell} ipython3
idata_trt["posterior_predictive"]["pred"].mean() - idata_ntrt["posterior_predictive"]["pred"].mean()
```

All perspectives on the question of causal inference here seem broadly convergent. Next we'll see an example where the choices an analyst makes can go quite wrong. 

+++

## Health Expenditure Data

We will begin with looking a health-expenditure data set analysed in _Bayesian Nonparametrics for Causal Inference and Missing Data_ . The telling feature about this data set is the absence of obvious causal impact on expenditure due to the presence of smoking. We follow the authors and try and model the effect of `smoke` on the logged out `log_y`.

```{code-cell} ipython3
df = pd.read_csv("../data/meps_bayes_np_health.csv", index_col=["Unnamed: 0"])
df = df[df["totexp"] > 0].reset_index(drop=True)
df["log_y"] = np.log(df["totexp"] + 1000)
df["loginc"] = np.log(df["income"])
df["smoke"] = np.where(df["smoke"] == "No", 0, 1)
df
```

### Some basic Summary Statistics

Lets review the basic summary statistics and see how they change across various strata of the population

```{code-cell} ipython3
raw_diff = df.groupby("smoke")[["log_y"]].mean()
print("Treatment Diff:", raw_diff["log_y"].iloc[0] - raw_diff["log_y"].iloc[1])
raw_diff
```

```{code-cell} ipython3
pd.set_option("display.max_rows", 500)
strata_df = df.groupby(["smoke", "sex", "race", "phealth"])[["log_y"]].agg(["count", "mean"])

global_avg = df["log_y"].mean()
strata_df["global_avg"] = global_avg
# strata_df["diff"] = strata_df[("log_y", "mean")] - strata_df["global_avg"]
strata_df.reset_index(inplace=True)
strata_df.columns = [" ".join(col).strip() for col in strata_df.columns.values]
strata_df["diff"] = strata_df["log_y mean"] - strata_df["global_avg"]
strata_df.style.background_gradient(axis=0)
```

```{code-cell} ipython3
strata_expected_df = strata_df.groupby("smoke")[["log_y count", "log_y mean", "diff"]].agg(
    {"log_y count": ["sum"], "log_y mean": "mean", "diff": "mean"}
)
print(
    "Treatment Diff:",
    strata_expected_df[("log_y mean", "mean")].iloc[0]
    - strata_expected_df[("log_y mean", "mean")].iloc[1],
)
strata_expected_df
```

It certaintly seems that there is little to no impact due to our treatment effect in the data. Can we recover this insight using the method of inverse propensity score weighting? But first, let's do some basic exploratory data analysis to confirm our intuition. 

```{code-cell} ipython3
fig, axs = plt.subplots(2, 2, figsize=(20, 8))
axs = axs.flatten()
axs[0].hist(
    df[df["smoke"] == 1]["log_y"],
    alpha=0.3,
    density=True,
    bins=30,
    label="Smoker",
    ec="black",
    color="red",
)
axs[0].hist(
    df[df["smoke"] == 0]["log_y"],
    alpha=0.5,
    density=True,
    bins=30,
    label="Non-Smoker",
    ec="black",
    color="grey",
)
axs[1].hist(
    df[df["smoke"] == 1]["log_y"],
    density=True,
    bins=30,
    cumulative=True,
    histtype="step",
    label="Smoker",
    color="red",
)
axs[1].hist(
    df[df["smoke"] == 0]["log_y"],
    density=True,
    bins=30,
    cumulative=True,
    histtype="step",
    label="Non-Smoker",
    color="grey",
)
axs[2].scatter(df["loginc"], df["log_y"], c=df["smoke"], cmap="Set1", alpha=0.6)
axs[2].set_xlabel("Log Income")
axs[3].scatter(df["age"], df["log_y"], c=df["smoke"], cmap="Set1", alpha=0.6)

axs[3].set_title("Log Outcome ~ Age")
axs[2].set_title("Log Outcome ~ Log Income")
axs[3].set_xlabel("Age")
axs[0].set_title("Empirical Densities")
axs[0].legend()
axs[1].legend()
axs[1].set_title("Empirical Cumulative \n Densities");
```

The plots would seem to confirm undifferentiated nature of the outcome across the two groups. With some hint of difference at the outer quantiles  of the distribution. 

```{code-cell} ipython3
qs = np.linspace(0.05, 0.99, 100)
quantile_diff = (
    df.groupby("smoke")[["totexp"]]
    .quantile(qs)
    .reset_index()
    .pivot("level_1", "smoke", "totexp")
    .rename({0: "Non-Smoker", 1: "Smoker"}, axis=1)
    .assign(diff=lambda x: x["Non-Smoker"] - x["Smoker"])
    .reset_index()
    .rename({"level_1": "quantile"}, axis=1)
)

fig, axs = plt.subplots(1, 2, figsize=(20, 6))
axs[0].plot(quantile_diff["quantile"], quantile_diff["Smoker"])
axs[0].plot(quantile_diff["quantile"], quantile_diff["Non-Smoker"])
axs[0].set_title("Q-Q plot comparing \n Smoker and Non-Smokers")
axs[1].plot(quantile_diff["quantile"], quantile_diff["diff"])
axs[1].set_title("Differences across the Quantiles");
```

### What could possibly go Wrong?

```{code-cell} ipython3
dummies = pd.concat(
    [
        pd.get_dummies(df["seatbelt"], drop_first=True, prefix="seatbelt"),
        pd.get_dummies(df["marital"], drop_first=True, prefix="marital"),
        pd.get_dummies(df["race"], drop_first=True, prefix="race"),
        pd.get_dummies(df["sex"], drop_first=True, prefix="sex"),
        pd.get_dummies(df["phealth"], drop_first=True, prefix="phealth"),
    ],
    axis=1,
)
idx = df.sample(1000, random_state=100).index
X = pd.concat(
    [
        df[
            [
                "age",
            ]
        ],
        dummies,
    ],
    axis=1,
)
X = X.iloc[idx]
t = df.iloc[idx]["smoke"]
y = df.iloc[idx]["log_y"]
X
```

```{code-cell} ipython3
m_ps_expend_bart, idata_expend_bart = make_propensity_model(
    X, t, bart=True, probit=True, samples=4000
)
m_ps_expend_logit, idata_expend_logit = make_propensity_model(X, t, bart=False, samples=1000)
```

### Non-Parametric BART Model Propensity Model is mis-specified

The flexibility of the BART model fit is poorly calibrated to recover the average treatment effect. Let's evaluate the weighted outcome distributions under the robust IVPw estimate. 

```{code-cell} ipython3
ps = idata_expend_bart["posterior"]["p"].mean(dim=("chain", "draw")).values
make_plot(
    X,
    idata_expend_bart,
    ylims=[(-100, 340), (-70, 380), (-130, 420)],
    lower_bins=np.arange(6, 15, 1),
    text_pos=(11, 200),
    method="robust",
    ps=ps,
)
```

This is a __disastrous__ result. Evaluated at the expected values of the posterior propensity score distribution the robust IPW estimator of ATE suggests a substantial difference in the treatment and control groups. What is going on?

What happens if we look at the posterior ATE distributions under different estimators?

```{code-cell} ipython3
qs = range(4000)
ate_dist = [get_ate(X, t, y, q, idata_expend_bart, method="doubly_robust") for q in qs]
ate_dist_df_dr = pd.DataFrame(ate_dist, columns=["ATE", "E(Y(1))", "E(Y(0))"])

ate_dist = [get_ate(X, t, y, q, idata_expend_bart, method="robust") for q in qs]
ate_dist_df_r = pd.DataFrame(ate_dist, columns=["ATE", "E(Y(1))", "E(Y(0))"])

ate_dist_df_dr.head()
```

```{code-cell} ipython3
plot_ate(ate_dist_df_r)
```

Deriving ATE estimates across draws from the posterior distribution and averaging these seems to give a more sensible figure, but still inflated. If instead we use the doubly robust estimator we recover a much more sensible figure. 

```{code-cell} ipython3
plot_ate(ate_dist_df_dr)
```

It's worth here expanding on the theory of doubly robust estimation. We showed above the code for implementing the compromise between the treatment assignment estimator and the response or outcome estimator. But why is this useful? Consider again the functional form of the doubly robust estimator.

$$ \hat{Y(1)} = \frac{1}{n} \sum_{0}^{N} \Bigg[ \frac{T(Y - m_{1}(X))}{p_{T}(X)} + m_{1}(X) \Bigg] $$

$$ \hat{Y(0)} = \frac{1}{n} \sum_{0}^{N} \Bigg[ \frac{(1-T)(Y - m_{0}(X))}{(1-p_{T}(X))} + m_{0}(X) \Bigg] $$

It is not immediately intuitive as to how these formulas effect a compromise between the outcome model and the treatment assignment model. But consider the extreme cases first imagine our model $m_{1}$ is a perfect fit to our outcome $Y$, then the numerator of the fraction is 0 and we end up with an average of the model predictions. Instead imagine model $m_{1}$ is mis-specified and we have some error $\epsilon > 0$ in the numerator. If the propensity score model is accurate then in the treated class our denominator should be high... say $\sim N(.9, .1)$, and as such the estimator adds a number close to $\epsilon$ back to the $m_{1}$ prediction. Similar reasoning goes through for the $Y(0)$ case. So as long as one of the two models is well-specified this estimator can recover accurate unbiased treatment effects.

+++

### How does Regression Help?

We've just seen an example of how a mis-specfied machine learning model can wildly bias the causal estimates in a study. We've seen one means of fixing it, but how would things work out if we just tried simpler exploratory regression modelling?

```{code-cell} ipython3
model_ps_reg_expend, idata_ps_reg_expend = make_prop_reg_model(X, t, y, idata_expend_bart)
```

```{code-cell} ipython3
az.summary(idata_ps_reg_expend, var_names=["b"])
```

This model is clearly too simple. It recovers only the biased estimate due to the mis-specified BART propensity model, but what if we just used the propensity as another feature in our covariate profile. Let it add precision, but fit the model we think actually reflects the causal story.

```{code-cell} ipython3
model_ps_reg_expend_h, idata_ps_reg_expend_h = make_prop_reg_model(
    X,
    t,
    y,
    idata_expend_bart,
    covariates=["phealth_Fair", "phealth_Good", "phealth_Poor", "phealth_Very Good"],
)
```

```{code-cell} ipython3
az.summary(idata_ps_reg_expend_h, var_names=["b"])
```

This is much healthier and we can see that the propensity score feature in conjunction with the health factors to arrive at a sensible treatement effect estimate.

+++

### Quantile Models

```{code-cell} ipython3
dummies = pd.concat(
    [
        pd.get_dummies(df["seatbelt"], drop_first=True, prefix="seatbelt"),
        pd.get_dummies(df["marital"], drop_first=True, prefix="marital"),
        pd.get_dummies(df["race"], drop_first=True, prefix="race"),
        pd.get_dummies(df["sex"], drop_first=True, prefix="sex"),
        pd.get_dummies(df["phealth"], drop_first=True, prefix="phealth"),
    ],
    axis=1,
)
idx = df.sample(1000, random_state=100).index
X = pd.concat([df[["age", "bmi", "smoke"]], dummies], axis=1)
X = X.iloc[idx]
t = df.iloc[idx]["smoke"]
y = df.iloc[idx]["log_y"]
X
```

```{code-cell} ipython3
y_stack = np.stack([y] * 3)
quantiles = np.array([[0.9, 0.95, 0.975]]).T

with pm.Model() as model_q:
    X_data = pm.MutableData("X", X)
    mu = pmb.BART("mu", X_data, y, shape=(3, X_data.shape[0]))
    sigma = pm.HalfNormal("sigma", 1)
    obs = pm.AsymmetricLaplace("obs", mu=mu, b=sigma, q=quantiles, observed=y_stack)
    idata = pm.sample_prior_predictive()
    idata.extend(pm.sample())

pm.model_to_graphviz(model_q)
```

```{code-cell} ipython3
idata
```

```{code-cell} ipython3
order, ax = pmb.plot_variable_importance(idata, bartrv=mu, X=X, figsize=(20, 6))
ax.set_xticklabels(ax.get_xticklabels(), rotation=45);
```

```{code-cell} ipython3
az.plot_trace(idata, var_names=["mu", "sigma"])
```

### Inferred Quantile Causal Effects

```{code-cell} ipython3
X["smoke"] = 1
X["phealth_Fair"] = 0
X["phealth_Good"] = 0
X["phealth_Poor"] = 0
X["phealth_Very Good"] = 0
with model_q:
    # update values of predictors:
    pm.set_data({"X": X})
    idata_smoke = pm.sample_posterior_predictive(idata)

idata_smoke
```

```{code-cell} ipython3
X["smoke"] = 1
X["phealth_Fair"] = 1
X["phealth_Good"] = 0
X["phealth_Poor"] = 0
X["phealth_Very Good"] = 0
with model_q:
    # update values of predictors:
    pm.set_data({"X": X})
    idata_smoke_health = pm.sample_posterior_predictive(idata)

idata_smoke_health
```

```{code-cell} ipython3
X["smoke"] = 0
X["phealth_Fair"] = 1
X["phealth_Good"] = 0
X["phealth_Poor"] = 0
X["phealth_Very Good"] = 0
with model_q:
    # update values of predictors:
    pm.set_data({"X": X})
    idata_health = pm.sample_posterior_predictive(idata)

idata_health
```

```{code-cell} ipython3
X["smoke"] = 0
X["phealth_Fair"] = 0
X["phealth_Good"] = 0
X["phealth_Poor"] = 0
X["phealth_Very Good"] = 0

with model_q:
    # update values of predictors:
    pm.set_data({"X": X})
    idata_non_smoke = pm.sample_posterior_predictive(idata)

idata_non_smoke
```

```{code-cell} ipython3
smoke_quantiles = idata_smoke["posterior_predictive"].mean(dim=("chain", "draw", "obs_dim_3"))
nonsmoke_quantiles = idata_non_smoke["posterior_predictive"].mean(
    dim=("chain", "draw", "obs_dim_3")
)
smoke_health_quantiles = idata_smoke_health["posterior_predictive"].mean(
    dim=("chain", "draw", "obs_dim_3")
)
health_quantiles = idata_health["posterior_predictive"].mean(dim=("chain", "draw", "obs_dim_3"))
quantiles_df = pd.DataFrame(
    {
        "nonsmoke_quantiles": nonsmoke_quantiles["obs"].values,
        "smoke_quantiles": smoke_quantiles["obs"].values,
        "smoke_health_quantiles": smoke_health_quantiles["obs"].values,
        "health_quantiles": health_quantiles["obs"].values,
    },
    index=[0.975, 0.95, 0.90],
)

np.exp(quantiles_df)
```

### Propensity Score Modelling


```{code-cell} ipython3
dummies = pd.concat(
    [
        pd.get_dummies(df["seatbelt"], drop_first=True, prefix="seatbelt"),
        pd.get_dummies(df["marital"], drop_first=True, prefix="marital"),
        pd.get_dummies(df["race"], drop_first=True, prefix="race"),
        pd.get_dummies(df["sex"], drop_first=True, prefix="sex"),
        pd.get_dummies(df["phealth"], drop_first=True, prefix="phealth"),
    ],
    axis=1,
)
idx = df.sample(1000, random_state=100).index
X = pd.concat(
    [
        df[
            [
                "age",
            ]
        ],
        dummies,
    ],
    axis=1,
)
X = X.iloc[idx]
t = df.iloc[idx]["smoke"]
y = df.iloc[idx]["log_y"]
X
```

```{code-cell} ipython3
import pytensor.tensor as pt

p = idata_expend["posterior"]["p"].mean(dim=("chain", "draw")).values
p = np.where(t, p, 1 - p)
t = df.iloc[idx]["smoke"].reset_index(drop=True).values
# p = pd.DataFrame([p, t]).T.values
K = 30

coords = {"N": np.arange(X.shape[0]), "K": np.arange(K) + 1, "one": [1]}


def norm_cdf(z):
    return 0.5 * (1 + pt.erf(z / np.sqrt(2)))


def stick_breaking(v):
    return v * pt.concatenate(
        [pt.ones_like(v[:, :1]), pt.extra_ops.cumprod(1 - v, axis=1)[:, :-1]], axis=1
    )


with pm.Model(coords=coords) as model_dpr:
    ps = pm.MutableData("ps", p[:, np.newaxis])
    trt = pm.MutableData("trt", t[:, np.newaxis])
    alpha = pm.Normal("alpha", 0.0, 1.0, dims="K")
    beta = pm.Normal("beta", 0.0, 1.0, dims="K")
    beta1 = pm.Normal("beta1", 0.0, 1.0, dims="K")
    beta2 = pm.Normal("beta2", 0.0, 1.0, dims="K")
    v = pm.Deterministic("v", norm_cdf(alpha + ps * beta + trt * beta1 + (ps * trt) * beta2))
    w = pm.Deterministic("w", stick_breaking(v))
    gamma = pm.Normal("gamma", 0.0, 1.0, dims="K")
    delta = pm.Normal("delta", 0.0, 1.0, dims="K")
    delta1 = pm.Normal("delta1", 0.0, 1.0, dims="K")
    delta2 = pm.Normal("delta2", 0.0, 1.0, dims="K")
    mu1 = pm.Deterministic("mu1", gamma + ps * delta + trt * delta1 + (ps * trt) * delta2)
    tau = pm.Gamma("tau", 1.0, 1.0, dims="K")
    y_obs = pm.MutableData("y", y)
    obs = pm.NormalMixture("obs", w, mu1, tau=tau, observed=y_obs)

    idata_dpr = pm.sample_prior_predictive()
    idata_dpr.extend(pm.sample(2000, nuts_sampler="numpyro", target_accept=0.99, chains=2))


pm.model_to_graphviz(model_dpr)
```

```{code-cell} ipython3
idata_dpr["posterior"]["w"].mean(dim=("chain", "draw", "w_dim_0")).round(2)
```

```{code-cell} ipython3
az.summary(idata_dpr, var_names=["delta1"])
```

```{code-cell} ipython3
with model_dpr:
    idata_dpr.extend(pm.sample_posterior_predictive(idata_dpr))

idata_dpr
```

```{code-cell} ipython3
az.plot_trace(idata_dpr, var_names=["alpha", "gamma", "beta", "beta1", "delta", "delta1", "tau"]);
```

```{code-cell} ipython3
X["smoke"] = 1
X["phealth_Fair"] = 0
X["phealth_Good"] = 0
X["phealth_Poor"] = 0
X["phealth_Very Good"] = 0
with model_dpr:
    # update values of predictors:
    pm.set_data({"trt": np.ones_like(t)[:, np.newaxis]})
    idata_smoke = pm.sample_posterior_predictive(idata_dpr)

idata_smoke
```

```{code-cell} ipython3
with model_dpr:
    # update values of predictors:
    pm.set_data({"trt": np.zeros_like(t)[:, np.newaxis]})
    idata_non_smoke = pm.sample_posterior_predictive(idata_dpr)
```

```{code-cell} ipython3
idata_smoke
```

```{code-cell} ipython3
smoke_quantiles = idata_smoke["posterior_predictive"].quantile(
    [0.5, 0.9, 0.95, 0.975], dim=("chain", "draw", "obs_dim_2")
)
nonsmoke_quantiles = idata_non_smoke["posterior_predictive"].quantile(
    [0.5, 0.9, 0.95, 0.975], dim=("chain", "draw", "obs_dim_2")
)

nonsmoke_quantiles - smoke_quantiles
```

## Authors
- Authored by [Nathaniel Forde](https://nathanielf.github.io/) in January 2024 

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
