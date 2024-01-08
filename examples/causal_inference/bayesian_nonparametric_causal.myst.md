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
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'  # high resolution figures
az.style.use("arviz-darkgrid")
rng = np.random.default_rng(42)
```

## Causal Inference and Propensity Scores

You always run the risk of being wrong. 

Your appetite for that risk is likely proportional to the strength of the claim you are making - less concerned with _casual_ assertions than __causal__ claims! There are few claims stronger than the assertion of a causal relationship. Each such statement is an inferential gamble underwritten by our faith in (occasionally arcance) methodology. 

In this notebook we will explain and motivate the usage of propensity scores in analysis of causal inference questions. We will avoid the impression of magic - our focus will be on the manner in which we (a) estimate propensity scores and (b) use them in the analysis of causal questions. We will see how they help avoid risks of selection bias in causal inference.

We will illustrate these patterns using two data sets: (i) the NHEFS data used througout Miguel Hernan's _Causal Inference: What If_ book and a second patient focused data set used throughout _Bayesian Nonparametrics for Causal Inference and Missing Data_ by Daniels, Linero and Roy. Throughout we will contrast the use of non-parametric BART models with simpler regression models for the estimation of propensity scores and causal effects.

+++

### NHEFS Data

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

### Propensity Score Model

+++

In this first step we define a model building function to capture the probability of treatment i.e. our propensity score for each observation. We specify two types of model which are to  be assessed. One which relies entirely on the Logistic regression and another which uses BART to model the relationships between and the covariates, the outcome and each other.

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
pm.model_to_graphviz(m_ps_logit)
```

```{code-cell} ipython3
m_ps_probit, idata_probit = make_propensity_model(X, t, bart=True, probit=False, samples=4000)
```

Once we have fitted these models we can compare how they attribute the propensity to treatment (in our case the propensity of quitting) to each and every such measured individual. One thing to note is how this sample seems to suggest a greater uncertainty of attributed score for the BART model.

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

```{code-cell} ipython3
ps_logit = idata_logit["posterior"]["p"].mean(dim=("chain", "draw")).round(2)
ps_logit
```

```{code-cell} ipython3
ps_probit = idata_probit["posterior"]["p"].mean(dim=("chain", "draw")).round(2)
ps_probit
```

### Using Propensity Scores: Weights and Pseudo Populations

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
    X["age"], X["wt71"], color=t.map(colors), s=(1 / ps_logit.values) * 10, ec="black", alpha=0.4
)
axs[2].set_xlabel("Age")
axs[3].set_xlabel("Age")
axs[3].set_ylabel("wt71")
axs[2].set_ylabel("wt71")
axs[2].set_title("Wt71~Age \n Size by IP Weights")
axs[3].set_title("Wt71~Age \n Size by IP Weights")
axs[3].scatter(
    X["age"], X["wt71"], color=t.map(colors), s=(1 / ps_probit.values) * 10, ec="black", alpha=0.4
)
red_patch = mpatches.Patch(color="red", label="Control")
blue_patch = mpatches.Patch(color="blue", label="Treated")
axs[2].legend(handles=[red_patch, blue_patch])
axs[3].legend(handles=[red_patch, blue_patch]);
```

### Estimated Expected Causal Effect (ATE)

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
    robust=True,
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
    if robust:
        p_of_t = X["trt"].mean()
        X["i_ps"] = np.where(t, (p_of_t / X["ps"]), (1 - p_of_t) / (1 - X["ps"]))
        n_ntrt = X[X["trt"] == 0].shape[0]
        n_trt = X[X["trt"] == 1].shape[0]
    else:
        X["ps"] = np.where(X["trt"], X["ps"], 1 - X["ps"])
        X["i_ps"] = 1 / X["ps"]
        n_ntrt = n_trt = len(X)
    X["log_y"] = y

    bins = np.arange(0.025, 0.85, 0.05)
    propensity0 = X[X["trt"] == 0]["ps"]
    propensity1 = X[X["trt"] == 1]["ps"]
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
    i_propensity0 = X[X["trt"] == 0]["i_ps"]
    i_propensity1 = X[X["trt"] == 1]["i_ps"]
    outcome_trt = X[X["trt"] == 1]["log_y"]
    outcome_ntrt = X[X["trt"] == 0]["log_y"]
    propensity0 = i_propensity0 * outcome_ntrt
    propensity1 = i_propensity1 * outcome_trt

    top0, _ = np.histogram(propensity0, bins=bins)
    top1, _ = np.histogram(propensity1, bins=bins)
    plot_weights(bins, top0, top1, ylims[1], axs[2])
    axs[2].set_ylabel("No. Patients", fontsize=14)
    axs[2].set_xlabel("Estimated IP Weighted Outcome \n Shifted", fontsize=14)
    axs[2].text(text_pos[0], text_pos[1], f"Control: E(Y) = {propensity0.sum() / n_ntrt}")
    axs[2].text(text_pos[0], text_pos[1] - 20, f"Treatment: E(Y) = {propensity1.sum() / n_trt}")
    axs[2].text(
        text_pos[0],
        text_pos[1] - 40,
        f"tau: E(Y(1) - Y(0)) = {propensity0.sum() / n_ntrt - propensity1.sum() / n_trt}",
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


make_plot(X, idata_logit)
```

```{code-cell} ipython3
def get_ate(X, t, y, i, idata, robust=False):
    X = X.copy()
    X["ps"] = idata["posterior"]["p"].stack(z=("chain", "draw"))[:, i].values
    X["trt"] = t
    if robust:
        p_of_t = X["trt"].mean()
        X["i_ps"] = np.where(t, (p_of_t / X["ps"]), (1 - p_of_t) / (1 - X["ps"]))
    else:
        X["ps"] = np.where(X["trt"], X["ps"], 1 - X["ps"])
        X["i_ps"] = 1 / X["ps"]
    X["outcome"] = y
    i_propensity0 = X[X["trt"] == 0]["i_ps"]
    i_propensity1 = X[X["trt"] == 1]["i_ps"]
    outcome_trt = X[X["trt"] == 1]["outcome"]
    outcome_ntrt = X[X["trt"] == 0]["outcome"]
    weighted_outcome_ntrt = i_propensity0 * outcome_ntrt
    weighted_outcome_trt = i_propensity1 * outcome_trt
    if robust:
        ntrt = weighted_outcome_ntrt.sum() / len(X[X["trt"] == 0])
        trt = weighted_outcome_trt.sum() / len(X[X["trt"] == 1])
    else:
        ntrt = weighted_outcome_ntrt.sum() / len(X)
        trt = weighted_outcome_trt.sum() / len(X)
    ate = ntrt - trt
    return [ate, trt, ntrt]


qs = range(4000)
ate_dist = [get_ate(X, t, y, q, idata_logit, robust=True) for q in qs]

ate_dist_df_logit = pd.DataFrame(ate_dist, columns=["ATE", "E(Y(1))", "E(Y(0))"])
ate_dist_df_logit.head()
```

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

```{code-cell} ipython3
make_plot(X, idata_probit)
```

```{code-cell} ipython3
ate_dist_probit = [get_ate(X, t, y, q, idata_probit, robust=True) for q in qs]
ate_dist_df_probit = pd.DataFrame(ate_dist_probit, columns=["ATE", "E(Y(1))", "E(Y(0))"])
ate_dist_df_probit.head()
```

```{code-cell} ipython3
plot_ate(ate_dist_df_probit)
```

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

### Regression with Propensity Scores


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
        mu = pm.math.dot(X, b)
        t_pred = pm.Normal("pred", mu, sigma, observed=y, dims="obs")

        idata = pm.sample_prior_predictive()
        idata.extend(pm.sample(samples, idata_kwargs={"log_likelihood": True}))
        idata.extend(pm.sample_posterior_predictive(idata))
    return model_ps_reg, idata


model_ps_reg, idata_ps_reg = make_prop_reg_model(t, y, idata_logit)
```

```{code-cell} ipython3
az.summary(idata_ps_reg)
```

```{code-cell} ipython3
model_ps_reg_bart, idata_ps_reg_bart = make_prop_reg_model(t, y, idata_probit)
```

```{code-cell} ipython3
az.summary(idata_ps_reg_bart)
```

## Health Expenditure Data

```{code-cell} ipython3
df = pd.read_csv("../data/meps_bayes_np_health.csv", index_col=["Unnamed: 0"])
df = df[df["totexp"] > 0].reset_index(drop=True)
df["log_y"] = np.log(df["totexp"] + 1000)
df["loginc"] = np.log(df["income"])
df["smoke"] = np.where(df["smoke"] == "No", 0, 1)
df
```

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
m_ps_expend, idata_expend = make_propensity_model(X, t, bart=True, probit=True, samples=1000)
```

```{code-cell} ipython3
make_plot(
    X,
    idata_expend,
    ylims=[(-100, 260), (-40, 270), (-130, 400)],
    lower_bins=np.arange(6, 45, 1),
    text_pos=(11, 200),
    robust=False,
)
```

```{code-cell} ipython3
qs = range(4000)
ate_dist = [get_ate(X, t, y, q, idata_expend, robust=True) for q in qs]
ate_dist_df = pd.DataFrame(ate_dist, columns=["ATE", "E(Y(1))", "E(Y(0))"])
ate_dist_df.head()
```

```{code-cell} ipython3
plot_ate(ate_dist_df)
```

```{code-cell} ipython3
model_ps_reg_expend, idata_ps_reg_expend = make_prop_reg_model(X, t, y, idata_expend)
```

```{code-cell} ipython3
az.summary(idata_ps_reg_expend, var_names=["b"])
```

```{code-cell} ipython3
model_ps_reg_expend_h, idata_ps_reg_expend_h = make_prop_reg_model(
    X,
    t,
    y,
    idata_expend,
    covariates=["phealth_Fair", "phealth_Good", "phealth_Poor", "phealth_Very Good"],
)
```

```{code-cell} ipython3
az.summary(idata_ps_reg_expend_h, var_names=["b"])
```

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
