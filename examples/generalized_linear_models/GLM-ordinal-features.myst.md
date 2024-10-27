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

+++ {"id": "nwglFsshedmq"}

(GLM-ordinal-features.ipynb)=
# GLM-ordinal-features

:::{post} Oct 27, 2024
:tags: ordinal-features, ordinal-regression, glm, bayesian-workflow, r-datasets
:category: beginner, reference
:author: Jonathan Sedar
:::

+++ {"id": "R_cn0vgIJaf9"}

## Ordinal Exogenous Feature: Worked Example with Bayesian Workflow

Here we use an **ordinal exogenous predictor feature** within a model:

```
y ~ x + e
y: Numeric
x: Ordinal Category
```

... this is in contrast to estimating an **ordinal endogenous target feature**,
which we show in another notebook


**Disclaimer:**
+ This Notebook is a worked example only, it's not intended to be an academic reference
+ The theory and math may be incorrect, incorrectly notated, or incorrectly used
+ The code may contain errors, inefficiencies, legacy methods, and the text may have typos
+ Use at your own risk!

## Contents

+ [Discussion](#Discussion)

+ [Setup](#Setup)

+ [0. Curate Dataset](#0.-Curate-Dataset)

+ [1. Model A: The Wrong Way - Simple Linear Coefficients](#1.-Model-A:-The-Wrong-Way---Simple-Linear-Coefficients)

+ [2. Model B: A Better Way - Dirichlet Hyperprior Allocator](#2.-Model-B:-A-Better-Way---Dirichlet-Hyperprior-Allocator)

---

---

+++ {"id": "MW3UCyzRJaf-"}

# Discussion

+++ {"id": "w8amu0EQJaf_"}

## Problem Statement

+ Human action and economics is all about expressing our ordinal preferences between limited options in the real-world.
+ We often encounter situations (and datasets) where a predictior feature is an ordinal category, recorded either:
  + As a totally subjective opinion which might be different between observations e.g. "bad, good, better, way better,
    best, actually the best, magnificent"  - these are difficult to work with and a symptom of poor survey design
  + On a partially subjective, standardized scale e.g. "strongly agree, agree, disagree, strongly disagree" - this is
    the approach of the familar [Likert scale](https://en.wikipedia.org/wiki/Likert_scale)
  + As a summary binning of a metric scale e.g. binning ages into age-groups [<30, 30 - 60, 60+], or medical
    self-scoring "[0-10%, ..., 90-100%]" - these are typically a misuse of the metric because the data has been
    compressed: losing infomation, and reasoning for the binning and the choices of bin-edges are usually not given
+ This latter binning is common practice in many industries from insurance to health, and erroneously encourages
  modellers to incorporate such features as a categorical (very bad choice) or a numeric value (a subtly bad choice).

> Our problem statement is that when faced with ordinal features we want to:
>
> 1) **Infer** a series of cutpoints that transform the ordinals into a linear (polynomial) scale
>
> 2) **Predict** the endogenous feature as usual, having captured the information from the ordinals
    

## Data & Models Demonstrated
    
+ This notebook takes the opportunity to:
  + Demonstrate a general method using a constrained Dirichlet prior, based on
    {cite:p}burkner2018 
  + Using the same health dataset as that paper `ICFCoreSetCWP.RData` available 
    in an R package [ordPens](https://cran.r-project.org/src/contrib/ordPens_1.1.0.tar.gz )
  + Extend a pymc-specific example by
    Austin Rochford {cite:p}`rochford2018`
  + Demonstrate a reasonably complete Bayesian workflow {cite:p}`gelman2020bayesian` including 
    data curation and grabbing data from an RDataset which is characteristically ugly

+ This notebook is a partner to another notebook (TBD) where we estimate an **ordinal endogenous target feature**.

+++ {"id": "CemVRXjtJaf_"}

---

---

+++ {"id": "32SMEqWMJaf_"}

# Setup

+++

:::{include} ../extra_installs.md
:::

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: CQixNaaKJ8fH
outputId: ca76e80a-0950-4b7a-93ea-a86d62061caf
---
# uncomment to install in a Google Colab environment
# !pip install pyreadr watermark
```

```{code-cell} ipython3
---
id: KHonk2PdJagA
jupyter:
  outputs_hidden: false
---
from copy import deepcopy

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pyreadr
import pytensor.tensor as pt

from matplotlib import gridspec
from pymc.testing import assert_no_rvs

import warnings  # isort:skip # suppress seaborn, it's far too chatty

warnings.simplefilter(action="ignore", category=FutureWarning)  # isort:skip
import seaborn as sns
```

```{code-cell} ipython3
:id: 1bkE8AVFJagA

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

sns.set_theme(
    style="darkgrid",
    palette="muted",
    context="notebook",
    rc={"figure.dpi": 72, "savefig.dpi": 144, "figure.figsize": (12, 4)},
)

# set target_accept quite high to minimise divergences in mdlb
SAMPLE_KWS = dict(
    progressbar=True,
    draws=500,
    tune=2000,
    target_accept=0.8,
    idata_kwargs=dict(log_likelihood=True),
)

USE_LOCAL_DATASET = False
```

+++ {"id": "YemOGTYcJagA"}

---

---

+++ {"id": "APsFdavWJagA"}

# 0. Curate Dataset

+++ {"id": "fiQFFNqhJagB"}

## 0.1 Extract

+++ {"id": "pwNFAyKzJagB"}

Annoyingly but not suprisingly for an R project, despite being a small, simple table, the dataset is only available in a
highly proprietary and obscure, obsolete R binary format, so we'll download, unpack and store locally as a normal CSV file

```{code-cell} ipython3
:id: -hJ0BqsHJagB

if USE_LOCAL_DATASET:
    dfr = pd.read_csv("icf_core_set_cwp.csv", index_col="rownames")
else:
    import os
    import tarfile
    import urllib.request

    url = "https://cran.r-project.org/src/contrib/ordPens_1.1.0.tar.gz"
    filehandle, _ = urllib.request.urlretrieve(url)
    rbytes = tarfile.open(filehandle).extractfile(member="ordPens/data/ICFCoreSetCWP.RData").read()
    fn = "ICFCoreSetCWP.RData"
    with open(fn, "wb") as f:
        f.write(rbytes)
    dfr = pyreadr.read_r(fn)["ICFCoreSetCWP"]
    os.remove(fn)
    dfr.to_csv("icf_core_set_cwp.csv")
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 691
id: HbZvu2rDJagB
outputId: 3db23e4f-3c64-4680-d880-dae21a245409
---
print(dfr.shape)
display(pd.concat((dfr.describe(include="all").T, dfr.isnull().sum(), dfr.dtypes), axis=1))
display(dfr.head())
```

+++ {"id": "MOVXxEJPJagB"}

**Observe:**

+ Looks okay - if this was a prpper project we'd want to know what those cryptic column headings actually mean
+ For this purpose we'll only use a couplf of the features so will press ahead

+++ {"id": "M7oFbrrsJagB"}

## 0.2 Clean

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 364
id: 5kFxNzDQJagB
outputId: 4245cc14-6850-475b-b839-e969a03633f5
---
fts_austin = ["d450", "d455", "phcs"]
df = dfr[fts_austin].copy()
display(pd.concat((df.describe(include="all").T, df.isnull().sum(), df.dtypes), axis=1))
df.head()
```

+++ {"id": "GhWmbmkPJagB"}

### ~~0.2.1 Clean Observations~~

```{code-cell} ipython3
:id: g6-FYfnTJagB

# Not needed
```

+++ {"id": "v0lm92QsJagB"}

### 0.2.2 Clean Features

+++ {"id": "YDccQXALJagB"}

#### ~~0.2.2.1 Rename Features~~

+++ {"id": "DM_lzCesJagB"}

Nothing really needed, will rename the index when we force dtype and set index

+++ {"id": "9zAlkU82JagB"}

### ~~0.2.2.2 Correct Features~~

```{code-cell} ipython3
:id: MsVLdwshJagC

# Seems not needed
```

+++ {"id": "DxzUSnI4JagC"}

### 0.2.2.3 Force Datatypes

+++ {"id": "-arZiqyMJagC"}

##### Force `d450` to string representation and ordered categorical dtype (supplied as an int which is unhelpful)

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: dfIcUcOEJagC
outputId: cb6c455f-1220-4e82-a93d-01ae600e5aeb
---
ft = "d450"
idx = df[ft].notnull()
df.loc[idx, ft] = df.loc[idx, ft].apply(lambda x: f"c{x}")
df[ft].unique()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: Hk-DR6akJagC
outputId: f7bfd499-ddfe-4ddf-807d-307dddc3f158
---
lvls = ["c0", "c1", "c2", "c3"]
df[ft] = pd.Categorical(df[ft].values, categories=lvls, ordered=True)
df[ft].cat.categories
```

+++ {"id": "JZAtKDQMJagC"}

##### Force `d455` to string representation and ordered categorical dtype (supplied as an int which is unhelpful)

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: wzNpQLS9JagC
outputId: 84dea368-b39f-478e-c252-b1bf4ebf25dc
---
ft = "d455"
idx = df[ft].notnull()
df.loc[idx, ft] = df.loc[idx, ft].apply(lambda x: f"c{x}")
df[ft].unique()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: Yiwey6naJagC
outputId: 18197b5d-e556-475f-945e-8de8c46ad015
---
lvls = ["c0", "c1", "c2", "c3", "c4"]
df[ft] = pd.Categorical(df[ft].values, categories=lvls, ordered=True)
df[ft].cat.categories
```

+++ {"id": "I1yJMuuIJagC"}

### 0.2.2.4 Create and set indexes

```{code-cell} ipython3
:id: SpWZQVjKJagC

df0 = df.reset_index()
df0["oid"] = df0["rownames"].apply(lambda n: f"o{str(n).zfill(3)}")
df = df0.drop("rownames", axis=1).set_index("oid").sort_index()
assert df.index.is_unique
```

+++ {"id": "p_8ozyA2JagC"}

## 0.3 Very limited quick EDA

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 382
id: o81OeFEKJagC
outputId: 38c657f8-31b8-4fef-9b9a-d6dd3774681a
---
print(df.shape)
display(pd.concat((df.describe(include="all").T, df.isnull().sum(), df.dtypes), axis=1))
display(df.head())
```

+++ {"id": "k7wl-XBoJagD"}

### 0.3.1 Univariate

+++ {"id": "7lWkS0HjJagD"}

`Numerics`

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 186
id: FYfSlaV1JagD
outputId: 0ae59d07-3979-422d-8c5f-f691fa0cb81e
---
fts = ["phcs"]
v_kws = dict(data=df, cut=0)
cs = sns.color_palette(n_colors=len(fts))
f, axs = plt.subplots(len(fts), 1, figsize=(12, 1 + 1.4 * len(fts)), squeeze=False)
for i, ft in enumerate(fts):
    ax = sns.violinplot(x=ft, **v_kws, ax=axs[0][i], color=cs[i])
    _ = ax.set_title(ft)
_ = f.suptitle("Univariate numerics")
_ = f.tight_layout()
```

+++ {"id": "haVg8sLoLUpG"}

**Observe:**

+ Fairly well-behaved target feature, suitable for use

+++ {"id": "p43qjcvJJagH"}

### 0.3.2 Bivariate `phcs` vs `['d450', 'd455']`

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 441
id: H-cc0MBLJagH
outputId: fb2d2128-bb13-435b-8455-cb7ecf814854
---
def plot_numeric_vs_cat(df, ftnum="phcs", ftcat="d450") -> plt.figure:
    f = plt.figure(figsize=(12, 3))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], figure=f)
    ax0 = f.add_subplot(gs[0])
    ax1 = f.add_subplot(gs[1], sharey=ax0)
    _ = ax0.set(title=f"Distribution of `{ftnum}` wihin each `{ftcat}` category")
    _ = ax1.set(title=f"Count obs per `{ftcat}` category", ylabel=False)

    kws_box = dict(
        orient="h",
        showmeans=True,
        whis=(3, 97),
        meanprops=dict(markerfacecolor="w", markeredgecolor="#333333", marker="d", markersize=12),
    )

    _ = sns.boxplot(x=ftnum, y=ftcat, hue=ftcat, data=df, **kws_box, ax=ax0)
    _ = sns.countplot(y=ftcat, hue=ftcat, data=df, ax=ax1)
    _ = ax0.invert_yaxis()
    _ = ax1.yaxis.label.set_visible(False)
    _ = plt.setp(ax1.get_yticklabels(), visible=False)
    _ = f.suptitle(f"Summary stats for {len(df)} obs in dataset")
    _ = f.tight_layout()


f = plot_numeric_vs_cat(df, ftnum="phcs", ftcat="d450")
f = plot_numeric_vs_cat(df, ftnum="phcs", ftcat="d455")
```

+++ {"id": "NFP5H1oeJagH"}

**Observe:**

`phcs vs d450`:
+ `c0` wider and higher: possibly a catch-all category because heavily observed too
+ `c3` fewer counts

`phcs vs d455`:
+ `c1` and `c2` look very similar
+ `c4` fewer counts

+++ {"id": "1e8KEDMmJagH"}

## 0.4 Transform dataset to `dfx` for model input

+++ {"id": "v5ZQFkmhJagH"}

**IMPORTANT NOTE**

+ Reminder that Bayesian inferential methods **do not need** a `test` dataset (nor k-fold cross validation)
  to fit parameters. We also do not need a `holdout` (out-of-sample) dataset to evaluate model performance,
  because we can use in-sample PPC, LOO-PIT and ELPD evaluations
+ So we use the entire dataset `df` as our model input
+ Depending on the real-world model implementation we might:
  + Separate out a `holdout` set (even though we dont need it) to eyeball the predictive outputs, but here we have a summarized dataset, so this isn't possible nor suitable
  + Create a `forecast` set to demonstrate how we would use the model and it's predictive outputs in Production.

**NOTE:**

+ This is an abbreviated / simplified transformation process which still allows
  for the potential to add more features in future

+++ {"id": "4aWXz6j_JagH"}

Map ordinal categorical to an ordinal numeric (int) based on its preexisting categorical order

```{code-cell} ipython3
:id: 8C5fJCB_JagH

map_int_to_cat_d450 = dict(enumerate(df["d450"].cat.categories))
MAP_CAT_TO_INT_D450 = {v: k for k, v in map_int_to_cat_d450.items()}
df["d450_idx"] = df["d450"].map(MAP_CAT_TO_INT_D450).astype(int)
df["d450_num"] = df["d450_idx"].copy()

map_int_to_cat_d455 = dict(enumerate(df["d455"].cat.categories))
MAP_CAT_TO_INT_D455 = {v: k for k, v in map_int_to_cat_d455.items()}
df["d455_idx"] = df["d455"].map(MAP_CAT_TO_INT_D455).astype(int)
df["d455_num"] = df["d455_idx"].copy()
```

+++ {"id": "fQrHcXvfJagH"}

Transform (zscore and scale) numerics

```{code-cell} ipython3
:id: LcENp7I3JagH

fts_num = ["d450_num", "d455_num"]
fts_non_num = ["d450_idx", "d455_idx"]
fts_y = ["phcs"]
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 238
id: IJrcR2AkJagH
outputId: 84cb0688-5614-4068-cbde-230af0d08754
---
MNS = np.nanmean(df[fts_num], axis=0)
SDEVS = np.nanstd(df[fts_num], axis=0)
dfx_num = (df[fts_num] - MNS) / SDEVS

icpt = pd.Series(np.ones(len(df)), name="intercept", index=dfx_num.index)

# concat including y_idx which will be used as observed
dfx = pd.concat((df[fts_y], icpt, df[fts_non_num], dfx_num), axis=1)
dfx.sample(5, random_state=42)
```

+++ {"id": "BytWql8rJagH"}

## 0.5 Create `forecast` set and convert to `dffx` for model input

+++ {"id": "hdJJvqc0JagH"}

Create `forecast` set

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 238
id: _cs_cls9JagI
outputId: c919000d-fc4c-46be-9a6d-5b5fc5fc7266
---
dff = df.groupby(["d450", "d455"]).size().reset_index()[["d450", "d455"]]
lvls_d450 = ["c0", "c1", "c2", "c3"]
dff["d450"] = pd.Categorical(dff["d450"].values, categories=lvls_d450, ordered=True)
lvls_d455 = ["c0", "c1", "c2", "c3", "c4"]
dff["d455"] = pd.Categorical(dff["d455"].values, categories=lvls_d455, ordered=True)
dff["phcs"] = 0.0
dff["oid"] = [f"o{str(n).zfill(3)}" for n in range(len(dff))]
dff.set_index("oid", inplace=True)
dff.head()
```

+++ {"id": "W2cYxoGuJagI"}

Convert to `dffx` for model input

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 238
id: DBretpaKJagI
outputId: cf120edf-aaf6-4ae2-97f8-1cc6888310c7
---
dff["d450_idx"] = dff["d450"].map(MAP_CAT_TO_INT_D450).astype(int)
dff["d455_idx"] = dff["d455"].map(MAP_CAT_TO_INT_D455).astype(int)
dff["d450_num"] = dff["d450_idx"].copy()
dff["d455_num"] = dff["d455_idx"].copy()

dffx_num = (dff[fts_num] - MNS) / SDEVS
icptf = pd.Series(np.ones(len(dff)), name="intercept", index=dffx_num.index)

# # concat including y_idx which will be used as observed
dffx = pd.concat((dff[fts_y], icptf, dff[fts_non_num], dffx_num), axis=1)
dffx.sample(5, random_state=42)
```

+++ {"id": "_yElRoLOJagI"}

---

---

+++ {"id": "N2cUJGMIJagI"}

# 1. Model A: The Wrong Way - Simple Linear Coefficients

+++ {"id": "lGfgwVD1JagI"}

This is a simple linear model where we acknowledge that the categorical features are ordered, but ignore that the
impact of the factor-values on the coefficient might not be equally-spaced, and instead just assume equal spacing:

$$
\begin{align}
\sigma_{\beta} &\sim \text{InverseGamma}(11, 10)  \\
\beta &\sim \text{Normal}(0, \sigma_{\beta}, \text{shape}=j)  \\
\\
\text{lm} &= \beta^{T}\mathbb{x}_{i,j}  \\
\epsilon &\sim \text{InverseGamma}(11, 10)  \\
\hat{y_{i}} &\sim \text{Normal}(\mu=\text{lm}, \epsilon)  \\
\end{align}
$$

where:
+ Observations $i$ contain numeric features $j$, and $\hat{y_{i}}$ is our estimate, here of `phcs`
+ The linear sub-model $\beta^{T}\mathbb{x}_{ij}$ lets us regress onto those features
+ Notably:
    + $\mathbb{x}_{i,d450}$ is treated as a numeric feature
    + $\mathbb{x}_{i,d455}$ is treated as a numeric feature

+++ {"id": "4NTdcod5JagI"}

## 1.1 Build Model Object

```{code-cell} ipython3
:id: WgA3vejIJagI

ft_y = "phcs"
fts_x = ["intercept", "d450_num", "d455_num"]

COORDS = dict(oid=dfx.index.values, y_nm=ft_y, x_nm=fts_x)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 35
id: aejLD20fJagI
outputId: 412b4cab-7f8b-4c3b-faf3-b9aa739938e5
---
with pm.Model(coords=COORDS) as mdla:
    # 0. create (Mutable)Data containers for obs (Y, X)
    y = pm.Data("y", dfx[ft_y].values, dims="oid")  # (i, )
    x = pm.Data("x", dfx[fts_x].values, dims=("oid", "x_nm"))  # (i, x)

    # 1. define priors for numeric exogs
    b_s = pm.InverseGamma("beta_sigma", alpha=11, beta=10)  # (1, )
    b = pm.Normal("beta", mu=0, sigma=b_s, dims="x_nm")  # (x, )

    # 2. define likelihood
    epsilon = pm.InverseGamma("epsilon", alpha=11, beta=10)
    _ = pm.Normal("phcs_hat", mu=pt.dot(x, b.T), sigma=epsilon, observed=y, dims="oid")

RVS_PPC = ["phcs_hat"]
RVS_SIMPLE_COMMON = ["beta_sigma", "beta", "epsilon"]

# display RVS
display(dict(unobserved=mdla.unobserved_RVs, observed=mdla.observed_RVs))
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 720
id: sxvwwMBPJagI
outputId: 293c88c8-bf62-480f-e8ee-8b0ad6f63569
---
display(pm.model_to_graphviz(mdla, formatting="plain"))
assert_no_rvs(mdla.logp())
mdla.debug(fn="logp", verbose=True)
mdla.debug(fn="random", verbose=True)
```

+++ {"id": "TcfwZvsQJagI"}

## 1.2 Sample Prior Predictive, View Diagnostics

```{code-cell} ipython3
:id: tQi_IDRuJagI

with mdla:
    ida = pm.sample_prior_predictive(
        var_names=RVS_PPC + RVS_SIMPLE_COMMON,
        samples=2000,
        return_inferencedata=True,
        random_seed=42,
    )
```

+++ {"id": "DSnsELyLJagI"}

### 1.2.1 In-Sample Prior PPC (Retrodictive Check)

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 255
id: a6YDSKYMJagI
outputId: 1a6cbacf-8a48-4405-903f-5c916c363f34
---
def plot_ppc_retrodictive(idata, group="prior", mdlname="mdla", ynm="y") -> plt.figure:
    """Convenience plot PPC retrodictive KDE"""
    f, axs = plt.subplots(1, 1, figsize=(12, 3))
    _ = az.plot_ppc(idata, group=group, kind="kde", var_names=RVS_PPC, ax=axs, observed=True)
    _ = f.suptitle(f"In-sample {group.title()} PPC Retrodictive KDE on `{ynm}` - `{mdlname}`")
    return f


f = plot_ppc_retrodictive(ida, "prior", "mdla", "phcs")
```

+++ {"id": "Mz8ftR_VJagI"}

**Observe:**

+ Values are wrong as expected, but range is reasonable

+++ {"id": "hRlR2bPDJagI"}

### 1.2.2 Quick look at selected priors

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 173
id: FN8eOipFJagI
outputId: 26b99de0-cbef-43bd-fbcb-608acfe36eca
---
def plot_posterior(
    idata,
    group="prior",
    rvs=RVS_SIMPLE_COMMON,
    coords=None,
    mdlname="mdla",
    n=1,
    nrows=1,
) -> plt.figure:
    """Convenience plot posterior (or prior) KDE"""
    m = int(np.ceil(n / nrows))
    f, axs = plt.subplots(nrows, m, figsize=(2.4 * m, 0.8 + nrows * 1.4))
    _ = az.plot_posterior(idata, group=group, ax=axs, var_names=rvs, coords=coords)
    _ = f.suptitle(f"{group.title()} distributions for rvs {rvs} - `{mdlname}")
    _ = f.tight_layout()
    return f


f = plot_posterior(ida, "prior", rvs=RVS_SIMPLE_COMMON, mdlname="mdla", n=1 + 3 + 1, nrows=1)
```

+++ {"id": "Rx_YsUCwJagJ"}

**Observe:**

+ `beta_sigma`, `beta: (levels)`, `epsilon` all have reasonable prior ranges as specified

+++ {"id": "PMUtbh_aJagJ"}

## 1.3 Sample Posterior, View Diagnostics

+++ {"id": "TmiH-SbMJagJ"}

### 1.3.1 Sample Posterior and PPC

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 67
  referenced_widgets: [5ac4d556355b4890a980ad4d4f53d792, 14ad4102c38a40da9a7387328c48429b,
    7f43a317059147bf90d614dc574ff130, d913e7dcfcb34696a5f766a25c4db9aa, b8eae6e849a64fa0b35642c162295e77,
    82adca499c4e4893a8ace68c81287ce5]
id: NDZ1i4wwJagJ
outputId: 76ca3d59-923e-495b-8895-3601a675ad71
---
with mdla:
    ida.extend(pm.sample(**SAMPLE_KWS), join="right")
    ida.extend(
        pm.sample_posterior_predictive(trace=ida.posterior, var_names=RVS_PPC),
        join="right",
    )
```

+++ {"id": "UuMXkGdkJagJ"}

### 1.3.2 Traces

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 606
id: m6lCwXTjJagJ
outputId: e8ac1e4e-79a7-414d-9cd0-a1aa7d5e3d16
---
def plot_traces_and_display_summary(idata, rvs, coords=None, mdlname="mdla") -> plt.figure:
    """Convenience to plot traces and display summary table for rvs"""
    _ = az.plot_trace(idata, var_names=rvs, coords=coords, figsize=(12, 1.4 * len(rvs)))
    f = plt.gcf()
    _ = f.suptitle(f"Posterior traces of {rvs} - `{mdlname}`")
    _ = f.tight_layout()
    _ = az.plot_energy(idata, fill_alpha=(0.8, 0.6), fill_color=("C0", "C8"), figsize=(12, 1.6))
    display(az.summary(idata, var_names=rvs))
    return f


f = plot_traces_and_display_summary(ida, rvs=RVS_SIMPLE_COMMON, mdlname="mdla")
```

+++ {"id": "nm1bL8HqJagJ"}

**Observe:**

+ Samples well-mixed and well-behaved
  + `ess_bulk` is good, `r_hat` is good
+ Marginal energy | energy transition looks reasonable
  + `E-BFMI > 0.3` so [apparently reasonable](https://python.arviz.org/en/stable/api/generated/arviz.bfmi.html#arviz.bfmi)

+++ {"id": "P5oMtpRUJagJ"}

### 1.3.3 In-Sample Posterior PPC (Retrodictive Check)

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 255
id: 4XRuM0y6JagJ
outputId: 9185a4c6-9258-4566-8e72-9fec8db5a976
---
f = plot_ppc_retrodictive(ida, "posterior", "mdla", "phcs")
```

+++ {"id": "ip_usejdJagJ"}

**Observe:**

+ In-sample PPC `phcs_hat` tracks the observed `phcs` moderately well: slightly overdispersed, perhaps a likelihood
  with fatter tails would be more appropriate (e.g. StudentT)

+++ {"id": "k91trZaJJagJ"}

### 1.3.4 In-Sample PPC LOO-PIT

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 187
id: 4-MDBU4fJagJ
outputId: 467795eb-b539-45eb-f296-6b03f3549376
---
def plot_loo_pit(idata, mdlname="mdla", y="phcs_hat", y_hat="phcs_hat"):
    """Convenience plot LOO-PIT KDE and ECDF"""
    f, axs = plt.subplots(1, 2, figsize=(12, 2.4))
    _ = az.plot_loo_pit(idata, y=y, y_hat=y_hat, ax=axs[0])
    _ = az.plot_loo_pit(idata, y=y, y_hat=y_hat, ax=axs[1], ecdf=True)
    _ = axs[0].set_title(f"Predicted `{y_hat}` LOO-PIT")
    _ = axs[1].set_title(f"Predicted `{y_hat}` LOO-PIT cumulative")
    _ = f.suptitle(f"In-sample LOO-PIT `{mdlname}`")
    _ = f.tight_layout()
    return f


f = plot_loo_pit(ida, "mdla")
```

+++ {"id": "doYARdL9JagJ"}

**Observe:**

+ `LOO-PIT` looks good, again slightly overdispersed but acceptable for use

+++ {"id": "OB14RVBUJagJ"}

### ~~1.3.5 Compare Log-Likelihood vs Other Models~~

```{code-cell} ipython3
:id: ZMXs0ZeMJagJ

# Nothing to compare yet
```

+++ {"id": "Px1R5oZeJagJ"}

## 1.4 Evaluate Posterior Parameters

+++ {"id": "hUjhq92mJagJ"}

### 1.4.1 Univariate

+++ {"id": "uHnzMgRyJagJ"}

Lots of parameters, let's take our time

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 173
id: 1bJK1kGrJagK
outputId: c04eab92-6d9e-444e-fef4-8e5bfd0cc8b0
---
f = plot_posterior(ida, "posterior", rvs=RVS_SIMPLE_COMMON, mdlname="mdla", n=5, nrows=1)
```

+++ {"id": "YAvUagbDJagK"}

**Observe:**

+ `beta_sigma`: `E ~ 10` indicates need for high variance in lcoations of `beta`s
+ `beta: intercept`: `E ~ 32` confirms the bulk of the variance in `beta`s locations is simply due to the intercept
    offset required to get the zscored values into range of `phcs`, no problem
+ `beta: d450_num`: `E ~ -3` negative, HDI94 does not span 0, substantial effect, smooth central distribution:
  + Higher values of `d450_num` create a reduction in `phcs_hat`
+ `beta: d455_num`: `E ~ -2` negative, HDI94 does not span 0, substantial effect, smooth central distribution
  + Higher values of `d455_num` create a smaller reduction in `phcs_hat`
+ `epsilon`: `E ~ 7` indicates quite a lot of variance still in the data, not yet handled by a modelled feature

+++ {"id": "J46NaedMJagK"}

## 1.5 Create PPC Forecast on simplified `forecast` set

+++ {"id": "yrrzYjmhJagK"}

Just for completeness, just compare to Figure 3 in the Bürkner paper and Rochford's
blogpost. Those plots summarize to a mean though, which seems unneccesary - let's
improve it a little with full sample posteriors

+++ {"id": "X4XB1eiwJagK"}

##### Replace dataset with `dffx` and rebuild

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 720
id: x3YSDHhyJagK
outputId: c77315fe-e798-4001-b362-15240a5b1e53
---
COORDS_F = deepcopy(COORDS)
COORDS_F["oid"] = dffx.index.values
mdla.set_data("y", dffx[ft_y].values, coords=COORDS_F)
mdla.set_data("x", dffx[fts_x].values, coords=COORDS_F)

display(pm.model_to_graphviz(mdla, formatting="plain"))
assert_no_rvs(mdla.logp())
mdla.debug(fn="logp", verbose=True)
mdla.debug(fn="random", verbose=True)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
  referenced_widgets: [fd6b9ad3a49146f3b552e8a703716147, cf40da57e90e47abbf51aa5fa82ba586]
id: 1o8-uBpvJagK
outputId: 53b7f11f-b39b-43ba-f3f0-563c71b0bf12
---
with mdla:
    ida_ppc = pm.sample_posterior_predictive(
        trace=ida.posterior, var_names=RVS_PPC, predictions=True
    )
```

+++ {"id": "mYQP_-3lJagK"}

### 1.5.2 View Predictions

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 440
id: cPDzQbRhJagK
outputId: 7a2b46b4-4da5-4cbf-8b5b-aba139a410ff
---
def plot_predicted_phcshat_d450_d455(idata, mdlname) -> plt.Figure:
    """Convenience to plot predicted phcs_hat vs d450 and d455"""
    phcs_hat = (
        az.extract(idata, group="predictions", var_names=["phcs_hat"])
        .to_dataframe()
        .drop(["chain", "draw"], axis=1)
    )
    dfppc = pd.merge(
        phcs_hat.reset_index(),
        dff[["d450", "d455"]].reset_index(),
        how="left",
        on="oid",
    )

    kws = dict(
        y="phcs_hat",
        data=dfppc,
        linestyles=":",
        estimator="mean",
        errorbar=("pi", 94),
        dodge=0.2,
    )

    f, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    _ = sns.pointplot(x="d450", hue="d455", **kws, ax=axs[0], palette="plasma_r")
    _ = sns.pointplot(x="d455", hue="d450", **kws, ax=axs[1], palette="viridis_r")
    _ = [axs[i].set_title(t) for i, t in enumerate(["d450 x d455", "d455 x d450"])]
    _ = f.suptitle(
        "Domain specific plot of posterior predicted `phcs_hat`"
        + f" vs `d450` and `d455` - `{mdlname}`"
    )
    _ = f.tight_layout()


plot_predicted_phcshat_d450_d455(idata=ida_ppc, mdlname="mdla")
```

+++ {"id": "fwOSEYPWJagK"}

**Observe:**

+ Compare this to the plots in
  [Austin Rochford's notebook](https://austinrochford.com/posts/2018-11-10-monotonic-predictors.html)
+ We see the linear responses and equal spacings of `d450` and of `d455` when treated as numeric values
+ Note here we plot the full posteriors on each datapoint (rather than summarise to a mean) which emphasises
  the large amount of variance still in the data & model

+++ {"id": "Ajjap-FZJagK"}

---

---

+++ {"id": "JB7H8-u4JagK"}

# 2. Model B: A Better Way - Dirichlet Hyperprior Allocator

+++ {"id": "bhFPL8KhJagK"}

This is an improved linear model where we acknowledge that the categorical features are ordinal _and_ allow the ordinal
values to have a non-equal spacing, For example, it might well be that `A > B > C`, but the spacing is not metric:
instead `A >>> B > C`. We achieve this using a Dirichlet hyperprior to allocate hetrogenously spaced sections of a
linear ceofficient:

$$
\begin{align}
\sigma_{\beta} &\sim \text{InverseGamma}(11, 10)  \\
\beta &\sim \text{Normal}(0, \sigma_{\beta}, \text{shape}=j)  \\
\\
\beta_{d450} &\sim \text{Normal}(0, \sigma_{\beta})  \\
\chi_{d450} &\sim \text{Dirichlet}(1, \text{shape}=k_{d450})  \\
\nu_{d450} &\sim \beta_{d450} * \sum_{i=0}^{i=k_{d450}}\chi_{d450} \\
\\
\beta_{d455} &\sim \text{Normal}(0, \sigma_{\beta})  \\
\chi_{d455} &\sim \text{Dirichlet}(1, \text{shape}=k_{d455})  \\
\nu_{d455} &\sim \beta_{d455} * \sum_{i=0}^{i=k_{d455}}\chi_{d455} \\
\\
lm &= \beta^{T}\mathbb{x}_{i,j} + \nu_{d450}[x_{i,d450}] + \nu_{d455}[x_{i,d455}]\\
\epsilon &\sim \text{InverseGamma}(11, 10)  \\
\hat{y_{i}} &\sim \text{Normal}(\mu=lm, \epsilon)  \\
\end{align}
$$

where:
+ Observations $i$ contain numeric features $j$ and ordinal categorical features
  $k$ (here `d450, d455`) which each have factor value levels $k_{d450}, k_{d455}$
+ $\hat{y_{i}}$ is our estimate, here of `phcs`
+ The linear sub-model $lm = \beta^{T}\mathbb{x}_{i,j} + \nu_{d450}[x_{i,d450}] + \nu_{d455}[x_{i,d455}]$ lets us
  regress onto those features
+ Notably:
    + $\mathbb{x}_{i,d450}$ is treated as an ordinal feature and used to index $\nu_{d450}[x_{i,d450}]$
    + $\mathbb{x}_{i,d455}$ is treated as an ordinal feature and used to index $\nu_{d455}[x_{i,d455}]$
+ NOTE: The above spec is not particuarly optimised / vectorised / DRY to aid explanation

+++ {"id": "F47aQhT2JagK"}

## 2.1 Build Model Object

```{code-cell} ipython3
:id: DWlbRSj_JagK

ft_y = "phcs"
fts_x = ["intercept"]
# NOTE fts_ord = ['d450_idx', 'd455_idx']

COORDS = dict(
    oid=dfx.index.values,
    y_nm=ft_y,
    x_nm=fts_x,
    d450_nm=list(map_int_to_cat_d450.values()),
    d455_nm=list(map_int_to_cat_d455.values()),
)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 196
id: ZyP0P29AJagK
outputId: f9aaa950-5d29-4823-a359-0067ee343305
---
with pm.Model(coords=COORDS) as mdlb:
    # NOTE: Spec not particuarly optimised / vectorised / DRY to aid explanation

    # 0. create (Mutable)Data containers for obs (Y, X)
    y = pm.Data("y", dfx[ft_y].values, dims="oid")  # (i, )
    x = pm.Data("x", dfx[fts_x].values, dims=("oid", "x_nm"))  # (i, x)
    idx_d450 = pm.Data("idx_d450", dfx["d450_idx"].values, dims="oid")  # (i, )
    idx_d455 = pm.Data("idx_d455", dfx["d455_idx"].values, dims="oid")  # (i, )

    # 1. define priors for numeric exogs
    b_s = pm.InverseGamma("beta_sigma", alpha=11, beta=10)  # (1, )
    b = pm.Normal("beta", mu=0, sigma=b_s, dims="x_nm")  # (x, )

    # 2. define nu
    def _get_nu(nm, dim):
        """Partition continous prior into ordinal chunks"""
        b0 = pm.Normal(f"beta_{nm}", mu=0, sigma=b_s)  # (1, )
        c0 = pm.Dirichlet(f"chi_{nm}", a=np.ones(len(COORDS[dim])), dims=dim)  # (lvls, )
        return pm.Deterministic(f"nu_{nm}", b0 * c0.cumsum(), dims=dim)  # (lvls, )

    nu_d450 = _get_nu("d450", "d450_nm")
    nu_d455 = _get_nu("d455", "d455_nm")

    # 3. define likelihood
    epsilon = pm.InverseGamma("epsilon", alpha=11, beta=10)
    _ = pm.Normal(
        "phcs_hat",
        mu=pt.dot(x, b.T) + nu_d450[idx_d450] + nu_d455[idx_d455],
        sigma=epsilon,
        observed=y,
        dims="oid",
    )


rvs_simple = RVS_SIMPLE_COMMON + ["beta_d450", "beta_d455"]
rvs_d450 = ["chi_d450", "nu_d450"]
rvs_d455 = ["chi_d455", "nu_d455"]

# display RVS
display(dict(unobserved=mdlb.unobserved_RVs, observed=mdlb.observed_RVs))
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 877
id: GlUShTelJagK
outputId: 846953d9-aa03-4c80-cfd5-a1fe38be6075
---
display(pm.model_to_graphviz(mdlb, formatting="plain"))
assert_no_rvs(mdlb.logp())
mdlb.debug(fn="logp", verbose=True)
mdlb.debug(fn="random", verbose=True)
```

+++ {"id": "ESP0PWjUJagL"}

## 2.2 Sample Prior Predictive, View Diagnostics

```{code-cell} ipython3
:id: Om4K71WcJagL

with mdlb:
    idb = pm.sample_prior_predictive(
        var_names=RVS_PPC + rvs_simple + rvs_d450 + rvs_d455,
        samples=2000,
        return_inferencedata=True,
        random_seed=42,
    )
```

+++ {"id": "ly-Wghq9JagL"}

### 2.2.1 In-Sample Prior PPC (Retrodictive Check)

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 255
id: 0YpdzHXLJagL
outputId: 0e7cc3d2-a864-4439-88ad-0e25375fa206
---
f = plot_ppc_retrodictive(idb, "prior", "mdlb", "phcs")
```

+++ {"id": "w_sXdKypJagL"}

**Observe:**

+ Values are wrong as expected, but range is reasonable

+++ {"id": "8PIphhhVJagL"}

### 2.2.2 Quick look at selected priors

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 683
id: QnsGBp2-JagL
outputId: 155ae8b3-50ed-48d5-ef00-6e3ab372758b
---
f = plot_posterior(idb, "prior", rvs=rvs_simple, mdlname="mdlb", n=5, nrows=1)
f = plot_posterior(idb, "prior", rvs=rvs_d450, mdlname="mdlb", n=4 * 2, nrows=2)
f = plot_posterior(idb, "prior", rvs=rvs_d455, mdlname="mdlb", n=5 * 2, nrows=2)
```

+++ {"id": "AOKrjKQgJagL"}

**Observe:**

+ Several new parameters!
+ `beta_sigma`, `beta: (levels)`, `epsilon`: all have reasonable prior ranges as specified
+ `*_d450`:
  + `chi_*`: obey the simplex constraint of the Dirichlet and span the range
  + `nu_*`: all reasonable as specified, note the ordering already present in the prior
+ `*_d455`:
  + `chi_*`: obey the simplex constraint of the Dirichlet and span the range
  + `nu_*`: all reasonable as specified, note the ordering already present in the prior

+++ {"id": "EYSLAdHvJagL"}

## 2.3 Sample Posterior, View Diagnostics

+++ {"id": "VAB-af9QJagL"}

### 2.3.1 Sample Posterior and PPC

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 104
  referenced_widgets: [ab1520d761ed4391aa348dd2545aea62, ed77bbddc4b841a891c19168713432be,
    09f77186189b44c1a54f61922bd05997, 818b1bffe7b24fbc91ec7970f414b7fa, fab11b7a600b468bb6e6729043894957,
    e58e85ebd77548219658075b7f365ef6]
id: uMoTtZA-JagL
outputId: 7bca1e61-988f-48fe-e63c-10890128dfbb
---
SAMPLE_KWS["target_accept"] = 0.9  # raise to mitigate some minor divergences
with mdlb:
    idb.extend(pm.sample(**SAMPLE_KWS), join="right")
    idb.extend(
        pm.sample_posterior_predictive(trace=idb.posterior, var_names=RVS_PPC),
        join="right",
    )
```

+++ {"id": "4sl9cFicJagL"}

### 2.3.2 Traces

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 1000
id: CWgQdiUoJagL
outputId: a9979558-7902-46bf-a040-6b5485e40e99
---
f = plot_traces_and_display_summary(idb, rvs=rvs_simple + rvs_d450 + rvs_d455, mdlname="mdlb")
```

+++ {"id": "q-1uU-m6JagL"}

**Observe:**

+ Samples well-mixed and well-behaved, but note we raised `target_accept=0.9` to mitigate / avoid divergences seen at `0.8`
  + `ess_bulk` a little low, `r_hat` is okay
+ Marginal energy | energy transition looks reasonable
  + `E-BFMI > 0.3` so [apparently reasonable](https://python.arviz.org/en/stable/api/generated/arviz.bfmi.html#arviz.bfmi)

+++ {"id": "0Eha32LhJagL"}

### 2.3.3 In-Sample Posterior PPC (Retrodictive Check)

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 255
id: 0RgDbOKyJagL
outputId: 769052e0-babd-4e49-ddaa-1efb59ada87b
---
f = plot_ppc_retrodictive(idb, "posterior", "mdlb", "phcs")
```

+++ {"id": "mAHvrR8tJagL"}

**Observe:**

+ In-sample PPC `phcs_hat` tracks the observed `phcs` moderately well: slightly overdispersed, perhaps a likelihood
  with fatter tails would be more appropriate (e.g. StudentT)

+++ {"id": "jB3nKvyrJagL"}

### 2.3.4 In-Sample PPC LOO-PIT

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 187
id: gzVgnVAqJagL
outputId: 76a9eef5-9aa9-4905-a364-4288b5518fca
---
f = plot_loo_pit(idb, "mdlb")
```

+++ {"id": "WM2VArBVJagL"}

**Observe:**

+ `LOO-PIT` looks good, again slightly overdispersed but acceptable for use

+++ {"id": "c0P-PWIwJagM"}

### 2.3.5 Compare Log-Likelihood vs Other Models

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 324
id: mwiaR6CcJagM
outputId: 0b2f95f5-7bf7-46c4-e389-4b99c2f7fe55
---
def plot_compare_log_likelihood(idata_dict={}, y_hat="phcs_hat") -> plt.figure:
    """Convenience to plot comparison for a dict of idatas"""
    dfcomp = az.compare(idata_dict, var_name=y_hat, ic="loo", method="stacking", scale="log")
    f, axs = plt.subplots(1, 1, figsize=(12, 2.4 + 0.3 * len(idata_dict)))
    _ = az.plot_compare(dfcomp, ax=axs, title=False, textsize=10, legend=False)
    _ = f.suptitle(
        "Model Performance Comparison: ELPD via In-Sample LOO-PIT "
        + " vs ".join(list(idata_dict.keys()))
        + "\n(higher & narrower is better)"
    )
    _ = f.tight_layout()
    display(dfcomp)
    return f


f = plot_compare_log_likelihood(idata_dict={"mdla": ida, "mdlb": idb})
```

+++ {"id": "VYS6htX1JagM"}

**Observe:**

+ Our new ordinal-respecting `mdlb` appears to be the winner, taking nearly all the weight and a higher `elpd_loo`

+++ {"id": "EogKjrVwJagM"}

## 2.4 Evaluate Posterior Parameters

+++ {"id": "W8KfHwERJagM"}

### 2.4.1 Univariate

+++ {"id": "AV6XgCapJagM"}

Lots of parameters, let's take our time

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 173
id: Wh8MPtTLJagM
outputId: 9a03b5bc-9e31-48be-fb0b-527e233f38f2
---
f = plot_posterior(idb, "posterior", rvs=rvs_simple, mdlname="mdlb", n=5, nrows=1)
```

+++ {"id": "qh9nt0IhJagM"}

**Observe:**

+ `beta_sigma`: `E ~ 30` indicates need for high variance in locations of `beta`s
+ `beta: intercept`: `E ~ 41` confirms the bulk of the variance in `beta`s locations is simply due to the intercept
    offset required to get the zscored values into range of `phcs`, no problem
+ `epsilon`: `E ~ 7` indicates quite a lot of variance still in the data, not yet handled by a modelled feature
+ `beta: d450`: `E ~ -9` negative, HDI94 does not span 0, substantial effect, smooth central distribution:
  + Higher indexes of `d450_idx` create a reduction in `phcs_hat`
+ `beta: d455`: `E ~ -7` negative, HDI94 does not span 0, substantial effect, smooth central distribution
  + Higher indexes of `d455_idx` create a reduction in `phcs_hat`

In general the bigger coefficient values here (vs `mdla`) suggest more disrimination between the values in the data and
better performance

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 272
id: WdqLVls4JagM
outputId: 9364c0b9-c62b-414f-9e03-3af44283454e
---
f = plot_posterior(idb, "posterior", rvs=rvs_d450, mdlname="mdlb", n=4 * 2, nrows=2)
```

+++ {"id": "ia14e6n-JagM"}

**Observe:**

Interesting pattern:
+ `chi_d450`: Non-linear response throughout the range
+ `nu_d450`: The non-linear effect `beta * chi.csum()` is clear, in particular `c0` is far from the trend of `c1, c2, c3`

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 272
id: QCDyC6gAJagM
outputId: 5770d257-1581-4d25-f12d-b9105824df7c
---
f = plot_posterior(idb, "posterior", rvs=rvs_d455, mdlname="mdlb", n=5 * 2, nrows=2)
```

+++ {"id": "fCdxIZ7XJagM"}

**Observe:**

Interesting pattern:
+ `chi_d455`: Non-linear response throughout the range
+ `nu_d455`: The non-linear effect `beta * chi.csum()` is clear, in particular `c2` is almost the same as `c1`

> Let's see those levels forestplotted to make even more clear

+++ {"id": "yoicNMViJagM"}

##### Monotonic priors forestplot

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 357
id: tx3ma1oRJagM
outputId: 1b324a3b-c4c3-419b-8804-0d8277c2ec85
---
def plot_posterior_forest(idata, group="posterior", rv="beta", mdlname="mdla"):
    """Convenience forestplot posterior (or prior) KDE"""
    f, axs = plt.subplots(1, 1, figsize=(12, 2.4))
    _ = az.plot_forest(idata[group], var_names=[rv], ax=axs, combined=True)
    _ = f.suptitle(f"Forestplot of {group.title()} level values for `{rv}` - `{mdlname}`")
    _ = f.tight_layout()
    return f


f = plot_posterior_forest(idb, "posterior", "nu_d450", "mdlb")
f = plot_posterior_forest(idb, "posterior", "nu_d455", "mdlb")
```

+++ {"id": "Rv1UUVf2JagM"}

**Observe:**

Here we see the same patterns in more detail, inparticular:
+ `nu_d450`: `c0` is an outlier with disproportionately less impact than `c1, c2, c3`
+ `nu_d455`: `c1, c2` overlap strongly and so have very similar impact to one another

+++ {"id": "iBabtpXpJagM"}

## 2.5 Create PPC Forecast on simplified `forecast` set

+++ {"id": "Y81MC7nlJagM"}

Just for completeness, just compare to Figure 3 in the Bürkner paper and Rochford's
blogpost.

Those plots summarize to a mean though, which seems unneccesary - let's improve it a little.

+++ {"id": "b09YNkSkJagM"}

### 2.5.1 Replace dataset with `dffx`, rebuild, and sample PPC

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 877
id: Z-0pSJVPJagM
outputId: e7a091f0-a8af-48be-ecab-58b7c263aeb7
---
COORDS_F = deepcopy(COORDS)
COORDS_F["oid"] = dffx.index.values
mdlb.set_data("y", dffx[ft_y].values, coords=COORDS_F)
mdlb.set_data("x", dffx[fts_x].values, coords=COORDS_F)
mdlb.set_data("idx_d450", dffx["d450_idx"].values, coords=COORDS_F)
mdlb.set_data("idx_d455", dffx["d455_idx"].values, coords=COORDS_F)

display(pm.model_to_graphviz(mdlb, formatting="plain"))
assert_no_rvs(mdlb.logp())
mdlb.debug(fn="logp", verbose=True)
mdlb.debug(fn="random", verbose=True)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
  referenced_widgets: [28826f8350af445d9c1ca0a3580ded88, 8e0c363af1aa48feb60083e0ff42cd5e]
id: Ilbo7p8pJagM
outputId: 4028df37-29dc-4e52-8fbb-65b86328c3ad
---
with mdlb:
    idb_ppc = pm.sample_posterior_predictive(
        trace=idb.posterior, var_names=RVS_PPC, predictions=True
    )
```

+++ {"id": "FLwaOHm6JagN"}

### 2.5.2 View Predictions

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 440
id: nErBbUVaJagN
outputId: d0ff8cd0-e3ee-4c5f-802d-a85fb2770498
---
plot_predicted_phcshat_d450_d455(idata=idb_ppc, mdlname="mdlb")
```

+++ {"id": "_6qGc41tJagN"}

**Observe:**

+ Compare this to Section 1.5.2 above, and the plots in
  [Austin Rochford's notebook](https://austinrochford.com/posts/2018-11-10-monotonic-predictors.html)
+ We see the non-linear responses and non-equal spacings of `d450` and of `d455` when treated as ordinal categories
+ In particular, note the behaviours we already saw in the posterior plots
  + LHS plot `d450`: all points for `c0` are all higher than the plot in Section 1.5.2 (also note the overlap of `d455: c1, c2` levels in the shaded points)
  + RHS plot `d455`: all points for `c1, c2` overlap strongly (also note `d455 c0` outlying)
+ Note here we plot the full posteriors on each datapoint (rather than summarise to a mean) which emphasises
  the large amount of variance still in the data & model

+++ {"id": "dioW8jMIJagN"}

---

---

+++

# Errata

## Authors

+ Created by [Jonathan Sedar](https://github.com/jonsedar) in Oct 2024

## Reference

:::{bibliography}
:filter: docname in docnames
:::

+++ {"id": "KBx5js66JagN"}

## Watermark

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: _JhXV8qjJagN
outputId: e145f47a-a63f-4b50-b1cb-33e1401b32b5
---
# tested running on Google Colab 2024-10-27
%load_ext watermark
%watermark -n -u -v -iv -w
```

:::{include} ../page_footer.md
:::
