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

Human action and economics is all about expressing our ordinal preferences between limited options in the real-world.

We often encounter real-world situations and datasets where a predictor feature is an ordinal category recording a
preference or summarizing a metric value, and is particularly common in insurance and health. For example:

+ As a totally subjective opinion which can be different between observations (e.g. `["bad", "medium", "good", "better",
  "way better", "best", "actually the best"]`)  - these are difficult to work with and a symptom of poor data design
+ On a subjective but standardized scale (e.g. `["strongly disagree", "disagree", "agree", "strongly agree"]`) 
  this is the approach of the familar [Likert scale](https://en.wikipedia.org/wiki/Likert_scale)
+ As a summary binning of a real objective value on a metric scale (e.g. binning ages into age-groups 
  `["<30", "30 to 60", "60+"]`), or a subjective value that's been mapped to a metric scale (e.g. medical health
  self-scoring `["0-10%", ..., "90-100%"]`) - these are typically a misuse of the metric because the data has been
  compressed (losing information), and the reason for the binning and the choices of bin-edges are usually not known

In all these cases the critical issue is that the categorical values and their ordinal rank doesn't necessarily relate
linearly to the target variable. For example in a 4-value Likert scale (shown above) the relative effect of 
`"strongly agree"` (rank `4`) is probably not `"agree"` (rank 3) plus 1: `3+1 != 4`. 

Another way to put it is the metric distance between ordinal categories is not known and can be unequal. For example in
that 4-value Likert scale (shown above) the difference between `"disagree" vs "agree"` is probably not the same as 
between `"agree" vs "strongly agree"`. 

These properties can unfortunately encourage modellers to incorporate ordinal features as either a categorical (with
infinite degrees of freedom - so we lose ordering / rank information), or as a numeric coefficient (which ignores the 
unequal spacing, non-linear response). Both are poor choices and have subtly negative effects on the model performance.

A final nuance is that we might not see the occurence of all valid categorial ordinal levels in the training dataset. 
For example we might know a range is measured `["c0", "c1", "c2", "c3"]` but only see `["c0", "c1", "c3"]`. This is a 
missing data problem which could further encourage the misuse of a numeric coefficient to average or "interpolate" a
value. What we should do is incorporate our knowledge of the data domain into the model structure to autoimpute a
coefficient value. This is actually the case in this dataset here (see Section 0)! 


## Data & Models Demonstrated

Our problem statement is that when faced with such ordinal features we want to:

1. **Infer** a series of prior allocators that transform the ordinal categories into a linear (polynomial) scale
2. **Predict** the endogenous feature as usual, having captured the information from the ordinals
    
    
This notebook takes the opportunity to:

+ Demonstrate a general method using a constrained Dirichlet prior, based on {cite:p}`burkner2018` and demonstrated in a 
  pymc-specific example by Austin Rochford {cite:p}`rochford2018`
+ Improve upon both those methods by structurally correcting for a missing level value in an ordinal feature
+ Demonstrate a reasonably complete Bayesian workflow {cite:p}`gelman2020bayesian` including data curation and grabbing
  data from an RDataset 

This notebook is a partner to another notebook (TBD) where we estimate an **ordinal endogenous target feature**.

+++ {"id": "CemVRXjtJaf_"}

---

---

+++ {"id": "32SMEqWMJaf_"}

# Setup

+++ {"id": "4YE-JcFC9I8Q"}

:::{include} ../extra_installs.md
:::

```{code-cell} ipython3
:id: CQixNaaKJ8fH

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
    chains=4,
    idata_kwargs=dict(log_likelihood=True),
)

USE_LOCAL_DATASET = False
```

+++ {"id": "YemOGTYcJagA"}

---

---

+++ {"id": "APsFdavWJagA"}

# 0. Curate Dataset

+++

We use the same health dataset as in {cite:p}`burkner2018`, named `ICFCoreSetCWP.RData`, available in an R package 
[ordPens](https://github.com/cran/ordPens)

Per the Bürkner paper (Section 4: Case Study) this dataset is from a study ** of Chronic Widespread Pain(CWP) wherein
420 patients were self-surveyed on 67 health features (each a subjective ordinal category) and also assigned a 
differently generated (and also subjective) measure of physical health. In the dataset these 67 features are named e.g.\
`b1602`, `b102`, ... `d450`, `d455`, ... `s770` etc, and the target feature is named `phcs`.

Per the Bürkner paper we will subselect 2 features `d450`, `d455` (which measure an impairment of patient
walking ability on a scale `[0 to 4]` [no problem to complete problem]) and use them to predict `phcs`.

Quite interestingly, for feature `d450`, the highest ordinal level value `4` is not seen in the dataset, so we have a 
missing data problem which could further encourage the misuse of a numeric coefficient to average or "interpolate" a
value. What we should do is incorporate our knowledge of the data domain into the model structure to auto-impute a
coefficient value. This means that our model can make predictions on new data where a `d450=4` value might be seen.

** _Just for completness (but not needed for this notebook) that study is reported in 
Gertheiss, J., Hogger, S., Oberhauser, C., & Tutz, G. (2011). Selection of ordinally
784 scaled independent variables with applications to international classification of functioning
785 core sets. Journal of the Royal Statistical Society: Series C (Applied Statistics), 60 (3),
786 377–395._

NOTE some boilerplate steps are included but ~~struck through~~ with and explanatory comment 
e.g. "Not needed in this simple example". This is to preserve the logical workflow which is 
more generally useful

+++ {"id": "fiQFFNqhJagB"}

## 0.1 Extract

+++ {"id": "pwNFAyKzJagB"}

Annoyingly but not suprisingly for an R project, despite being a small, simple table, the dataset is only available in 
an obscure R binary format, and tarred, so we'll download, unpack and store locally as a normal CSV file.
This uses the rather helpful [`pyreadr`](https://github.com/ofajardo/pyreadr) package.

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
outputId: d8487bb5-acbe-4cfa-f5a1-5f7d9f42f377
---
print(dfr.shape)
display(pd.concat((dfr.describe(include="all").T, dfr.isnull().sum(), dfr.dtypes), axis=1))
display(dfr.head())
```

+++ {"id": "MOVXxEJPJagB"}

**Observe:**

+ Looks okay - if this was a proper project we'd want to know what those cryptic column headings actually mean
+ For this purpose we'll only use a couple of the features [`d450`, `d455`] and will press ahead

+++ {"id": "M7oFbrrsJagB"}

## 0.2 Clean

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 364
id: 5kFxNzDQJagB
outputId: 74ffd824-5a3c-442b-8968-4638f575de39
---
fts_austin = ["d450", "d455", "phcs"]
df = dfr[fts_austin].copy()
display(pd.concat((df.describe(include="all").T, df.isnull().sum(), df.dtypes), axis=1))
df.head()
```

+++ {"id": "GhWmbmkPJagB"}

### ~~0.2.1 Clean Observations~~

+++ {"id": "g6-FYfnTJagB"}

Not needed in this simple example

+++ {"id": "v0lm92QsJagB"}

### 0.2.2 Clean Features

+++ {"id": "YDccQXALJagB"}

#### ~~0.2.2.1 Rename Features~~

+++ {"id": "DM_lzCesJagB"}

Nothing really needed, will rename the index when we force dtype and set index

+++ {"id": "9zAlkU82JagB"}

### ~~0.2.2.2 Correct Features~~

+++

Not needed in this simple example

+++ {"id": "DxzUSnI4JagC"}

### 0.2.2.3 Force Datatypes

+++ {"id": "-arZiqyMJagC"}

##### Force `d450` to string representation and ordered categorical dtype (supplied as an int which is unhelpful)

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: dfIcUcOEJagC
outputId: 10fe2f37-4f50-41a5-f8bb-fb40bb1f647a
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
outputId: bb70e394-2623-470a-8621-efacdb107b72
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
outputId: 2da81148-a046-4c95-c314-5d80d88b408f
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
outputId: 7ccd7545-9cf4-48f2-ded8-34383d40bd3e
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
outputId: 955dc115-301d-49e1-a56e-f9549293a6fe
---
print(df.shape)
display(pd.concat((df.describe(include="all").T, df.isnull().sum(), df.dtypes), axis=1))
display(df.head())
```

+++ {"id": "k7wl-XBoJagD"}

### 0.3.1 Univariate target `phcs`

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 186
id: FYfSlaV1JagD
outputId: 80c8b2a4-f9c8-45d6-fa7c-c953c9d8ea5d
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

+ `phcs` is a subjective scored measure of physical healt, see {cite:p}`burkner2018` for details
+ Seems well-behaved, unimodal, smooth

+++ {"id": "p43qjcvJJagH"}

### 0.3.2 Target `phcs` vs `['d450', 'd455']`

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 441
id: H-cc0MBLJagH
outputId: d2336185-7f78-41b0-d676-16327d6f8b43
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
+ `c4` is not observed: it's missing from the data despite being valid in the data domain

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

df["d450_idx"] = df["d450"].cat.codes.astype(int)
df["d450_num"] = df["d450_idx"].copy()

df["d455_idx"] = df["d455"].cat.codes.astype(int)
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
outputId: 2aa8d64b-dbfd-46f4-e3d2-a428f6f4884c
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

**NOTE:** We depart from the datasets used in {cite:p}`rochford2018` and {cite:p}`burkner2018` to make sure our 
`forecast` dataset contains all valid levels of `d450` and `d455`. Specifically, the observed dataset does not contain 
the domain-valid `d450 = 4`, so we will make sure that our forecast set does include it.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 238
id: _cs_cls9JagI
outputId: 9cc9cb72-52e0-4c7f-9bb0-06e17d4c4a39
---
LVLS_D450_D455 = ["c0", "c1", "c2", "c3", "c4"]
dff = pd.merge(
    pd.Series(LVLS_D450_D455, name="d450"), pd.Series(LVLS_D450_D455, name="d455"), how="cross"
)
dff["d450"] = pd.Categorical(dff["d450"].values, categories=LVLS_D450_D455, ordered=True)
dff["d455"] = pd.Categorical(dff["d455"].values, categories=LVLS_D450_D455, ordered=True)
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
outputId: 62e65d97-17c3-43bc-e291-7621a477df89
---
dff["d450_idx"] = dff["d450"].cat.codes.astype(int)
dff["d455_idx"] = dff["d455"].cat.codes.astype(int)
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
outputId: 52fea489-cf9c-4dbd-bfa7-ab39d5513bfa
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
```

##### Verify the built model structure matches our intent, and validate the parameterization

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 720
id: sxvwwMBPJagI
outputId: 7f26512f-5051-404a-ee5d-63810ba08850
---
display(pm.model_to_graphviz(mdla, formatting="plain"))
display(dict(unobserved=mdla.unobserved_RVs, observed=mdla.observed_RVs))
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
outputId: 9c0b5535-743d-46ca-97ac-42f2b0b8866c
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
outputId: 700d6da8-e867-43f5-db71-ffd0956d825e
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
  height: 100
  referenced_widgets: [06cc3d00728540e5b575070062775484, 0a4168d5e6624e969ad00bedd2857e3a,
    24f3c3466b984e90871abe6010a00a2b, a013b3649b4f4cffac7cba760e293ce9, 75baebed1bf74beca21c1286c418236b,
    7b9eaf22b36d4d24b61b1e776f0c3048, 525e4c51751f461889ce090e0a6d8942, a9b2452905344c76b4d5eaf98f62cbac,
    2663c7804b6c4d47a9f0c7f5120f0258, 28295446a86f427792864bfb5e310ba6]
id: NDZ1i4wwJagJ
outputId: d6c45daa-aae2-41b6-cc49-fe3cd82d8d10
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
  height: 619
id: m6lCwXTjJagJ
outputId: 9980067e-fe78-47e2-f486-6e0854f14e23
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
outputId: 4b1370cc-865a-462d-850e-801d38cbf460
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
outputId: 14deb0a9-e51d-44e2-91f7-acbf7e1faca0
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
outputId: c5aa010c-cc07-4f3a-ff4b-42a7324801cd
---
f = plot_posterior(ida, "posterior", rvs=RVS_SIMPLE_COMMON, mdlname="mdla", n=5, nrows=1)
```

+++ {"id": "YAvUagbDJagK"}

**Observe:**

+ `beta_sigma`: `E ~ 10` indicates need for high variance in locations of `beta`s
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
outputId: 649c2ff0-09a1-4006-f7a5-4ae0500b3a42
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
  referenced_widgets: [0a60c8d42b27478a9c684057d6fa9e15, 12afca8696704967a11afcda55c9999b]
id: 1o8-uBpvJagK
outputId: 96e6e8a6-e45c-46a2-f2f3-d45bd40544b0
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
outputId: 2956df0c-3794-4430-9bbe-27d315c7eafd
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

+ Compare this to the final plots in {cite:p}`rochford2018` and Figure 12 in {cite:p}`burkner2018`
+ We see the linear responses and equal spacings of `d450` and of `d455` when treated as numeric values
+ We also see that `mdla` technically can make predictions for `d450=c4` which is not seen in the data. However, this
  prediction is a purely linear extrapolation and although helpful in a sense, could be completely and misleadingly
  wrong in this model specification
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
+ Observations $i$ contain numeric features $j$ and ordinal categorical features $k$ (here `d450, d455`) which each 
  have factor value levels $k_{d450}, k_{d455}$ and note per Section 0, these are both in range `[0 - 4]` as recorded 
  by notebook variable `LVLS_D450_D455`
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
    d450_nm=LVLS_D450_D455,  # not list(df[d450].cat.categories) because c4 missing
    d455_nm=list(df["d455"].cat.categories),
)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 196
id: ZyP0P29AJagK
outputId: 2f5e3717-7549-43d0-a334-7875b3871dcd
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
```

##### Verify the built model structure matches our intent, and validate the parameterization

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 877
id: GlUShTelJagK
outputId: 049b9d63-b975-4dec-e42b-3a4e07372c5b
---
display(pm.model_to_graphviz(mdlb, formatting="plain"))
display(dict(unobserved=mdlb.unobserved_RVs, observed=mdlb.observed_RVs))
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
outputId: 3b769a22-07d0-4a3b-dac9-cb2f16c645cd
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
outputId: be8acc83-5c8b-4b0f-8eac-1b8aec23c0f2
---
f = plot_posterior(idb, "prior", rvs=rvs_simple, mdlname="mdlb", n=5, nrows=1)
f = plot_posterior(idb, "prior", rvs=rvs_d450, mdlname="mdlb", n=5 * 2, nrows=2)
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
  height: 117
  referenced_widgets: [0d589c6414744b46b26ab53684a46a6c, 186760fa868747e0a725b0ebfc236c60,
    a1015cbb012a43c48d86a5bdf03fa0dc, c01d887394474462a3cfc601ea40dc13, 7ff6e1d26b3a4c40bf34d7cafd84f51d,
    1fabd430cd72478780ac4b10b1bef4ec, e4addbe432cd42ce92f694e7517d0a6c, e0ec99304ad442a881d0522376972bc1,
    8756d6e881344e099dcb7b2f737e3958, c209b8e19bed4966a4aa3445babac4da]
id: uMoTtZA-JagL
outputId: 97c84404-6427-4bf4-ccb5-eb0057097c69
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
outputId: 933701f3-4e28-4383-b21d-9d6c6e9288e7
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
outputId: 3950cd3e-a882-456f-f25a-047f13890e4f
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
outputId: bf747f62-d0cb-41a3-f55d-ab03e6a42c3a
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
outputId: 53c87454-13b0-40be-b4cf-08764dab5e80
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
outputId: 5614b32c-41ec-4fae-d0f0-d5119c05c415
---
f = plot_posterior(idb, "posterior", rvs=rvs_simple, mdlname="mdlb", n=5, nrows=1)
```

+++ {"id": "qh9nt0IhJagM"}

**Observe:**

+ `beta_sigma`: `E ~ 12` indicates need for high variance in locations of `beta`s
+ `beta: intercept`: `E ~ 41` confirms the bulk of the variance in `beta`s locations is simply due to the intercept
    offset required to get the zscored values into range of `phcs`, no problem
+ `epsilon`: `E ~ 7` indicates quite a lot of variance still in the data, not yet handled by a modelled feature
+ `beta: d450`: `E ~ -12` negative, HDI94 does not span 0, substantial effect, smooth central distribution:
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
outputId: 14ab8ce0-8e52-4b85-ae23-23605ba1db87
---
f = plot_posterior(idb, "posterior", rvs=rvs_d450, mdlname="mdlb", n=5 * 2, nrows=2)
```

+++ {"id": "ia14e6n-JagM"}

**Observe:**

Interesting pattern:
+ `chi_d450`: Non-linear response throughout the range
+ `nu_d450`: The non-linear effect `beta * chi.csum()` is clear, in particular `c0` is far from the trend of `c1, c2, c3`

Note in particular that the posterior distribution of `chi_d450 = c4` (and thus `nu_d450 = c4`) is almost exactly the 
same value as for its prior, because it hasn't been evidenced in the dataset. The constraint of the Dirichlet has in
turn scaled the values for `c0` to `c3` and the scale of `beta_450`. 

For comparison you can try the inferior alternative by setting `COORDS['d450_nm']=list(df[d450].cat.categories)` in the
model spec in Section 2.1 and re-running and seeing what happens

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 272
id: QCDyC6gAJagM
outputId: 8a2d7aef-8afe-42ba-ce7e-a321455f26eb
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
outputId: 9be55708-399c-4f71-9ecf-443e916c2395
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

Here we see the same patterns in more detail, in particular:
+ `nu_d450`: 
  + `c0` is an outlier with disproportionately less impact than `c1, c2, c3`
  + `c4` has been auto-imputed and takes the prior value which has very wide variance around a linear extrapolation
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
outputId: 61c38a1c-8e54-4fcb-aa59-2d9e3c8cd2cc
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
  referenced_widgets: [5bdb3b1ed6c045f38ac7267a07d6dcd7, 17a29d68378e4306af111358bd850e5e]
id: Ilbo7p8pJagM
outputId: ddb942f9-46c6-4ce6-dccd-72924598acb8
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
outputId: bff0b44b-8beb-40d3-bf0f-55ea05f5de98
---
plot_predicted_phcshat_d450_d455(idata=idb_ppc, mdlname="mdlb")
```

+++ {"id": "_6qGc41tJagN"}

**Observe:**

+ Compare this to the final plots in {cite:p}`rochford2018` and Figure 12 in {cite:p}`burkner2018`
+ We see the non-linear responses and non-equal spacings of `d450` and of `d455` when treated as ordinal categories
+ In particular, note the behaviours we already saw in the posterior plots
  + LHS plot `d450`: all points for `c0` are all higher than the plot in Section 1.5.2 (also note the overlap of `d455: c1, c2` levels in the shaded points)
  + RHS plot `d455`: all points for `c1, c2` overlap strongly (also note `d455 c0` outlying)
+ We also see that `mdlb` can make predictions for `d450=c4` which is not seen in the data
+ Note here we plot the full posteriors on each datapoint (rather than summarise to a mean) which emphasises
  the large amount of variance still in the data & model

+++ {"id": "dioW8jMIJagN"}

---

---

+++ {"id": "OK4z3NA49I8c"}

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
outputId: b5b3518f-b295-42cc-a525-d8f6767270b2
---
# tested running on Google Colab 2024-10-28
%load_ext watermark
%watermark -n -u -v -iv -w
```

+++ {"id": "VXNIouB29I8c"}

:::{include} ../page_footer.md
:::
