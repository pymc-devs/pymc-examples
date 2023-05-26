---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
substitutions:
  extra_dependencies: bambi seaborn
---

(conditional_autoregressive_priors)=
# Conditional Autoregressive (CAR) Models for Spatial Data

:::{post} Jul 29, 2022 
:tags: spatial
:category: beginner, tutorial
:author: Conor Hassan 
:::

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
# THESE ARE THE LIBRARIES THAT ARE NOT DEPENDENCIES ON PYMC
# import mapclassify
import libpysal

# import bokeh
from bokeh.plotting import output_notebook

# import geopandas
from geopandas import read_file
```

```{code-cell} ipython3
az.style.use("arviz-darkgrid")
output_notebook()
# plt.rcParams["figure.constrained_layout.use"] = False
```

# Conditional Autoregressive (CAR) model

A *conditional autoregressive CAR prior* on a set of random effects $\{\phi_i\}_{i=1}^N$ models the random effect $\phi_i$ as having a mean, that is the weighted average the random effects of observation $i$'s adjacent neighbours. Mathematically, this can be expressed as 
$$\phi_i \big | \mathbf{\phi}_{j\sim i}\sim\text{Normal}\bigg(\alpha\frac{\sum_{j=1}^{n_i}w_{ij}\phi_j}{n_i}, \sigma_{i}^2\bigg ),$$
where ${j\sim i}$ indicates the set of adjacent neighbours to observation $i$, $n_i$ denotes the number of adjacent neighbours that observation $i$ has, $w_{ij}$ is the weighting of the spatial relationship between observation $i$ and $j$. If $i$ and $j$ are not adjacent, then $w_{ij}=0$. Lastly, $\sigma_i^2$ is a spatially varying variance parameter for each area. Note that information such as an adjacency matrix, indicating the neighbour relationships, and a weight matrix $\textbf{w}$, indicating the weights of the spatial relationships, is required as input data. The parameters that we infer are $\{\phi\}_{i=1}^N, \{\sigma_i\}_{i=1}^N$, and $\alpha$. 

## Model specification 

Here we will demonstrate the implementation of a CAR model using a canonical example: the lip cancer risk data in Scotland between 1975 and 1980. The original data is from [1]. This dataset includes observed lip cancer case counts $\{y_i\}_{i=1}^N$ at $N=56$ spatial units in Scotland, with the expected number of cases $\{E_i\}_{i=1}^N$ as an offset term, an intercept parameter, and and a parameter for an area-specific continuous variable for the proportion of the population employed in agriculture, fishing, or forestry, denoted by $\{x_i\}_{i=1}^N$. We want to model how the lip cancer rates relate to the distribution of employment among industries, as exposure to sunlight is a risk factor. Mathematically, the model is 
\begin{align*} 
y_i &\sim \text{Poisson}\big (\lambda_i),\\
\log \lambda_i &= \beta_0+\beta_1x_i + \phi_i + \log E_i,\\
\phi_i \big | \mathbf{\phi}_{j\sim i}&\sim\text{Normal}\big(\alpha\sum_{j=1}^{n_i}w_{ij}\phi_j, \sigma_{i}^2\big ), \\
\beta_0, \beta_1 &\sim \text{Normal}\big (0, a\big ),
\end{align*}
where $a$ is the some chosen hyperparameter for the variance of the prior distribution of the regression coefficients. 

## Preparing the data 

We need to load in the dataset to access the variables $\{y_i, x_i, E_i\}_{i=1}^N$. But more unique to the use of CAR models, is the creation of the necessary spatial adjacency matrix. For the models that we fit, all neighbours are weighted as $1$, circumventing the need for a weight matrix. The dataset can be accessed via the `pm.get_data` function.

```{code-cell} ipython3
df_scot_cancer = pd.read_csv(pm.get_data("scotland_lips_cancer.csv"))
```

```{code-cell} ipython3
df_scot_cancer.head()
```

```{code-cell} ipython3
# observed cancer counts
y = df_scot_cancer["CANCER"].values

# number of observations
N = len(y)

# expected cancer counts E for each county: this is calculated using age-standardized rates of the local population
E = df_scot_cancer["CEXP"].values
logE = np.log(E)
```

```{code-cell} ipython3
# proportion of the population engaged in agriculture, forestry, or fishing
x = df_scot_cancer["AFF"].values / 10.0
```

Below are the steps that we take to create the necessary adjacency matrix, where the entry $i,j$ of the matrix is $1$ if observations $i$ and $j$ are considered neighbours, and $0$ otherwise.

```{code-cell} ipython3
# spatial adjacency information: column `ADJ` contains list entries which are preprocessed to obtain adj as a list of lists
adj = (
    df_scot_cancer["ADJ"].apply(lambda x: [int(val) for val in x.strip("][").split(",")]).to_list()
)
```

```{code-cell} ipython3
# change to Python indexing (i.e. -1)
for i in range(len(adj)):
    for j in range(len(adj[i])):
        adj[i][j] = adj[i][j] - 1
```

```{code-cell} ipython3
# storing the adjacency matrix as a two-dimensional np.array
adj_matrix = np.zeros((N, N), dtype="int32")

for area in range(N):
    adj_matrix[area, adj[area]] = 1
```

## Visualizing the data 

An important aspect of modelling spatial data is the ability to effectively visualize the spatial nature of the data, and whether the model that you have chosen captures this spatial dependency. 

We load in an alternate version of the *Scottish lip cancer* dataset, from the `libpysal` package, to use for plotting.

```{code-cell} ipython3
_ = libpysal.examples.load_example("Scotlip")
pth = libpysal.examples.get_path("scotlip.shp")
spat_df = read_file(pth)
spat_df["PROP"] = spat_df["CANCER"] / np.exp(spat_df["CEXP"])
spat_df.head()
```

We initially plot the observed number of cancer counts over the expected number of cancer counts for each area. The spatial dependency that we observe in this plot indicates that we may need to consider a spatial model for the data.

```{code-cell} ipython3
scotland_map = spat_df.plot(
    column="PROP",
    scheme="QUANTILES",
    k=4,
    cmap="BuPu",
    legend=True,
    legend_kwds={"loc": "center left", "bbox_to_anchor": (1, 0.5)},
)
```

## Writing some models in **PyMC**

+++

#### Our first model: an *independent* random effects model
We begin by fitting an independent random effect's model. We are not modelling any *spatial dependency* between the areas. This model is equivalent to a Poisson regression model with a normal random effect, and mathematically looks like
\begin{align*} 
y_i &\sim \text{Poisson}\big (\lambda_i),\\
\log \lambda_i &= \beta_0+\beta_1x_i + \theta_i + \log E_i,\\
\theta_i &\sim\text{Normal}\big(\mu=0, \tau=\tau_{\text{ind}}\big ), \\
\beta_0, \beta_1 &\sim \text{Normal}\big (\mu=0, \tau = 1e^{-5}\big ), \\
\tau_{\text{ind}} &\sim \text{Gamma}\big (\alpha=3.2761, \beta=1.81\big),
\end{align*} 
where $\tau_\text{ind}$ is an unknown parameter for the precision of the independent random effects. The values for the $\text{Gamma}$ prior are chosen specific to our second model and thus we will delay explaining our choice until then.

```{code-cell} ipython3
with pm.Model(coords={"area_idx": np.arange(N)}) as independent_model:
    beta0 = pm.Normal("beta0", mu=0.0, tau=1.0e-5)
    beta1 = pm.Normal("beta1", mu=0.0, tau=1.0e-5)
    # variance parameter of the independent random effect
    tau_ind = pm.Gamma("tau_ind", alpha=3.2761, beta=1.81)

    # independent random effect
    theta = pm.Normal("theta", mu=0, tau=tau_ind, dims="area_idx")

    # exponential of the linear predictor -> the mean of the likelihood
    mu = pm.Deterministic("mu", pt.exp(logE + beta0 + beta1 * x + theta), dims="area_idx")

    # likelihood of the observed data
    y_i = pm.Poisson("y_i", mu=mu, observed=y, dims="area_idx")

    # saving the residual between the observation and the mean response for the area
    res = pm.Deterministic("res", y_i - mu, dims="area_idx")

    # sampling the model
    independent_idata = pm.sample(2000, tune=2000, cores=8)
```

We can plot the residuals of this first model.

```{code-cell} ipython3
independent_stacked = az.extract_dataset(independent_idata)
spat_df["INDEPENDENT_RES"] = independent_stacked.res.mean(axis=1)
```

```{code-cell} ipython3
independent_map = spat_df.plot(
    column="INDEPENDENT_RES",
    scheme="QUANTILES",
    k=5,
    cmap="BuPu",
    legend=True,
    legend_kwds={"loc": "center left", "bbox_to_anchor": (1, 0.5)},
)
```

The mean of the residuals for the areas appear spatially correlated. This leads us to explore the addition of a spatially dependent random effect, by using a **conditional autoregressive (CAR)** prior.

#### Our second model: a *spatial* random effects model (with fixed spatial dependence)
Let us fit a model that has two random effects for each area: an *independent* random effect, and a *spatial* random effect first. This models looks
\begin{align*} 
y_i &\sim \text{Poisson}\big (\lambda_i),\\
\log \lambda_i &= \beta_0+\beta_1x_i + \theta_i + \phi_i + \log E_i,\\
\theta_i &\sim\text{Normal}\big(\mu=0, \tau=\tau_{\text{ind}}\big ), \\
\phi_i \big | \mathbf{\phi}_{j\sim i} &\sim \text{Normal}\big(\mu=\alpha\sum_{j=1}^{n_i}\phi_j, \tau=\tau_{\text{spat}}\big ),\\
\beta_0, \beta_1 &\sim \text{Normal}\big (\mu = 0, \tau = 1e^{-5}\big), \\
\tau_{\text{ind}} &\sim \text{Gamma}\big (\alpha=3.2761, \beta=1.81\big), \\
\tau_{\text{spat}} &\sim \text{Gamma}\big (\alpha=1, \beta=1\big ),
\end{align*} 
where the line $\phi_i \big | \mathbf{\phi}_{j\sim i} \sim \text{Normal}\big(\mu=\alpha\sum_{j=1}^{n_i}\phi_j, \tau=\tau_{\text{spat}}\big )$ denotes the CAR prior, $\tau_\text{spat}$ is an unknown parameter for the precision of the spatial random effects, and $\alpha$ captures the degree of spatial dependence between the areas. In this instance, we fix $\alpha=0.95$. 

*Side note:* Here we explain the prior's used for the precision of the two random effect terms. As we have two random effects $\theta_i$ and $\phi_i$ for each $i$, they are independently unidentifiable, but the sum $\theta_i + \phi_i$ is identifiable. However, by scaling the priors of this precision in this manner, one may be able to interpret the proportion of variance explained by each of the random effects.

```{code-cell} ipython3
with pm.Model(coords={"area_idx": np.arange(N)}) as fixed_spatial_model:
    beta0 = pm.Normal("beta0", mu=0.0, tau=1.0e-5)
    beta1 = pm.Normal("beta1", mu=0.0, tau=1.0e-5)
    # variance parameter of the independent random effect
    tau_ind = pm.Gamma("tau_ind", alpha=3.2761, beta=1.81)
    # variance parameter of the spatially dependent random effects
    tau_spat = pm.Gamma("tau_spat", alpha=1.0, beta=1.0)

    # area-specific model parameters
    # independent random effect
    theta = pm.Normal("theta", mu=0, tau=tau_ind, dims="area_idx")
    # spatially dependent random effect, alpha fixed
    phi = pm.CAR("phi", mu=np.zeros(N), tau=tau_spat, alpha=0.95, W=adj_matrix, dims="area_idx")

    # exponential of the linear predictor -> the mean of the likelihood
    mu = pm.Deterministic("mu", pt.exp(logE + beta0 + beta1 * x + theta + phi), dims="area_idx")

    # saving the residual between the observation and the mean response for the area
    res = pm.Deterministic("res", y_i - mu, dims="area_idx")

    # likelihood of the observed data
    pm.Poisson("y_i", mu=mu, observed=y, dims="area_idx")

    # sampling the model
    fixed_spatial_idata = pm.sample(2000, tune=3000, cores=8)
```

We can see by plotting the residuals of the second model, by accounting for spatial dependency with the CAR prior, the residuals of the model appear more independent with respect to the spatial location of the observation.

```{code-cell} ipython3
fixed_spatial_stacked = az.extract_dataset(fixed_spatial_idata)
spat_df["SPATIAL_RES"] = fixed_spatial_stacked.res.mean(axis=1)
```

```{code-cell} ipython3
fixed_spatial_map = spat_df.plot(
    column="SPATIAL_RES",
    scheme="quantiles",
    k=5,
    cmap="BuPu",
    legend=True,
    legend_kwds={"loc": "center left", "bbox_to_anchor": (1, 0.5)},
)
```

If we wanted to be *fully Bayesian* about the model that we specify, we would estimate the spatial dependence parameter $\alpha$. This leads to ... 

#### Our third model: a *spatial* random effects model, with unknown spatial dependence
The only difference between model 3 and model 2, is that in model 3, $\alpha$ is unknown, so we put a prior $\alpha\sim\text{Beta}\big (\alpha = 1, \beta=1\big )$ over it.

```{code-cell} ipython3
with pm.Model(coords={"area_idx": np.arange(N)}) as car_model:
    beta0 = pm.Normal("beta0", mu=0.0, tau=1.0e-5)
    beta1 = pm.Normal("beta1", mu=0.0, tau=1.0e-5)
    # variance parameter of the independent random effect
    tau_ind = pm.Gamma("tau_ind", alpha=3.2761, beta=1.81)
    # variance parameter of the spatially dependent random effects
    tau_spat = pm.Gamma("tau_spat", alpha=1.0, beta=1.0)

    # prior for alpha
    alpha = pm.Beta("alpha", alpha=1, beta=1)

    # area-specific model parameters
    # independent random effect
    theta = pm.Normal("theta", mu=0, tau=tau_ind, dims="area_idx")
    # spatially dependent random effect
    phi = pm.CAR("phi", mu=np.zeros(N), tau=tau_spat, alpha=alpha, W=adj_matrix, dims="area_idx")

    # exponential of the linear predictor -> the mean of the likelihood
    mu = pm.Deterministic("mu", pt.exp(logE + beta0 + beta1 * x + theta + phi), dims="area_idx")

    # likelihood of the observed data
    pm.Poisson("y_i", mu=mu, observed=y, dims="area_idx")

    # sampling the model
    car_idata = pm.sample(
        2000, tune=3000, cores=8, step=[pm.NUTS(max_treedepth=20, target_accept=0.8)]
    )
```

```{code-cell} ipython3
car_stacked = az.extract_dataset(car_idata)
```

Comparing the regression parameters $\beta_0$ and $\beta_1$ between the three models that we have fit, we can see that accounting for the spatial dependence between observations has the ability to greatly impact the interpretation of the effect of covariates on the response variable.

```{code-cell} ipython3
beta_density = az.plot_density(
    [independent_idata, fixed_spatial_idata, car_idata],
    data_labels=["Independent", "Spatial with alpha fixed", "Spatial with alpha random"],
    var_names=["beta0", "beta1"],
    shade=0.1,
    backend="bokeh",
)
```

As you can see from the warning messages given above, we have a difficult time effectively sampling from the third model. This is not an issue with PyMC, and is a common issue among software packages for models of this form: the model structure is unidentifiable. Though this should not be taken as guaranteed to be accurate, we can plot the marginal posterior for $\alpha$, and see that it is very near $1$.

```{code-cell} ipython3
alpha_density = az.plot_density([car_idata], var_names=["alpha"], shade=0.1, backend="bokeh")
```

This motivates having the ability that is equivalent to setting $\alpha=1$. In the definition of the CAR prior, $\alpha \in(0, 1)$. If $\alpha \rightarrow 1$, we get an alternate prior called the *intrinsic conditional autoregressive (ICAR)* prior. The ICAR prior has some desirable properties compared to the CAR prior, and is more widely used in spatial models, specifically the BYM {cite:p}`besag1991bayesian`, Leroux {cite:p}`leroux2000estimation` and BYM2 {cite:p}`riebler2016intuitive` models. Currently, work is being done to include the ICAR prior within PyMC.

+++

## Authors

* Adapted from a previous PyMC example notebook, authored by Junpeng Lao {ref}`conditional_autoregressive_model` by Conor Hassan on July, 2022.

+++

## References 

:::{bibliography}
:filter: docname in docnames 
:::

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p aesara,aeppl,xarray
```

:::{include} ../page_footer.md
:::
