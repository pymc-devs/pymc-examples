---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: pymc_env
  language: python
  name: python3
---

(GLM-out-of-sample-predictions)=
# Out-Of-Sample Predictions

:::{post} December, 2023
:tags: generalized linear model, logistic regression, out of sample predictions, patsy
:category: beginner
:::

```{code-cell} ipython3
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

from scipy.special import expit as inverse_logit
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
```

```{code-cell} ipython3
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
```

## Generate Sample Data

We want to fit a logistic regression model where there is a multiplicative interaction between two numerical features.

```{code-cell} ipython3
# Number of data points
n = 250
# Create features
x1 = rng.normal(loc=0.0, scale=2.0, size=n)
x2 = rng.normal(loc=0.0, scale=2.0, size=n)
# Define target variable
intercept = -0.5
beta_x1 = 1
beta_x2 = -1
beta_interaction = 2
z = intercept + beta_x1 * x1 + beta_x2 * x2 + beta_interaction * x1 * x2
p = inverse_logit(z)
# note binomial with n=1 is equal to a Bernoulli
y = rng.binomial(n=1, p=p, size=n)
df = pd.DataFrame(dict(x1=x1, x2=x2, y=y))
df.head()
```

Let us do some exploration of the data:

```{code-cell} ipython3
sns.pairplot(data=df, kind="scatter");
```

- $x_1$ and $x_2$ are not correlated.
- $x_1$ and $x_2$ do not seem to separate the $y$-classes independently.
- The distribution of $y$ is not highly unbalanced.

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.scatterplot(x="x1", y="x2", data=df, hue="y")
ax.legend(title="y")
ax.set(title="Sample Data", xlim=(-9, 9), ylim=(-9, 9));
```

## Prepare Data for Modeling

```{code-cell} ipython3
labels = ["Intercept", "x1", "x2", "x1:x2"]
df["Intercept"] = np.ones(len(df))
df["x1:x2"] = df["x1"] * df["x2"]
# reorder columns to be in the same order as labels
df = df[labels]
x = df.to_numpy()
```

Now we do a train-test split.

```{code-cell} ipython3
indices = rng.permutation(x.shape[0])
train_prop = 0.7
train_size = int(train_prop * x.shape[0])
training_idx, test_idx = indices[:train_size], indices[train_size:]
x_train, x_test = x[training_idx, :], x[test_idx, :]
y_train, y_test = y[training_idx], y[test_idx]
```

## Define and Fit the Model

We now specify the model in PyMC.

```{code-cell} ipython3
coords = {"coeffs": labels}

with pm.Model(coords=coords) as model:
    # data containers
    X = pm.MutableData("X", x_train)
    y = pm.MutableData("y", y_train)
    # priors
    b = pm.Normal("b", mu=0, sigma=1, dims="coeffs")
    # linear model
    mu = pm.math.dot(X, b)
    # link function
    p = pm.Deterministic("p", pm.math.invlogit(mu))
    # likelihood
    pm.Bernoulli("obs", p=p, observed=y)

pm.model_to_graphviz(model)
```

```{code-cell} ipython3
with model:
    idata = pm.sample()
```

```{code-cell} ipython3
az.plot_trace(idata, var_names="b", compact=False);
```

The chains look good.

```{code-cell} ipython3
az.summary(idata, var_names="b")
```

And we do a good job of recovering the true parameters for this simulated dataset.

```{code-cell} ipython3
az.plot_posterior(
    idata, var_names=["b"], ref_val=[intercept, beta_x1, beta_x2, beta_interaction], figsize=(15, 4)
);
```

## Generate Out-Of-Sample Predictions

Now we generate predictions on the test set.

```{code-cell} ipython3
with model:
    pm.set_data({"X": x_test, "y": y_test})
    idata.extend(pm.sample_posterior_predictive(idata))
```

```{code-cell} ipython3
# Compute the point prediction by taking the mean and defining the category via a threshold.
p_test_pred = idata.posterior_predictive["obs"].mean(dim=["chain", "draw"])
y_test_pred = (p_test_pred >= 0.5).astype("int").to_numpy()
```

## Evaluate Model

First let us compute the accuracy on the test set.

```{code-cell} ipython3
print(f"accuracy = {np.mean(y_test==y_test_pred): 0.3f}")
```

Next, we plot the [roc curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) and compute the [auc](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve).

```{code-cell} ipython3
fpr, tpr, thresholds = roc_curve(
    y_true=y_test, y_score=p_test_pred, pos_label=1, drop_intermediate=False
)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots()
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
roc_display = roc_display.plot(ax=ax, marker="o", markersize=4)
ax.set(title="ROC");
```

The model is performing as expected (we of course know the data generating process, which is almost never the case in practical applications).

+++

## Model Decision Boundary

Finally we will describe and plot the model decision boundary, which is the space defined as

$$\mathcal{B} = \{(x_1, x_2) \in \mathbb{R}^2 \: | \: p(x_1, x_2) = 0.5\}$$

where $p$ denotes the probability of belonging to the class $y=1$ output by the model. To make this set explicit, we simply write the condition in terms of the model parametrization:

$$0.5 = \frac{1}{1 + \exp(-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_{12} x_1x_2))}$$

which implies

$$0 = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_{12} x_1x_2$$

Solving for $x_2$ we get the formula

$$x_2 = - \frac{\beta_0 + \beta_1 x_1}{\beta_2 + \beta_{12}x_1}$$

Observe that this curve is a hyperbola centered at the singularity point $x_1 = - \beta_2 / \beta_{12}$.

+++

Let us now plot the model decision boundary using a grid:

```{code-cell} ipython3
def make_grid():
    x1_grid = np.linspace(start=-9, stop=9, num=300)
    x2_grid = x1_grid
    x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)
    x_grid = np.stack(arrays=[x1_mesh.flatten(), x2_mesh.flatten()], axis=1)
    return x1_grid, x2_grid, x_grid


x1_grid, x2_grid, x_grid = make_grid()

with model:
    # Create features on the grid.
    x_grid_ext = np.hstack(
        (
            np.ones((x_grid.shape[0], 1)),
            x_grid,
            (x_grid[:, 0] * x_grid[:, 1]).reshape(-1, 1),
        )
    )
    # set the observed variables
    pm.set_data({"X": x_grid_ext})
    # calculate pushforward values of `p`
    ppc_grid = pm.sample_posterior_predictive(idata, var_names=["p"])
```

```{code-cell} ipython3
# grid of predictions
grid_df = pd.DataFrame(x_grid, columns=["x1", "x2"])
grid_df["p"] = ppc_grid.posterior_predictive.p.mean(dim=["chain", "draw"])
p_grid = grid_df.pivot(index="x2", columns="x1", values="p").to_numpy()
```

Now we compute the model decision boundary on the grid for visualization purposes.

```{code-cell} ipython3
def calc_decision_boundary(idata, x1_grid):
    # posterior mean of coefficients
    intercept = idata.posterior["b"].sel(coeffs="Intercept").mean().data
    b1 = idata.posterior["b"].sel(coeffs="x1").mean().data
    b2 = idata.posterior["b"].sel(coeffs="x2").mean().data
    b1b2 = idata.posterior["b"].sel(coeffs="x1:x2").mean().data
    # decision boundary equation
    return -(intercept + b1 * x1_grid) / (b2 + b1b2 * x1_grid)
```

We finally get the plot and the predictions on the test set:

```{code-cell} ipython3
fig, ax = plt.subplots()

# data
sns.scatterplot(
    x=x_test[:, 1].flatten(),
    y=x_test[:, 2].flatten(),
    hue=y_test,
    ax=ax,
)

# decision boundary
ax.plot(x1_grid, calc_decision_boundary(idata, x1_grid), color="black", linestyle=":")

# grid of predictions
ax.contourf(x1_grid, x2_grid, p_grid, alpha=0.3)

ax.legend(title="y", loc="center left", bbox_to_anchor=(1, 0.5))
ax.set(title="Model Decision Boundary", xlim=(-9, 9), ylim=(-9, 9), xlabel="x1", ylabel="x2");
```

Note that we have computed the model decision boundary by using the mean of the posterior samples. However, we can generate a better (and more informative!) plot if we use the complete distribution (similarly for other metrics like accuracy and AUC).

+++

## References

- [Bayesian Analysis with Python (Second edition) - Chapter 4](https://github.com/aloctavodia/BAP/blob/master/code/Chp4/04_Generalizing_linear_models.ipynb)
- [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/)

+++



+++

## Authors
- Created by [Juan Orduz](https://github.com/juanitorduz).
- Updated by [Benjamin T. Vincent](https://github.com/drbenvincent) to PyMC v4 in June 2022
- Re-executed by [Benjamin T. Vincent](https://github.com/drbenvincent) with PyMC v5 in December 2022
- Updated by [Christian Luhmann](https://github.com/cluhmann)  in December 2023 ([pymc-examples#616](https://github.com/pymc-devs/pymc-examples/pull/616))

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor
```

:::{include} ../page_footer.md
:::
