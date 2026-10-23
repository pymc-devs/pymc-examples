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

(GLM-out-of-sample-predictions)=
# Out-Of-Sample Predictions

:::{post} October, 2026
:tags: generalized linear model, logistic regression, out of sample predictions
:category: beginner
:::

```{code-cell} ipython3
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import xarray as xr

from sklearn.metrics import RocCurveDisplay, auc, roc_curve
from sklearn.model_selection import train_test_split
```

```{code-cell} ipython3
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

# color each class from the end of the colormap that represents certainty in that class
cmap = plt.get_cmap("RdBu_r")
class_palette = {0: cmap(0.1), 1: cmap(0.9)}
marker_style = {"s": 50, "edgecolor": "white"}
```

## Generate Sample Data

We want to fit a logistic regression model where there is a multiplicative interaction between two numerical features. We write the model as a function that takes the features $x_1$ and $x_2$, and optionally the observed labels $y$, and returns a PyMC model. The same function generates the data, fits the model, and makes predictions.

```{code-cell} ipython3
labels = ["Intercept", "x1", "x2", "x1:x2"]


def make_model(x1, x2, y=None):
    coords = {"coeffs": labels, "obs_idx": np.arange(len(x1))}
    with pm.Model(coords=coords) as model:
        # data containers
        x1_data = pm.Data("x1", x1, dims="obs_idx")
        x2_data = pm.Data("x2", x2, dims="obs_idx")
        X = pm.math.stack([pm.math.ones_like(x1_data), x1_data, x2_data, x1_data * x2_data], axis=1)
        # priors
        b = pm.Normal("b", mu=0, sigma=1, dims="coeffs")
        # link function
        p = pm.Deterministic("p", pm.math.invlogit(pm.math.dot(X, b)), dims="obs_idx")
        # likelihood
        pm.Bernoulli("obs", p=p, observed=y, dims="obs_idx")
    return model
```

```{code-cell} ipython3
n = 250
x1 = rng.normal(loc=0.0, scale=2.0, size=n)
x2 = rng.normal(loc=0.0, scale=2.0, size=n)

intercept = -0.5
beta_x1 = 1
beta_x2 = -1
beta_interaction = 2

# fix the coefficients to their true values and take a single draw of the labels
true_model = pm.do(
    model=make_model(x1=x1, x2=x2),
    vars_to_interventions={"b": [intercept, beta_x1, beta_x2, beta_interaction]},
)
prior_draw = pm.sample_prior_predictive(draws=1, model=true_model, random_seed=rng)
y = prior_draw.prior["obs"].isel(chain=0, draw=0).to_numpy()

df = pd.DataFrame(dict(x1=x1, x2=x2, y=y))
df.head()
```

Let us do some exploration of the data:

```{code-cell} ipython3
sns.pairplot(data=df, hue="y", palette=class_palette, kind="scatter", plot_kws=marker_style);
```

- $x_1$ and $x_2$ are not correlated.
- $x_1$ and $x_2$ do not seem to separate the $y$-classes independently.

```{code-cell} ipython3
class_counts = df["y"].value_counts().sort_index()
class_counts.plot.bar(color=[class_palette[label] for label in class_counts.index], rot=0);
```

- The distribution of $y$ is not highly unbalanced.

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.scatterplot(x="x1", y="x2", data=df, hue="y", palette=class_palette, ax=ax, **marker_style)
ax.legend(title="y")
ax.set(title="Sample Data", xlim=(-9, 9), ylim=(-9, 9));
```

## Prepare Data for Modeling

+++

Now we do a train-test split.

```{code-cell} ipython3
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1, x2, y, train_size=0.7, random_state=RANDOM_SEED
)
```

## Define and Fit the Model

We build the model from the training features and labels and sample from its posterior.

```{code-cell} ipython3
model = make_model(x1=x1_train, x2=x2_train, y=y_train)

pm.model_to_graphviz(model)
```

```{code-cell} ipython3
with model:
    idata = pm.sample()
```

```{code-cell} ipython3
az.plot_trace_dist(idata, var_names="b", compact=False);
```

The chains look good.

```{code-cell} ipython3
az.summary(idata, var_names="b")
```

The 89% interval of every coefficient contains its true value, so the model recovers the parameters of this simulated dataset.

```{code-cell} ipython3
true_values = xr.DataArray(
    [intercept, beta_x1, beta_x2, beta_interaction],
    coords={"coeffs": labels},
    dims="coeffs",
    name="b",
)
pc = az.plot_dist(idata, var_names=["b"], figure_kwargs={"figsize": (15, 4)})
az.add_lines(pc, true_values);
```

## Generate Out-Of-Sample Predictions

Now we generate predictions on the test set.

```{code-cell} ipython3
test_model = make_model(x1=x1_test, x2=x2_test)
pm.sample_posterior_predictive(
    idata, model=test_model, var_names=["obs"], predictions=True, extend_inferencedata=True
);
```

```{code-cell} ipython3
# Compute the point prediction by taking the mean and defining the category via a threshold.
p_test_pred = idata.predictions["obs"].mean(dim=["chain", "draw"])
y_test_pred = (p_test_pred >= 0.5).astype("int").to_numpy()
```

## Evaluate Model

First let us compute the accuracy on the test set.

```{code-cell} ipython3
accuracy = np.mean(y_test == y_test_pred)
print(f"accuracy = {accuracy: 0.3f}")
```

Next, we plot the [roc curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) and compute the [auc](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve).

```{code-cell} ipython3
fpr, tpr, _ = roc_curve(y_true=y_test, y_score=p_test_pred, pos_label=1, drop_intermediate=False)
roc_auc = auc(x=fpr, y=tpr)

fig, ax = plt.subplots()
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(
    ax=ax, curve_kwargs={"marker": "o", "markersize": 4}
)
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
x1_grid = np.linspace(start=-9, stop=9, num=100)
x2_grid = x1_grid

with pm.Model(coords={"coeffs": labels, "x1": x1_grid, "x2": x2_grid}) as decision_model:
    # placeholder whose values are taken from the posterior draws in `idata`
    b = pm.Flat("b", dims="coeffs")
    b_intercept, b_x1, b_x2, b_x1x2 = (b[i] for i in range(len(labels)))

    # broadcast x1 along columns and x2 along rows to evaluate p on the full grid
    x1_mesh, x2_mesh = x1_grid[None, :], x2_grid[:, None]
    logit_grid = b_intercept + b_x1 * x1_mesh + b_x2 * x2_mesh + b_x1x2 * x1_mesh * x2_mesh
    pm.Deterministic("p_grid", pm.math.invlogit(logit_grid), dims=("x2", "x1"))

grid_draws = pm.compute_deterministics(idata.posterior, model=decision_model)
```

We finally get the plot and the predictions on the test set:

```{code-cell} ipython3
# posterior probability that each grid point lies on the y = 1 side of the decision boundary
prob_class_1 = (grid_draws["p_grid"] > 0.5).mean(dim=["chain", "draw"])


def plot_decision_surface(ax):
    fill = ax.contourf(
        x1_grid, x2_grid, prob_class_1, levels=np.linspace(0, 1, 11), cmap=cmap, alpha=0.3
    )
    # decision boundary with a 90% credible band
    boundary = ax.contour(
        x1_grid,
        x2_grid,
        prob_class_1,
        levels=[0.05, 0.5, 0.95],
        colors="black",
        linestyles=[":", "-", ":"],
    )
    sns.scatterplot(x=x1_test, y=x2_test, hue=y_test, palette=class_palette, ax=ax, **marker_style)
    return fill, boundary


fig, ax = plt.subplots()
fill, boundary = plot_decision_surface(ax)
fig.colorbar(fill, ax=ax, label="P(p > 0.5)")
ax.clabel(boundary, fmt="%.2f")
ax.legend(title="y", loc="upper left")
ax.set(title="Model Decision Boundary", xlim=(-9, 9), ylim=(-9, 9), xlabel="x1", ylabel="x2")

# zoom on the asymptotes of the boundary at the posterior mean, x1 = -b2 / b12 and x2 = -b1 / b12
b_mean = idata.posterior["b"].mean(dim=["chain", "draw"])
x1_center = float(-b_mean.sel(coeffs="x2") / b_mean.sel(coeffs="x1:x2"))
x2_center = float(-b_mean.sel(coeffs="x1") / b_mean.sel(coeffs="x1:x2"))
half_width = 1.5

inset = ax.inset_axes([0.03, 0.03, 0.32, 0.32])
plot_decision_surface(inset)
inset.get_legend().remove()
inset.set(
    xlim=(x1_center - half_width, x1_center + half_width),
    ylim=(x2_center - half_width, x2_center + half_width),
    xticks=[],
    yticks=[],
)
for spine in inset.spines.values():
    spine.set(visible=True, edgecolor="black", linewidth=1.5)
ax.indicate_inset_zoom(inset, edgecolor="black");
```

The solid line is the 0.5 contour of the posterior probability that a point falls on the $y = 1$ side of the decision boundary, and it traces both branches of the hyperbola. The dotted lines are the 0.05 and 0.95 contours: outside the band they enclose, the posterior puts at least 95% probability on one side of the boundary. The accuracy and AUC above are computed from point predictions, and evaluating them per posterior draw in the same way would give their posterior distributions.

+++

## References

- [Bayesian Analysis with Python (Second edition) - Chapter 4](https://github.com/aloctavodia/BAP/blob/master/code/Chp4/04_Generalizing_linear_models.ipynb)
- [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/)

+++

## Authors
- Created by [Juan Orduz](https://github.com/juanitorduz).
- Updated by [Benjamin T. Vincent](https://github.com/drbenvincent) to PyMC v4 in June 2022
- Re-executed by [Benjamin T. Vincent](https://github.com/drbenvincent) with PyMC v5 in December 2022
- Updated by [Christian Luhmann](https://github.com/cluhmann)  in December 2023 ([pymc-examples#616](https://github.com/pymc-devs/pymc-examples/pull/616))
- Updated by [Jesse Grabowski](https://github.com/jessegrabowski) to PyMC v6 and ArviZ v1 in October 2026

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor
```

:::{include} ../page_footer.md
:::
