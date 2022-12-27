---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: 'Python 3.7.6 64-bit (''website_projects'': conda)'
  metadata:
    interpreter:
      hash: fbddea5140024843998ae64bf59a7579a9660d103062604797ea5984366c686c
  name: python3
---

# GLM in PyMC3: Out-Of-Sample Predictions

In this notebook I explore the [glm](https://docs.pymc.io/api/glm.html) module of [PyMC3](https://docs.pymc.io/). I am particularly interested in the model definition using [patsy](https://patsy.readthedocs.io/en/latest/) formulas, as it makes the model evaluation loop faster (easier to include features and/or interactions). There are many good resources on this subject, but most of them evaluate the model in-sample. For many applications we require doing predictions on out-of-sample data. This experiment was motivated by the discussion of the thread ["Out of sample" predictions with the GLM sub-module](https://discourse.pymc.io/t/out-of-sample-predictions-with-the-glm-sub-module/773) on the (great!) forum [discourse.pymc.io/](https://discourse.pymc.io/), thank you all for your input!

**Resources**


- [PyMC3 Docs: Example Notebooks](https://docs.pymc.io/nb_examples/index.html)
    
    - In particular check [GLM: Logistic Regression](https://docs.pymc.io/notebooks/GLM-logistic.html)

- [Bambi](https://bambinos.github.io/bambi/), a more complete implementation of the GLM submodule which also allows for mixed-effects models.

- [Bayesian Analysis with Python (Second edition) - Chapter 4](https://github.com/aloctavodia/BAP/blob/master/code/Chp4/04_Generalizing_linear_models.ipynb)
- [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/)

+++

## Prepare Notebook

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style(style="darkgrid", rc={"axes.facecolor": ".9", "grid.color": ".8"})
sns.set_palette(palette="deep")
sns_c = sns.color_palette(palette="deep")

import arviz as az
import patsy
import pymc3 as pm

from pymc3 import glm

plt.rcParams["figure.figsize"] = [7, 6]
plt.rcParams["figure.dpi"] = 100
```

## Generate Sample Data

We want to fit a logistic regression model where there is a multiplicative interaction between two numerical features.

```{code-cell} ipython3
SEED = 42
np.random.seed(SEED)

# Number of data points.
n = 250
# Create features.
x1 = np.random.normal(loc=0.0, scale=2.0, size=n)
x2 = np.random.normal(loc=0.0, scale=2.0, size=n)
epsilon = np.random.normal(loc=0.0, scale=0.5, size=n)
# Define target variable.
intercept = -0.5
beta_x1 = 1
beta_x2 = -1
beta_interaction = 2
z = intercept + beta_x1 * x1 + beta_x2 * x2 + beta_interaction * x1 * x2
p = 1 / (1 + np.exp(-z))
y = np.random.binomial(n=1, p=p, size=n)

df = pd.DataFrame(dict(x1=x1, x2=x2, y=y))

df.head()
```

Let us do some exploration of the data:

```{code-cell} ipython3
sns.pairplot(
    data=df, kind="scatter", height=2, plot_kws={"color": sns_c[1]}, diag_kws={"color": sns_c[2]}
);
```

- $x_1$ and $x_2$ are not correlated.
- $x_1$ and $x_2$ do not seem to separate the $y$-classes independently.
- The distribution of $y$ is not highly unbalanced.

```{code-cell} ipython3
fig, ax = plt.subplots()
sns_c_div = sns.diverging_palette(240, 10, n=2)
sns.scatterplot(x="x1", y="x2", data=df, hue="y", palette=[sns_c_div[0], sns_c_div[-1]])
ax.legend(title="y", loc="center left", bbox_to_anchor=(1, 0.5))
ax.set(title="Sample Data", xlim=(-9, 9), ylim=(-9, 9));
```

## Prepare Data for Modeling

I wanted to use the *`classmethod`* `from_formula` (see [documentation](https://docs.pymc.io/api/glm.html)), but I was not able to generate out-of-sample predictions with this approach (if you find a way please let me know!). As a workaround, I created the features from a formula using [patsy](https://patsy.readthedocs.io/en/latest/) directly and then use *`class`* `pymc3.glm.linear.GLM` (this was motivated by going into the [source code](https://github.com/pymc-devs/pymc3/blob/master/pymc3/glm/linear.py)).

```{code-cell} ipython3
# Define model formula.
formula = "y ~ x1 * x2"
# Create features.
y, x = patsy.dmatrices(formula_like=formula, data=df)
y = np.asarray(y).flatten()
labels = x.design_info.column_names
x = np.asarray(x)
```

As pointed out on the [thread](https://discourse.pymc.io/t/out-of-sample-predictions-with-the-glm-sub-module/773) (thank you @Nicky!), we need to keep the labels of the features in the design matrix.

```{code-cell} ipython3
print(f"labels = {labels}")
```

Now we do a train-test split.

```{code-cell} ipython3
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=SEED)
```

## Define and Fit the Model

We now specify the model in PyMC3.

```{code-cell} ipython3
with pm.Model() as model:
    # Set data container.
    data = pm.Data("data", x_train)
    # Define GLM family.
    family = pm.glm.families.Binomial()
    # Set priors.
    priors = {
        "Intercept": pm.Normal.dist(mu=0, sd=10),
        "x1": pm.Normal.dist(mu=0, sd=10),
        "x2": pm.Normal.dist(mu=0, sd=10),
        "x1:x2": pm.Normal.dist(mu=0, sd=10),
    }
    # Specify model.
    glm.GLM(y=y_train, x=data, family=family, intercept=False, labels=labels, priors=priors)
    # Configure sampler.
    trace = pm.sample(5000, chains=5, tune=1000, target_accept=0.87, random_seed=SEED)
```

```{code-cell} ipython3
# Plot chains.
az.plot_trace(data=trace);
```

```{code-cell} ipython3
az.summary(trace)
```

The chains look good.

+++

## Generate Out-Of-Sample Predictions

Now we generate predictions on the test set.

```{code-cell} ipython3
# Update data reference.
pm.set_data({"data": x_test}, model=model)
# Generate posterior samples.
ppc_test = pm.sample_posterior_predictive(trace, model=model, samples=1000)
```

```{code-cell} ipython3
# Compute the point prediction by taking the mean
# and defining the category via a threshold.
p_test_pred = ppc_test["y"].mean(axis=0)
y_test_pred = (p_test_pred >= 0.5).astype("int")
```

## Evaluate Model

First let us compute the accuracy on the test set.

```{code-cell} ipython3
from sklearn.metrics import accuracy_score

print(f"accuracy = {accuracy_score(y_true=y_test, y_pred=y_test_pred): 0.3f}")
```

Next, we plot the [roc curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) and compute the [auc](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve).

```{code-cell} ipython3
from sklearn.metrics import RocCurveDisplay, auc, roc_curve

fpr, tpr, thresholds = roc_curve(
    y_true=y_test, y_score=p_test_pred, pos_label=1, drop_intermediate=False
)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots()
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
roc_display = roc_display.plot(ax=ax, marker="o", color=sns_c[4], markersize=4)
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
# Construct grid.
x1_grid = np.linspace(start=-9, stop=9, num=300)
x2_grid = x1_grid

x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)

x_grid = np.stack(arrays=[x1_mesh.flatten(), x2_mesh.flatten()], axis=1)

# Create features on the grid.
x_grid_ext = patsy.dmatrix(formula_like="x1 * x2", data=dict(x1=x_grid[:, 0], x2=x_grid[:, 1]))

x_grid_ext = np.asarray(x_grid_ext)

# Generate model predictions on the grid.
pm.set_data({"data": x_grid_ext}, model=model)
ppc_grid = pm.sample_posterior_predictive(trace, model=model, samples=1000)
```

Now we compute the model decision boundary on the grid for visualization purposes.

```{code-cell} ipython3
numerator = -(trace["Intercept"].mean(axis=0) + trace["x1"].mean(axis=0) * x1_grid)
denominator = trace["x2"].mean(axis=0) + trace["x1:x2"].mean(axis=0) * x1_grid
bd_grid = numerator / denominator

grid_df = pd.DataFrame(x_grid, columns=["x1", "x2"])
grid_df["p"] = ppc_grid["y"].mean(axis=0)
grid_df.sort_values("p", inplace=True)

p_grid = grid_df.pivot(index="x2", columns="x1", values="p").to_numpy()
```

We finally get the plot and the predictions on the test set:

```{code-cell} ipython3
fig, ax = plt.subplots()
cmap = sns.diverging_palette(240, 10, n=50, as_cmap=True)
sns.scatterplot(
    x=x_test[:, 1].flatten(),
    y=x_test[:, 2].flatten(),
    hue=y_test,
    palette=[sns_c_div[0], sns_c_div[-1]],
    ax=ax,
)
sns.lineplot(x=x1_grid, y=bd_grid, color="black", ax=ax)
ax.contourf(x1_grid, x2_grid, p_grid, cmap=cmap, alpha=0.3)
ax.legend(title="y", loc="center left", bbox_to_anchor=(1, 0.5))
ax.lines[0].set_linestyle("dotted")
ax.set(title="Model Decision Boundary", xlim=(-9, 9), ylim=(-9, 9), xlabel="x1", ylabel="x2");
```

**Remark:** Note that we have computed the model decision boundary by using the mean of the posterior samples. However, we can generate a better (and more informative!) plot if we use the complete distribution (similarly for other metrics like accuracy and auc). One way of doing this is by storing and computing it inside the model definition as a `Deterministic` variable as in [Bayesian Analysis with Python (Second edition) - Chapter 4](https://github.com/aloctavodia/BAP/blob/master/code/Chp4/04_Generalizing_linear_models.ipynb).

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w
```
