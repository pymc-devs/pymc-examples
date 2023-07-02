---
jupytext:
  notebook_metadata_filter: substitutions
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: pymc_env
  language: python
  name: pymc_env
---

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": []}

(interventional_distribution)=
# Conditional vs interventional distributions

:::{post} July, 2023
:tags: causal inference, do-operator
:category: beginner, explanation
:author: Benjamin T. Vincent
:::

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": []}

:::{attention}
This notebook relies on experimental functionality currently in the [pymc-experimental](https://github.com/pymc-devs/pymc-experimental) repository. In the near future this will be moved into the main [pymc](https://github.com/pymc-devs/pymc) repository.
:::

+++

In this post we are going to go beyond _statistical_ concepts and cover some important _causal_ concepts. In particular we are going to examine how we can ask "what-it?" questions based on possible interventions we could make, or could have made in the past.

So intervention is not necessarily something we actually have to carry out in the real world - hence the "what-if?" nature of the questions. But we can ask, given what we know, what do we believe if we intervene (or had intervened) on a system.

This notion of intervention can be carried out by the $\operatorname{do}$ operator. We will learn what this mysterious sounding thing is, how it works, and how we can do it in PyMC.

+++

## Set up the notebook

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: []
---
import arviz as az
import graphviz as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pymc_experimental as pmx
import seaborn as sns

from packaging import version
```

```{code-cell} ipython3
RANDOM_SEED = 123
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
sns.color_palette("tab10")
%config InlineBackend.figure_format = 'retina'
```

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": []}

Check we have the necessary versions to get the new experimental functionality.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: []
---
assert version.parse(pm.__version__) >= version.parse("5.5.0")
assert version.parse(pmx.__version__) >= version.parse("0.0.7")

# import the new functionality
from pymc_experimental.model_transform.conditioning import do, observe
```

## The $\operatorname{do}$ operator

The $\operatorname{do}$ operator implements an intervention that we want to make. It consists of 2 simple steps:
1. It takes a given node in a graph and sets that node at the desired value.
2. It removes any causal influence of this node by other nodes. It does this by removing all incoming edges into that node.

Here is a visual demonstration of that using an example from {cite:t}`pearl2000causality`.

![](sprinkler.png)

On the left of the figure we have a causal directed acyclic graph describing the causal relationships between season, whether a sprinkler has been on, whether it has rained, if the grass is wet, and if the grass is slippery. The joint distribution could be described as: 

$$
P(x_1, x_2, x_3, x_4, x_5) = P(x_1) P(x_3|x_1) P(x_2|x_1) P(x_4|x_3, x_2) P(x_5|x_4)
$$

On the right of the figure we have applied the $\operatorname{do}$ operator to examine what will happen if we set the sprinkler to be on. You can see that we have now set the value of that node, $x_3=1$ and we have removed the incoming edge (influence) of season, meaning that once we turn on the sprinkler manually, it's not influenced by the season anymore.

We could now describe this _interventional distribution_ as:

$$
P(x_1, x_2, \operatorname{do}(x_3=1), x_4, x_5) = P(x_1) P(x_2|x_1) P(x_4|x_3=1, x_2) P(x_5|x_4)
$$

Interested readers should check out the richly diagrammed and well-explained blog post [Causal Effects via the Do-operator](https://towardsdatascience.com/causal-effects-via-the-do-operator-5415aefc834a) {cite:p}`Talebi2022dooperator` as a good place to start.

+++ {"editable": true, "raw_mimetype": "", "slideshow": {"slide_type": ""}, "tags": []}

## Three different causal DAGS

:::{note}
This section takes heavy inspiration from the post [Causal Inference 2: Illustrating Interventions via a Toy Example](https://www.inference.vc/causal-inference-2-illustrating-interventions-in-a-toy-example/) {cite:p}`Huszár2019causal2`. Imitation is the sincerest form of flattery.
:::

If we think about how 2 variables, $x$ and $y$, are related we can come up with many different causal DAGS. Below we consider just 3 possibilities, which we'll label DAG 1, 2, and 3.

1. $x$ causally influences $y$
2. $y$ causally influences $x$
3. $z$ causally influences both $x$ and $y$

We can draw these more graphically below:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
g = gr.Digraph()

# DAG 1
g.node(name="x1", label="x")
g.node(name="y1", label="y")
g.edge(tail_name="x1", head_name="y1")

# DAG 2
g.node(name="y2", label="y")
g.node(name="x2", label="x")
g.edge(tail_name="y2", head_name="x2")

# DAG 3
g.node(name="z", label="z")
g.node(name="x", label="x")
g.node(name="y", label="y")
g.edge(tail_name="z", head_name="x")
g.edge(tail_name="z", head_name="y")

g
```

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": []}

We can also imagine implementing such causal DAGS in Python code to generate `N` random numbers. Each of these will give rise to specific joint distributions, $P(x, y)$, and in fact, because Ferenc Huszár was clever in his blog post, we'll see later that these will all give rise to the same joint distributions.

**DAG 1**

```{code-block} python
x = rng.normal(loc=0, scale=1, size=N)
y = x + 1 + np.sqrt(3) * rng.normal(size=N)
```

**DAG 2**

```{code-block} python
y = 1 + 2 * rng.normal(size=N)
x = (y - 1) / 4 + np.sqrt(3) * rng.normal(size=N) / 2
```

**DAG 3**

```{code-block} python
z = rng.normal(size=N)
y = z + 1 + np.sqrt(3) * rng.normal(size=N)
x = z
```

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": []}

However, we are going to implement these using Bayesian causal DAGS with PyMC. Let's see how we can do this, then generate samples from them using `pm.sample_prior_predictive`. As we go with each DAG, we'll package the data up in `DataFrame`'s for plotting later, and also plot the graphviz representation of the PyMC models. You'll see that while these are a fraction more visually complex, they do actually match up with the causal DAGs we've specified above.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: []
---
# number of samples to generate
N = 1_000_000
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: []
---
with pm.Model() as model1:
    x = pm.Normal("x")
    temp = pm.Normal("temp")
    y = pm.Deterministic("y", x + 1 + np.sqrt(3) * temp)
    idata1 = pm.sample_prior_predictive(samples=N)

df1 = az.extract(idata1.prior, var_names=["x", "y"]).to_dataframe()

pm.model_to_graphviz(model1)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: []
---
with pm.Model() as model2:
    y = pm.Normal("y", mu=1, sigma=2)
    temp = pm.Normal("temp")
    x = pm.Deterministic("x", (y - 1) / 4 + np.sqrt(3) * temp / 2)
    idata2 = pm.sample_prior_predictive(samples=N)

df2 = az.extract(idata2.prior, var_names=["x", "y"]).to_dataframe()

pm.model_to_graphviz(model2)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: []
---
with pm.Model() as model3:
    z = pm.Normal("z")
    temp = pm.Normal("temp")
    y = pm.Deterministic("y", z + 1 + np.sqrt(3) * temp)
    x = pm.Deterministic("x", z)
    idata3 = pm.sample_prior_predictive(samples=N)

df3 = az.extract(idata3.prior, var_names=["x", "y"]).to_dataframe()

pm.model_to_graphviz(model3)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": []}

### Joint distributions, $P(x,y)$

First, let's take a look at the joint distributions for each of the DAGs to convince ourselves that these are actually the same.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
fig, ax = plt.subplots(1, 3, figsize=(12, 8), sharex=True, sharey=True)

for i, df in enumerate([df1, df2, df3]):
    az.plot_kde(
        df["x"],
        df["y"],
        hdi_probs=[0.25, 0.5, 0.75, 0.9, 0.95],
        contour_kwargs={"colors": None},
        contourf_kwargs={"alpha": 0.5},
        ax=ax[i],
    )
    ax[i].set(
        title=f"$P(x, y)$, DAG {i+1}",
        xlim=[-4, 4],
        xticks=np.arange(-4, 4 + 1, step=2),
        ylim=[-6, 8],
        yticks=np.arange(-6, 8 + 1, step=2),
        aspect="equal",
    )
    ax[i].axvline(x=2, ls="--", c="k")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": []}

The dashed lines at $x=2$ help us imagine the conditional distribution $P(y|x=2)$ that we'll examine in the next section. Seeing as the joint distributions are the same, it is intuitive to imagine that the conditional distributions $P(y|x=2)$ will be identical for each of the 3 DAGs.

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": []}

### Conditional distributions, $P(y|x=2)$

+++

In the MCMC spirit of representing probability distributions by samples, let's now calculate the conditional distributions. If we picked all the values where $x$ was _exactly_ 2, then we might not end up with any samples at all, so what we'll do is to take a very narrow slice of samples around 2. So these will be approximations - as the number of samples increases and the width of the slice decreases, then our approximation would become more accurate.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: []
---
# Extract samples from P(y|x≈2)
conditional1 = df1.query("1.99 < x < 2.01")["y"]
conditional2 = df2.query("1.99 < x < 2.01")["y"]
conditional3 = df3.query("1.99 < x < 2.01")["y"]
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
# put the conditional distributions into a convenient long-format data frame
df1_new = pd.DataFrame({"Conditional": conditional1, "DAG": 1})
df2_new = pd.DataFrame({"Conditional": conditional2, "DAG": 2})
df3_new = pd.DataFrame({"Conditional": conditional3, "DAG": 3})
df_conditional = pd.concat([df1_new, df2_new, df3_new])
df_conditional.reset_index(drop=True, inplace=True)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": []}

### Interventional distributions, $P(y|\operatorname{do}(x=2))$

In turn for each of the 3 DAGs, let's use the $\operatorname{do}$ operator, setting $x=2$. This will give us a new DAG and we'll plot the graphviz representation and then take samples to represent the interventional distribution.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: []
---
model1_do = do(model1, {"x": 2})
pm.model_to_graphviz(model1_do)
```

:::{important}
Let's just take a moment to reflect on what we've done here! We took a model (`model1`) and then used the $\operatorname{do}$ function and specified an intervention we wanted to make. In this case it was to set $x=2$. We then got back a new model where the original DAG has been mutilated in the way that we set out above. Namely, we defined $x=2$ _and_ removed edges from incoming nodes to $x$. In this first DAG, there were no incoming edges, but this is the case in DAG2 and DAG 3 below.
:::

```{code-cell} ipython3
model2_do = do(model2, {"x": 2})
pm.model_to_graphviz(model2_do)
```

```{code-cell} ipython3
model3_do = do(model3, {"x": 2})
pm.model_to_graphviz(model3_do)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": []}

So we can see that in DAG 1, the $x$ variable still has causal influence on $y$. However, in DAGs 2 and 3, $y$ is no longer causally influenced by $x$. So in DAGs 2 and 3, our intervention $\operatorname{do}(x=2)$ have no influence on $y$.

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": []}

Next we'll sample from each of these interventional distributions. Note that we are using the mutilated models, `model1_do`, `model2_do`, and `model3_do`. 

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: []
---
with model1_do:
    idata1_do = pm.sample_prior_predictive(samples=N)

with model2_do:
    idata2_do = pm.sample_prior_predictive(samples=N)

with model3_do:
    idata3_do = pm.sample_prior_predictive(samples=N)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
# put the interventional distributions into a convenient long-format data frame
df1_new = pd.DataFrame(
    {
        "Interventional": az.extract(idata1_do.prior, var_names="y").squeeze().data,
        "DAG": 1,
    }
)
df2_new = pd.DataFrame(
    {
        "Interventional": az.extract(idata2_do.prior, var_names="y").squeeze().data,
        "DAG": 2,
    }
)
df3_new = pd.DataFrame(
    {
        "Interventional": az.extract(idata3_do.prior, var_names="y").squeeze().data,
        "DAG": 3,
    }
)
df_interventional = pd.concat([df1_new, df2_new, df3_new])
df_interventional.reset_index(drop=True, inplace=True)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": []}

So let's compare the conditional and interventional distributions for all 3 DAGs.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

sns.kdeplot(
    df_conditional,
    x="Conditional",
    hue="DAG",
    common_norm=True,
    ax=ax[0],
    palette="tab10",
    lw=3,
)
ax[0].set(xlabel="y", title="Conditional distributions\n$P(y|x=2)$")

sns.kdeplot(
    df_interventional,
    x="Interventional",
    hue="DAG",
    common_norm=True,
    ax=ax[1],
    palette="tab10",
    lw=3,
)
ax[1].set(xlabel="y", title="Interventional distributions\n$P(y|\\operatorname{do}(x=2))$");
```

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": []}

We can see, as expected, that the conditional distributions are the same for all 3 DAGs. The story is different for the interventional distributions however. Here, DAG 1 differs because it is the only one where our $\operatorname{do}(x=2)$ intervention causally effects $y$. This intervention severed any causal influence of $x$ on $y$ in DAGs 2 and 3.

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": []}

## Authors
- Authored by [Benjamin T. Vincent](https://github.com/drbenvincent) in July 2023

+++

## References

:::{bibliography}
:filter: docname in docnames
:::

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,xarray
```

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": []}

:::{include} ../page_footer.md
:::
