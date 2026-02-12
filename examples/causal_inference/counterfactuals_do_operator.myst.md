---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: eabm
  language: python
  name: python3
---

(counterfactuals_do_operator)=
# Counterfactual generation using pymc do-operator

:::{post} August, 2023
:tags: causality, causal inference, do-operator, counterfactuals
:category: beginner, reference
:author: Shekhar Khandelwal
:::

```{code-cell} ipython3
import warnings

import arviz.preview as az
import numpy as np
import pandas as pd
import pymc as pm

warnings.filterwarnings("ignore")
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'  # high resolution figures
az.style.use("arviz-variat")
rng = np.random.default_rng(42)
SEED = 8927
```

# Introduction

In the realm of data science and analytics, understanding the causal relationships between variables is paramount. While traditional statistical methods have provided insights into these relationships, the advent of probabilistic programming has ushered in a new era of causal analysis. In this article, we will explore the power of counterfactuals in causal analysis using the PyMC framework, with a special focus on the “do-operator.”
Counterfactuals are essentially “what-if” scenarios that allow us to understand the potential outcomes had a different action been taken or a different condition been present. By leveraging the PyMC framework and its “do-operator,” we can programmatically simulate these scenarios, giving us a deeper understanding of the relationships between predictors and target variables.

Through a step-by-step guide, we will delve into the process of building a PyMC model skeleton, generating data using the do-operator, and validating the relationships captured by the model. Furthermore, we will explore the magic of the do-operator in simulating different ‘what-if’ scenarios, akin to programmatic A/B testing.

- Step 1. Build a pymc model skeleton
- Step 2. Use model skeleton and generate data using do-operator to infuse relationship between predictors and target variable (ssshhh, that’s a hidden superhero feature of do-operator ;) )
- Step 3. Use observe-operator to assign generated data on the model skeleton
- Step 4. Create samples and validate that the infused relationship between predictors and target variable are captured by the model samples (isn’t that what we expect a predictive model to do ;) )
- Step 5. Use do-operator to time travel, and generate target variable with different ‘what-if’ scenarios (basically mimic A/B testing…programatically)

+++

### Step 1. Build a pymc model skeleton

For this demo, we are building a very simple Linear Regression model.
- Predictor — ‘a’, ‘b’, ‘c’
- Target Variable — ‘y’
- Coefficients —
>- ‘beta_ay’ -> coefficient of |a|
>- ‘beta_by’ -> coefficient of |b|
>- ‘beta_cy’ -> coefficient of |c|

```{code-cell} ipython3
with pm.Model(coords={"i": [0]}) as model_generative:
    # priors
    beta_y0 = pm.Normal("beta_y0")
    beta_ay = pm.Normal("beta_ay")
    beta_by = pm.Normal("beta_by")
    beta_cy = pm.Normal("beta_cy")
    # observation noise on Y
    sigma_y = pm.HalfNormal("sigma_y")
    # core nodes and causal relationships
    a = pm.Normal("a", mu=0, sigma=1, dims="i")
    b = pm.Normal("b", mu=0, sigma=1, dims="i")
    c = pm.Normal("c", mu=0, sigma=1, dims="i")
    y_mu = pm.Deterministic(
        "y_mu", beta_y0 + (beta_ay * a) + (beta_by * b) + (beta_cy * c), dims="i"
    )
    y = pm.Normal("y", mu=y_mu, sigma=sigma_y, dims="i")


pm.model_to_graphviz(model_generative)
```

### Step 2. Use model skeleton and generate data using do-operator to infuse relationship between predictors and target variable. We will use this generated data for modelling later.

Let’s first define the predictors relationship with target variable.

```{code-cell} ipython3
true_values = {"beta_ay": 1.5, "beta_by": 0.7, "beta_cy": 0.3, "sigma_y": 0.2, "beta_y0": 0.0}
```

Basically what we are saying here is, we are intentionally defining the coefficient values, which we expect predictive model to predict later on.

Now the magic begins. We will use do-operator to use this dictionary and sample data variables. How do we do this ? Simple by passing two arguments to pymc do-operator. First, the model skeleton object. And second, the coefficient dictionary.

```{code-cell} ipython3
model_simulate = pm.do(model_generative, true_values)
```

This will create a new model object with the coefficent variables values infused. 

```{code-cell} ipython3
model_simulate.to_graphviz()
```

The gray shades on the coefficient variables depicts the tale. Check the previous model graph, it was all white.

Now, all we have to do is generate samples, the known pymc way.

Lets generate 100 samples.

```{code-cell} ipython3
N = 100

with model_simulate:
    simulate = pm.sample_prior_predictive(samples=N)
```

We know that this generates an Arviz object, and since we have called sample_prior_predictive, hence the object will only contain priors.

```{code-cell} ipython3
simulate
```

Extract the sampled prior data into a pandas dataframe.

```{code-cell} ipython3
observed = {
    "a": simulate.prior["a"].values.flatten(),
    "b": simulate.prior["b"].values.flatten(),
    "c": simulate.prior["c"].values.flatten(),
    "y": simulate.prior["y"].values.flatten(),
}

df = pd.DataFrame(observed)
print(df.shape)
df.head(5)
```

Ok, so now we are all set with a sample data.

+++

### Step 3. Use observe-operator to assign generated data on the model skeleton

Now, this is another cool feature of pymc newly introduced observe method. Observe method, takes in a model skeleton and the dictionary with the data for the variables we want to infuse into the model.

```{code-cell} ipython3
data_dict = {"a": df["a"], "b": df["b"], "c": df["c"], "y": df["y"]}
model_inference = pm.observe(model_generative, data_dict)
model_inference.set_dim("i", N, coord_values=np.arange(N))
pm.model_to_graphviz(model_inference)
```

See the gray matter again. This time we have observed data infused into the model, and we have to sample for the coefficient and other parameters.

So, lets sample.

### Step 4. Create samples and validate that the infused relationship between predictors and target variable are captured by the model samples

```{code-cell} ipython3
with model_inference:
    idata = pm.sample(random_seed=SEED)
```

Now, lets validate if model captured the infused coefficient values in the data.

```{code-cell} ipython3
pc = az.plot_dist(
    idata,
    var_names=list(true_values.keys()),
)
az.add_lines(pc, true_values);
```

BAM ! Pretty nice fit !

Now, lets do what we are supposed to do ! Counterfactuals.

Basically, this is about generating target variable values with different predictor values. Basically, answering what if questions !

_What-if there was all ‘b’ values as 0 ?_

_What-if all ‘b’ values were double ?_

How to do this ? Here you go..

### Step 5. Use do-operator to time travel, and generate target variable with different ‘what-if’ scenarios.
Since, we want to experiment with ‘b’, lets first assign observed values to ‘a’ and ‘c’. Not to ‘y’, because that’s what we want to sample. Correct !

```{code-cell} ipython3
model_counterfactual = pm.do(model_inference, {"a": df["a"], "c": df["c"]})
```

Now, lets begin the fun part. Let’s generate counterfactuals.

### _Scenario 1 :- What if all values for ‘b’ were 0 ?_

```{code-cell} ipython3
model_b0 = pm.do(model_counterfactual, {"b": np.zeros(N, dtype="int32")}, prune_vars=True)
model_b1 = pm.do(model_counterfactual, {"b": df["b"]}, prune_vars=True)
```

Just sample.

```{code-cell} ipython3
# Sample when 'b' was 0: P(y | (a,c), do(b=0))
idata_b0 = pm.sample_posterior_predictive(
    idata,
    model=model_b0,
    predictions=True,
    var_names=["y_mu"],
    random_seed=SEED,
)
# Sample when 'b' was as observed: P(y | (a,c), do(b=observed))
idata_b1 = pm.sample_posterior_predictive(
    idata,
    model=model_b1,
    predictions=True,
    var_names=["y_mu"],
    random_seed=SEED,
)
```

Some basic python and here we have the counterfactuals.

```{code-cell} ipython3
df["b_scenario_1"] = 0
df["y_scenario_1"] = (
    idata_b0.predictions.y_mu.mean(("chain", "draw")).values.reshape(1, -1).flatten()
)
df.head(5)
```

### _Scenario 2: What if ‘b’ was 5 times as observed_

```{code-cell} ipython3
model_b0 = pm.do(model_counterfactual, {"b": 5 * df["b"]}, prune_vars=True)
model_b1 = pm.do(model_counterfactual, {"b": df["b"]}, prune_vars=True)
```

Sample.

```{code-cell} ipython3
# Sample when 'b' was 5 times b: P(y | (a,c), do(b=5*b))
idata_b0 = pm.sample_posterior_predictive(
    idata,
    model=model_b0,
    predictions=True,
    var_names=["y_mu"],
    random_seed=SEED,
)
# Sample when 'b' was as observed: P(y | (a,c), do(b=observed))
idata_b1 = pm.sample_posterior_predictive(
    idata,
    model=model_b1,
    predictions=True,
    var_names=["y_mu"],
    random_seed=SEED,
)

df["b_scenario_2"] = 5 * df["b"]
df["y_scenario_2"] = (
    idata_b0.predictions.y_mu.mean(("chain", "draw")).values.reshape(1, -1).flatten()
)
df.head(5)
```

Ok, so now you got the idea. It's an open playground. Go back in time, change whatever you want to change, and see how output changes.

This opens the door for many more possibilities in various use cases. Especially, Causal Analytics !

+++

## Authors
- Authored by [Shekhar Khandelwal](https://github.com/shekharkhandelwal1983) in August 2023
- Updated by Osvaldo Martin in February 2026 

+++

## References

https://medium.com/@khandelwal-shekhar/counterfactuals-for-causal-analysis-via-pymc-do-operator-234ba04e4e80

https://www.pymc-labs.io/blog-posts/causal-analysis-with-pymc-answering-what-if-with-the-new-do-operator/

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor
```

:::{include} ../page_footer.md
:::
