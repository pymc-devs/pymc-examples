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

(data_container)=
# Using shared variables (`Data` container adaptation)

:::{post} Dec 16, 2021
:tags: posterior predictive, shared data 
:category: beginner
:author: Juan Martin Loyola, Kavya Jaiswal, Oriol Abril
:::

```{code-cell} ipython3
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

from numpy.random import default_rng

print(f"Running on PyMC3 v{pm.__version__}")
```

```{code-cell} ipython3
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
RANDOM_SEED = 8927
rng = default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
```

## The Data class

The {class}`pymc.Data` container class wraps the theano shared variable class and lets the model be aware of its inputs and outputs. This allows one to change the value of an observed variable to predict or refit on new data. All variables of this class must be declared inside a model context and specify a name for them.

In the following example, this is demonstrated with fictional temperature observations.

```{code-cell} ipython3
df_data = pd.DataFrame(columns=["date"]).set_index("date")
dates = pd.date_range(start="2020-05-01", end="2020-05-20")

for city, mu in {"Berlin": 15, "San Marino": 18, "Paris": 16}.items():
    df_data[city] = rng.normal(loc=mu, size=len(dates))

df_data.index = dates
df_data.index.name = "date"
df_data.head()
```

PyMC3 can also keep track of the dimensions (like dates or cities) and coordinates (such as the actual date times or city names) of multi-dimensional data. That way, when wrapping your data around the `Data` container when building your model, you can specify the dimension names and coordinates of random variables, instead of specifying the shapes of those random variables as numbers.

More generally, there are two ways to specify new dimensions and their coordinates:
- Entering the dimensions in the `dims` kwarg of a `pm.Data` variable with a pandas Series or DataFrame. The name of the index and columns will be remembered as the dimensions, and PyMC3 will infer that the values of the given columns must be the coordinates.
- Using the new `coords` argument to {class}`pymc.Model` to set the coordinates explicitly.

For more explanation about dimensions, coordinates and their big benefits, we encourage you to take a look at the {ref}`ArviZ documentation <arviz:xarray_for_arviz>`.

This is a lot of explanation -- let's see how it's done! We will use a hierarchical model: it assumes a mean temperature for the European continent and models each city relative to the continent mean:

```{code-cell} ipython3
# The data has two dimensions: date and city
coords = {"date": df_data.index, "city": df_data.columns}
```

```{code-cell} ipython3
with pm.Model(coords=coords) as model:
    europe_mean = pm.Normal("europe_mean_temp", mu=15.0, sigma=3.0)
    city_offset = pm.Normal("city_offset", mu=0.0, sigma=3.0, dims="city")
    city_temperature = pm.Deterministic("city_temperature", europe_mean + city_offset, dims="city")

    data = pm.Data("data", df_data, dims=("date", "city"))
    pm.Normal("likelihood", mu=city_temperature, sigma=0.5, observed=data)

    idata = pm.sample(
        2000,
        tune=2000,
        target_accept=0.85,
        return_inferencedata=True,
        random_seed=RANDOM_SEED,
    )
```

We can plot the digraph for our model using:

```{code-cell} ipython3
pm.model_to_graphviz(model)
```

And we see that the model did remember the coords we gave it:

```{code-cell} ipython3
model.coords
```

Coordinates are automatically stored into the {class}`arviz.InferenceData` object:

```{code-cell} ipython3
idata.posterior.coords
```

```{code-cell} ipython3
az.plot_trace(idata, var_names=["europe_mean_temp", "city_temperature"]);
```

We can get the data container variable from the model using:

```{code-cell} ipython3
model["data"].get_value()
```

Note that we used a theano method {meth}`theano.compile.sharedvalue.SharedVariable.get_value` of class {class}`theano.compile.sharedvalue.SharedVariable` to get the value of the variable. This is because our variable is actually a `SharedVariable`.

```{code-cell} ipython3
type(data)
```

The methods and functions related to the Data container class are:

- `data_container.get_value` (method inherited from the theano SharedVariable): gets the value associated with the `data_container`.
- `data_container.set_value` (method inherited from the theano SharedVariable): sets the value associated with the `data_container`.
- {func}`pymc.set_data`: PyMC3 function that sets the value associated with each Data container variable indicated in the dictionary `new_data` with it corresponding new value.

+++

## Using Data container variables to fit the same model to several datasets

This and the next sections are an adaptation of the notebook ["Advanced usage of Theano in PyMC3"](../Advanced_usage_of_Theano_in_PyMC3.html#using-shared-variables) using `pm.Data`.

We can use `Data` container variables in PyMC3 to fit the same model to several datasets without the need to recreate the model each time (which can be time consuming if the number of datasets is large):

```{code-cell} ipython3
:tags: [hide-output]

# We generate 10 datasets
true_mu = [rng.random() for _ in range(10)]
observed_data = [mu + rng.random(20) for mu in true_mu]

with pm.Model() as model:
    data = pm.Data("data", observed_data[0])
    mu = pm.Normal("mu", 0, 10)
    pm.Normal("y", mu=mu, sigma=1, observed=data)

# Generate one trace for each dataset
traces = []
for data_vals in observed_data:
    with model:
        # Switch out the observed dataset
        pm.set_data({"data": data_vals})
        traces.append(pm.sample(return_inferencedata=True))
```

## Using Data container variables to predict on new data

We can also sometimes use `Data` container variables to work around limitations in the current PyMC3 API. A common task in machine learning is to predict values for unseen data, and one way to achieve this is to use a `Data` container variable for our observations:

```{code-cell} ipython3
x = rng.random(100)
y = x > 0

with pm.Model() as model:
    x_shared = pm.Data("x_shared", x)
    coeff = pm.Normal("x", mu=0, sigma=1)

    logistic = pm.math.sigmoid(coeff * x_shared)
    pm.Bernoulli("obs", p=logistic, observed=y)

    # fit the model
    trace = pm.sample(return_inferencedata=True, tune=2000)
```

```{code-cell} ipython3
new_values = [-1, 0, 1.0]
with model:
    # Switch out the observations and use `sample_posterior_predictive` to predict
    pm.set_data({"x_shared": new_values})
    post_pred = pm.sample_posterior_predictive(trace)
```

The same concept applied to a more complex model can be seen in the notebook {ref}`bayesian_neural_network_advi`.

+++

## Applied example: height of toddlers as a function of age

+++

This example is taken from Osvaldo Martin's book: [Bayesian Analysis with Python: Introduction to statistical modeling and probabilistic programming using PyMC3 and ArviZ, 2nd Edition](https://www.amazon.com/Bayesian-Analysis-Python-Introduction-probabilistic-ebook/dp/B07HHBCR9G) {cite:p}`martin2018bayesian`.

+++

The World Health Organization and other health institutions around the world collect data
for newborns and toddlers and design [growth charts standards](http://www.who.int/childgrowth/en/). These charts are an essential component of the paediatric toolkit and also as a measure of the general well-being of
populations in order to formulate health policies, and plan interventions and
monitor their effectiveness.

An example of such data is the lengths (heights) of newborn / toddler girls as a function of age (in months):

```{code-cell} ipython3
try:
    data = pd.read_csv("../data/babies.csv")
except FileNotFoundError:
    data = pd.read_csv(pm.get_data("babies.csv"))
data.plot.scatter("Month", "Length", alpha=0.4);
```

To model this data we are going to use this model:

```{code-cell} ipython3
with pm.Model(coords={"time_idx": np.arange(len(data))}) as model_babies:
    α = pm.Normal("α", sigma=10)
    β = pm.Normal("β", sigma=10)
    γ = pm.HalfNormal("γ", sigma=10)
    δ = pm.HalfNormal("δ", sigma=10)

    month = pm.Data("month", data.Month.values.astype(float), dims="time_idx")

    μ = pm.Deterministic("μ", α + β * month**0.5, dims="time_idx")
    ε = pm.Deterministic("ε", γ + δ * month, dims="time_idx")

    length = pm.Normal("length", mu=μ, sigma=ε, observed=data.Length, dims="time_idx")

    idata_babies = pm.sample(tune=2000, return_inferencedata=True)
```

The following figure shows the result of our model. The expected length, $\mu$, is represented with a blue curve, and two semi-transparent orange bands represent the 60% and 94% highest posterior density intervals of posterior predictive length measurements:

```{code-cell} ipython3
with model_babies:
    idata_babies.extend(
        az.from_pymc3(posterior_predictive=pm.sample_posterior_predictive(idata_babies))
    )
```

```{code-cell} ipython3
ax = az.plot_hdi(
    data.Month,
    idata_babies.posterior_predictive["length"],
    hdi_prob=0.6,
    fill_kwargs={"alpha": 0.8},
)
ax.plot(
    data.Month,
    idata_babies.posterior["μ"].mean(("chain", "draw")),
    label="Posterior predictive mean",
)
ax = az.plot_lm(
    idata=idata_babies,
    y="length",
    x="month",
    kind_pp="hdi",
    y_kwargs={"color": "k", "ms": 6, "alpha": 0.15},
    y_hat_fill_kwargs=dict(fill_kwargs={"alpha": 0.4}),
    axes=ax,
)
```

At the moment of writing Osvaldo's daughter is two weeks ($\approx 0.5$ months) old, and thus he wonders how her length compares to the growth chart we have just created. One way to answer this question is to ask the model for the distribution of the variable length for babies of 0.5 months. Using PyMC3 we can ask this questions with the function `sample_posterior_predictive` , as this will return samples of _Length_ conditioned on the obseved data and the estimated distribution of parameters, that is including uncertainties. 

The only problem is that by default this function will return predictions for _Length_ for the observed values of _Month_, and $0.5$ months (the value Osvaldo cares about) has not been observed, -- all measures are reported for integer months. The easier way to get predictions for non-observed values of _Month_ is to pass new values to the `Data` container we defined above in our model. To do that, we need to use `pm.set_data` and then we just have to sample from the posterior predictve distribution:

```{code-cell} ipython3
ages_to_check = [0.5, 0.75]
with model_babies:
    pm.set_data({"month": ages_to_check})
    # we use two values instead of only 0.5 months to avoid triggering
    # https://github.com/pymc-devs/pymc3/issues/3640
    predictions = pm.sample_posterior_predictive(idata_babies)

    # add the generation predictions also to the inferencedata object
    # this is not necessary but allows for example storing data, posterior and predictions in the same file
    az.from_pymc3_predictions(
        predictions,
        idata_orig=idata_babies,
        inplace=True,
        # we update the dimensions and coordinates, we no longer have use for "time_idx"
        # as unique id. We'll now use the age in months as coordinate for better labeling and indexing
        # We duplicate the constant_data as coords though
        coords={"age (months)": ages_to_check},
        dims={"length": ["age (months)"], "month": ["age (months)"]},
    )
```

Now we can plot the expected distribution of lengths for 2-week old babies and compute additional quantities -- for example the percentile of a child given her length. Here, let's imagine that the child we're interested in has a length of 51.5:

```{code-cell} ipython3
ref_length = 51.5

az.plot_posterior(
    idata_babies,
    group="predictions",
    ref_val={"length": [{"age (months)": 0.5, "ref_val": ref_length}]},
    labeller=az.labels.DimCoordLabeller(),
);
```

## Authors
* Authored by [Juan Martin Loyola](https://github.com/jmloyola) in March, 2019 ([pymc#3389](https://github.com/pymc-devs/pymc/pull/3389))
* Updated by [Kavya Jaiswal](https://github.com/KavyaJaiswal) and [Oriol Abril](https://github.com/OriolAbril) in December, 2021 ([pymc-examples#151](https://github.com/pymc-devs/pymc-examples/pull/151))

+++

## References

:::{bibliography}
:filter: docname in docnames
:::

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p theano,xarray
```

:::{include} ../page_footer.md
:::
