---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: pymc-marketing
  language: python
  name: pymc-marketing
---

(ModelBuilder usage example)=
# ModelBuilder usage example

:::{post} Aug 18, 2023
:tags: ModelBuilder, model deployment,
:category: intermediate, tutorial
:author: Michał Raczycki
:::

+++

# Deploying MMMs and CLVs in Production: Saving and Loading Models

+++

In this article, we'll tackle the historically challenging process of deploying Bayesian models built with PyMC. Introducing a revolutionary deployment module, we bring unprecedented simplicity and efficiency to the deployment of PyMC models. As we prioritize user-friendly solutions, let's delve into how this innovation can significantly elevate your data science projects.

+++


Recent release of PyMC-Marketing by [Labs](https://www.pymc-labs.io) proves to be a big hit [(PyMC-Marketing)](https://www.pymc-labs.io/blog-posts/pymc-marketing-a-bayesian-approach-to-marketing-data-science/). In the feedback one could see an ongoing theme, many of you have been requesting easy and robust way of deploying models to production. It’s been a long-standing problem with PyMC ( and most other Probabilistic Programming Languages). The reason for that is that there’s no obvious way, and doesn’t matter which approach you try it proves to be tricky. That is why we’re happy to announce the release of `ModelBuilder`, brand new PyMC-experimental module that addresses this need, and improves on the deployment process significantly.

The ModelBuilder module is a new feature of PyMC based models. It provides 2 easy-to-use methods: save() and load() that can be used after the model has been fit.save() allow easy preservation of the model to .netcdf format, and load() gives one-line replication of the original model. Users can control the prior settings with model_config, and customize the sampling process using sampler_config. Default values of those are working just fine, so first time give it a try without changing, and provide your own model_config and model_sampler if afterwards you want to try to customize it more for your use case!

+++

For this notebook I'll use the example model used in [MMM Example Notebook](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_example.html), but ommit the details of data generation and plotting functionalities, since they're out of scope for this introduction, I highly recommend to see that part as well, but for now let's focus on today's topic: Groundbreaking deployment improvements in PyMC-Marketing!

```{code-cell} ipython3
import arviz as az
import numpy as np
import pandas as pd

from pymc_marketing.mmm import DelayedSaturatedMMM
```

```{code-cell} ipython3
az.style.use("arviz-darkgrid")
```

Let's load the dataset:

```{code-cell} ipython3
url = "https://raw.githubusercontent.com/pymc-labs/pymc-marketing/main/datasets/mmm_example.csv"
df = pd.read_csv(url)
df
```

But for our model we need much smaller dataset, many of the previous features were contributing to generation of others, now as our target variable is computed we can filter out not needed columns:

```{code-cell} ipython3
columns_to_keep = [
    "date_week",
    "y",
    "x1",
    "x2",
    "event_1",
    "event_2",
    "dayofyear",
]
seed: int = sum(map(ord, "mmm"))
rng = np.random.default_rng(seed=seed)

data = df[columns_to_keep].copy()

data["t"] = range(df.shape[0])
data.head()
```

## _Model Creation_
After we have our dataset ready, we could proceed straight to our model definition, but first to show the full potential of one of the new features: `model_config` we need to use some of our data to define our prior for sigma parameter for each of the channels. `model_config` is a customizable dictionary with keys corresponding to priors within the model, and values containing a dictionaries with parameters necessary to initialize them. Later on we'll learn that through the `save()` method we can preserve our priors contained inside the `model_config`, to allow complete replication of our model.

+++

### model_config

+++

`default_model_config` attribute of every model inheriting from `ModelBuilder` will allow you to see which priors are available for customization. To see it simply initialize a dummy model:

```{code-cell} ipython3
dummy_model = DelayedSaturatedMMM(date_column="", channel_columns="", adstock_max_lag=4)
dummy_model.default_model_config
```

You can change only the prior parameters that you wish, no need to alter all of them, unless you'd like to!
In this case we'll just simply replace our sigma for beta_channel with our computed one:

+++

First, let's compute the share of spend per channel:

```{code-cell} ipython3
total_spend_per_channel = data[["x1", "x2"]].sum(axis=0)

spend_share = total_spend_per_channel / total_spend_per_channel.sum()

spend_share
```

Next, we specify the `sigma`parameter per channel:

```{code-cell} ipython3
# The scale necessary to make a HalfNormal distribution have unit variance
HALFNORMAL_SCALE = 1 / np.sqrt(1 - 2 / np.pi)

n_channels = 2

prior_sigma = HALFNORMAL_SCALE * n_channels * spend_share.to_numpy()

prior_sigma.tolist()
```

```{code-cell} ipython3
custom_beta_channel_prior = {"beta_channel": {"sigma": prior_sigma, "dims": ("channel",)}}
my_model_config = dummy_model.default_model_config | custom_beta_channel_prior
```

As mentioned in the original notebook: "_For the prior specification there is no right or wrong answer. It all depends on the data, the context and the assumptions you are willing to make. It is always recommended to do some prior predictive sampling and sensitivity analysis to check the impact of the priors on the posterior. We skip this here for the sake of simplicity. If you are not sure about specific priors, the `DelayedSaturatedMMM` class has some default priors that you can use as a starting point._"

+++

The second feature that we can use for model definition is `sampler_config`. Similar to `model_config`, it's a dictionary that gets saved and contains things you'd usually pass to the `fit()` kwargs. It's not mandatory to create your own `sampler_config`; if not provided, both `model_config` and `sampler_config` will default to the forms specified by PyMC Labs experts, which allows for the usage of all model functionalities. The default `sampler_config` is left empty because the default sampling parameters usually prove sufficient for a start.

```{code-cell} ipython3
dummy_model.default_sampler_config
```

```{code-cell} ipython3
my_sampler_config = {
    "tune": 1000,
    "draws": 1000,
    "chains": 4,
    "target_accept": 0.95,
}
```

Let's finally assemble our model!

```{code-cell} ipython3
mmm = DelayedSaturatedMMM(
    model_config=my_model_config,
    sampler_config=my_sampler_config,
    date_column="date_week",
    channel_columns=["x1", "x2"],
    control_columns=[
        "event_1",
        "event_2",
        "t",
    ],
    adstock_max_lag=8,
    yearly_seasonality=2,
)
```

An important thing to note here is that in the new version of `DelayedSaturatedMMM`, we don't pass our dataset to the class constructor itself. This is due to a reason I've mentioned before - it supports `sklearn` transformers and validations that require a usual X, y split and typically expect the data to be passed to the `fit()` method.

+++

## _Model Fitting_

+++

Let's split the dataset:

```{code-cell} ipython3
X = data.drop("y", axis=1)
y = data["y"]
```

All that's left now is to finally fit the model:

As you can see below, you can still pass the sampler kwargs directly to `fit()` method. However, only those kwargs passed using `sampler_config` will be saved. Therefore, only these will be available after loading the model.

```{code-cell} ipython3
mmm.fit(X=X, y=y, random_seed=rng)
```

The `fit()` method automatically builds the model using the priors from `model_config`, and assigns the created model to our instance. You can access it as a normal attribute.

```{code-cell} ipython3
type(mmm.model)
```

```{code-cell} ipython3
mmm.graphviz()
```

posterior trace can be accessed by `fit_result` attribute:

```{code-cell} ipython3
mmm.fit_result
```

If you wish to inspect the entire inference data, use the `idata` attribute. Within `idata`, you can find the entire dataset passed to the model under `fit_data`.

```{code-cell} ipython3
mmm.idata
```

## `Save` and `load`

+++

All the data passed to the model on initialisation is stored in `idata.attrs`. This will be used later in the `save()` method to convert both this data and all the fit data into the netCDF format.

+++

Simply specify the path to which you'd like to save your model:

```{code-cell} ipython3
mmm.save("my_saved_model.nc")
```

And pass it to the `load()` method when it's needed again on the target system:

```{code-cell} ipython3
loaded_model = DelayedSaturatedMMM.load("my_saved_model.nc")
```

```{code-cell} ipython3
loaded_model.graphviz()
```

```{code-cell} ipython3
loaded_model.idata
```

A model loaded in this way is ready to be used for sampling and prediction, and has access to all previous samples and data.

```{code-cell} ipython3
with loaded_model.model:
    new_predictions = loaded_model.sample_posterior_predictive(
        X, extend_idata=True, combined=False, random_seed=rng
    )
new_predictions
```

```{code-cell} ipython3
az.plot_ppc(loaded_model.idata);
```

## Summary:

+++

In summary, this article introduces the revolutionary ModelBuilder, a new [PyMC-experimental](https://github.com/pymc-devs/pymc-experimental) module that simplifies the deployment of PyMC Bayesian models. It addresses a historic challenge faced by users of PyMC and most PPLs by offering a user-friendly and efficient approach to model deployment. The ModelBuilder provides two straightforward methods, save() and load(), which streamline the model preservation and replication process post fitting. Users are offered flexibility in controlling the prior settings with `model_config` and customizing the sampling process via `sampler_config`.

The use of an example model from the [MMM Example Notebook](https://www.pymc-marketing.io/en/stable/notebooks/index.html) demonstrates the practical implementation of `ModelBuilder`, emphasizing its ability to enhance model sharing among teams without the necessity for extensive domain knowledge about the model. The deployment improvements in [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing) brought about by ModelBuilder are not only user-friendly but also significantly enhance efficiency, making PyMC models more accessible for a wider audience.

+++

Even though this introduction is using `DelayedSaturatedMMM`, functionalities from `ModelBuilder` are available in the CLV models as well.

+++

## Authors
- Authored by [Michał Raczycki](https://github.com/michaelraczycki) in August 2023

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,aeppl,xarray
```

:::
{include} ../page_footer.md
:::
