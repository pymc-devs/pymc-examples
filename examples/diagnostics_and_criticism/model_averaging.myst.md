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

(model_averaging)=
# Model Averaging

:::{post} Aug 2022
:tags: model comparison, model averaging
:category: intermediate
:author: Osvaldo Martin
:::

```{code-cell} ipython3
---
papermill:
  duration: 4.910288
  end_time: '2020-11-29T12:13:07.788552'
  exception: false
  start_time: '2020-11-29T12:13:02.878264'
  status: completed
---
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

print(f"Running on PyMC3 v{pm.__version__}")
```

```{code-cell} ipython3
---
papermill:
  duration: 0.058811
  end_time: '2020-11-29T12:13:07.895012'
  exception: false
  start_time: '2020-11-29T12:13:07.836201'
  status: completed
---
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")
```

+++ {"papermill": {"duration": 0.068882, "end_time": "2020-11-29T12:13:08.020372", "exception": false, "start_time": "2020-11-29T12:13:07.951490", "status": "completed"}}

When confronted with more than one model we have several options. One of them is to perform model selection, using for example a given Information Criterion as exemplified the PyMC examples {ref}`pymc:model_comparison` and the {ref}`GLM-model-selection`. Model selection is appealing for its simplicity, but we are discarding information about the uncertainty in our models. This is somehow similar to computing the full posterior and then just keep a point-estimate like the posterior mean; we may become overconfident of what we really know. You can also browse the {doc}`blog/tag/model-comparison` tag to find related posts. 

One alternative is to perform model selection but discuss all the different models together with the computed values of a given Information Criterion. It is important to put all these numbers and tests in the context of our problem so that we and our audience can have a better feeling of the possible limitations and shortcomings of our methods. If you are in the academic world you can use this approach to add elements to the discussion section of a paper, presentation, thesis, and so on.

Yet another approach is to perform model averaging. The idea now is to generate a meta-model (and meta-predictions) using a weighted average of the models. There are several ways to do this and PyMC includes 3 of them that we are going to briefly discuss, you will find a more thorough explanation in the work by {cite:t}`Yao_2018`. PyMC integrates with ArviZ for model comparison. 


## Pseudo Bayesian model averaging

Bayesian models can be weighted by their marginal likelihood, this is known as Bayesian Model Averaging. While this is theoretically appealing, it is problematic in practice: on the one hand the marginal likelihood is highly sensible to the specification of the prior, in a way that parameter estimation is not, and on the other, computing the marginal likelihood is usually a challenging task. An alternative route is to use the values of WAIC (Widely Applicable Information Criterion) or LOO (pareto-smoothed importance sampling Leave-One-Out cross-validation), which we will call generically IC, to estimate weights. We can do this by using the following formula:

$$w_i = \frac {e^{ - \frac{1}{2} dIC_i }} {\sum_j^M e^{ - \frac{1}{2} dIC_j }}$$

Where $dIC_i$ is the difference between the i-esim information criterion value and the lowest one. Remember that the lowest the value of the IC, the better. We can use any information criterion we want to compute a set of weights, but, of course, we cannot mix them. 

This approach is called pseudo Bayesian model averaging, or Akaike-like weighting and is an heuristic way to compute the relative probability of each model (given a fixed set of models) from the information criteria values. Look how the denominator is just a normalization term to ensure that the weights sum up to one.

## Pseudo Bayesian model averaging with Bayesian Bootstrapping

The above formula for computing weights is a very nice and simple approach, but with one major caveat it does not take into account the uncertainty in the computation of the IC. We could compute the standard error of the IC (assuming a Gaussian approximation) and modify the above formula accordingly. Or we can do something more robust, like using a [Bayesian Bootstrapping](http://www.sumsar.net/blog/2015/04/the-non-parametric-bootstrap-as-a-bayesian-model/) to estimate, and incorporate this uncertainty.

## Stacking

The third approach implemented in PyMC is known as _stacking of predictive distributions_ by {cite:t}`Yao_2018`. We want to combine several models in a metamodel in order to minimize the divergence between the meta-model and the _true_ generating model, when using a logarithmic scoring rule this is equivalent to:

$$\max_{w} \frac{1}{n} \sum_{i=1}^{n}log\sum_{k=1}^{K} w_k p(y_i|y_{-i}, M_k)$$

Where $n$ is the number of data points and $K$ the number of models. To enforce a solution we constrain $w$ to be $w_k \ge 0$ and  $\sum_{k=1}^{K} w_k = 1$. 

The quantity $p(y_i|y_{-i}, M_k)$ is the leave-one-out predictive distribution for the $M_k$ model. Computing it requires fitting each model $n$ times, each time leaving out one data point. Fortunately we can approximate the exact leave-one-out predictive distribution using LOO (or even WAIC), and that is what we do in practice.

## Weighted posterior predictive samples

Once we have computed the weights, using any of the above 3 methods,  we can use them to get a weighted posterior predictive samples. PyMC offers functions to perform these steps in a simple way, so let see them in action using an example.

The following example is taken from the superb book {cite:t}`mcelreath2018statistical` by Richard McElreath. You will find more PyMC examples from this book in the repository [Statistical-Rethinking-with-Python-and-PyMC](https://github.com/pymc-devs/pymc-resources/tree/main/Rethinking_2). We are going to explore a simplified version of it. Check the book for the whole example and a more thorough discussion of both, the biological motivation for this problem and a theoretical/practical discussion of using Information Criteria to compare, select and average models.

Briefly, our problem is as follows: We want to explore the composition of milk across several primate species, it is hypothesized that females from species of primates with larger brains produce more _nutritious_ milk (loosely speaking this is done _in order to_ support the development of such big brains). This is an important question for evolutionary biologists and try to give an answer we will use 3 variables, two predictor variables: the proportion of neocortex compare to the total mass of the brain and the logarithm of the body mass of the mothers. And for predicted variable, the kilocalories per gram of milk. With these variables we are going to build 3 different linear models:
 
1. A model using only the neocortex variable
2. A model using only the logarithm of the mass variable
3. A model using both variables

Let start by uploading the data and centering the `neocortex` and `log mass` variables, for better sampling.

```{code-cell} ipython3
---
papermill:
  duration: 1.114901
  end_time: '2020-11-29T12:13:09.196103'
  exception: false
  start_time: '2020-11-29T12:13:08.081202'
  status: completed
---
d = pd.read_csv(
    "https://raw.githubusercontent.com/pymc-devs/resources/master/Rethinking_2/Data/milk.csv",
    sep=";",
)
d = d[["kcal.per.g", "neocortex.perc", "mass"]].rename({"neocortex.perc": "neocortex"}, axis=1)
d["log_mass"] = np.log(d["mass"])
d = d[~d.isna().any(axis=1)].drop("mass", axis=1)
d.iloc[:, 1:] = d.iloc[:, 1:] - d.iloc[:, 1:].mean()
d.head()
```

+++ {"papermill": {"duration": 0.048113, "end_time": "2020-11-29T12:13:09.292526", "exception": false, "start_time": "2020-11-29T12:13:09.244413", "status": "completed"}}

Now that we have the data we are going to build our first model using only the `neocortex`.

```{code-cell} ipython3
---
papermill:
  duration: 75.962348
  end_time: '2020-11-29T12:14:25.303027'
  exception: false
  start_time: '2020-11-29T12:13:09.340679'
  status: completed
---
with pm.Model() as model_0:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", 10)

    mu = alpha + beta * d["neocortex"]

    kcal = pm.Normal("kcal", mu=mu, sigma=sigma, observed=d["kcal.per.g"])
    trace_0 = pm.sample(2000, return_inferencedata=True)
```

+++ {"papermill": {"duration": 0.049578, "end_time": "2020-11-29T12:14:25.401979", "exception": false, "start_time": "2020-11-29T12:14:25.352401", "status": "completed"}}

The second model is exactly the same as the first one, except we now use the logarithm of the mass

```{code-cell} ipython3
---
papermill:
  duration: 8.996265
  end_time: '2020-11-29T12:14:34.447153'
  exception: false
  start_time: '2020-11-29T12:14:25.450888'
  status: completed
---
with pm.Model() as model_1:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=1)
    sigma = pm.HalfNormal("sigma", 10)

    mu = alpha + beta * d["log_mass"]

    kcal = pm.Normal("kcal", mu=mu, sigma=sigma, observed=d["kcal.per.g"])

    trace_1 = pm.sample(2000, return_inferencedata=True)
```

+++ {"papermill": {"duration": 0.049839, "end_time": "2020-11-29T12:14:34.547268", "exception": false, "start_time": "2020-11-29T12:14:34.497429", "status": "completed"}}

And finally the third model using the `neocortex` and `log_mass` variables

```{code-cell} ipython3
---
papermill:
  duration: 19.373847
  end_time: '2020-11-29T12:14:53.971081'
  exception: false
  start_time: '2020-11-29T12:14:34.597234'
  status: completed
---
with pm.Model() as model_2:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=1, shape=2)
    sigma = pm.HalfNormal("sigma", 10)

    mu = alpha + pm.math.dot(beta, d[["neocortex", "log_mass"]].T)

    kcal = pm.Normal("kcal", mu=mu, sigma=sigma, observed=d["kcal.per.g"])

    trace_2 = pm.sample(2000, return_inferencedata=True)
```

+++ {"papermill": {"duration": 0.050236, "end_time": "2020-11-29T12:14:54.072799", "exception": false, "start_time": "2020-11-29T12:14:54.022563", "status": "completed"}}

Now that we have sampled the posterior for the 3 models, we are going to compare them visually. One option is to use the `forestplot` function that supports plotting more than one trace.

```{code-cell} ipython3
---
papermill:
  duration: 0.967337
  end_time: '2020-11-29T12:14:55.090748'
  exception: false
  start_time: '2020-11-29T12:14:54.123411'
  status: completed
---
traces = [trace_0, trace_1, trace_2]
az.plot_forest(traces, figsize=(10, 5));
```

+++ {"papermill": {"duration": 0.052958, "end_time": "2020-11-29T12:14:55.196722", "exception": false, "start_time": "2020-11-29T12:14:55.143764", "status": "completed"}}

Another option is to plot several traces in a same plot is to use `plot_density`. This plot is somehow similar to a forestplot, but we get truncated KDE (kernel density estimation) plots (by default 95% credible intervals) grouped by variable names together with a point estimate (by default the mean).

```{code-cell} ipython3
---
papermill:
  duration: 2.61715
  end_time: '2020-11-29T12:14:57.866426'
  exception: false
  start_time: '2020-11-29T12:14:55.249276'
  status: completed
---
ax = az.plot_density(
    traces,
    var_names=["alpha", "sigma"],
    shade=0.1,
    data_labels=["Model 0 (neocortex)", "Model 1 (log_mass)", "Model 2 (neocortex+log_mass)"],
)

ax[0, 0].set_xlabel("Density")
ax[0, 0].set_ylabel("")
ax[0, 0].set_title("95% Credible Intervals: alpha")

ax[0, 1].set_xlabel("Density")
ax[0, 1].set_ylabel("")
ax[0, 1].set_title("95% Credible Intervals: sigma")
```

+++ {"papermill": {"duration": 0.055089, "end_time": "2020-11-29T12:14:57.977616", "exception": false, "start_time": "2020-11-29T12:14:57.922527", "status": "completed"}}

Now that we have sampled the posterior for the 3 models, we are going to use WAIC (Widely applicable information criterion) to compare the 3 models. We can do this using the `compare` function included with ArviZ.

```{code-cell} ipython3
---
papermill:
  duration: 0.239084
  end_time: '2020-11-29T12:14:58.272998'
  exception: false
  start_time: '2020-11-29T12:14:58.033914'
  status: completed
---
model_dict = dict(zip(["model_0", "model_1", "model_2"], traces))
comp = az.compare(model_dict)
comp
```

+++ {"papermill": {"duration": 0.056609, "end_time": "2020-11-29T12:14:58.387481", "exception": false, "start_time": "2020-11-29T12:14:58.330872", "status": "completed"}}

We can see that the best model is `model_2`, the one with both predictor variables. Notice the DataFrame is ordered from lowest to highest WAIC (_i.e_ from _better_ to _worst_ model). Check the {ref}`pymc:model_comparison` for a more detailed discussion on model comparison.

We can also see that we get a column with the relative `weight` for each model (according to the first equation at the beginning of this notebook). This weights can be _vaguely_ interpreted as the probability that each model will make the correct predictions on future data. Of course this interpretation is conditional on the models used to compute the weights, if we add or remove models the weights will change. And also is dependent on the assumptions behind WAIC (or any other Information Criterion used). So try to not overinterpret these `weights`. 

Now we are going to use computed `weights` to generate predictions based not on a single model, but on the weighted set of models. This is one way to perform model averaging. Using PyMC we can call the `sample_posterior_predictive_w` function as follows:

```{code-cell} ipython3
---
papermill:
  duration: 31.463179
  end_time: '2020-11-29T12:15:29.907492'
  exception: false
  start_time: '2020-11-29T12:14:58.444313'
  status: completed
---
ppc_w = pm.sample_posterior_predictive_w(
    traces=traces,
    models=[model_0, model_1, model_2],
    weights=comp.weight.sort_index(ascending=True),
    progressbar=True,
)
```

+++ {"papermill": {"duration": 0.058454, "end_time": "2020-11-29T12:15:30.024455", "exception": false, "start_time": "2020-11-29T12:15:29.966001", "status": "completed"}}

Notice that we are passing the weights ordered by their index. We are doing this because we pass `traces` and `models` ordered from model 0 to 2, but the computed weights are ordered from lowest to highest WAIC (or equivalently from larger to lowest weight). In summary, we must be sure that we are correctly pairing the weights and models.

We are also going to compute PPCs for the lowest-WAIC model.

```{code-cell} ipython3
---
papermill:
  duration: 25.204481
  end_time: '2020-11-29T12:15:55.287049'
  exception: false
  start_time: '2020-11-29T12:15:30.082568'
  status: completed
---
ppc_2 = pm.sample_posterior_predictive(trace=trace_2, model=model_2, progressbar=False)
```

+++ {"papermill": {"duration": 0.058214, "end_time": "2020-11-29T12:15:55.404271", "exception": false, "start_time": "2020-11-29T12:15:55.346057", "status": "completed"}}

A simple way to compare both kind of predictions is to plot their mean and hpd interval.

```{code-cell} ipython3
---
papermill:
  duration: 0.301319
  end_time: '2020-11-29T12:15:55.764128'
  exception: false
  start_time: '2020-11-29T12:15:55.462809'
  status: completed
---
mean_w = ppc_w["kcal"].mean()
hpd_w = az.hdi(ppc_w["kcal"].flatten())

mean = ppc_2["kcal"].mean()
hpd = az.hdi(ppc_2["kcal"].flatten())

plt.plot(mean_w, 1, "C0o", label="weighted models")
plt.hlines(1, *hpd_w, "C0")
plt.plot(mean, 0, "C1o", label="model 2")
plt.hlines(0, *hpd, "C1")

plt.yticks([])
plt.ylim(-1, 2)
plt.xlabel("kcal per g")
plt.legend();
```

+++ {"papermill": {"duration": 0.05969, "end_time": "2020-11-29T12:15:55.884685", "exception": false, "start_time": "2020-11-29T12:15:55.824995", "status": "completed"}}

As we can see the mean value is almost the same for both predictions but the uncertainty in the weighted model is larger. We have effectively propagated the uncertainty about which model we should select to the posterior predictive samples. You can now try with the other two methods for computing weights `stacking` (the default and recommended method) and `pseudo-BMA`.

**Final notes:** 

There are other ways to average models such as, for example, explicitly building a meta-model that includes all the models we have. We then perform parameter inference while jumping between the models. One problem with this approach is that jumping between models could hamper the proper sampling of the posterior.

Besides averaging discrete models we can sometimes think of continuous versions of them. A toy example is to imagine that we have a coin and we want to estimated its degree of bias, a number between 0 and 1 having a 0.5 equal chance of head and tails (fair coin). We could think of two separate models one with a prior biased towards heads and one towards tails. We could fit both separate models and then average them using, for example, IC-derived weights. An alternative, is to build a hierarchical model to estimate the prior distribution, instead of contemplating two discrete models we will be computing a continuous model that includes these the discrete ones as particular cases. Which approach is better? That depends on our concrete problem. Do we have good reasons to think about two discrete models, or is our problem better represented with a continuous bigger model?

+++

## Authors

* Authored by Osvaldo Martin in June 2017 ([pymc#2273](https://github.com/pymc-devs/pymc/pull/2273))
* Updated by Osvaldo Martin in December 2017 ([pymc#2741](https://github.com/pymc-devs/pymc/pull/2741))
* Updated by Marco Gorelli in November 2020 ([pymc#4271](https://github.com/pymc-devs/pymc/pull/4271))
* Moved from pymc to pymc-examples repo in December 2020 ([pymc-examples#8](https://github.com/pymc-devs/pymc-examples/pull/8))
* Updated by Raul Maldonado in February 2021 ([pymc#25](https://github.com/pymc-devs/pymc-examples/pull/25))
* Updated Markdown and styling by @reshamas in August 2022, ([pymc-examples#414](https://github.com/pymc-devs/pymc-examples/pull/414))

+++

## References

:::{bibliography}
:filter: docname in docnames
:::

+++

## Watermark

```{code-cell} ipython3
---
papermill:
  duration: 0.127595
  end_time: '2020-11-29T12:16:06.392237'
  exception: false
  start_time: '2020-11-29T12:16:06.264642'
  status: completed
---
%load_ext watermark
%watermark -n -u -v -iv -w
```

:::{include} ../page_footer.md
:::
