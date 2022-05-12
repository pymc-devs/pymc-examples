---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# GLM: Logistic Regression

* This is a reproduction with a few slight alterations of [Bayesian Log Reg](http://jbencook.github.io/portfolio/bayesian_logistic_regression.html) by J. Benjamin Cook

* Author: Peadar Coyle and J. Benjamin Cook
* How likely am I to make more than $50,000 US Dollars?
* Exploration of model selection techniques too - I use WAIC to select the best model. 
* The convenience functions are all taken from Jon Sedars work.
* This example also has some explorations of the features so serves as a good example of Exploratory Data Analysis and how that can guide the model creation/ model selection process.

```{code-cell} ipython3
import warnings

from collections import OrderedDict
from time import time

import arviz as az
import bambi as bmb
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn
import theano as thno
import theano.tensor as T

from formulae import design_matrices
from scipy import integrate
from scipy.optimize import fmin_powell

print(f"Running on PyMC3 v{pm.__version__}")
```

```{code-cell} ipython3
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

%config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")
```

```{code-cell} ipython3
def run_models(df, upper_order=5):
    """
    Convenience function:
    Fit a range of pymc3 models of increasing polynomial complexity.
    Suggest limit to max order 5 since calculation time is exponential.
    """

    models, traces = OrderedDict(), OrderedDict()

    for k in range(1, upper_order + 1):

        nm = f"k{k}"
        fml = create_poly_modelspec(k)

        models[nm] = bmb.Model(fml, df, family="bernoulli")
        traces[nm] = models[nm].fit(draws=1000, tune=1000, init="adapt_diag")

    return models, traces


def plot_traces(traces, model, retain=0):
    """
    Convenience function:
    Plot traces with overlaid means and values
    """
    summary = az.summary(traces, stat_funcs={"mean": np.mean}, extend=False)
    ax = az.plot_trace(
        traces,
        lines=tuple([(k, {}, v["mean"]) for k, v in summary.iterrows()]),
    )

    for i, mn in enumerate(summary["mean"].values):
        ax[i, 0].annotate(
            f"{mn:.2f}",
            xy=(mn, 0),
            xycoords="data",
            xytext=(5, 10),
            textcoords="offset points",
            rotation=90,
            va="bottom",
            fontsize="large",
            color="C0",
        )


def create_poly_modelspec(k=1):
    """
    Convenience function:
    Create a polynomial modelspec string for patsy
    """
    return (
        "income ~ educ + hours + age " + " ".join([f"+ np.power(age,{j})" for j in range(2, k + 1)])
    ).strip()
```

The [Adult Data Set](http://archive.ics.uci.edu/ml/datasets/Adult) is commonly used to benchmark machine learning algorithms. The goal is to use demographic features, or variables, to predict whether an individual makes more than \\$50,000 per year. The data set is almost 20 years old, and therefore, not perfect for determining the probability that I will make more than \$50K, but it is a nice, simple dataset that can be used to showcase a few benefits of using Bayesian logistic regression over its frequentist counterpart.


The motivation for myself to reproduce this piece of work was to learn how to use Odd Ratio in Bayesian Regression.

```{code-cell} ipython3
raw_data = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    header=None,
    names=[
        "age",
        "workclass",
        "fnlwgt",
        "education-categorical",
        "educ",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "captial-gain",
        "capital-loss",
        "hours",
        "native-country",
        "income",
    ],
)
```

```{code-cell} ipython3
raw_data.head(10)
```

## Scrubbing and cleaning
We need to remove any null entries in Income. 
And we also want to restrict this study to the United States.

```{code-cell} ipython3
data = raw_data[~pd.isnull(raw_data["income"])]
```

```{code-cell} ipython3
data[data["native-country"] == " United-States"].sample(5)
```

```{code-cell} ipython3
income = 1 * (data["income"] == " >50K")
```

```{code-cell} ipython3
data = data[["age", "educ", "hours"]]

# Scale age by 10, it helps with model convergence.
data["age"] = data["age"] / 10.0
data["age2"] = np.square(data["age"])
data["income"] = income
```

```{code-cell} ipython3
income.value_counts()
```

## Exploring the data 
Let us get a feel for the parameters. 
* We see that age is a tailed distribution. Certainly not Gaussian!
* We don't see much of a correlation between many of the features, with the exception of Age and Age2. 
* Hours worked has some interesting behaviour. How would one describe this distribution?

```{code-cell} ipython3
g = seaborn.pairplot(data)
```

```{code-cell} ipython3
# Compute the correlation matrix
corr = data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = seaborn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
seaborn.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    vmax=0.3,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
    ax=ax,
);
```

We see here not many strong correlations. The highest is 0.30 according to this plot. We see a weak-correlation between hours and income 
(which is logical), we see a slightly stronger correlation between education and income (which is the kind of question we are answering).

+++

## The model
We will use a simple model, which assumes that the probability of making more than $50K 
is a function of age, years of education and hours worked per week. We will use PyMC3 
do inference. 

In Bayesian statistics, we treat everything as a random variable and we want to know the posterior probability distribution of the parameters
(in this case the regression coefficients)
The posterior is equal to the likelihood $$p(\theta | D) = \frac{p(D|\theta)p(\theta)}{p(D)}$$

Because the denominator is a notoriously difficult integral, $p(D) = \int p(D | \theta) p(\theta) d \theta $ we would prefer to skip computing it. Fortunately, if we draw examples from the parameter space, with probability proportional to the height of the posterior at any given point, we end up with an empirical distribution that converges to the posterior as the number of samples approaches infinity.  

What this means in practice is that we only need to worry about the numerator. 

Getting back to logistic regression, we need to specify a prior and a likelihood in order to draw samples from the posterior. We could use sociological knowledge about the effects of age and education on income, but instead, let's use the default prior specification for GLM coefficients that PyMC3 gives us, which is $p(θ)=N(0,10^{12}I)$. This is a very vague prior that will let the data speak for themselves.

The likelihood is the product of n Bernoulli trials, $\prod^{n}_{i=1} p_{i}^{y} (1 - p_{i})^{1-y_{i}}$,
where $p_i = \frac{1}{1 + e^{-z_i}}$, 

$z_{i} = \beta_{0} + \beta_{1}(age)_{i} + \beta_2(age)^{2}_{i} + \beta_{3}(educ)_{i} + \beta_{4}(hours)_{i}$ and $y_{i} = 1$ if income is greater than 50K and $y_{i} = 0$ otherwise. 

With the math out of the way we can get back to the data. Here I use PyMC3 to draw samples from the posterior. The sampling algorithm used is NUTS, which is a form of Hamiltonian Monte Carlo, in which parameteres are tuned automatically. Notice, that we get to borrow the syntax of specifying GLM's from R, very convenient! I use a convenience function from above to plot the trace information from the first 1000 parameters.

```{code-cell} ipython3
model = bmb.Model("income ~ age + age2 + educ + hours", data, family="bernoulli")

trace = model.fit(draws=1000, tune=1000, init="adapt_diag")
```

```{code-cell} ipython3
plot_traces(trace, model);
```

## Some results 
One of the major benefits that makes Bayesian data analysis worth the extra computational effort in many circumstances is that we can be explicit about our uncertainty. Maximum likelihood returns a number, but how certain can we be that we found the right number? Instead, Bayesian inference returns a distribution over parameter values.

I'll use ArviZ to look at the distribution of some of these factors.

```{code-cell} ipython3
az.plot_pair(
    trace,
    var_names=["age", "educ"],
    kind="hexbin",
    figsize=(7, 7),
    marginals=True,
    marginal_kwargs={"quantiles": np.arange(1, 5) / 5},
);
```

So how do age and education affect the probability of making more than \$50K? 

To answer this question, we can show how the probability of making more than \$50K changes with age for a few different education levels. Here, we assume that the number of hours worked per week is fixed at 50. PyMC3 gives us a convenient way to plot the posterior predictive distribution. We need to give the function a linear model and a set of points to evaluate. We will pass in three different linear models: 
- one with educ == 12 (finished high school)
- one with educ == 16 (finished undergrad) 
- one with educ == 19 (three years of grad school).

```{code-cell} ipython3
def lm_full(trace, age, educ, hours):
    shape = np.broadcast(age, educ, hours).shape
    x_norm = np.asarray([np.broadcast_to(x, shape) for x in [age / 10.0, educ, hours]])
    return 1 / (
        1
        + np.exp(
            -(
                trace["Intercept"]
                + trace["age"] * x_norm[0]
                + trace["age2"] * (x_norm[0] ** 2)
                + trace["educ"] * x_norm[1]
                + trace["hours"] * x_norm[2]
            )
        )
    )


# Linear model with hours == 50 and educ == 12
lm = lambda x, samples: lm_full(samples, x, 12.0, 50.0)

# Linear model with hours == 50 and educ == 16
lm2 = lambda x, samples: lm_full(samples, x, 16.0, 50.0)

# Linear model with hours == 50 and educ == 19
lm3 = lambda x, samples: lm_full(samples, x, 19.0, 50.0)
```

Each curve shows how the probability of earning more than \\$50K changes with age. The red curve represents 19 years of education, the green curve represents 16 years of education and the blue curve represents 12 years of education. For all three education levels, the probability of making more than \$50K increases with age until approximately age 60, when the probability begins to drop off. Notice that each curve is a little blurry. This is because we are actually plotting 100 different curves for each level of education. Each curve is a draw from our posterior distribution. Because the curves are somewhat translucent, we can interpret dark, narrow portions of a curve as places where we have low uncertainty and light, spread out portions of the curve as places where we have somewhat higher uncertainty about our coefficient values.

```{code-cell} ipython3
# Plot the posterior predictive distributions of P(income > $50K) vs. age
pm.plot_posterior_predictive_glm(
    trace, eval=np.linspace(25, 75, 1000), lm=lm, samples=100, color="C0", alpha=0.15
)
pm.plot_posterior_predictive_glm(
    trace,
    eval=np.linspace(25, 75, 1000),
    lm=lm2,
    samples=100,
    color="C1",
    alpha=0.15,
)
pm.plot_posterior_predictive_glm(
    trace, eval=np.linspace(25, 75, 1000), lm=lm3, samples=100, color="C2", alpha=0.15
)

blue_line = mlines.Line2D(["lm"], [], color="C0", label="High School Education")
green_line = mlines.Line2D(["lm2"], [], color="C1", label="Bachelors")
red_line = mlines.Line2D(["lm3"], [], color="C2", label="Grad School")
plt.legend(handles=[blue_line, green_line, red_line], loc="lower right")
plt.ylabel("P(Income > $50K)")
plt.xlabel("Age")
plt.show()
```

```{code-cell} ipython3
odds_ratio = np.exp(trace.posterior["educ"])

ax = az.plot_kde(odds_ratio, rug=True, quantiles=np.arange(1, 5) / 5)
ax.set_xlabel("Odds Ratio");
```

Finally, we can find a credible interval (remember kids - credible intervals are Bayesian and confidence intervals are frequentist) for this quantity. This may be the best part about Bayesian statistics: we get to interpret credibility intervals the way we've always wanted to interpret them. We are 95% confident that the odds ratio lies within our interval!

```{code-cell} ipython3
lb, ub = np.percentile(odds_ratio, 2.5), np.percentile(odds_ratio, 97.5)

print(f"P({lb:.3f} < O.R. < {ub:.3f}) = 0.95")
```

## Model selection 

One question that was immediately asked was what effect does age have on the model, and why should it be $age^2$ versus age? We'll run the model with a few changes to see what effect higher order terms have on this model in terms of WAIC.

```{code-cell} ipython3
models_lin, traces_lin = run_models(data, 3)
```

```{code-cell} ipython3
model_trace_dict = dict()
for nm in ["k1", "k2", "k3"]:
    model_trace_dict.update({nm: traces_lin[nm]})

dfwaic = az.compare(model_trace_dict, ic="WAIC", scale="deviance")
az.plot_compare(dfwaic);
```

WAIC confirms our decision to use age^2.

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p xarray
```
