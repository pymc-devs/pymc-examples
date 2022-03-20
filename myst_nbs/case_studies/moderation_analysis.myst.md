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

# Bayesian moderation analysis

**Author:** [Benjamin T. Vincent](https://github.com/drbenvincent)

This notebook covers Bayesian [moderation analysis](https://en.wikipedia.org/wiki/Moderation_(statistics)). This is appropriate when we believe that one predictor variable (the moderator) may influence the linear relationship between another predictor variable and an outcome. Here we look at an example where we look at the relationship between hours of training and muscle mass, where it may be that age (the moderating variable) affects this relationship.

This is not intended as a one-stop solution to a wide variety of data analysis problems, rather, it is intended as an educational exposition to show how moderation analysis works and how to conduct Bayesian parameter estimation in PyMC3.

Note that this is sometimes mixed up with [mediation analysis](https://en.wikipedia.org/wiki/Mediation_(statistics)). Mediation analysis is appropriate when we believe the effect of a predictor variable upon an outcome variable is (partially, or fully) mediated through a 3rd mediating variable. Readers are referred to the textbook by Hayes (2018) as a comprehensive (albeit Frequentist) guide to moderation and related models.

```{code-cell} ipython3
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import xarray as xr

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
```

```{code-cell} ipython3
az.style.use("arviz-darkgrid")
%config InlineBackend.figure_format = 'retina'
```

Entirely unrelated to the core concepts, we first define a helper function for plotting purposes. This function will create a Matplotlib `ScalarMappable` so we can use a consistent colormap across both data points and posterior predictive lines. We can use `scalarMap.cmap` to use as a colormap, and `scalarMap.to_rgba(moderator_value)` to grab a colour for a given moderator value.

```{code-cell} ipython3
def make_scalarMap(m):
    return ScalarMappable(norm=Normalize(vmin=np.min(m), vmax=np.max(m)), cmap="viridis")
```

# Does the effect of training upon muscularity decrease with age?

I've taken inspiration from [this example](https://www.spss-tutorials.com/spss-regression-with-moderation-interaction-effect/) by Ruben Geert van den Berg which examines whether age influences (moderates) the effect of training on muscle percentage. We might speculate that more training results in higher muscle mass, at least for younger people. But it might be the case that the relationship between training and muscle mass changes with age - perhaps training is less effective at increasing muscle mass in older age?

The schematic box and arrow notation often used to represent moderation is shown by an arrow from the moderating variable to the line between a predictor and an outcome variable.

![](moderation_figure.png)

It can be useful to use consistent notation, so we will define:
- $x$ as the main predictor variable. In this example it is training.
- $y$ as the outcome variable. In this example it is muscle percentage.
- $m$ as the moderator. In this example it is age.

## The moderation model

While the visual schematic (above) is a useful shorthand to understand complex models when you already know what moderation is, you can't derive it from the diagram alone. So let us formally specify the moderation model - it defines an outcome variable $y$ as:

$$
y \sim \mathrm{Normal}(\beta_0 + \beta_1 \cdot x + \beta_2 \cdot x \cdot m + \beta_3 \cdot m, \sigma^2)
$$

where $y$, $x$, and $m$ are your observed data, and the following are the model parameters:
- $\beta_0$ is the intercept, its value does not have that much importance in the interpretation of this model.
- $\beta_1$ is the rate at which $y$ (muscle percentage) increases per unit of $x$ (training hours). 
- $\beta_2$ is the coefficient for the interaction term $x \cdot m$.
- $\beta_3$ is the rate at which $y$ (muscle percentage) increases per unit of $m$ (age). 
- $\sigma$ is the standard deviation of the observation noise.

We can see that the mean $y$ is simply a multiple linear regression with an interaction term between the two predictors, $x$ and $m$. 

We can get some insight into why this is the case by thinking about this as a multiple linear regression with $x$ and $m$ as predictor variables, but where the value of $m$ influences the relationship between $x$ and $y$. This is achieved by making the regression coefficient for $x$ is a function of $m$:

$$
y \sim \mathrm{Normal}(\beta_0 + f(m) \cdot x + \beta_3 \cdot m, \sigma^2)
$$

and if we define that as a linear function, $f(m) = \beta_1 + \beta_2 \cdot m$, we get

$$
y \sim \mathrm{Normal}(\beta_0 + (\beta_1 + \beta_2 \cdot m) \cdot x + \beta_3 \cdot m, \sigma^2)
$$

We can use $f(m) = \beta_1 + \beta_2 \cdot m$ later to visualise the moderation effect.

+++

## Import data
First, we will load up our example data and do some basic data visualisation. The dataset is taken from the [example blog post](https://www.spss-tutorials.com/spss-regression-with-moderation-interaction-effect/) by Ruben Geert van den Berg, but it is unclear if this corresponds to real life research data or if it was simulated.

```{code-cell} ipython3
def load_data():
    try:
        df = pd.read_csv("../data/muscle-percent-males-interaction.csv")
    except:
        df = pd.read_csv(pm.get_data("muscle-percent-males-interaction.csv"))

    x = df["thours"].values
    m = df["age"].values
    y = df["mperc"].values
    return (x, y, m)


x, y, m = load_data()

# Make a scalar color map for this dataset. Just for plotting.
scalarMap = make_scalarMap(m)
```

```{code-cell} ipython3
fig, ax = plt.subplots(1, 3, figsize=(14, 3))

ax[0].hist(x, alpha=0.5)
ax[0].set(xlabel="training, $x$")

ax[1].hist(m, alpha=0.5)
ax[1].set(xlabel="age, $m$")

ax[2].hist(y, alpha=0.5)
ax[2].set(xlabel="muscle percentage, $y$");
```

## Define the PyMC3 model and conduct inference

```{code-cell} ipython3
def model_factory(x, m, y):
    with pm.Model() as model:
        x = pm.Data("x", x)
        m = pm.Data("m", m)
        # priors
        β0 = pm.Normal("β0", mu=0, sd=10)
        β1 = pm.Normal("β1", mu=0, sd=10)
        β2 = pm.Normal("β2", mu=0, sd=10)
        β3 = pm.Normal("β3", mu=0, sd=10)
        σ = pm.HalfCauchy("σ", 1)
        # likelihood
        y = pm.Normal("y", mu=β0 + (β1 * x) + (β2 * x * m) + (β3 * m), sd=σ, observed=y)

    return model
```

```{code-cell} ipython3
model = model_factory(x, m, y)
```

Plot the model graph to confirm it is as intended.

```{code-cell} ipython3
pm.model_to_graphviz(model)
```

```{code-cell} ipython3
with model:
    result = pm.sample(
        2000,
        tune=2000,
        target_accept=0.9,
        random_seed=42,
        return_inferencedata=True,
        idata_kwargs={"dims": {"x": ["obs_id"], "m": ["obs_id"], "y": ["obs_id"]}},
    )
```

Visualise the trace to check for convergence.

```{code-cell} ipython3
az.plot_trace(result);
```

We have good chain mixing and the posteriors for each chain look very similar, so no problems in that regard.

+++

## Visualise the important parameters

First we will use a pair plot to look at joint posterior distributions. This might help us identify any estimation issues with the interaction term (see the discussion below about multicollinearity).

```{code-cell} ipython3
az.plot_pair(
    result,
    marginals=True,
    point_estimate="median",
    figsize=(12, 12),
    scatter_kwargs={"alpha": 0.01},
);
```

And just for the sake of completeness, we can plot the posterior distributions for each of the $\beta$ parameters and use this to arrive at research conclusions.

```{code-cell} ipython3
az.plot_posterior(result, var_names=["β1", "β2", "β3"], figsize=(14, 4));
```

For example, from an estimation (in contrast to a hypothesis testing) perspective, we could look at the posterior over $\beta_2$ and claim a credibly less than zero moderation effect.

+++

## Posterior predictive checks
Define a set of quantiles of $m$ that we are interested in visualising.

```{code-cell} ipython3
m_quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
```

### Visualisation in data space
Here we will plot the data alongside model posterior predictive checks. This can be a useful visual method of comparing the model predictions against the data.

```{code-cell} ipython3
def plot_data(x, moderator, y, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = plt.gcf()

    h = ax.scatter(x, y, c=moderator, cmap=scalarMap.cmap)
    ax.set(xlabel="x", ylabel="y")
    # colourbar for moderator
    cbar = fig.colorbar(h)
    cbar.ax.set_ylabel("moderator")
    return ax
```

```{code-cell} ipython3
def posterior_prediction_plot(result, x, moderator, m_quantiles, ax=None):
    """Plot posterior predicted `y`"""
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    post = result.posterior.stack(sample=("chain", "draw"))
    xi = xr.DataArray(np.linspace(np.min(x), np.max(x), 20), dims=["x_plot"])
    m_levels = result.constant_data["m"].quantile(m_quantiles).rename({"quantile": "m_level"})

    for p, m in zip(m_quantiles, m_levels):
        y = post.β0 + post.β1 * xi + post.β2 * xi * m + post.β3 * m
        region = y.quantile([0.025, 0.5, 0.975], dim="sample")
        ax.fill_between(
            xi,
            region.sel(quantile=0.025),
            region.sel(quantile=0.975),
            alpha=0.2,
            color=scalarMap.to_rgba(m),
            edgecolor="w",
        )
        ax.plot(
            xi,
            region.sel(quantile=0.5),
            color=scalarMap.to_rgba(m),
            linewidth=2,
            label=f"{p*100}th percentile of moderator",
        )

    ax.legend(fontsize=9)
    ax.set(xlabel="x", ylabel="y")
    return ax
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 6))
ax = plot_data(x, m, y, ax=ax)
posterior_prediction_plot(result, x, m, m_quantiles, ax=ax)
ax.set_title("Data and posterior prediction");
```

### Spotlight graph
We can also visualise the 'moderation effect' by plotting $\beta_1 + \beta_2 \cdot m$ as a function of the $m$. This was named a spotlight graph, see Spiller et. al. (2015) and McClelland et. al. (2016).

```{code-cell} ipython3
def plot_moderation_effect(m, m_quantiles, ax=None):

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    post = result.posterior.stack(sample=("chain", "draw"))

    # calculate 95% CI region and median
    xi = xr.DataArray(np.linspace(np.min(m), np.max(m), 20), dims=["x_plot"])
    rate = post.β1 + post.β2 * xi
    region = rate.quantile([0.025, 0.5, 0.975], dim="sample")

    ax.fill_between(
        xi,
        region.sel(quantile=0.025),
        region.sel(quantile=0.975),
        alpha=0.2,
        color="k",
        edgecolor="w",
    )

    ax.plot(xi, region.sel(quantile=0.5), color="k", linewidth=2)

    # plot points at each percentile of m
    percentile_list = np.array(m_quantiles) * 100
    m_levels = np.percentile(m, percentile_list)
    for p, m in zip(percentile_list, m_levels):
        ax.plot(
            m,
            np.mean(post.β1) + np.mean(post.β2) * m,
            "o",
            c=scalarMap.to_rgba(m),
            markersize=10,
            label=f"{p}th percentile of moderator",
        )

    ax.legend(fontsize=9)

    ax.set(
        title="Spotlight graph",
        xlabel="$moderator$",
        ylabel=r"$\beta_1 + \beta_2 \cdot moderator$",
    )
```

```{code-cell} ipython3
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

plot_moderation_effect(m, m_quantiles, ax[0])
az.plot_posterior(result, var_names="β2", ax=ax[1]);
```

The expression $\beta_1 + \beta_2 \cdot \text{moderator}$ defines the rate of change of the outcome (muscle percentage) per unit of $x$ (training hours/week). We can see that as age (the moderator) increases, this effect of training hours/week on muscle percentage decreases.

+++

# Related issues: mean centering and multicollinearity

Readers should be aware that there are issues around mean-centering and multicollinearity. The original [SPSS Moderation Regression Tutorial](https://www.spss-tutorials.com/spss-regression-with-moderation-interaction-effect/) did mean-centre the predictor variables $x$ and $m$. This will have a downstream effect upon the interaction term $x \cdot m$.

One effect of mean centering is to change the interpretation of the parameter estimates. In this notebook, we did not mean center the variables which will affect the parameter estimates and their interpretation. It is not that one is correct or incorrect, but one must be cognisant of how mean-centering (or not) affects the interpretation of parameter estimates. Readers are again directed to Hayes (2018) for a more in-depth consideration of mean-centering in moderation analyses.

Another issue, particularly relevant to moderation analysis is [multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity), where one predictor variable is well-described as a linear combination of other predictors. This is clearly the case in moderation analysis as the interaction term $m \cdot x$ is by definition a linear combination of $x$ and $m$.

Iacobucci et al (2016) explored the issues of mean-centering and multicollinearity and conclude:
 > When all is said and done, should a researcher mean center the X1 and X2 variables before computing a product term X1X2 to include in a moderated multiple regression? It depends. Mean centering is advisable when: (1) the predictor variables are measured on scales with arbitrary zeros and the researcher seeks to enhance the interpretation of the regression results vis-à-vis the variables’ means rather than the arbitrary zero points, or (2) the research questions involve testing the main effect terms in addition to the interaction term and the researcher seeks to obtain these statistical tests without the interference of the so-called nonessential multicollinearity. On the other hand, mean centering may be bypassed when: (1) the research question involves primarily the test of the interaction term, with no regard for the lower order main effect terms, or (2) the research question involves primarily the assessment of the overall fit of the model, the R2, with no interest in apportioning the explained variability across the predictors, main effects or interaction alike.
 
This was critiqued however by McClelland (2016) who claimed that Iacobucci et al (2016) made a number of errors, and that multicollinearity is a red herring:

> Multicollinearity is irrelevant to the search for moderator variables, contrary to the implications of Iacobucci, Schneider, Popovich, and Bakamitsos (Behavior Research Methods, 2016, this issue). Multicollinearity is like the red herring in a mystery novel that distracts the statistical detective from the pursuit of a true moderator relationship.

They state:

> Researchers using MMR [moderated multiple regression] need not compute any multicollinearity diagnostics nor worry about it at all. They need not use mean-centering or the orthogonal transformation or do anything else to avoid the purported problems of multicollinearity. The only purpose of those transformations is to facilitate understanding of MMR models.

Bearing in mind McClelland et. al. (2016) took a frequentist hypothesis testing (not a Bayesian approach) their take-home points can be paraphrased as:
1. Fit the regression model, $y \sim \mathrm{Normal}(\beta_0 + \beta_1 \cdot x + \beta_2 \cdot x \cdot m + \beta_3 \cdot m, \sigma^2)$, with original (not mean-centred) data.
2. If the main interest is on the moderation effect, then focus upon $\beta_2$.
3. Transformations are useful if conditional relationships are to be highlighted.
4. "... researchers who wish to examine all possible conditional relationships or to help their readers who might want to consider other conditional relationships, should construct the [spotlight] graph..."

But readers are strongly encouraged to read McClelland et. al. (2016) for more details, as well as the reply from Iacobucci et. al. (2017). Readers should also be aware that there are conflicting opinions and recommendations about mean centering etc in textbooks (see Further Reading below), some of which are published before 2016. None of these textbooks explicitly cite McClelland et. al. (2016), so it is unclear if the textbook authors are unaware of, agree with, or disagree with McClelland et. al. (2016).

# Further reading
- Further information about the 'moderation effect', or what McClelland et. al. (2016) called a spotlight graphs, can be found in Bauer and Curran (2005) and Spiller et. al. (2013). Although these papers take a frequentist (not Bayesian) perspective.
- Zhang & Wang (2017) compare maximum likelihood and Bayesian methods for moderation analysis with missing predictor variables.
- Multicollinearity, data centering, and linear models with interaction terms are also discussed in a number of prominent Bayesian text books (Gelman et al, 2013; Kruschke, 2014; Gelman, Hill, & Vehtari, 2020; McElreath, 2020).

+++

# References

- Bauer, D. J., & Curran, P. J. (2005). Probing interactions in fixed and multilevel regression: Inferential and graphical techniques. Multivariate Behavioral Research, 40, 373–400.
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). Bayesian data analysis. CRC press.
- Gelman, A., Hill, J., & Vehtari, A. (2020). Regression and other stories. Cambridge University Press.
- Hayes, A. F. (2018). Introduction to Mediation, Moderation, and Conditional Process Analysis: A Regression‐Based Approach. New York: Guilford Press. Retrieved from https://doi.org/10.1111/jedm.12050
- Iacobucci, D., Schneider, M.J., Popovich, D.L. et al. (2016) Mean centering helps alleviate “micro” but not “macro” multicollinearity. Behav Res 48, 1308–1317. https://doi.org/10.3758/s13428-015-0624-x
- Iacobucci, D., Schneider, M.J., Popovich, D.L. et al. (2017) Mean centering, multicollinearity, and moderators in multiple regression: The reconciliation redux. Behav Res 49, 403–404. https://doi.org/10.3758/s13428-016-0827-9
- Kruschke, J. (2014). Doing Bayesian data analysis: A tutorial with R, JAGS, and Stan.
- McClelland, G.H., Irwin, J.R., Disatnik, D. et al. (2017) Multicollinearity is a red herring in the search for moderator variables: A guide to interpreting moderated multiple regression models and a critique of Iacobucci, Schneider, Popovich, and Bakamitsos (2016). Behav Res 49, 394–402 . https://doi.org/10.3758/s13428-016-0785-2
- McElreath, R. (2020). Statistical rethinking: A Bayesian course with examples in R and Stan. CRC press.
- Spiller, S. A., Fitzsimons, G. J., Lynch, J. G., & McClelland, G. H. (2013). Spotlights, floodlights, and the magic number zero: Simple effects tests in moderated regression. Journal of Marketing Research, 50, 277–288.
- van den Berg, R. G. (n.d.) SPSS Moderation Regression Tutorial [blog post]. Retrieved from [https://www.spss-tutorials.com/spss-regression-with-moderation-interaction-effect/](https://www.spss-tutorials.com/spss-regression-with-moderation-interaction-effect/)
- Zhang, Q., & Wang, L. (2017). Moderation analysis with missing data in the predictors. Psychological methods, 22(4), 649.

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p theano,xarray
```
