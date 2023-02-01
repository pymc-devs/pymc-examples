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

(dirichlet_mixture_of_multinomials)=
# Dirichlet mixtures of multinomials

:::{post} Jan 8, 2022
:tags: mixture model, 
:category: advanced
:author: Byron J. Smith, Abhipsha Das, Oriol Abril-Pla
:::

+++

This example notebook demonstrates the use of a
Dirichlet mixture of multinomials
(a.k.a [Dirichlet-multinomial](https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution) or DM)
to model categorical count data.
Models like this one are important in a variety of areas, including
natural language processing, ecology, bioinformatics, and more.

The Dirichlet-multinomial can be understood as draws from a [Multinomial distribution](https://en.wikipedia.org/wiki/Multinomial_distribution)
where each sample has a slightly different probability vector, which is itself drawn from a common [Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution).
This contrasts with the Multinomial distribution, which assumes that all observations arise from a single fixed probability vector.
This enables the Dirichlet-multinomial to accommodate more variable (a.k.a, over-dispersed) count data than the Multinomial.

Other examples of over-dispersed count distributions are the
[Beta-binomial](https://en.wikipedia.org/wiki/Beta-binomial_distribution)
(which can be thought of as a special case of the DM) or the
[Negative binomial](https://en.wikipedia.org/wiki/Negative_binomial_distribution)
distributions.

The DM is also an example of marginalizing
a mixture distribution over its latent parameters.
This notebook will demonstrate the performance benefits that come from taking that approach.

```{code-cell} ipython3
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import scipy as sp
import scipy.stats
import seaborn as sns

print(f"Running on PyMC3 v{pm.__version__}")
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
```

## Simulation data

+++

Let us simulate some over-dispersed, categorical count data
for this example.

Here we are simulating from the DM distribution itself,
so it is perhaps tautological to fit that model,
but rest assured that data like these really do appear in
the counts of different:

1. words in text corpuses {cite:p}`madsen2005modelingdirichlet`,
2. types of RNA molecules in a cell {cite:p}`nowicka2016drimseq`,
3. items purchased by shoppers {cite:p}`goodhardt1984thedirichlet`.

Here we will discuss a community ecology example, pretending that we have observed counts of $k=5$ different
tree species in $n=10$ different forests.

Our simulation will produce a two-dimensional matrix of integers (counts)
where each row, (zero-)indexed by $i \in (0...n-1)$, is an observation (different forest), and each
column $j \in (0...k-1)$ is a category (tree species).
We'll parameterize this distribution with three things:
- $\mathrm{frac}$ : the expected fraction of each species,
  a $k$-dimensional vector on the simplex (i.e. sums-to-one)
- $\mathrm{total\_count}$ : the total number of items tallied in each observation,
- $\mathrm{conc}$ : the concentration, controlling the overdispersion of our data,
  where larger values result in our distribution more closely approximating the multinomial.
  
Here, and throughout this notebook, we've used a
[convenient reparameterization](https://mc-stan.org/docs/2_26/stan-users-guide/reparameterizations.html#dirichlet-priors)
of the Dirichlet distribution
from one to two parameters,
$\alpha=\mathrm{conc} \times \mathrm{frac}$, as this
fits our desired interpretation.
  
Each observation from the DM is simulated by:
1. first obtaining a value on the $k$-simplex simulated as
   $p_i \sim \mathrm{Dirichlet}(\alpha=\mathrm{conc} \times \mathrm{frac})$,
2. and then simulating $\mathrm{counts}_i \sim \mathrm{Multinomial}(\mathrm{total\_count}, p_i)$.

Notice that each observation gets its _own_
latent parameter $p_i$, simulated independently from
a common Dirichlet distribution.

```{code-cell} ipython3
true_conc = 6.0
true_frac = np.array([0.45, 0.30, 0.15, 0.09, 0.01])
trees = ["pine", "oak", "ebony", "rosewood", "mahogany"]  # Tree species observed
# fmt: off
forests = [  # Forests observed
    "sunderbans", "amazon", "arashiyama", "trossachs", "valdivian",
    "bosc de poblet", "font groga", "monteverde", "primorye", "daintree",
]
# fmt: on
k = len(trees)
n = len(forests)
total_count = 50

true_p = sp.stats.dirichlet(true_conc * true_frac).rvs(size=n)
observed_counts = np.vstack([sp.stats.multinomial(n=total_count, p=p_i).rvs() for p_i in true_p])

observed_counts
```

## Multinomial model

+++

The first model that we will fit to these data is a plain
multinomial model, where the only parameter is the
expected fraction of each category, $\mathrm{frac}$, which we will give a Dirichlet prior.
While the uniform prior ($\alpha_j=1$ for each $j$) works well, if we have independent beliefs about the fraction of each tree,
we could encode this into our prior, e.g.
increasing the value of $\alpha_j$ where we expect a higher fraction of species-$j$.

```{code-cell} ipython3
coords = {"tree": trees, "forest": forests}
with pm.Model(coords=coords) as model_multinomial:
    frac = pm.Dirichlet("frac", a=np.ones(k), dims="tree")
    counts = pm.Multinomial(
        "counts", n=total_count, p=frac, observed=observed_counts, dims=("forest", "tree")
    )

pm.model_to_graphviz(model_multinomial)
```

Interestingly, NUTS frequently runs into numerical problems on this model, perhaps an example of the
["Folk Theorem of Statistical Computing"](https://statmodeling.stat.columbia.edu/2008/05/13/the_folk_theore/).

Because of a couple of identities of the multinomial distribution,
we could reparameterize this model in a number of ways&mdash;we
would obtain equivalent models by exploding our $n$ observations
of $\mathrm{total\_count}$ items into $(n \times \mathrm{total\_count})$
independent categorical trials, or collapsing them down into
one Multinomial draw with $(n \times \mathrm{total\_count})$ items.
(Importantly, this is _not_ true for the DM distribution.)

Rather than _actually_ fixing our problem through reparameterization,
here we'll instead switch to the Metropolis step method,
which ignores some of the geometric pathologies of our naïve model.

**Important**: switching to Metropolis does not not _fix_ our model's issues, rather it _sweeps them under the rug_.
In fact, if you try running this model with NUTS (PyMC3's default step method), it will break loudly during sampling.
When that happens, this should be a **red alert** that there is something wrong in our model.

You'll also notice below that we have to increase considerably the number of draws we take from the posterior;
this is because Metropolis is much less efficient at
exploring the posterior than NUTS.

```{code-cell} ipython3
with model_multinomial:
    trace_multinomial = pm.sample(
        draws=5000, chains=4, step=pm.Metropolis(), return_inferencedata=True
    )
```

Let's ignore the warning about inefficient sampling for now.

```{code-cell} ipython3
az.plot_trace(data=trace_multinomial, var_names=["frac"]);
```

The trace plots look fairly good;
visually, each parameter appears to be moving around the posterior well,
although some sharp parts of the KDE plot suggests that
sampling sometimes gets stuck in one place for a few steps.

```{code-cell} ipython3
summary_multinomial = az.summary(trace_multinomial, var_names=["frac"])

summary_multinomial = summary_multinomial.assign(
    ess_bulk_per_sec=lambda x: x.ess_bulk / trace_multinomial.posterior.sampling_time,
)

summary_multinomial
```

Likewise, diagnostics in the parameter summary table all look fine.
Here I've added a column estimating the effective sample size per
second of sampling.

Nonetheless, the fact that we were unable to use NUTS is still a red flag, and we should be
very cautious in using these results.

```{code-cell} ipython3
az.plot_forest(trace_multinomial, var_names=["frac"])
for j, (y_tick, frac_j) in enumerate(zip(plt.gca().get_yticks(), reversed(true_frac))):
    plt.vlines(frac_j, ymin=y_tick - 0.45, ymax=y_tick + 0.45, color="black", linestyle="--")
```

Here we've drawn a forest-plot, showing the mean and 94% HDIs from our posterior approximation.
Interestingly, because we know what the underlying
frequencies are for each species (dashed lines), we can comment on the accuracy
of our inferences.
And now the issues with our model become apparent;
notice that the 94% HDIs _don't include the true values_ for
tree species 0, 2, 3.
We might have seen _one_ HDI miss, but _three_???

...what's going on?

Let's troubleshoot this model using a posterior-predictive check, comparing our data to simulated data conditioned on our posterior estimates.

```{code-cell} ipython3
with model_multinomial:
    pp_samples = az.from_pymc3(
        posterior_predictive=pm.fast_sample_posterior_predictive(trace=trace_multinomial)
    )

# Concatenate with InferenceData object
trace_multinomial.extend(pp_samples)
```

```{code-cell} ipython3
cmap = plt.get_cmap("tab10")

fig, axs = plt.subplots(k, 1, sharex=True, sharey=True, figsize=(6, 8))
for j, ax in enumerate(axs):
    c = cmap(j)
    ax.hist(
        trace_multinomial.posterior_predictive.counts.sel(tree=trees[j]).values.flatten(),
        bins=np.arange(total_count),
        histtype="step",
        color=c,
        density=True,
        label="Post.Pred.",
    )
    ax.hist(
        (trace_multinomial.observed_data.counts.sel(tree=trees[j]).values.flatten()),
        bins=np.arange(total_count),
        color=c,
        density=True,
        alpha=0.25,
        label="Observed",
    )
    ax.axvline(
        true_frac[j] * total_count,
        color=c,
        lw=1.0,
        alpha=0.45,
        label="True",
    )
    ax.annotate(
        f"{trees[j]}",
        xy=(0.96, 0.9),
        xycoords="axes fraction",
        ha="right",
        va="top",
        color=c,
    )

axs[-1].legend(loc="upper center", fontsize=10)
axs[-1].set_xlabel("Count")
axs[-1].set_yticks([0, 0.5, 1.0])
axs[-1].set_ylim(0, 0.6);
```

Here we're plotting histograms of the predicted counts
against the observed counts for each species.

_(Notice that the y-axis isn't full height and clips the distributions for species-4 in purple.)_

And now we can start to see why our posterior HDI deviates from the _true_ parameters for three of five species (vertical lines).
See that for all of the species the observed counts are frequently quite far from the predictions
conditioned on the posterior distribution.
This is particularly obvious for (e.g.) species-2 where we have one observation of more than 20
trees of this species, despite the posterior predicitive mass being concentrated far below that.

This is overdispersion at work, and a clear sign that we need to adjust our model to accommodate it.

Posterior predictive checks are one of the best ways to diagnose model misspecification,
and this example is no different.

+++

## Dirichlet-Multinomial Model - Explicit Mixture

+++

Let's go ahead and model our data using the DM distribution.

For this model we'll keep the same prior on the expected frequencies of each
species, $\mathrm{frac}$.
We'll also add a strictly positive parameter, $\mathrm{conc}$, for the concentration.

In this iteration of our model we'll explicitly include the latent multinomial
probability, $p_i$, modeling the $\mathrm{true\_p}_i$ from our simulations (which we would not
observe in the real world).

```{code-cell} ipython3
with pm.Model(coords=coords) as model_dm_explicit:
    frac = pm.Dirichlet("frac", a=np.ones(k), dims="tree")
    conc = pm.Lognormal("conc", mu=1, sigma=1)
    p = pm.Dirichlet("p", a=frac * conc, dims=("forest", "tree"))
    counts = pm.Multinomial(
        "counts", n=total_count, p=p, observed=observed_counts, dims=("forest", "tree")
    )

pm.model_to_graphviz(model_dm_explicit)
```

Compare this diagram to the first.
Here the latent, Dirichlet distributed $p$ separates the multinomial from the expected frequencies, $\mathrm{frac}$,
accounting for overdispersion of counts relative to the simple multinomial model.

```{code-cell} ipython3
with model_dm_explicit:
    trace_dm_explicit = pm.sample(chains=4, return_inferencedata=True)
```

We got a warning, although we'll ignore it for now.
More interesting is how much longer it took to sample this model than the
first.
This may be because our model has an additional ~$(n \times k)$ parameters,
but it seems like there are other geometric challenges for NUTS as well.

We'll see if we can fix these in the next model, but for now let's take a look at the traces.

```{code-cell} ipython3
az.plot_trace(data=trace_dm_explicit, var_names=["frac", "conc"]);
```

Obviously some sampling issues, but it's hard to see where divergences are occurring.

```{code-cell} ipython3
az.plot_forest(trace_dm_explicit, var_names=["frac"])
for j, (y_tick, frac_j) in enumerate(zip(plt.gca().get_yticks(), reversed(true_frac))):
    plt.vlines(frac_j, ymin=y_tick - 0.45, ymax=y_tick + 0.45, color="black", linestyle="--")
```

On the other hand, since we know the ground-truth for $\mathrm{frac}$,
we can congratulate ourselves that
the HDIs include the true values for all of our species!

Modeling this mixture has made our inferences robust to the overdispersion of counts,
while the plain multinomial is very sensitive.
Notice that the HDI is much wider than before for each $\mathrm{frac}_i$.
In this case that makes the difference between correct and incorrect inferences.

```{code-cell} ipython3
summary_dm_explicit = az.summary(trace_dm_explicit, var_names=["frac", "conc"])
summary_dm_explicit = summary_dm_explicit.assign(
    ess_bulk_per_sec=lambda x: x.ess_bulk / trace_dm_explicit.posterior.sampling_time,
)

summary_dm_explicit
```

This is great, but _we can do better_.
The larger $\hat{R}$ value for $\mathrm{frac}_4$ is mildly concerning, and it's surprising
that our $\mathrm{ESS} \; \mathrm{sec}^{-1}$ is relatively small.

+++

## Dirichlet-Multinomial Model - Marginalized

+++

Happily, the Dirichlet distribution is conjugate to the multinomial
and therefore there's a convenient, closed-form for the marginalized
distribution, i.e. the Dirichlet-multinomial distribution, which was added to PyMC3 in [3.11.0](https://github.com/pymc-devs/pymc3/releases/tag/v3.11.0).

Let's take advantage of this, marginalizing out the explicit latent parameter, $p_i$,
replacing the combination of this node and the multinomial
with the DM to make an equivalent model.

```{code-cell} ipython3
with pm.Model(coords=coords) as model_dm_marginalized:
    frac = pm.Dirichlet("frac", a=np.ones(k), dims="tree")
    conc = pm.Lognormal("conc", mu=1, sigma=1)
    counts = pm.DirichletMultinomial(
        "counts", n=total_count, a=frac * conc, observed=observed_counts, dims=("forest", "tree")
    )

pm.model_to_graphviz(model_dm_marginalized)
```

The plate diagram shows that we've collapsed what had been the latent Dirichlet and the multinomial
nodes together into a single DM node.

```{code-cell} ipython3
with model_dm_marginalized:
    trace_dm_marginalized = pm.sample(chains=4, return_inferencedata=True)
```

It samples much more quickly and without any of the warnings from before!

```{code-cell} ipython3
az.plot_trace(data=trace_dm_marginalized, var_names=["frac", "conc"]);
```

Trace plots look fuzzy and KDEs are clean.

```{code-cell} ipython3
summary_dm_marginalized = az.summary(trace_dm_marginalized, var_names=["frac", "conc"])
summary_dm_marginalized = summary_dm_marginalized.assign(
    ess_mean_per_sec=lambda x: x.ess_bulk / trace_dm_marginalized.posterior.sampling_time,
)
assert all(summary_dm_marginalized.r_hat < 1.03)

summary_dm_marginalized
```

We see that $\hat{R}$ is close to $1$ everywhere
and $\mathrm{ESS} \; \mathrm{sec}^{-1}$ is much higher.
Our reparameterization (marginalization) has greatly improved the sampling!
(And, thankfully, the HDIs look similar to the other model.)

This all looks very good, but what if we didn't have the ground-truth?

Posterior predictive checks to the rescue (again)!

```{code-cell} ipython3
with model_dm_marginalized:
    pp_samples = az.from_pymc3(
        posterior_predictive=pm.fast_sample_posterior_predictive(trace_dm_marginalized)
    )

# Concatenate with InferenceData object
trace_dm_marginalized.extend(pp_samples)
```

```{code-cell} ipython3
cmap = plt.get_cmap("tab10")

fig, axs = plt.subplots(k, 2, sharex=True, sharey=True, figsize=(8, 8))
for j, row in enumerate(axs):
    c = cmap(j)
    for _trace, ax in zip([trace_dm_marginalized, trace_multinomial], row):
        ax.hist(
            _trace.posterior_predictive.counts.sel(tree=trees[j]).values.flatten(),
            bins=np.arange(total_count),
            histtype="step",
            color=c,
            density=True,
            label="Post.Pred.",
        )
        ax.hist(
            (_trace.observed_data.counts.sel(tree=trees[j]).values.flatten()),
            bins=np.arange(total_count),
            color=c,
            density=True,
            alpha=0.25,
            label="Observed",
        )
        ax.axvline(
            true_frac[j] * total_count,
            color=c,
            lw=1.0,
            alpha=0.45,
            label="True",
        )
    row[1].annotate(
        f"{trees[j]}",
        xy=(0.96, 0.9),
        xycoords="axes fraction",
        ha="right",
        va="top",
        color=c,
    )

axs[-1, -1].legend(loc="upper center", fontsize=10)
axs[0, 1].set_title("Multinomial")
axs[0, 0].set_title("Dirichlet-multinomial")
axs[-1, 0].set_xlabel("Count")
axs[-1, 1].set_xlabel("Count")
axs[-1, 0].set_yticks([0, 0.5, 1.0])
axs[-1, 0].set_ylim(0, 0.6)
ax.set_ylim(0, 0.6);
```

_(Notice, again, that the y-axis isn't full height, and clips the distributions for species-4 in purple.)_

Compared to the multinomial (plots on the right), PPCs for the DM (left) show that the observed data is
an entirely reasonable realization of our model.
This is great news!

+++

## Model Comparison

+++

Let's go a step further and try to put a number on how much better our DM model is
relative to the raw multinomial.
We'll use leave-one-out cross validation to compare the
out-of-sample predictive ability of the two.

```{code-cell} ipython3
az.compare(
    {"multinomial": trace_multinomial, "dirichlet_multinomial": trace_dm_marginalized}, ic="loo"
)
```

Unsurprisingly, the DM outclasses the multinomial by a mile, assigning a weight of nearly
100% to the over-dispersed model.
We can conclude that between the two, the DM should be greatly favored for prediction,
parameter inference, etc.

+++

## Conclusions

Obviously the DM is not a perfect model in every case, but it is often a better choice than the multinomial, much more robust while taking on just one additional parameter.

There are a number of shortcomings to the DM that we should keep in mind when selecting a model.
The biggest problem is that, while more flexible than the multinomial, the DM
still ignores the possibility of underlying correlations between categories.
If one of our tree species relies on another, for instance, the model we've used here
will not effectively account for this.
In that case, swapping the vanilla Dirichlet distribution for something fancier (e.g. the [Generalized Dirichlet](https://en.wikipedia.org/wiki/Generalized_Dirichlet_distribution) or [Logistic-Multivariate Normal](https://en.wikipedia.org/wiki/Logit-normal_distribution#Multivariate_generalization)) may be worth considering.

+++

## References


:::{bibliography}
:filter: docname in docnames
:::

+++

## Authors
* Authored by [Byron J. Smith](https://github.com/bsmith89) on Jan, 2021 ([pymc-examples#18](https://github.com/pymc-devs/pymc-examples/pull/18))
* Updated by Abhipsha Das and Oriol Abril-Pla on August, 2021 ([pymc-examples#212](https://github.com/pymc-devs/pymc-examples/pull/212))

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p theano,xarray
```

:::{include} page_footer.md
:::

```{code-cell} ipython3

```
