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

(diagnosing_with_divergences)=
# Diagnosing Biased Inference with Divergences

:::{post} Feb, 2018
:tags: hierarchical model, diagnostics
:category: intermediate
:author: Agustina Arroyuelo
:::

```{code-cell} ipython3
from collections import defaultdict

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

print(f"Running on PyMC v{pm.__version__}")
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")
SEED = [20090425, 20180125]
```

This notebook is a PyMC port of [Michael Betancourt's post on mc-stan](http://mc-stan.org/documentation/case-studies/divergences_and_bias.html). For detailed explanation of the underlying mechanism please check the original post and Betancourt's excellent paper, [A Conceptual Introduction to Hamiltonian Monte Carlo](https://arxiv.org/abs/1701.02434).

+++

Bayesian statistics is all about building a model and estimating the parameters in that model. However, a naive or direct parameterization of our probability model can sometimes be ineffective, you can check out Thomas Wiecki's blog post, [Why hierarchical models are awesome, tricky, and Bayesian](http://twiecki.github.io/blog/2017/02/08/bayesian-hierchical-non-centered/) on the same issue in PyMC. Suboptimal parameterization often leads to slow sampling, and more problematic, biased MCMC estimators. 

Markov chain Monte Carlo (MCMC) approximates expectations with respect to a given target distribution, 

$$ \mathbb{E}{\pi} [ f ] = \int \mathrm{d}q \, \pi (q) \, f(q)$$ 

using the states of a Markov chain, ${q{0}, \ldots, q_{N} }$, 

$$ \mathbb{E}{\pi} [ f ] \approx \hat{f}{N} = \frac{1}{N + 1} \sum_{n = 0}^{N} f(q_{n}) $$  

These estimators, however, are guaranteed to be accurate only asymptotically as the chain grows to be infinitely long, 

$$ \lim_{N \rightarrow \infty} \hat{f}{N} = \mathbb{E}{\pi} [ f ]$$  

To be useful in applied analyses, we need MCMC estimators to converge to the true expectation values sufficiently quickly that they are reasonably accurate before we exhaust our finite computational resources. This fast convergence requires strong ergodicity conditions to hold, in particular geometric ergodicity between a Markov transition and a target distribution. Geometric ergodicity is a necessary condition for MCMC estimators to satisfy the Bayesian central limit theorem. This ensures not only that they are unbiased even after only a finite number of iterations but also that we can empirically quantify their precision using the MCMC standard error.

Unfortunately, proving geometric ergodicity is impossible for any nontrivial problem. Instead we must rely on empirical diagnostics that identify obstructions to geometric ergodicity, and hence, well-behaved MCMC estimators. For a general Markov transition and target distribution, the best known diagnostic is the split $\hat{R}$ statistic over an ensemble of Markov chains initialized from diffuse points in parameter space; to do any better we need to exploit the particular structure of a given transition or target distribution.
 
Hamiltonian Monte Carlo, for example, is especially powerful in this regard as its failures to be geometrically ergodic with respect to any target distribution manifest in distinct behaviors that have been developed into sensitive diagnostics. One of these behaviors is the appearance of divergences that indicate the Hamiltonian Markov chain has encountered regions of high curvature in the target distribution which it cannot adequately explore.

In this notebook we aim to identify divergences and the underlying pathologies in `PyMC`.

+++

## The Eight Schools Model

The hierarchical model of the Eight Schools dataset (Rubin 1981) as seen in `Stan`:

$$\mu \sim \mathcal{N}(0, 5)$$
$$\tau \sim \text{Half-Cauchy}(0, 5)$$
$$\theta_{n} \sim \mathcal{N}(\mu, \tau)$$
$$y_{n} \sim \mathcal{N}(\theta_{n}, \sigma_{n}),$$  

where $n \in \{1, \ldots, 8 \}$ and the $\{ y_{n}, \sigma_{n} \}$ are given as data.  

Inferring the hierarchical hyperparameters, $\mu$ and $\sigma$, together with the group-level parameters, $\theta_{1}, \ldots, \theta_{8}$, allows the model to pool data across the groups and reduce their posterior variance. Unfortunately, the direct *centered* parameterization also squeezes the posterior distribution into a particularly challenging geometry that obstructs geometric ergodicity and hence biases MCMC estimation.

```{code-cell} ipython3
# Data of the Eight Schools Model
J = 8
y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
```

## A Centered Eight Schools Implementation  

`Stan` model:

```C
data {
  int<lower=0> J;
  real y[J];
  real<lower=0> sigma[J];
}

parameters {
  real mu;
  real<lower=0> tau;
  real theta[J];
}

model {
  mu ~ normal(0, 5);
  tau ~ cauchy(0, 5);
  theta ~ normal(mu, tau);
  y ~ normal(theta, sigma);
}
```
Similarly, we can easily implement it in `PyMC`

```{code-cell} ipython3
def centered_eight_model():
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=5)
        tau = pm.HalfCauchy("tau", beta=5)
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=J)
        pm.Normal("obs", mu=theta, sigma=sigma, observed=y)
    return model
```

Unfortunately, this direct implementation of the model exhibits a pathological geometry that frustrates geometric ergodicity. Even more worrisome, the resulting bias is subtle and may not be obvious upon inspection of the sampled values alone. To understand this bias, let's consider first a short Markov chain, commonly used when computational expediency is a motivating factor, and only afterwards a longer Markov chain.

+++

### A Dangerously-Short Markov Chain

```{code-cell} ipython3
with centered_eight_model():
    short_trace = pm.sample(600, tune=500, chains=2, random_seed=SEED)
```

In the [original post](http://mc-stan.org/documentation/case-studies/divergences_and_bias.html) a single chain of 1200 sample is applied. However, since split $\hat{R}$ is not implemented in `ArviZ` we fit 2 chains with 600 sample each instead.  

The $\hat{R}$ diagnostic doesn’t indicate any problems (values are all close to 1). You could try re-running the model with a different seed and see if this still holds.

```{code-cell} ipython3
az.summary(short_trace).round(2)
```

Moreover, the trace plots all look fine. Let's consider, for example, the hierarchical standard deviation $\tau$, or more specifically, its logarithm, $log(\tau)$. Because $\tau$ is constrained to be positive, its logarithm will allow us to better resolve behavior for small values. Indeed the chains seems to be exploring both small and large values reasonably well.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Trace plot of log(tau)
    name: nb-divergence-traceplot
  image:
    alt: log-tau
---
# plot the trace of log(tau)
short_trace.posterior["log_tau"] = np.log(short_trace.posterior["tau"])
ax = az.plot_trace(short_trace.posterior["log_tau"], legend=True)
ax[0, 1].set_xlabel("Draw")
ax[0, 1].set_ylabel("log(tau)")
ax[0, 1].set_title("")

ax[0, 0].set_xlabel("log(tau)")
ax[0, 0].set_title("Probability density function of log(tau)");
```

Unfortunately, the resulting estimate for the mean of $log(\tau)$ is strongly biased away from the true value, here shown in grey.

```{code-cell} ipython3
# plot the estimate for the mean of log(τ) cumulating mean
mean_log_tau = [
    short_trace.posterior["log_tau"][:, :i].mean()
    for i in short_trace.posterior.coords["draw"].values
]
plt.figure(figsize=(15, 4))
plt.axhline(0.7657852, lw=2.5, color="gray")
plt.plot(mean_log_tau, lw=2.5)
plt.ylim(0, 2)
plt.xlabel("Iteration")
plt.ylabel("MCMC mean of log(tau)")
plt.title("MCMC estimation of log(tau)");
```

Hamiltonian Monte Carlo, however, is not so oblivious to these issues as several samples in our run ended with a divergence.

```{code-cell} ipython3
# display the total number and percentage of divergent
divergent = short_trace.sample_stats["diverging"].values
print("Number of Divergent %d" % divergent.nonzero()[0].size)
divperc = divergent.mean() * 100
print("Percentage of Divergent %.1f" % divperc)
```

Even with a single short chain these divergences are able to identity the bias and advise skepticism of any resulting MCMC estimators.

Additionally, because the divergent transitions, here shown in green, tend to be located near the pathologies we can use them to identify the location of the problematic neighborhoods in parameter space.

```{code-cell} ipython3
def pairplot_divergence(trace, ax=None, divergence=True, color="C3", divergence_color="C2"):
    theta = az.extract(trace, var_names="theta").values[0]
    logtau = az.extract(trace, var_names="log_tau").values
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(theta, logtau, "o", color=color, alpha=0.5)
    if divergence:
        divergent = az.extract(trace.sample_stats, var_names="diverging").values
        ax.plot(theta[divergent], logtau[divergent], "o", color=divergence_color)
    ax.set_xlabel("theta[0]")
    ax.set_ylabel("log(tau)")
    ax.set_title("scatter plot between log(tau) and theta[0]")
    return ax


pairplot_divergence(short_trace);
```

It is important to point out that the pathological samples from the trace are not necessarily concentrated at the funnel: when a divergence is encountered, the subtree being constructed is rejected and the transition samples uniformly from the existing discrete trajectory. Consequently, divergent samples will not be located exactly in the region of high curvature.

+++

There are many other ways to explore and visualize the pathological region in the parameter space. For example, with `arviz.plot_parallel` we can reproduce Figure 5b in [Visualization in Bayesian workflow](https://arxiv.org/pdf/1709.01449.pdf)

```{code-cell} ipython3
az.plot_parallel(short_trace);
```

### A Safer, Longer Markov Chain  

Given the potential insensitivity of split $\hat{R}$ on single short chains, `Stan` recommend always running multiple chains as long as possible to have the best chance to observe any obstructions to geometric ergodicity. Because it is not always possible to run long chains for complex models, however, divergences are an incredibly powerful diagnostic for biased MCMC estimation.

```{code-cell} ipython3
with centered_eight_model():
    longer_trace = pm.sample(2000, chains=2, tune=1000, random_seed=SEED)
```

```{code-cell} ipython3
def report_trace(trace):
    # plot the trace of log(tau)
    trace.posterior["log_tau"] = np.log(trace.posterior["tau"])
    az.plot_trace(trace, var_names=["log_tau"])

    # plot the estimate for the mean of log(τ) cumulating mean
    mean_log_tau = [
        trace.posterior["log_tau"][:, :i].mean() for i in trace.posterior.coords["draw"].values
    ]
    plt.figure(figsize=(15, 4))
    plt.axhline(0.7657852, lw=2.5, color="gray")
    plt.plot(mean_log_tau, lw=2.5)
    plt.ylim(0, 2)
    plt.xlabel("Iteration")
    plt.ylabel("MCMC mean of log(tau)")
    plt.title("MCMC estimation of log(tau)")
    plt.show()

    # display the total number and percentage of divergent
    divergent = trace.sample_stats["diverging"].values
    print("Number of Divergent %d" % divergent.nonzero()[0].size)
    divperc = divergent.mean() * 100
    print("Percentage of Divergent %.1f" % divperc)

    # scatter plot between log(tau) and theta[0]
    # for the identification of the problematic neighborhoods in parameter space
    pairplot_divergence(trace);
```

```{code-cell} ipython3
report_trace(longer_trace)
```

```{code-cell} ipython3
az.summary(longer_trace).round(2)
```

Similar to the result in `Stan`,  $\hat{R}$ does not indicate any serious issues. However, the effective sample size per iteration has drastically fallen, indicating that we are exploring less efficiently the longer we run. This odd behavior is a clear sign that something problematic is afoot. As shown in the trace plot, the chain occasionally "sticks" as it approaches small values of $\tau$, exactly where we saw the divergences concentrating. This is a clear indication of the underlying pathologies. These sticky intervals induce severe oscillations in the MCMC estimators early on, until they seem to finally settle into biased values.   

In fact the sticky intervals are the Markov chain trying to correct the biased exploration. If we ran the chain even longer then it would eventually get stuck again and drag the MCMC estimator down towards the true value. Given an infinite number of iterations this delicate balance asymptotes to the true expectation as we’d expect given the consistency guarantee of MCMC. Stopping after any finite number of iterations, however, destroys this balance and leaves us with a significant bias. 

More details can be found in Betancourt's [paper](https://arxiv.org/abs/1701.02434) on HMC.

+++

## Mitigating Divergences by Adjusting PyMC's Adaptation Routine

Divergences in Hamiltonian Monte Carlo arise when the Hamiltonian transition encounters regions of extremely large curvature, such as the opening of the hierarchical funnel. Unable to accurate resolve these regions, the transition malfunctions and flies off towards infinity. With the transitions unable to completely explore these regions of extreme curvature, we lose geometric ergodicity and our MCMC estimators become biased.

Algorithm implemented in `Stan` uses a heuristic to quickly identify these misbehaving trajectories, and hence label divergences, without having to wait for them to run all the way to infinity. This heuristic can be a bit aggressive, however, and sometimes label transitions as divergent even when we have not lost geometric ergodicity.

To resolve this potential ambiguity we can adjust the step size, $\epsilon$, of the Hamiltonian transition. The smaller the step size the more accurate the trajectory and the less likely it will be mislabeled as a divergence. In other words, if we have geometric ergodicity between the Hamiltonian transition and the target distribution then decreasing the step size will reduce and then ultimately remove the divergences entirely. If we do not have geometric ergodicity, however, then decreasing the step size will not completely remove the divergences.

Like `Stan`, the step size in `PyMC` is tuned automatically during warm up, but we can coerce smaller step sizes by tweaking the configuration of `PyMC`'s adaptation routine. In particular, we can increase the `target_accept` parameter from its default value of 0.8 closer to its maximum value of 1.

+++

### Adjusting Adaptation Routine

```{code-cell} ipython3
acceptance_runs = dict()
for target_accept in [0.85, 0.90, 0.95, 0.99]:
    with centered_eight_model():
        acceptance_runs[target_accept] = pm.sample(
            5000,
            chains=2,
            tune=2000,
            target_accept=target_accept,
            random_seed=SEED,
            progressbar=False,
        )
```

```{code-cell} ipython3
longer_trace.sample_stats["diverging"].sum().item()
```

```{code-cell} ipython3
df = pd.DataFrame(
    [
        longer_trace.sample_stats["step_size"].mean().item(),
        acceptance_runs[0.85].sample_stats["step_size"].mean().item(),
        acceptance_runs[0.90].sample_stats["step_size"].mean().item(),
        acceptance_runs[0.95].sample_stats["step_size"].mean().item(),
        acceptance_runs[0.99].sample_stats["step_size"].mean().item(),
    ],
    columns=["Step_size"],
)
df["Divergent"] = pd.Series(
    [
        longer_trace.sample_stats["diverging"].sum().item(),
        acceptance_runs[0.85].sample_stats["diverging"].sum().item(),
        acceptance_runs[0.90].sample_stats["diverging"].sum().item(),
        acceptance_runs[0.95].sample_stats["diverging"].sum().item(),
        acceptance_runs[0.99].sample_stats["diverging"].sum().item(),
    ]
)
df["delta_target"] = pd.Series([".80", ".85", ".90", ".95", ".99"])
df
```

Here, the number of divergent transitions dropped dramatically when delta was increased to 0.99. 

This behavior also has a nice geometric intuition. The more we decrease the step size the more the Hamiltonian Markov chain can explore the neck of the funnel. Consequently, the marginal posterior distribution for $log (\tau)$ stretches further and further towards negative values with the decreasing step size. 

Since in `PyMC` after tuning we have a smaller step size than `Stan`, the geometery is better explored.

However, the Hamiltonian transition is still not geometrically ergodic with respect to the centered implementation of the Eight Schools model. Indeed, this is expected given the observed bias.

```{code-cell} ipython3
_, ax = plt.subplots(1, 1, figsize=(10, 6))

acceptance_runs[0.99].posterior["log_tau"] = np.log(acceptance_runs[0.99].posterior["tau"])
pairplot_divergence(acceptance_runs[0.99], ax=ax, color="C3", divergence=False)

pairplot_divergence(longer_trace, ax=ax, color="C1", divergence=False)

ax.legend(["Centered, delta=0.99", "Centered, delta=0.85"]);
```

```{code-cell} ipython3
logtau0 = longer_trace.posterior["log_tau"]
logtau2 = np.log(acceptance_runs[0.90].posterior["tau"])
logtau1 = acceptance_runs[0.99].posterior["log_tau"]

plt.figure(figsize=(15, 4))
plt.axhline(0.7657852, lw=2.5, color="gray")

mlogtau0 = [logtau0[:, :i].mean() for i in longer_trace.posterior.coords["draw"].values]
plt.plot(mlogtau0, label="Centered, delta=0.85", lw=2.5)
mlogtau2 = [logtau2[:, :i].mean() for i in acceptance_runs[0.90].posterior.coords["draw"].values]
plt.plot(mlogtau2, label="Centered, delta=0.90", lw=2.5)
mlogtau1 = [logtau1[:, :i].mean() for i in acceptance_runs[0.99].posterior.coords["draw"].values]
plt.plot(mlogtau1, label="Centered, delta=0.99", lw=2.5)
plt.ylim(0, 2)
plt.xlabel("Iteration")
plt.ylabel("MCMC mean of log(tau)")
plt.title("MCMC estimation of log(tau)")
plt.legend();
```

## A Non-Centered Eight Schools Implementation  
 
Although reducing the step size improves exploration, ultimately it only reveals the true extent the pathology in the centered implementation. Fortunately, there is another way to implement hierarchical models that does not suffer from the same pathologies.  

In a non-centered parameterization we do not try to fit the group-level parameters directly, rather we fit a latent Gaussian variable from which we can recover the group-level parameters with a scaling and a translation.  

$$\mu \sim \mathcal{N}(0, 5)$$
$$\tau \sim \text{Half-Cauchy}(0, 5)$$
$$\tilde{\theta}_{n} \sim \mathcal{N}(0, 1)$$
$$\theta_{n} = \mu + \tau \cdot \tilde{\theta}_{n}.$$

Stan model:

```C
data {
  int<lower=0> J;
  real y[J];
  real<lower=0> sigma[J];
}

parameters {
  real mu;
  real<lower=0> tau;
  real theta_tilde[J];
}

transformed parameters {
  real theta[J];
  for (j in 1:J)
    theta[j] = mu + tau * theta_tilde[j];
}

model {
  mu ~ normal(0, 5);
  tau ~ cauchy(0, 5);
  theta_tilde ~ normal(0, 1);
  y ~ normal(theta, sigma);
}
```

```{code-cell} ipython3
def non_centered_eight_model():
    with pm.Model() as NonCentered_eight:
        mu = pm.Normal("mu", mu=0, sigma=5)
        tau = pm.HalfCauchy("tau", beta=5)
        theta_tilde = pm.Normal("theta_t", mu=0, sigma=1, shape=J)
        theta = pm.Deterministic("theta", mu + tau * theta_tilde)
        obs = pm.Normal("obs", mu=theta, sigma=sigma, observed=y)
    return NonCentered_eight
```

```{code-cell} ipython3
with non_centered_eight_model():
    fit_ncp80 = pm.sample(5000, chains=2, tune=1000, random_seed=SEED, target_accept=0.80)
```

```{code-cell} ipython3
az.summary(fit_ncp80).round(2)
```

As shown above, the effective sample size per iteration has drastically improved, and the trace plots no longer show any "stickyness". However, we do still see the rare divergence. These infrequent divergences do not seem concentrate anywhere in parameter space, which is indicative of the divergences being false positives.

```{code-cell} ipython3
report_trace(fit_ncp80)
```

As expected of false positives, we can remove the divergences entirely by decreasing the step size.

```{code-cell} ipython3
with non_centered_eight_model():
    fit_ncp90 = pm.sample(5000, chains=2, tune=1000, random_seed=SEED, target_accept=0.90)

# display the total number and percentage of divergent
divergent = fit_ncp90.sample_stats["diverging"].values
print("Number of Divergent %d" % divergent.nonzero()[0].size)
```

The more agreeable geometry of the non-centered implementation allows the Markov chain to explore deep into the neck of the funnel, capturing even the smallest values of `tau` ($\tau$) that are consistent with the measurements. Consequently, MCMC estimators from the non-centered chain rapidly converge towards their true expectation values.

```{code-cell} ipython3
_, ax = plt.subplots(1, 1, figsize=(10, 6))

fit_ncp80.posterior["log_tau"] = np.log(fit_ncp80.posterior["tau"])
pairplot_divergence(fit_ncp80, ax=ax, color="C0", divergence=False)
pairplot_divergence(acceptance_runs[0.99], ax=ax, color="C3", divergence=False)
acceptance_runs[0.90].posterior["log_tau"] = np.log(acceptance_runs[0.90].posterior["tau"])
pairplot_divergence(acceptance_runs[0.90], ax=ax, color="C1", divergence=False)

ax.legend(["Non-Centered, delta=0.80", "Centered, delta=0.99", "Centered, delta=0.90"]);
```

```{code-cell} ipython3
plt.figure(figsize=(15, 4))
plt.axhline(0.7657852, lw=2.5, color="gray")
mlogtaun = [
    fit_ncp80.posterior["log_tau"][:, :i].mean() for i in fit_ncp80.posterior.coords["draw"].values
]
plt.plot(mlogtaun, color="C0", lw=2.5, label="Non-Centered, delta=0.80")
mlogtau2 = [logtau2[:, :i].mean() for i in acceptance_runs[0.90].posterior.coords["draw"].values]
plt.plot(mlogtau2, color="C2", label="Centered, delta=0.90", lw=2.5)
mlogtau1 = [logtau1[:, :i].mean() for i in acceptance_runs[0.99].posterior.coords["draw"].values]
plt.plot(mlogtau1, color="C1", label="Centered, delta=0.99", lw=2.5)
plt.ylim(0, 2)
plt.xlabel("Iteration")
plt.ylabel("MCMC mean of log(tau)")
plt.title("MCMC estimation of log(tau)")
plt.legend();
```

## Authors
* Adapted from Michael Betancourt's post January 2017, [Diagnosing Biased Inference with Divergences](https://mc-stan.org/users/documentation/case-studies/divergences_and_bias.html)
* Updated by Agustina Arroyuelo in February 2018, ([pymc#2861](https://github.com/pymc-devs/pymc/pull/2861))
* Updated by [@CloudChaoszero](https://github.com/CloudChaoszero) in January 2021, ([pymc-examples#25](https://github.com/pymc-devs/pymc-examples/pull/25))
* Updated Markdown and styling by @reshamas in August 2022, ([pymc-examples#402](https://github.com/pymc-devs/pymc-examples/pull/402))
* Updated by @fonnesbeck in August 2024

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w
```

:::{include} ../page_footer.md
:::

```{code-cell} ipython3

```
