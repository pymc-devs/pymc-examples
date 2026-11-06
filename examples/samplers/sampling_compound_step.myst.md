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

(sampling_compound_step)=
# Compound Steps in Sampling

:::{post} October 25, 2026
:tags: sampling, step method
:category: beginner, explanation
:author: Yuan Huang, Meenal Jhajharia, Jesse Grabowski
:::

This notebook explains how the compound steps work in `pymc.sample` function when sampling multiple random variables. We are going to answer the following questions associated with compound steps:

- How do compound steps work?
- What happens when PyMC assigns step methods by default?
- How to specify the step methods? What is the order to apply the step methods at each iteration? Is there a way to specify the order of the step methods? 
- What are the issues with mixing discrete and continuous samplers, especially with HMC/NUTS?
- What happens to sample statistics that occur in multiple step methods?

```{code-cell} ipython3
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
```

```{code-cell} ipython3
az.style.use("arviz-variat")
```

## Compound steps

+++

When sampling a model with multiple free random variables, compound steps are needed in the `pm.sample` function. When compound steps are involved, the function takes a list of `step` to generate a list of `methods` for different random variables. For example, in the following code:
```python
with pm.Model() as m:
    rv1 = ... # random variable 1 (continuous)
    rv2 = ... # random variable 2 (continuous)
    rv3 = ... # random variable 3 (categorical)
    #...
    step1 = pm.Metropolis([rv1, rv2], blocked=True)
    step2 = pm.CategoricalGibbsMetropolis([rv3])
    trace = pm.sample(..., step=[step1, step2])
```
`blocked=True` makes `pm.Metropolis` propose `rv1` and `rv2` jointly. Without it, `pm.Metropolis` and `pm.Slice` assign a separate step to each variable in the list.

The compound step now contains a list of `methods`. At each sampling step, it iterates over these methods, taking a `point` as input. Each method proposes a new `point` as its output. If the Metropolis-Hastings criterion rejects the proposal, the method returns the input `point` unchanged.

+++

## Compound steps by default
To conduct Markov chain Monte Carlo (MCMC) sampling to generate posterior samples in PyMC, we specify a step method object that corresponds to a particular MCMC algorithm, such as Metropolis, Slice sampling, or the No-U-Turn Sampler (NUTS). PyMC's step methods can be assigned manually, or assigned automatically by PyMC. For automatic assignment, each step method scores how well it can sample a variable with its `competence` method, and PyMC assigns every free variable the step method with the highest score. In general:

- Binary variables, including categorical variables with two categories, will be assigned to `BinaryGibbsMetropolis`
- Categorical variables with more than two categories will be assigned to `CategoricalGibbsMetropolis`
- Other discrete variables will be assigned to `Metropolis`
- Continuous variables will be assigned to `NUTS`

Take the following example:

```{code-cell} ipython3
with pm.Model() as m:
    n_ = pm.Data("n", np.asarray([10, 15]))
    p = pm.Beta("p", 1.0, 1.0)
    ni = pm.Bernoulli("ni", 0.5)
    k = pm.Binomial("k", p=p, n=n_[ni], observed=4)
    trace = pm.sample()
```

There are two free parameters in the model we would like to sample from, a continuous variable `p` (sampled on the log-odds scale as `p_logodds__`) and a binary variable `ni`.

```{code-cell} ipython3
m.free_RVs
```

As the sampling log above shows, `NUTS` was assigned to `p` and `BinaryGibbsMetropolis` was assigned to `ni`.

+++

## Specify compound steps
Auto-assignment can be overridden for any subset of variables by specifying them manually prior to sampling:

```{code-cell} ipython3
with m:
    step1 = pm.Metropolis([p])
    step2 = pm.BinaryMetropolis([ni])
    trace = pm.sample(step=[step1, step2])
```

`pm.sample` calls these step methods for us, but we can also call them by hand. Start from the initial point of the model:

```{code-cell} ipython3
point = m.initial_point()
point
```

Pass the `point` to the first step method, `step1`, which samples `p`:

```{code-cell} ipython3
point, state = step1.step(point=point)
point, state
```

`step1` only proposes a new value for `p_logodds__`, so `ni` is unchanged whether the proposal is accepted or rejected.

Similarly, passing the updated `point` to `step2` proposes a new value for `ni` and leaves `p_logodds__` as it is:

```{code-cell} ipython3
point, state = step2.step(point=point)
point, state
```

A compound step works exactly like this, iterating over all the steps in the list. In effect, it is Metropolis-Hastings within Gibbs sampling.

`pm.sample()` builds a `pm.step_methods.CompoundStep` internally. We can also build one explicitly:

```{code-cell} ipython3
with m:
    comp_step1 = pm.step_methods.CompoundStep([step1, step2])
    trace1 = pm.sample(step=comp_step1)
comp_step1.methods
```

```{code-cell} ipython3
# These are the Sample Stats for Compound Step based sampling
list(trace1.sample_stats.data_vars)
```

Note: In a compound step, a sample stats variable may be present in more than one step method, like `accept`. PyMC stacks such a stat along an extra dimension, `accept_dim_0`, in the order of the step methods: index 0 belongs to `step1` and index 1 to `step2`.

For these Metropolis-type steps, `accept` holds the Metropolis-Hastings ratio, which can exceed 1. A value above 1 means the proposal had a higher posterior density than the current point, and the step always accepts such a proposal.

```{code-cell} ipython3
trace1.sample_stats["accept"].sel(chain=1).values
```

## Order of step methods

+++

By default, the step methods we pass to `pm.sample` run first, followed by the automatically assigned ones. Automatically assigned variables that get the same step method share a single step, and these steps run in the order in which their first variable appears in the model. When we specify the steps, the order of the list sets the order of the updates:

```{code-cell} ipython3
with m:
    comp_step2 = pm.step_methods.CompoundStep([step2, step1])
    trace2 = pm.sample(step=comp_step2)
comp_step2.methods
```

In the sampling process, every iteration applies the steps in the same order, in Gibbs-like fashion. Each of the Metropolis-type steps used here proposes a new value and compares the acceptance ratio with a draw $u \sim \text{Uniform}(0, 1)$. The proposal is accepted if $\log u < \log \pi(\theta_{\text{proposed}}) - \log \pi(\theta_{\text{current}})$, where $\pi$ is the posterior density. Other step methods, such as `NUTS`, use their own acceptance rules.

+++

Each step method gets its own `accept`. When the step order is reversed, the two panels swap: index 0 now belongs to `BinaryMetropolis` and index 1 to `Metropolis`. Clipping `accept` at 1 turns the Metropolis-Hastings ratio into the acceptance probability.

```{code-cell} ipython3
accept_prob1 = trace1["sample_stats"].to_dataset()[["accept"]].clip(max=1)
az.plot_dist(accept_prob1, var_names="accept", point_estimate="mean");
```

```{code-cell} ipython3
accept_prob2 = trace2["sample_stats"].to_dataset()[["accept"]].clip(max=1)
az.plot_dist(accept_prob2, var_names="accept", point_estimate="mean");
```

## Issues with mixing discrete and continuous sampling

+++

A recurrent issue/concern is the validity of mixing discrete and continuous sampling, especially mixing other samplers with NUTS. While in the book [Bayesian Data Analysis 3rd edition](http://www.stat.columbia.edu/~gelman/book/) Chapter 12.4, there is a small paragraph on "Combining Hamiltonian Monte Carlo with Gibbs sampling", which suggests that this could be a valid way to do, the Stan developers are always skeptical about how practical it is. (Here are more discussions about this issue [1](http://discourse.mc-stan.org/t/mcmc-sampling-does-not-work-when-execute/1918/47), [2](http://discourse.mc-stan.org/t/constraining-latent-factor-model-baysian-probabalisic-matrix-factorization-to-remove-multimodality/2152/21)). 

The concern with mixing discrete and continuous sampling is that the change in discrete parameters will affect the continuous distribution's geometry so that the adaptation (i.e., the tuned mass matrix and step size) may be inappropriate for the Hamiltonian Monte Carlo sampling. HMC/NUTS is hypersensitive to its tuning parameters (mass matrix and step size). Another issue is that we also don't know how many iterations we have to run to get a decent sample when the discrete parameters change. Though it hasn't been fully evaluated, it seems that if the discrete parameter is in low dimensions (e.g., 2-class mixture models, outlier detection with explicit discrete labeling), the mixing of discrete sampling with HMC/NUTS works OK. However, it is much less efficient than marginalizing out the discrete parameters. And sometimes it can be observed that the Markov chains get stuck quite often. In order to evaluate this more properly, one can use a simulation-based method to look at the posterior coverage and establish the computational correctness, as explained in [Cook, Gelman, and Rubin 2006](https://amstat.tandfonline.com/doi/abs/10.1198/106186006x136976).

+++

## Authors
- Authored by Yuan Huang in July 2018 ([pymc#3014](https://github.com/pymc-devs/pymc/pull/3014))
- Updated by Junpeng Lao in March 2019
- Updated by Alexandre Andorra in June 2020 ([pymc#3955](https://github.com/pymc-devs/pymc/pull/3955))
- Updated by Michael Osthege in June 2020
- Updated by Marco Gorelli in September 2020 ([pymc#4073](https://github.com/pymc-devs/pymc/pull/4073)) and October 2020 ([pymc#4165](https://github.com/pymc-devs/pymc/pull/4165))
- Updated by Meenal Jhajharia in April 2021 ([pymc-examples#141](https://github.com/pymc-devs/pymc-examples/pull/141))
- Updated by Oriol Abril-Pla in November 2021 ([pymc-examples#239](https://github.com/pymc-devs/pymc-examples/pull/239)) and February 2022 ([pymc-examples#283](https://github.com/pymc-devs/pymc-examples/pull/283))
- Updated by Michael Osthege in January 2022 ([pymc-examples#266](https://github.com/pymc-devs/pymc-examples/pull/266))
- Updated by Hector Munoz in May 2022 ([pymc-examples#321](https://github.com/pymc-devs/pymc-examples/pull/321))
- Updated by Thomas Wiecki in December 2022 ([pymc-examples#481](https://github.com/pymc-devs/pymc-examples/pull/481))
- Updated by [Jesse Grabowski](https://github.com/jessegrabowski) in October 2026

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w
```

:::{include} ../page_footer.md
:::

```{code-cell} ipython3
:tags: [remove-cell]

n_iterations = 10
rng = np.random.default_rng(seed=251)
with m:
    steps = (pm.Metropolis([p], rng=rng), pm.BinaryMetropolis([ni], rng=rng))

point = m.initial_point()
path = [(float(point["p_logodds__"]), int(point["ni"]))]
for _ in range(n_iterations):
    for step in steps:
        point, _ = step.step(point=point)
        path.append((float(point["p_logodds__"]), int(point["ni"])))
path = np.array(path)

logp_fn = m.compile_logp()
grid = np.linspace(-2.6, 1.2, 500)
lane_density = np.array(
    [[np.exp(logp_fn({"p_logodds__": x, "ni": lane})) for x in grid] for lane in (0, 1)]
)
lane_density /= lane_density.max()

ridge_height = 0.9


def ridge(x, lane):
    return lane + ridge_height * np.interp(x, grid, lane_density[lane])


# A state and the BinaryMetropolis step after it share an offset, so lane jumps stay vertical
offset_per_iteration = 0.03
x_shown = path[:, 0] + offset_per_iteration * ((np.arange(len(path)) + 1) // 2)
lanes = path[:, 1].astype(int)
y_shown = np.array([ridge(x, lane) for x, lane in zip(x_shown, lanes, strict=True)])

step_colors = ("C3", "C1")
arrow_style = {"arrowstyle": "-|>", "mutation_scale": 20, "lw": 0, "shrinkA": 0, "shrinkB": 5}
min_arrow_length = 0.05

fig, ax = plt.subplots(figsize=(6, 6))
for lane in (1, 0):
    ax.fill_between(grid, lane, ridge(grid, lane), color="C0", alpha=0.35, lw=0)
    ax.plot(grid, ridge(grid, lane), color="C0", lw=2)

for i in range(len(path) - 1):
    color = step_colors[i % len(steps)]
    if lanes[i] == lanes[i + 1]:
        segment_x = np.linspace(x_shown[i], x_shown[i + 1], 50)
        segment_y = ridge(segment_x, lanes[i])
    else:
        segment_x = x_shown[i : i + 2]
        segment_y = y_shown[i : i + 2]
    ax.plot(segment_x, segment_y, color=color, lw=3, zorder=2)
    if np.hypot(segment_x[-1] - segment_x[0], segment_y[-1] - segment_y[0]) > min_arrow_length:
        ax.annotate(
            "",
            xy=(segment_x[-1], segment_y[-1]),
            xytext=(segment_x[-2], segment_y[-2]),
            arrowprops={**arrow_style, "color": color},
            zorder=3,
        )

# Rejected steps leave the state unchanged, so consecutive repeats share one dot and label
run_starts = np.flatnonzero(np.r_[True, np.any(path[1:] != path[:-1], axis=1)])
run_ends = np.r_[run_starts[1:], len(path)] - 1
ax.scatter(x_shown[run_starts], y_shown[run_starts], s=45, color="0.2", zorder=4)
for start, end in zip(run_starts, run_ends, strict=True):
    label = str(start) if start == end else f"{start}-{end}"
    ax.annotate(
        label,
        (x_shown[start], y_shown[start]),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=11,
    )

ax.set(
    xlim=(grid[0], grid[-1]),
    ylim=(-0.05, 2.15),
    yticks=[0, 1],
    yticklabels=["ni = 0", "ni = 1"],
    xlabel="p_logodds__",
)
ax.plot([], [], color=step_colors[0], lw=3, label="Metropolis on p")
ax.plot([], [], color=step_colors[1], lw=3, label="BinaryMetropolis on ni")
ax.legend(loc="upper right", frameon=False);
```
