---
jupytext:
  default_lexer: ipython3
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
myst:
  substitutions:
    extra_dependencies: pymc-extras pyarrow
---

(streaming_tick_data)=

# Streaming variational inference on high-frequency tick data

:::{post} August 2026
:tags: variational inference, minibatch, out-of-core, hierarchical model, time series
:category: advanced, tutorial
:author: Yicheng Yang
:::

+++

Exchanges publish per-trade archives — Binance, for instance, distributes
aggregate trades as daily and monthly files — and a multi-symbol,
multi-month slice of such an archive does not fit in memory. PyMC's
{class}`~pymc.Minibatch` randomly slices tensor *inputs*; it is not itself a
disk-backed reader, so it cannot help once the array no longer fits. This
notebook fits a hierarchical hurdle–Student-t model of *next-event price
moves* by streaming minibatches from disk with pymc-extras'
{class}`~pymc_extras.variational.dataloader.DataLoader`, on 300,000 synthetic
rows that are generated inside the notebook. It tries to teach three things:

1. **When minibatch VI is even valid.** Two acceptance gates that most
   time-series models fail — and a model class that passes both.
2. **The mechanics, end to end** on an archive small enough to rebuild here:
   an on-disk global shuffle, streaming ADVI with `total_size` rescaling, and
   what a stopping rule has to be able to see before it can fire.
3. **Honest reporting.** Mean-field variational inference produces some
   numbers you should trust and some you should not; with known ground truth
   we measure which are which instead of guessing.

The notebook runs in about a minute. Everything below executes on synthetic
data with known ground truth, which is what makes the recovery checks
possible: nothing here is a claim about any real market.

+++

## Two gates: when minibatch VI is valid, and when it is worth it

Minibatch variational inference rests on one identity: if the likelihood
factors over rows given the parameters, then the batch log-likelihood scaled by
$N/B$ is an unbiased estimator of the full-data log-likelihood
{cite:p}`hoffman2013stochastic`. Two gates shaped this notebook's model — the
first is about *validity*, the second about whether streaming is doing any
real work — and they are worth internalizing before reaching for `total_size`
on your own data:

**Gate 1 — validity: the likelihood must factor over rows, sampled uniformly.**
No latent path coupling observations, no label shared across rows, and every
row given the same inclusion probability by the batching scheme (unequal
inclusion breaks the plain $N/B$ rescaling). This is exactly why the classic
stochastic volatility model of the {ref}`stochastic_volatility` notebook
*cannot* be minibatched: its latent volatility path ties every observation to
its neighbours, so a random subset of rows does not carry $B/N$ of the
log-likelihood. Any model whose rows share one outcome (for example, every tick
of a match sharing the final result) fails the same gate through
pseudo-replication — the effective sample size is the number of outcomes, not
the number of rows.

**Gate 2 — non-triviality: no low-dimensional sufficient statistics.** This
one is not about validity but about honesty of the *showcase*: a Normal
likelihood with per-cell means and variances collapses to per-cell
$(\sum y, \sum y^2, n)$, so one linear scan computes the exact posterior
inputs and "streaming inference" degenerates into a glorified `groupby` —
valid, but theater. What provably breaks the collapse: a Student-t likelihood,
a mixture (the hurdle below), and row-level continuous covariates through
nonlinear links.

The model below passes both gates, and each ingredient that makes it pass is
also substantively motivated by the data.

+++

:::{include} ../extra_installs.md
:::

:::{note}
The {class}`~pymc_extras.variational.dataloader.DataLoader` merged into
pymc-extras after the v0.14.0 release. Until the next release, install
pymc-extras from `main`:
`pip install "pymc-extras @ git+https://github.com/pymc-devs/pymc-extras"`.
:::

```{code-cell} ipython3
import json
import logging
import os
import tempfile
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pymc as pm
import pytensor.tensor as pt

from pymc_extras.variational.dataloader import DataLoader, parquet_source
from scipy import stats
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'
RANDOM_SEED = 20260731
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-variat")
# the loss figures carry the fit story; keep the fit logger's lines out of the output
logging.getLogger("pymc").setLevel(logging.ERROR)
# pytensor's predictive path emits two benign internals warnings (a 0/0 inside its
# own allclose diagnostics; numba falling back to object mode for the python
# `random`); neither is actionable here, so they are filtered by origin
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"pytensor\.tensor\.type")
warnings.filterwarnings("ignore", message=r"Numba will use object mode")
```

## The model: hierarchical hurdle–Student-t next-event returns

One row is one trade-to-trade transition $i$ on symbol $s$ at UTC hour $h$:
the return $y_i = 10^4 \, (\log p_{i+1} - \log p_i)$ in basis points, and the
move indicator $m_i = \mathbb{1}[y_i \neq 0]$. Two consequences of indexing by
*events* rather than by clock time are worth stating before the algebra.

First, a large share of consecutive trades leave the price unchanged, because
prices live on a discrete grid and many trades do not cross to a new level. A
purely continuous likelihood puts zero probability mass on that outcome. The
fix is a hurdle: model *whether* the price moves separately from *how far* it
moves given that it does.

Second, each row is a transition between consecutive observed events, not a
fixed-duration return. So $1 - \pi$ is the probability that the next event
leaves the price unchanged, and $\sigma$ is the Student-t scale *conditional on
a move*. Neither is clock-time volatility: converting to a clock would
additionally require a model of event arrival times
{cite:p}`engle1998autoregressive`.

$$
\begin{aligned}
m_i &\sim \text{Bernoulli}(\pi_i) \\
y_i \mid m_i = 1 &\sim \text{StudentT}(\nu,\ \mu_i,\ \sigma_i) \\
\operatorname{logit} \pi_i &= \kappa_0 + b^{(\kappa)}_s
  + B(h)^\top\!\left(c + b^{(\pi h)}_s\right)
  + \lambda_a a_i + \lambda_q q_i \\
\log \sigma_i &= \alpha_0 + b^{(\alpha)}_s
  + B(h)^\top\!\left(g + b^{(\sigma h)}_s\right)
  + (\beta_a + b^{(\beta a)}_s)\, a_i + \beta_q q_i \\
\mu_i &= \theta_d\, d_i + \theta_r\, \text{ylag}_i
\end{aligned}
$$

The covariates are a simulated trade sign $d_i$, a notional-like $q_i$, an
activity-like $a_i$, and $\text{ylag}_i$, the most recent previous nonzero
return on that symbol. In the generator below they are drawn directly rather
than engineered from a trade tape: only $\text{ylag}$ is produced
sequentially, so it is the one covariate that could not see the future, and
$q$ and $a$ are standardized once over the whole sample. On a real archive
both would instead be trailing statistics built causally from the tape and
standardized with constants frozen on a burn-in window — a construction that
is out of scope here, and one that is easy to get subtly wrong. $B(h)$ is the first two sine/cosine harmonics of hour-of-day,
the standard way to encode intraday periodicity smoothly
{cite:p}`andersen1997intraday`; heavy-tailed Student-t noise for returns goes
back at least to {cite:t}`bollerslev1987conditionally`. All symbol effects
$b_s \sim \mathcal{N}(0, \tau)$ are partially pooled {cite:p}`gelman2006data`,
so a thin symbol borrows the global intraday shape where its own data run out —
that is the hierarchical payoff we will measure. They are parameterized
*centered*, deliberately: the non-centered trick pays off when groups are
data-poor, and the generator below gives every symbol at least several hundred
rows.
The degrees of freedom are
shared across symbols, parameterized $\nu = 1 + \operatorname{softplus}(\eta)$;
the floor of 1 (rather than the more comfortable 2) leaves room for tails too
heavy to carry a finite variance — and the estimand below is chosen to stay
meaningful exactly there.

The reported estimand is **event-return dispersion on the event clock**:
$\pi_{s,h}$ together with the conditional-move 90% half-width
$\sigma_{s,h} \cdot t^{-1}_{0.95}(\nu)$. A quantile is the safe choice here: it
stays finite for any $\nu > 0$, whereas a moment-based dispersion requires
$\nu > 2$ — a constraint the model does not impose, even though the generator
below happens to use $\nu = 3.5$.

+++

## Synthetic data with known truth

Notebooks in this collection do not download data at build time, so the
executed path uses a seeded synthetic generator: the schema an exchange
archive would have after feature construction, a hurdle at exactly zero,
heavy conditional tails, and twelve symbols with a deliberately thin tail so
shrinkage is visible. Because the truth is known, recovery is checkable. Every
statement below about what the fit recovers is conditional on this generator;
none of it is a measurement of any market.

```{code-cell} ipython3
n_symbols = 12
counts = np.array(
    [90_700, 55_000, 40_000, 30_000, 24_000, 19_000, 15_000, 11_000, 8_000, 4_600, 1_800, 900]
)
thin = [10, 11]  # the two symbols with the least data

truth = {
    "kappa0": np.log(0.7 / 0.3),  # logit(0.7): see the annotation below
    "c": np.array([0.25, -0.15, 0.10, 0.05]),
    "lambda_a": 0.35,
    "lambda_q": 0.20,
    "alpha0": np.log(0.05),  # conditional-move scale, in basis points
    "g": np.array([0.20, -0.12, 0.08, 0.04]),
    "beta_a": 0.18,
    "beta_q": 0.12,
    "theta_d": 0.02,
    "theta_r": 0.25,
    "nu": 3.5,
}


def standardize(x, axis=0):
    return (x - x.mean(axis=axis, keepdims=True)) / x.std(axis=axis, keepdims=True)


z_truth = {
    "z_k": standardize(rng.standard_normal(n_symbols)),
    "z_ph": standardize(rng.standard_normal((n_symbols, 4))),
    "z_al": standardize(rng.standard_normal(n_symbols)),
    "z_sh": standardize(rng.standard_normal((n_symbols, 4))),
    "z_ba": standardize(rng.standard_normal(n_symbols)),
}
```

```{code-cell} ipython3
def hour_basis(hour):
    w = 2 * np.pi * np.asarray(hour, dtype=float) / 24.0
    return np.column_stack([np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)])


sym = np.repeat(np.arange(n_symbols), counts)
n = len(sym)
hour = rng.integers(0, 24, size=n)
d = rng.choice([-1.0, 1.0], size=n)
a = standardize(rng.standard_normal(n))  # trailing activity (already standardized)
q = standardize(0.3 * a + np.sqrt(1 - 0.3**2) * rng.standard_normal(n))

B = hour_basis(hour)
logit_pi = (
    truth["kappa0"]
    + 0.30 * z_truth["z_k"][sym]
    + B @ truth["c"]
    + (B * (0.15 * z_truth["z_ph"][sym])).sum(1)
    + truth["lambda_a"] * a
    + truth["lambda_q"] * q
)
log_sigma = (
    truth["alpha0"]
    + 0.35 * z_truth["z_al"][sym]
    + B @ truth["g"]
    + (B * (0.12 * z_truth["z_sh"][sym])).sum(1)
    + (truth["beta_a"] + 0.10 * z_truth["z_ba"][sym]) * a
    + truth["beta_q"] * q
)

m = (rng.random(n) < 1 / (1 + np.exp(-logit_pi))).astype(np.int8)
t_draw = rng.standard_t(truth["nu"], size=n)

# sequential generation per symbol so ylag feeds back causally
y = np.zeros(n)
ylag = np.zeros(n)
sigma = np.exp(log_sigma)
for s in range(n_symbols):
    idx = np.flatnonzero(sym == s)
    last = 0.0
    for i in idx:
        ylag[i] = last
        if m[i]:
            y[i] = truth["theta_d"] * d[i] + truth["theta_r"] * last + sigma[i] * t_draw[i]
            last = y[i]

print(
    f"{n:,} rows, zero-move share {(m == 0).mean():.1%}, "
    f"median nonzero |y| {np.median(np.abs(y[m == 1])):.3f} bp, "
    f"corr(a, q) {np.corrcoef(a, q)[0, 1]:.2f}"
)
```

Each constant above is a setting, not a measurement, and it is worth reading
them the way one would read a specification for a simulator:

* `kappa0` = $\operatorname{logit}(0.7)$. If every other term in the logit were
  zero, the move probability would be 0.7. The harmonics, the covariates and
  the symbol effects all shift it, so the *realized* zero share is the number
  printed above, not $30\%$ by construction.
* `theta_r` $= 0.25$ imposes positive first-order dependence in the conditional
  location: the next move is generated partly from the previous nonzero one. It
  is a synthetic dependence parameter — note that the classical bid–ask bounce
  of {cite:t}`roll1984simple` runs the other way, inducing *negative* serial
  dependence, which this generator does not encode.
* `theta_d` $= 0.02$ is the coefficient on the simulated trade sign: holding
  everything else fixed, sign $+1$ versus $-1$ differs by 0.04 bp in
  conditional location.
* $a$ and $q$ are constructed with a target correlation of 0.3 rather than
  orthogonally, so the fit faces mildly collinear covariates instead of a
  textbook design; the realized sample correlation is printed below.
* `ylag` is generated per symbol in sequence, so row $i$ sees only the most
  recent previously generated nonzero move. That discipline applies to the lag
  construction; the standardization of $a$ and $q$ still uses full-sample
  constants, an offline simplification kept for readability.

With that in mind, here is the feature that forces the hurdle:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
moves = y[m == 1]
ax.hist(moves, bins=201, range=(-1.5, 1.5), log=True, color="C0", label="nonzero moves")
ax.bar(
    [0.0],
    [(m == 0).sum()],
    width=0.02,
    color="C1",
    label=f"exactly zero ({(m == 0).mean():.0%} of rows)",
)
ax.set_xlabel("next-event return (bp)")
ax.set_ylabel("count (log scale)")
ax.set_title("A continuous density puts zero mass on the most common outcome")
ax.legend();
```

## The on-disk global shuffle

The trap in streaming ordered data is subtle enough to deserve its own
section. A bounded runtime shuffle buffer only *block*-shuffles a strongly
ordered stream: with tick data sorted by symbol and time, early optimization
steps would only ever see early dates and the first symbols, and an early
stopping decision would be biased by construction. The fix is to shuffle
**once, globally, on disk at ETL time**: every row gets a deterministic hash
key, rows are scattered across shards by that key, and each shard is sorted by
it. After that, sequential reads follow one fixed, data-independent uniform
permutation — replayed identically each epoch, which is single-shuffle SGD
semantics rather than fresh per-step subsampling — and the loader can run
with `shuffle=False` at full speed.

```{code-cell} ipython3
data_dir = tempfile.mkdtemp(prefix="ticks_")
table = pa.table(
    {
        "y_bp": y.astype(np.float32),
        "m": m,
        "d": d.astype(np.int8),
        "q_std": q.astype(np.float32),
        "a_std": a.astype(np.float32),
        "ylag_bp": ylag.astype(np.float32),
        "hour": hour.astype(np.int8),
        "sym": sym.astype(np.int16),
    }
)


def splitmix64(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return x ^ (x >> np.uint64(31))


key = splitmix64(np.arange(n, dtype=np.uint64))
order = np.argsort(key, kind="stable")
n_shards = 8
for i in range(n_shards):
    part = table.take(order[i::n_shards])
    # Row groups are the loader's batches on the shuffle=False path, so size them here.
    pq.write_table(part, os.path.join(data_dir, f"shard_{i:03d}.parquet"), row_group_size=1024)

head = pq.read_table(os.path.join(data_dir, "shard_000.parquet")).slice(0, 10_000)
print(
    f"first 10k rows of one shard cover {len(np.unique(head['hour']))} hours "
    f"and {len(np.unique(head['sym']))} symbols"
)
```

The printed check is the one to keep: the head of any shard must already mix
every hour and nearly every symbol, asserted rather than assumed. Note that
the cell above is the pedagogical miniature — it sorts the whole table in
memory, which is exactly what one cannot do once the table stops fitting. The
out-of-core version of the same permutation is two passes: hash-scatter rows
into shards with bounded row-group appends, then sort each shard
independently, so no step holds more than one shard.

+++

## Streaming the model

With `shuffle=False` the `DataLoader` passes source blocks through verbatim —
one block per Parquet row group, in a frozen column order — which is why the
shards were written with `row_group_size=1024` above. The dataset size comes
from the Parquet metadata: `loader.total_size` is `N`, and `len(loader)` is
`total_size // batch_size`, the number of *full* batches — which, as the cell
below shows, is not the same as the number of blocks an epoch yields on this
path. The model reads
one `pm.Data` placeholder; everything derived — the Fourier basis, the
integer symbol index — is computed inside the graph, so advancing the stream
is a single `set_value` per step.

```{code-cell} ipython3
columns = ["y_bp", "m", "d", "q_std", "a_std", "ylag_bp", "hour", "sym"]
loader = DataLoader(
    parquet_source(data_dir, columns=columns),
    batch_size=1024,
    shuffle=False,  # the shards are already globally shuffled on disk
    total_size="auto",
)

# Count one epoch instead of inferring it: on the verbatim path a block is a row
# group, and the last group of each shard is short, so the block count is not
# total_size // batch_size.
block_rows = [b.shape[0] for b in loader]
steps_per_epoch = len(block_rows)
assert sum(block_rows) == loader.total_size == n
n_full = block_rows.count(1024)
print(
    f"N = {loader.total_size:,} rows -> {steps_per_epoch} blocks per epoch "
    f"({n_full} of 1024 rows + {steps_per_epoch - n_full} shorter), "
    f"conserving {sum(block_rows):,} rows; len(loader) = {len(loader)}"
)
```

Two things that assertion pins down. First, **nothing is dropped**: verbatim
pass-through streams every row, and the short trailing group of each shard is
simply a shorter batch. The `total_size` rescaling copes, because PyMC scales
the minibatch log-likelihood by `N / b` with `b` read from the batch actually
installed, so a short block is weighted up by exactly its own size and every
row enters the epoch exactly once, with no tempering correction to track. One
precision on Gate 1: the replay is a fixed permutation rather than fresh
random sampling at each step, so what holds here is equal inclusion *over an
epoch* — single-shuffle SGD semantics — not independent uniform batches.
(Only the shuffle-buffer path drops a trailing partial batch; this notebook
never uses it.)

Second, the printed comparison makes the earlier caveat concrete: because the
short trailing blocks are streamed rather than dropped, an epoch yields a few
more blocks than `total_size // batch_size`. Anything that needs a true epoch
boundary — the stopping horizon below, for instance — should count blocks
rather than trust the division.

```{code-cell} ipython3
def build_model(symbols, batch_init, total_size):
    coords = {"symbol": list(symbols), "harmonic": ["sin1", "cos1", "sin2", "cos2"]}
    with pm.Model(coords=coords) as model:
        batch = pm.Data("batch", batch_init)
        y_ = batch[:, 0]
        d_ = batch[:, 2]
        q_ = batch[:, 3]
        a_ = batch[:, 4]
        ylag_ = batch[:, 5]
        w = 2.0 * np.pi * batch[:, 6] / 24.0
        B_ = pt.stack([pt.sin(w), pt.cos(w), pt.sin(2 * w), pt.cos(2 * w)], axis=1)
        sym_ = pt.cast(batch[:, 7], "int32")

        kappa0 = pm.Normal("kappa0", 0.0, 5.0)
        c = pm.Normal("c", 0.0, 0.5, dims="harmonic")
        lambda_a = pm.Normal("lambda_a", 0.0, 0.5)
        lambda_q = pm.Normal("lambda_q", 0.0, 0.5)

        alpha0 = pm.Normal("alpha0", 0.0, 5.0)
        g = pm.Normal("g", 0.0, 0.3, dims="harmonic")
        beta_a = pm.Normal("beta_a", 0.0, 0.3)
        beta_q = pm.Normal("beta_q", 0.0, 0.3)

        theta_d = pm.Normal("theta_d", 0.0, 0.25)
        theta_r = pm.Normal("theta_r", 0.0, 0.25)

        tau_k, tau_ph, tau_al, tau_sh, tau_ba = (
            pm.LogNormal(name, 0.0, 1.5)
            for name in ["tau_k", "tau_ph", "tau_al", "tau_sh", "tau_ba"]
        )
        b_k = pm.Normal("b_k", 0.0, tau_k, dims="symbol")
        b_ph = pm.Normal("b_ph", 0.0, tau_ph, dims=("symbol", "harmonic"))
        b_al = pm.Normal("b_al", 0.0, tau_al, dims="symbol")
        b_sh = pm.Normal("b_sh", 0.0, tau_sh, dims=("symbol", "harmonic"))
        b_ba = pm.Normal("b_ba", 0.0, tau_ba, dims="symbol")

        eta = pm.Normal("eta", 5.0, 1.0)
        nu = pm.Deterministic("nu", 1.0 + pt.softplus(eta))

        logit_pi = (
            kappa0
            + b_k[sym_]
            + (B_ * (c + b_ph[sym_])).sum(axis=-1)
            + lambda_a * a_
            + lambda_q * q_
        )
        log_sigma = (
            alpha0
            + b_al[sym_]
            + (B_ * (g + b_sh[sym_])).sum(axis=-1)
            + (beta_a + b_ba[sym_]) * a_
            + beta_q * q_
        )
        mu = theta_d * d_ + theta_r * ylag_

        def hurdle_logp(value, logit_pi, mu, log_sigma, nu):
            # one coherent mixed distribution: an atom at exactly zero plus a
            # Student-t density off zero — the move indicator is value != 0,
            # never a separate parameter, so logp and random describe the SAME law
            sigma = pt.exp(log_sigma)
            t_ll = (
                pt.gammaln((nu + 1.0) / 2.0)
                - pt.gammaln(nu / 2.0)
                - 0.5 * pt.log(nu * np.pi)
                - log_sigma
                - (nu + 1.0) / 2.0 * pt.log1p(((value - mu) / sigma) ** 2 / nu)
            )
            moved = pt.neq(value, 0.0)
            return pt.where(moved, -pt.softplus(-logit_pi) + t_ll, -pt.softplus(logit_pi))

        def hurdle_random(logit_pi, mu, log_sigma, nu, rng=None, size=None):
            # simulate the same law: first whether the price moves, then how far
            pi = 1.0 / (1.0 + np.exp(-logit_pi))
            move = rng.random(size=size) < pi
            draw = mu + np.exp(log_sigma) * rng.standard_t(nu, size=size)
            return np.where(move, draw, 0.0)

        pm.CustomDist(
            "y_obs",
            logit_pi,
            mu,
            log_sigma,
            nu,
            logp=hurdle_logp,
            random=hurdle_random,
            observed=y_,
            total_size=total_size,
        )
    return model
```

Two implementation notes worth pausing on. The likelihood is a
{class}`~pymc.CustomDist` with a `logp`, not a `pm.Potential`: `CustomDist`
gives the term *observed-RV semantics*, which is what makes PyMC's own
`total_size` minibatch rescaling apply {cite:p}`kucukelbir2015automatic`. And
the hurdle's two logs are written with `softplus` —
$\log \pi = -\operatorname{softplus}(-x)$,
$\log(1-\pi) = -\operatorname{softplus}(x)$ — which is exact and stable at both
tails. The exact float comparison `value != 0` is safe here because the zeros
are *structural* — a transition either changes the price or it does not, and
the smallest genuine move on any listed tick grid sits dozens of orders of
magnitude above the float32 flush-to-zero threshold.

```{code-cell} ipython3
model = build_model(
    [f"SYM{i:02d}" for i in range(n_symbols)], next(iter(loader)), loader.total_size
)
pm.model_to_graphviz(model)
```

The plate diagram makes the two-link structure visible at a glance: five
partially pooled effect families on the symbol plate feeding two linear
predictors, one shared tail parameter, and a single observed node whose batch
dimension is whatever the placeholder currently holds.

The loop below is the whole streaming adapter: a `pm.fit` callback that pushes
the next block into the placeholder after each step. pymc-extras#710 proposes
an interim wrapper around this same data-advance lifecycle while
pymc-extras#635 develops the longer-term ADVI API, so the twenty lines here are
deliberately the minimal version a notebook can own.

```{code-cell} ipython3
class StreamAdvance:
    """pm.fit callback: put the next minibatch into the placeholder after each step."""

    def __init__(self, model, loader):
        self._shared = model["batch"]
        self._stream = self._endless(loader)

    @staticmethod
    def _endless(loader):
        while True:
            yield from loader

    def prime(self):
        self._shared.set_value(next(self._stream), borrow=True)

    def __call__(self, approx, losses, i):
        self._shared.set_value(next(self._stream), borrow=True)


stream = StreamAdvance(model, loader)
stream.prime()

with model:
    advi = pm.ADVI(random_seed=RANDOM_SEED)
advi.fit(6_000, obj_optimizer=pm.adam(learning_rate=0.02), callbacks=[stream], progressbar=False)
approx = advi.fit(
    2_500, obj_optimizer=pm.adam(learning_rate=0.005), callbacks=[stream], progressbar=False
)
```

The two-stage learning rate is not decoration: at a constant 0.02 the Adam
updates near the optimum are larger than the posterior standard deviations of
the sharpest parameters, so the mean jitters around the mode instead of
settling into it. Annealing was adopted after measuring exactly that on the
prototype.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 3), layout="constrained")
ax.plot(approx.hist, lw=0.5)
ax.axvline(6_000, color="C1", ls="--", lw=1, label="anneal: lr 0.02 to 0.005")
ax.set_ylim(-450, 1200)
ax.set_xlabel("step")
ax.set_ylabel("negative ELBO")
ax.set_title(
    "Streaming ADVI loss, y clipped to the plateau (early losses are "
    "off scale).\nThe visible band is batch-resampling noise — "
    "remember it for the stopping section"
)
ax.legend();
```

## Did it recover the truth?

```{code-cell} ipython3
idata = approx.sample(2_000, random_seed=RANDOM_SEED)
post = idata.posterior

scalar_params = [
    "kappa0",
    "alpha0",
    "lambda_a",
    "lambda_q",
    "beta_a",
    "beta_q",
    "theta_d",
    "theta_r",
    "nu",
]
rows = {name: [float(post[name].mean()), float(post[name].std())] for name in scalar_params}
# the intercepts share a ridge with their group means; only the sums are identified
for icpt, b in [("kappa0", "b_k"), ("alpha0", "b_al"), ("beta_a", "b_ba")]:
    s = post[icpt] + post[b].mean("symbol")
    rows[f"{icpt} + mean({b})"] = [float(s.mean()), float(s.std())]
recovery = pd.DataFrame(rows, index=["mean", "sd"]).T
recovery["truth"] = [truth[p] for p in scalar_params] + [
    truth["kappa0"],
    truth["alpha0"],
    truth["beta_a"],
]
recovery["abs_error"] = (recovery["mean"] - recovery["truth"]).abs()
recovery.round(4)
```

Read the table bottom-up. The raw intercepts look off by 0.3 and 1.3 — but a
global coefficient and the mean of its group effects are only *jointly*
identified: the likelihood constrains their sum, and the prior only weakly
splits it. The identified sums in the last three rows land within about one
hundredth of the truth, comparable to the other likelihood-identified rows.
The same translation ridge exists for every global-coefficient/group-effect
pair in this model (including the vector pairs $c$/$b^{(\pi h)}$ and
$g$/$b^{(\sigma h)}$, whose sums we spare you) — so raw split rows like
`kappa0`, `alpha0`, and `beta_a` are prior-dependent decompositions, not
estimates of their generating values; a sum-to-zero constraint on the symbol
effects is the standard reparameterization when the split itself matters.
Notice also what we did *not* print: z-scores against the truth. Some would
exceed 2, because the mean-field standard deviations in the third column are
razor-thin — a z-score against a point truth mixes the sampling noise of this
particular synthetic dataset with variational overconfidence along exactly
such ridges. The stance behind that omission is a risk-management one, and it
is worth stating plainly: material disagreement between independent-seed fits
is enough to withhold a claim about a width, while agreement only removes the
instability warning — it does not validate calibration. Replication can veto a
width; it can never certify one.

The more interesting check is hierarchical: what happened to the per-symbol
effects of the two symbols with 1,800 and 900 rows?

```{code-cell} ipython3
# plot the identified CONTRAST b - mean(b): the table above showed the absolute
# level belongs to a ridge with the intercept, so comparing raw b to raw truth
# would only display the arbitrary level split. Hand-rolled rather than
# az.plot_forest because the display is bespoke: truth overlay + thin shading.
bc = post["b_al"] - post["b_al"].mean("symbol")
b_mean = bc.mean(("chain", "draw")).values
b_sd = bc.std(("chain", "draw")).values
b_truth = 0.35 * z_truth["z_al"]  # the generator's scale times its z draws
b_truth = b_truth - b_truth.mean()

fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
x = np.arange(n_symbols)
ax.errorbar(
    x, b_mean, yerr=2 * b_sd, fmt="o", color="C0", capsize=3, label="mean ± 2 mean-field sd"
)
ax.scatter(x, b_truth, marker="x", color="C1", s=60, zorder=3, label="truth")
for t in thin:
    ax.axvspan(t - 0.4, t + 0.4, color="C3", alpha=0.12)
ax.set_xticks(x)
ax.set_xticklabels([f"S{i}" for i in x])
ax.set_xlabel("symbol (shaded = thin: 1,800 and 900 rows)")
ax.set_ylabel(r"$b^{(\alpha)}_s - \bar{b}^{(\alpha)}$")
ax.set_title("Symbol effects against the generating values (thin symbols shaded)")
ax.legend()

order = np.argsort(-b_sd)
print(
    "posterior sd of the contrast, widest first: "
    + ", ".join(f"S{i:02d} {b_sd[i]:.3f} ({counts[i]:,} rows)" for i in order[:3])
    + f" ... narrowest S{order[-1]:02d} {b_sd[order[-1]]:.3f} ({counts[order[-1]]:,} rows)"
)
```

What the figure supports is narrower than the usual slogan, so it is worth
being exact. The posterior spread is widest for the symbols with the least
data — the printout orders them — which is the hierarchy expressing that it
knows less about those groups {cite:p}`gelman2006data`. The point estimates
track their generating values across the board, including the thin ones.

What the figure does *not* show is shrinkage in the strict sense: that would
need an unpooled per-symbol fit to compare against, and this notebook does not
run one. Nor are the bars calibrated intervals — the paragraph above declined
to treat mean-field widths that way, and that applies to their ordering too.
And whether pooling improves held-out prediction on real data is a separate
question again, which nothing here tests.

What do those parameters mean as *objects*? For one symbol, holding the
covariates at $a = q = 0$, the fit implies two curves over the event clock: the
probability that the next event leaves the price unchanged, and the 90%
half-width of the move given that one happens. Both are built from
global-plus-symbol sums, so neither sits on the translation ridge.

```{code-cell} ipython3
s_idx, hours = 0, np.arange(24)
Bh = hour_basis(hours)  # (24, 4)
# .dataset: idata.posterior is a DataTree in arviz 1.x
st = post.dataset.stack(sample=("chain", "draw"))


def pick(name, *dims):
    return st[name].transpose(*dims, "sample").values


logit_pi = (
    pick("kappa0")[None, :]
    + pick("b_k", "symbol")[s_idx][None, :]
    + Bh @ (pick("c", "harmonic") + pick("b_ph", "symbol", "harmonic")[s_idx])
)
log_sigma = (
    pick("alpha0")[None, :]
    + pick("b_al", "symbol")[s_idx][None, :]
    + Bh @ (pick("g", "harmonic") + pick("b_sh", "symbol", "harmonic")[s_idx])
)
# transform each draw, then summarise: the median of a function, not a function of medians
zero_prob = np.median(1.0 / (1.0 + np.exp(logit_pi)), axis=1)
half90 = np.median(np.exp(log_sigma) * stats.t.ppf(0.95, pick("nu"))[None, :], axis=1)

t_logit = (
    truth["kappa0"]
    + 0.30 * z_truth["z_k"][s_idx]
    + Bh @ (truth["c"] + 0.15 * z_truth["z_ph"][s_idx])
)
t_logsig = (
    truth["alpha0"]
    + 0.35 * z_truth["z_al"][s_idx]
    + Bh @ (truth["g"] + 0.12 * z_truth["z_sh"][s_idx])
)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
for ax, post_curve, truth_curve, ylab in [
    (axes[0], zero_prob, 1.0 / (1.0 + np.exp(t_logit)), "P(next event does not move)"),
    (axes[1], half90, np.exp(t_logsig) * stats.t.ppf(0.95, truth["nu"]), "90% half-width (bp)"),
]:
    ax.plot(hours, post_curve, color="C0", lw=2, label="posterior median")
    ax.plot(hours, truth_curve, color="C1", ls="--", lw=1.5, label="generator truth")
    ax.set_xlabel("UTC hour")
    ax.set_ylabel(ylab)
axes[0].legend(fontsize=9)
fig.suptitle(f"SYM{s_idx:02d} on the event clock, at a = q = 0", fontsize=11);
```

The half-width is a quantile of the *outcome* distribution, not a credible band
for the curve: it answers "how far does a move go", and it stays finite for any
$\nu > 0$, which is why the notebook reports it instead of a conditional
standard deviation — the Student-t variance exists only for $\nu > 2$, and
nothing in this model guarantees that. Under this generator, the fit recovers
both shapes; on real data the same two curves would be estimates, and would
need out-of-sample evaluation before being used for anything.

+++

## In-sample posterior predictive adequacy

A recovery table checks parameters; a posterior predictive check asks the
model to reproduce the data features it exists to describe. The `CustomDist`
carries a `random` implementation alongside its `logp`, so
{func}`~pymc.sample_posterior_predictive` works out of the box. The two
features that matter for this model are the exact-zero share and the heavy
conditional tail:

```{code-cell} ipython3
eval_batch = next(iter(loader))
model["batch"].set_value(eval_batch, borrow=True)  # same path the callback uses
with model:
    idata = pm.sample_posterior_predictive(
        idata,
        var_names=["y_obs"],
        random_seed=RANDOM_SEED,
        extend_inferencedata=True,
        progressbar=False,
    )

pp = idata.posterior_predictive["y_obs"].stack(sample=("chain", "draw")).values.T
y_eval = eval_batch[:, 0]

zero_share = (pp == 0).mean(axis=1)
med_nonzero = np.array([np.median(np.abs(d[d != 0])) for d in pp[:500]])
q99_nonzero = np.array([np.quantile(np.abs(d[d != 0]), 0.99) for d in pp[:500]])


def check_row(observed, draws):
    lo, hi = np.quantile(draws, [0.05, 0.95])
    return [observed, draws.mean(), lo, hi]


y_nonzero = np.abs(y_eval[y_eval != 0])
pd.DataFrame(
    [
        check_row((y_eval == 0).mean(), zero_share),
        check_row(np.median(y_nonzero), med_nonzero),
        check_row(np.quantile(y_nonzero, 0.99), q99_nonzero),
    ],
    columns=["observed", "pp mean", "pp 5%", "pp 95%"],
    index=["zero share", "median |move| (bp)", "q99 |move| (bp)"],
).round(3)
```

```{code-cell} ipython3
# hand-rolled ECDF: the bespoke feature is the annotated discrete jump at zero,
# which the predictive must reproduce in both location and height
fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
grid = np.linspace(-0.5, 0.5, 801)
for draw in pp[:60]:
    ax.plot(grid, np.searchsorted(np.sort(draw), grid) / draw.size, color="C0", alpha=0.08, lw=1)
ax.plot(
    grid, np.searchsorted(np.sort(y_eval), grid) / y_eval.size, color="k", lw=1.6, label="observed"
)
ax.plot([], [], color="C0", label="posterior predictive (60 draws)")
zero_jump = (y_eval == 0).mean()
below = (y_eval < 0).mean()
ax.annotate(
    f"vertical step at exactly 0:\nthe zero share ({zero_jump:.0%})",
    (0.03, below + zero_jump / 2),
    fontsize=9,
    ha="left",
    va="center",
)
ax.set_xlabel("next-event return (bp)")
ax.set_ylabel("ECDF")
ax.set_title("Posterior predictive ECDF: the step at zero is the hurdle")
ax.legend(loc="lower right");
```

All three statistics sit inside their predictive 90% bands. The heading is literal: the
evaluation batch was seen during the fit, so this is adequacy, not
generalisation. A held-out split is the next thing to add before any of this
is used to compare models.

+++

## What a stopping rule has to be able to see

A streamed ELBO trace is noisy: every value is a one-batch, one-Monte-Carlo
estimate, so consecutive losses differ mostly because the batch changed. That
makes "has it converged?" a signal-detection problem, and the horizon over
which you look is the whole game.

Write the standardised *block contrast* at horizon $w$: average the $w$ losses
before a point, average the $w$ losses after it, and divide the difference by
the noise scale of that difference,

$$
z_w(t) = \frac{\bar L_{t-2w:t-w} - \bar L_{t-w:t}}
              {\hat\sigma \sqrt{2/w}},
$$

with $\hat\sigma$ estimated from successive differences. Positive $z_w$ means
the loss fell. Averaging divides the noise by $\sqrt{w}$ while a steady drift
accumulates linearly in $w$, so the detectable drift shrinks like $w^{-3/2}$:
the horizon does not merely smooth the picture, it sets what is visible at all.

```{code-cell} ipython3
# Stage 2 only: after the last planned optimizer change, which is where a
# stopping decision would actually be taken.
stage2 = np.asarray(approx.hist[6_000:], dtype=float)
sigma_hat = np.mean(np.abs(np.diff(stage2))) * np.sqrt(np.pi) / 2.0


def signed_z(losses, w):
    """Standardised contrast between adjacent blocks of w losses."""
    csum = np.concatenate([[0.0], np.cumsum(losses)])
    t = np.arange(2 * w, len(losses))
    older = (csum[t - w] - csum[t - 2 * w]) / w
    newer = (csum[t] - csum[t - w]) / w
    return t, (older - newer) / (sigma_hat * np.sqrt(2.0 / w))


horizons = [(1, "per step"), (steps_per_epoch, "one epoch"), (2 * steps_per_epoch, "two epochs")]
# control: a horizon that is NOT a whole number of passes, to test whether the
# variance collapse below is really about epoch alignment
control = [(steps_per_epoch + steps_per_epoch // 2, "1.5 epochs")]
for w, label in horizons + control:
    _, z = signed_z(stage2, w)
    print(
        f"{label:>10s} (w={w:4d}):  mean z {z.mean():+.3f}   sd {z.std():.2f}"
        f"   |mean|/sd {abs(z.mean()) / z.std():.2f}"
    )
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 3.2), layout="constrained")
for (w, label), color in zip(horizons, ["C7", "C0", "C1"]):
    t, z = signed_z(stage2, w)
    ax.plot(
        t,
        z,
        color=color,
        lw=0.9 if w == 1 else 1.6,
        alpha=0.5 if w == 1 else 1.0,
        label=f"{label} (w={w})",
    )
ax.axhline(0.0, color="k", lw=0.8)
ax.set_xlabel("step within stage 2")
ax.set_ylabel(r"standardised block contrast $z_w$")
ax.set_title("The same trace, three horizons: what a stop rule is allowed to see")
ax.legend(fontsize=9);
```

Read the printout as three signal-to-noise ratios. At $w = 1$ the standardised
contrast has unit spread — which is what the $\sqrt{2/w}$ scaling is built to
produce — and a mean indistinguishable from zero: a single increment says
nothing about the trend.

At the two epoch-aligned horizons the spread collapses far below what
independent noise would predict — and the control row explains why. The
shuffle is done once on disk and replayed in the same order every epoch, so a
window of exactly one or two passes averages over the *same rows* every time
and the batch-composition noise, which dominates the per-step view, cancels.
Widen the window to one and a half passes and it does not: the control sits
at roughly twenty times the spread of its neighbours on either side, even
though it is *wider* than the one-epoch window. Nothing but the alignment
changed.

What survives the cancellation is Monte-Carlo noise and the parameter drift
itself, and against that much smaller yardstick the drift becomes visible —
the ratio in the last column rises with the horizon. Two lessons sit on top of
each other here. The horizon is not a smoothing preference, it decides whether
there is anything to see at all; and with a fixed replay order the useful
horizons are the ones commensurate with an epoch, which is a property of how
the data was shuffled, not of the optimizer.

:::{admonition} Why a naive rule does not merely stay silent — it fires
:class: warning
It is tempting to accumulate per-step evidence with a one-sided cumulative-sum
statistic {cite:p}`page1954continuous`, $S \leftarrow \max(0,\ S + (\kappa -
\max(z, 0)))$, stopping when $S$ exceeds a threshold. The rectification is the problem. Under symmetric noise
$\mathbb{E}[\max(z,0)]$ is a fraction of a standard deviation, so any
allowance $\kappa$ above that value makes $S$ climb at a constant rate *on
noise alone* and cross any fixed threshold after a fixed number of steps —
whatever the data is doing. A rule built that way announces convergence on
schedule rather than on evidence. The fix is not a better threshold but a
wider horizon, plus a second yardstick that asks whether the remaining
improvement is negligible relative to the reduction already achieved. That
design is under review as
[pymc-extras#733](https://github.com/pymc-devs/pymc-extras/pull/733); this
notebook shows the observation problem it exists to solve rather than shipping
a second copy of it.
:::

How much would an online rule be worth here? The retrospective answer is a
benchmark you can only compute afterwards — which is exactly why the online
version is needed:

```{code-cell} ipython3
# t99: the first step at which a trailing average of width one epoch has covered
# 99% of stage 2's total smoothed reduction. Retrospective by construction.
kernel = np.ones(steps_per_epoch) / steps_per_epoch
smoothed = np.convolve(stage2, kernel, mode="valid")
total_drop = smoothed[0] - smoothed.min()
# "valid" convolution starts at the first full window, so index j of `smoothed`
# is the trailing average ending at stage-2 step j + steps_per_epoch - 1.
t99 = int(np.argmax(smoothed <= smoothed.min() + 0.01 * total_drop)) + steps_per_epoch - 1
print(
    f"stage 2 smoothed reduction: {total_drop:,.2f} nats; "
    f"99% of it reached by step {t99:,} of {len(stage2):,} "
    f"({t99 / len(stage2):.0%} of the stage-2 budget)"
)
```

Two honest qualifications about any stop, this one included. What such a rule
detects is a *loss plateau* — a necessary signal, not a proof that the
posterior has converged; the recovery and predictive checks above are the kind
of independent evidence a stop should be paired with. And a plateau in a noisy
loss can only ever be established relative to a horizon: at any finite step, an
improvement small enough is indistinguishable from none.

## Where the pieces live

The streaming stack this notebook exercises is being contributed to
pymc-extras. The map is by role; each pull request's current state is on
GitHub rather than frozen into this page.

| Component | Role | Form in this notebook |
| --- | --- | --- |
| [`DataLoader` / `parquet_source`](https://github.com/pymc-devs/pymc-extras/pull/698) | Turns an out-of-core source into minibatches and owns `total_size` | Imported and used directly |
| [`Trainer`](https://github.com/pymc-devs/pymc-extras/pull/710) | Wraps the data-advance lifecycle around `pm.fit`, while [#635](https://github.com/pymc-devs/pymc-extras/pull/635) develops the longer-term ADVI API | The `StreamAdvance` adapter above, owned by the notebook |
| [`CheckLossConvergence`](https://github.com/pymc-devs/pymc-extras/pull/733) | Loss-based stopping on growing block horizons with two yardsticks | The observation problem it solves is shown; the implementation is not copied here |
| [streaming Pathfinder](https://github.com/pymc-devs/pymc-extras/pull/722) | Quasi-Newton alternative returning a Gaussian proposal from a short optimizer run, with `pareto_k` as its own veto | Not run on this target — see below |

+++

### Choosing a path: why the fast one is not taken here

ADVI is not the only way to fit a model on a stream. Streaming Pathfinder runs
a short quasi-Newton trajectory on minibatch gradients — a few hundred steps
rather than thousands — and returns a Gaussian *proposal*, importance-corrected
against the exact full-data log-density, with the Pareto-$k$ diagnostic as its
own veto. It is the right tool when an approximately-placed starting point is
what you need, for example as an initialisation for MCMC.

It is not run on this model, and the reason is a decision taken *before*
fitting rather than a result:

```{code-cell} ipython3
from pymc.blocking import DictToArrayBijection

n_free = DictToArrayBijection.map(model.initial_point()).data.size
print(f"free parameters in the unconstrained space: {n_free}")
```

pymc-extras#722 states a measured operating range: non-hierarchical targets of
at most a few tens of parameters. This target is hierarchical and an order of
magnitude larger than that, so applying it here would be using a method
outside the range its author documented. That is an applicability decision,
not a prediction about what its diagnostic would report. The ADVI results
above stand or fall on their own recovery and predictive checks, independently
of this choice.

+++

## Authors

* Authored by [Yicheng Yang](https://github.com/YichengYang-Ethan) in August
  2026 for the Google Summer of Code project *Streaming Variational Inference
  for Large Datasets* (PyMC / NumFOCUS).

+++

## References

:::{bibliography}
:filter: docname in docnames
:::

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,pymc_extras
```

:::{include} ../page_footer.md
:::
