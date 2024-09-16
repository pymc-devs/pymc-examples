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

+++ {"id": "XShKDkNir2PX"}

(lkj_prior_for_multivariate_normal)=
# LKJ Cholesky Covariance Priors for Multivariate Normal Models

+++ {"id": "QxSKBbjKr2PZ"}

While the [inverse-Wishart distribution](https://en.wikipedia.org/wiki/Inverse-Wishart_distribution) is the conjugate prior for the covariance matrix of a multivariate normal distribution, it is [not very well-suited](https://github.com/pymc-devs/pymc3/issues/538#issuecomment-94153586) to modern Bayesian computational methods.  For this reason, the [LKJ prior](http://www.sciencedirect.com/science/article/pii/S0047259X09000876) is recommended when modeling the covariance matrix of a multivariate normal distribution.

To illustrate modelling covariance with the LKJ distribution, we first generate a two-dimensional normally-distributed sample data set.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 17Thh2DHr2Pa
outputId: 90631275-86c9-4f4a-f81a-22465d0c8b8c
---
import arviz as az
import numpy as np
import pymc as pm

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

print(f"Running on PyMC v{pm.__version__}")
```

```{code-cell} ipython3
:id: Sq6K4Ie4r2Pc

%config InlineBackend.figure_format = 'retina'
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: eA491vJMr2Pc
outputId: 30ea38db-0767-4e89-eb09-68927878018e
---
N = 10000

mu_actual = np.array([1.0, -2.0])
sigmas_actual = np.array([0.7, 1.5])
Rho_actual = np.matrix([[1.0, -0.4], [-0.4, 1.0]])

Sigma_actual = np.diag(sigmas_actual) * Rho_actual * np.diag(sigmas_actual)

x = rng.multivariate_normal(mu_actual, Sigma_actual, size=N)
Sigma_actual
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 628
id: ZmFDGQ8Jr2Pd
outputId: 03ba3248-370c-4ff9-8626-ba601423b9c1
---
var, U = np.linalg.eig(Sigma_actual)
angle = 180.0 / np.pi * np.arccos(np.abs(U[0, 0]))

fig, ax = plt.subplots(figsize=(8, 6))

e = Ellipse(mu_actual, 2 * np.sqrt(5.991 * var[0]), 2 * np.sqrt(5.991 * var[1]), angle=angle)
e.set_alpha(0.5)
e.set_facecolor("C0")
e.set_zorder(10)
ax.add_artist(e)

ax.scatter(x[:, 0], x[:, 1], c="k", alpha=0.05, zorder=11)
ax.set_xlabel("y")
ax.set_ylabel("z")

rect = plt.Rectangle((0, 0), 1, 1, fc="C0", alpha=0.5)
ax.legend([rect], ["95% density region"], loc=2);
```

+++ {"id": "d6320GCir2Pd"}

The sampling distribution for the multivariate normal model is $\mathbf{x} \sim N(\mu, \Sigma)$, where $\Sigma$ is the covariance matrix of the sampling distribution, with $\Sigma_{ij} = \textrm{Cov}(x_i, x_j)$. The density of this distribution is

$$f(\mathbf{x}\ |\ \mu, \Sigma^{-1}) = (2 \pi)^{-\frac{k}{2}} |\Sigma|^{-\frac{1}{2}} \exp\left(-\frac{1}{2} (\mathbf{x} - \mu)^{\top} \Sigma^{-1} (\mathbf{x} - \mu)\right).$$

The LKJ distribution provides a prior on the correlation matrix, $\mathbf{C} = \textrm{Corr}(x_i, x_j)$, which, combined with priors on the standard deviations of each component, [induces](http://www3.stat.sinica.edu.tw/statistica/oldpdf/A10n416.pdf) a prior on the covariance matrix, $\Sigma$. Since inverting $\Sigma$ is numerically unstable and inefficient, it is computationally advantageous to use the [Cholesky decompositon](https://en.wikipedia.org/wiki/Cholesky_decomposition) of $\Sigma$, $\Sigma = \mathbf{L} \mathbf{L}^{\top}$, where $\mathbf{L}$ is a lower-triangular matrix. This decompositon allows computation of the term $(\mathbf{x} - \mu)^{\top} \Sigma^{-1} (\mathbf{x} - \mu)$ using back-substitution, which is more numerically stable and efficient than direct matrix inversion.

PyMC supports LKJ priors for the Cholesky decomposition of the covariance matrix via the {class}`pymc.LKJCholeskyCov` distribution. This distribution has parameters `n` and `sd_dist`, which are the dimension of the observations, $\mathbf{x}$, and the PyMC distribution of the component standard deviations, respectively. It also has a hyperparamter `eta`, which controls the amount of correlation between components of $\mathbf{x}$. The LKJ distribution has the density $f(\mathbf{C}\ |\ \eta) \propto |\mathbf{C}|^{\eta - 1}$, so $\eta = 1$ leads to a uniform distribution on correlation matrices, while the magnitude of correlations between components decreases as $\eta \to \infty$.

In this example, we model the standard deviations with $\textrm{Exponential}(1.0)$ priors, and the correlation matrix as $\mathbf{C} \sim \textrm{LKJ}(\eta = 2)$.

```{code-cell} ipython3
:id: 7GcM6oENr2Pe

with pm.Model() as m:
    packed_L = pm.LKJCholeskyCov(
        "packed_L", n=2, eta=2.0, sd_dist=pm.Exponential.dist(1.0, shape=2), compute_corr=False
    )
```

+++ {"id": "6Cscu-CRr2Pe"}

Since the Cholesky decompositon of $\Sigma$ is lower triangular, `LKJCholeskyCov` only stores the diagonal and sub-diagonal entries, for efficiency:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: JMWeTjDjr2Pe
outputId: e4f767a3-c1d7-4016-a3cf-91089c925bdb
---
packed_L.eval()
```

+++ {"id": "59FtijDir2Pe"}

We use {func}`expand_packed_triangular <pymc.expand_packed_triangular>` to transform this vector into the lower triangular matrix $\mathbf{L}$, which appears in the Cholesky decomposition $\Sigma = \mathbf{L} \mathbf{L}^{\top}$.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: YxBbFyUxr2Pf
outputId: bd37c630-98dd-437b-bb13-89281aeccc44
---
with m:
    L = pm.expand_packed_triangular(2, packed_L)
    Sigma = L.dot(L.T)

L.eval().shape
```

+++ {"id": "SwdNd_0Jr2Pf"}

Often however, you'll be interested in the posterior distribution of the correlations matrix and of the standard deviations, not in the posterior Cholesky covariance matrix *per se*. Why? Because the correlations and standard deviations are easier to interpret and often have a scientific meaning in the model. As of PyMC v4, the `compute_corr` argument is set to `True` by default, which returns a tuple consisting of the Cholesky decomposition, the correlations matrix, and the standard deviations.

```{code-cell} ipython3
:id: ac3eQeMJr2Pf

coords = {"axis": ["y", "z"], "axis_bis": ["y", "z"], "obs_id": np.arange(N)}
with pm.Model(coords=coords) as model:
    chol, corr, stds = pm.LKJCholeskyCov(
        "chol", n=2, eta=2.0, sd_dist=pm.Exponential.dist(1.0, shape=2)
    )
    cov = pm.Deterministic("cov", chol.dot(chol.T), dims=("axis", "axis_bis"))
```

+++ {"id": "cpEupNzWr2Pg"}

To complete our model, we place independent, weakly regularizing priors, $N(0, 1.5),$ on $\mu$:

```{code-cell} ipython3
:id: iTI4uiBdr2Pg

with model:
    mu = pm.Normal("mu", 0.0, sigma=1.5, dims="axis")
    obs = pm.MvNormal("obs", mu, chol=chol, observed=x, dims=("obs_id", "axis"))
```

+++ {"id": "QOCi1RKvr2Ph"}

We sample from this model using NUTS and give the trace to {ref}`arviz` for summarization:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 608
id: vBPIQDWrr2Ph
outputId: f039bfb8-1acf-42cb-b054-bc2c97697f96
---
with model:
    trace = pm.sample(
        random_seed=rng,
        idata_kwargs={"dims": {"chol_stds": ["axis"], "chol_corr": ["axis", "axis_bis"]}},
    )
az.summary(trace, var_names="~chol", round_to=2)
```

+++ {"id": "X8ucBpcRr2Ph"}

Sampling went smoothly: no divergences and good r-hats (except for the diagonal elements of the correlation matrix - however, these are not a concern, because, they should be equal to 1 for each sample for each chain and the variance of a constant value isn't defined. If one of the diagonal elements has `r_hat` defined, it's likely due to tiny numerical errors). 
 
You can also see that the sampler recovered the true means, correlations and standard deviations. As often, that will be clearer in a graph:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 228
id: dgOKiSLdr2Pi
outputId: a29bde4b-c4dc-49f4-e65d-c3365c1933e1
---
az.plot_trace(
    trace,
    var_names="chol_corr",
    coords={"axis": "y", "axis_bis": "z"},
    lines=[("chol_corr", {}, Rho_actual[0, 1])],
);
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 628
id: dtBWyd5Jr2Pi
outputId: 94ee6945-a564-487a-e447-3c447057f0bf
---
az.plot_trace(
    trace,
    var_names=["~chol", "~chol_corr"],
    compact=True,
    lines=[
        ("mu", {}, mu_actual),
        ("cov", {}, Sigma_actual),
        ("chol_stds", {}, sigmas_actual),
    ],
);
```

+++ {"id": "NnLWJyCMr2Pi"}

The posterior expected values are very close to the true value of each component! How close exactly? Let's compute the percentage of closeness for $\mu$ and $\Sigma$:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: yDlyVSizr2Pj
outputId: 69c22c57-db27-4f43-ab94-7b88480a21f9
---
mu_post = trace.posterior["mu"].mean(("chain", "draw")).values
(1 - mu_post / mu_actual).round(2)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: rFF947Grr2Pj
outputId: 398332a0-a142-4ad0-dadf-bde13ef2b00b
---
Sigma_post = trace.posterior["cov"].mean(("chain", "draw")).values
(1 - Sigma_post / Sigma_actual).round(2)
```

+++ {"id": "DMDwKtp0r2Pj"}

So the posterior means are within 1% of the true values of $\mu$ and $\Sigma$.

Now let's replicate the plot we did at the beginning, but let's overlay the posterior distribution on top of the true distribution -- you'll see there is excellent visual agreement between both:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 628
id: _dwHYuj1r2Pj
outputId: 9b53b875-af25-4f79-876f-a02e72bba5a9
---
var_post, U_post = np.linalg.eig(Sigma_post)
angle_post = 180.0 / np.pi * np.arccos(np.abs(U_post[0, 0]))

fig, ax = plt.subplots(figsize=(8, 6))

e = Ellipse(
    mu_actual,
    2 * np.sqrt(5.991 * var[0]),
    2 * np.sqrt(5.991 * var[1]),
    angle=angle,
    linewidth=3,
    linestyle="dashed",
)
e.set_edgecolor("C0")
e.set_zorder(11)
e.set_fill(False)
ax.add_artist(e)

e_post = Ellipse(
    mu_post,
    2 * np.sqrt(5.991 * var_post[0]),
    2 * np.sqrt(5.991 * var_post[1]),
    angle=angle_post,
    linewidth=3,
)
e_post.set_edgecolor("C1")
e_post.set_zorder(10)
e_post.set_fill(False)
ax.add_artist(e_post)

ax.scatter(x[:, 0], x[:, 1], c="k", alpha=0.05, zorder=11)
ax.set_xlabel("y")
ax.set_ylabel("z")

line = Line2D([], [], color="C0", linestyle="dashed", label="True 95% density region")
line_post = Line2D([], [], color="C1", label="Estimated 95% density region")
ax.legend(
    handles=[line, line_post],
    loc=2,
);
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: kJCfuzGtr2Pq
outputId: da547b05-d812-4959-aff6-cf4a12faca15
---
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,xarray
```
