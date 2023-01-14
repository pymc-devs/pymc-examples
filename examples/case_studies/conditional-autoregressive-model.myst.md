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

```{code-cell} ipython3
import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy.stats as stats
import theano
import theano.tensor as tt

from pymc3.distributions import continuous, distribution
from theano import scan, shared
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")
floatX = "float32"
```

# Conditional Autoregressive (CAR) model
A walkthrough of implementing a Conditional Autoregressive (CAR) model in `PyMC3`, with `WinBUGS`/`PyMC2` and `Stan` code as references.

+++

As a probabilistic language, there are some fundamental differences between `PyMC3` and other alternatives such as `WinBUGS`, `JAGS`, and `Stan`. In this notebook, I will summarise some heuristics and intuition I got over the past two years using `PyMC3`. I will outline some thinking in how I approach a modelling problem using `PyMC3`, and how thinking in linear algebra solves most of the programming problems. I hope this notebook will shed some light onto the design and features of `PyMC3`, and similar languages that are built on linear algebra packages with a static world view (e.g., Edward, which is based on Tensorflow).  


For more resources comparing between PyMC3 codes and other probabilistic languages:
* [PyMC3 port of "Doing Bayesian Data Analysis" - PyMC3 vs WinBUGS/JAGS/Stan](https://github.com/aloctavodia/Doing_bayesian_data_analysis)
* [PyMC3 port of "Bayesian Cognitive Modeling" - PyMC3 vs WinBUGS/JAGS/Stan](https://github.com/junpenglao/Bayesian-Cognitive-Modeling-in-Pymc3)
* [PyMC3 port of "Statistical Rethinking" - PyMC3 vs Stan](https://github.com/aloctavodia/Statistical-Rethinking-with-Python-and-PyMC3)

+++

## Background information
Suppose we want to implement a [Conditional Autoregressive (CAR) model](http://www.statsref.com/HTML/index.html?car_models.html) with examples in [WinBUGS/PyMC2](http://glau.ca/?p=340) and [Stan](http://mc-stan.org/documentation/case-studies/mbjoseph-CARStan.html).  
For the sake of brevity, I will not go into the details of the CAR model. The essential idea is autocorrelation, which is informally "correlation with itself". In a CAR model, the probability of values estimated at any given location $y_i$ are conditional on some neighboring values $y_j, _{j \neq i}$ (in another word, correlated/covariated with these values):  

$$y_i \mid y_j, j \neq i \sim \mathcal{N}(\alpha \sum_{j = 1}^n b_{ij} y_j, \sigma_i^{2})$$

where $\sigma_i^{2}$ is a spatially varying covariance parameter, and $b_{ii} = 0$. 

Here we will demonstrate the implementation of a CAR model using a canonical example: the lip cancer risk data in Scotland between 1975 and 1980. The original data is from (Kemp et al. 1985). This dataset includes observed lip cancer case counts at 56 spatial units in Scotland, with the expected number of cases as intercept, and an area-specific continuous variable coded for the proportion of the population employed in agriculture, fishing, or forestry (AFF). We want to model how lip cancer rates (`O` below) relate to AFF (`aff` below), as exposure to sunlight is a risk factor.

$$O_i \sim \mathcal{Poisson}(\text{exp}(\beta_0 + \beta_1*aff + \phi_i + \log(\text{E}_i)))$$
$$\phi_i \mid \phi_j, j \neq i \sim \mathcal{N}(\alpha \sum_{j = 1}^n b_{ij} \phi_j, \sigma_i^{2})$$

Setting up the data:

```{code-cell} ipython3
# Read the data from file containing columns: NAME, CANCER, CEXP, AFF, ADJ, WEIGHTS
df_scot_cancer = pd.read_csv(pm.get_data("scotland_lips_cancer.csv"))

# name of the counties
county = df_scot_cancer["NAME"].values

# observed
O = df_scot_cancer["CANCER"].values
N = len(O)

# expected (E) rates, based on the age of the local population
E = df_scot_cancer["CEXP"].values
logE = np.log(E)

# proportion of the population engaged in agriculture, forestry, or fishing (AFF)
aff = df_scot_cancer["AFF"].values / 10.0

# Spatial adjacency information: column (ADJ) contains list entries which are preprocessed to obtain adj as list of lists
adj = (
    df_scot_cancer["ADJ"].apply(lambda x: [int(val) for val in x.strip("][").split(",")]).to_list()
)

# Change to Python indexing (i.e. -1)
for i in range(len(adj)):
    for j in range(len(adj[i])):
        adj[i][j] = adj[i][j] - 1

# spatial weight: column (WEIGHTS) contains list entries which are preprocessed to obtain weights as list of lists
weights = (
    df_scot_cancer["WEIGHTS"]
    .apply(lambda x: [int(val) for val in x.strip("][").split(",")])
    .to_list()
)
Wplus = np.asarray([sum(w) for w in weights])
```

## A WinBUGS/PyMC2 implementation

The classical `WinBUGS` implementation (more information [here](http://glau.ca/?p=340)):

```stan
model
{
   for (i in 1 : regions) {
      O[i] ~ dpois(mu[i])
      log(mu[i]) <- log(E[i]) + beta0 + beta1*aff[i]/10 + phi[i] + theta[i]
      theta[i] ~ dnorm(0.0,tau.h)
   }
   phi[1:regions] ~ car.normal(adj[], weights[], Wplus[], tau.c)

   beta0 ~ dnorm(0.0, 1.0E-5)  # vague prior on grand intercept
   beta1 ~ dnorm(0.0, 1.0E-5)  # vague prior on covariate effect

   tau.h ~ dgamma(3.2761, 1.81)    
   tau.c ~ dgamma(1.0, 1.0)  

   sd.h <- sd(theta[]) # marginal SD of heterogeneity effects
   sd.c <- sd(phi[])   # marginal SD of clustering (spatial) effects

   alpha <- sd.c / (sd.h + sd.c)
}
```

The main challenge to porting this model to `PyMC3` is the `car.normal` function in `WinBUGS`. It is a likelihood function that conditions each realization on some neighbour realization (a smoothed property). In `PyMC2`, it could be implemented as a [custom likelihood function (a `@stochastic` node) `mu_phi`](http://glau.ca/?p=340):  

```python
@stochastic
def mu_phi(tau=tau_c, value=np.zeros(N)):
    # Calculate mu based on average of neighbours 
    mu = np.array([ sum(weights[i]*value[adj[i]])/Wplus[i] for i in xrange(N)])
    # Scale precision to the number of neighbours
    taux = tau*Wplus
    return normal_like(value,mu,taux)
```

We can just define `mu_phi` similarly and wrap it in a `pymc3.DensityDist`, however, doing so usually results in a very slow model (both in compiling and sampling). In general, porting pymc2 code into pymc3 (or even generally porting `WinBUGS`, `JAGS`, or `Stan` code into `PyMC3`) that use a `for` loops tend to perform poorly in `theano`, the backend of `PyMC3`.  

The underlying mechanism in `PyMC3` is very different compared to `PyMC2`, using `for` loops to generate RV or stacking multiple RV with arguments such as `[pm.Binomial('obs%'%i, p[i], n) for i in range(K)]` generate unnecessary large number of nodes in `theano` graph, which then slows down compilation appreciably.  

The easiest way is to move the loop out of `pm.Model`. And usually is not difficult to do. For example, in `Stan` you can have a `transformed data{}` block; in `PyMC3` you just need to compute it before defining your Model.

If it is absolutely necessary to use a `for` loop, you can use a theano loop (i.e., `theano.scan`), which you can find some introduction on the [theano website](http://deeplearning.net/software/theano/tutorial/loop.html) and see a usecase in PyMC3 [timeseries distribution](https://github.com/pymc-devs/pymc3/blob/master/pymc3/distributions/timeseries.py#L125-L130).

+++

## PyMC3 implementation using `theano.scan`

So lets try to implement the CAR model using `theano.scan`. First we create a `theano` function with `theano.scan` and check if it really works by comparing its result to the for-loop.

```{code-cell} ipython3
value = np.asarray(
    np.random.randn(
        N,
    ),
    dtype=theano.config.floatX,
)

maxwz = max([sum(w) for w in weights])
N = len(weights)
wmat = np.zeros((N, maxwz))
amat = np.zeros((N, maxwz), dtype="int32")
for i, w in enumerate(weights):
    wmat[i, np.arange(len(w))] = w
    amat[i, np.arange(len(w))] = adj[i]

# defining the tensor variables
x = tt.vector("x")
x.tag.test_value = value
w = tt.matrix("w")
# provide Theano with a default test-value
w.tag.test_value = wmat
a = tt.matrix("a", dtype="int32")
a.tag.test_value = amat


def get_mu(w, a):
    a1 = tt.cast(a, "int32")
    return tt.sum(w * x[a1]) / tt.sum(w)


results, _ = theano.scan(fn=get_mu, sequences=[w, a])
compute_elementwise = theano.function(inputs=[x, w, a], outputs=results)

print(compute_elementwise(value, wmat, amat))


def mu_phi(value):
    N = len(weights)
    # Calculate mu based on average of neighbours
    mu = np.array([np.sum(weights[i] * value[adj[i]]) / Wplus[i] for i in range(N)])
    return mu


print(mu_phi(value))
```

Since it produces the same result as the original for-loop, we will wrap it as a new distribution with a log-likelihood function in `PyMC3`.

```{code-cell} ipython3
class CAR(distribution.Continuous):
    """
    Conditional Autoregressive (CAR) distribution

    Parameters
    ----------
    a : list of adjacency information
    w : list of weight information
    tau : precision at each location
    """

    def __init__(self, w, a, tau, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = a = tt.as_tensor_variable(a)
        self.w = w = tt.as_tensor_variable(w)
        self.tau = tau * tt.sum(w, axis=1)
        self.mode = 0.0

    def get_mu(self, x):
        def weight_mu(w, a):
            a1 = tt.cast(a, "int32")
            return tt.sum(w * x[a1]) / tt.sum(w)

        mu_w, _ = scan(fn=weight_mu, sequences=[self.w, self.a])

        return mu_w

    def logp(self, x):
        mu_w = self.get_mu(x)
        tau = self.tau
        return tt.sum(continuous.Normal.dist(mu=mu_w, tau=tau).logp(x))
```

We then use it in our `PyMC3` version of the CAR model:

```{code-cell} ipython3
with pm.Model() as model1:
    # Vague prior on intercept
    beta0 = pm.Normal("beta0", mu=0.0, tau=1.0e-5)
    # Vague prior on covariate effect
    beta1 = pm.Normal("beta1", mu=0.0, tau=1.0e-5)

    # Random effects (hierarchial) prior
    tau_h = pm.Gamma("tau_h", alpha=3.2761, beta=1.81)
    # Spatial clustering prior
    tau_c = pm.Gamma("tau_c", alpha=1.0, beta=1.0)

    # Regional random effects
    theta = pm.Normal("theta", mu=0.0, tau=tau_h, shape=N)
    mu_phi = CAR("mu_phi", w=wmat, a=amat, tau=tau_c, shape=N)

    # Zero-centre phi
    phi = pm.Deterministic("phi", mu_phi - tt.mean(mu_phi))

    # Mean model
    mu = pm.Deterministic("mu", tt.exp(logE + beta0 + beta1 * aff + theta + phi))

    # Likelihood
    Yi = pm.Poisson("Yi", mu=mu, observed=O)

    # Marginal SD of heterogeniety effects
    sd_h = pm.Deterministic("sd_h", tt.std(theta))
    # Marginal SD of clustering (spatial) effects
    sd_c = pm.Deterministic("sd_c", tt.std(phi))
    # Proportion sptial variance
    alpha = pm.Deterministic("alpha", sd_c / (sd_h + sd_c))

    infdata1 = pm.sample(
        1000,
        tune=500,
        cores=4,
        init="advi",
        target_accept=0.9,
        max_treedepth=15,
        return_inferencedata=True,
    )
```

Note: there are some hidden problems with the model, some regions of the parameter space are quite difficult to sample from. Here I am using ADVI as initialization, which gives a smaller variance of the mass matrix. It keeps the sampler around the mode.

```{code-cell} ipython3
az.plot_trace(infdata1, var_names=["alpha", "sd_h", "sd_c"]);
```

We also got a lot of Rhat warning, that's because the Zero-centre phi introduce unidentification to the model:

```{code-cell} ipython3
summary1 = az.summary(infdata1)
summary1[summary1["r_hat"] > 1.05]
```

```{code-cell} ipython3
az.plot_forest(
    infdata1,
    kind="ridgeplot",
    var_names=["phi"],
    combined=False,
    ridgeplot_overlap=3,
    ridgeplot_alpha=0.25,
    colors="white",
    figsize=(9, 7),
);
```

```{code-cell} ipython3
az.plot_posterior(infdata1, var_names=["alpha"]);
```

`theano.scan` is much faster than using a python for loop, but it is still quite slow. One approach for improving it is to use linear algebra. That is, we should try to find a way to use matrix multiplication instead of looping (if you have experience in using MATLAB, it is the same philosophy). In our case, we can totally do that.  

For a similar problem, you can also have a look of [my port of Lee and Wagenmakers' book](https://github.com/junpenglao/Bayesian-Cognitive-Modeling-in-Pymc3). For example, in Chapter 19, the Stan code use [a for loop to generate the likelihood function](https://github.com/stan-dev/example-models/blob/master/Bayesian_Cognitive_Modeling/CaseStudies/NumberConcepts/NumberConcept_1_Stan.R#L28-L59), and I [generate the matrix outside and use matrix multiplication etc](http://nbviewer.jupyter.org/github/junpenglao/Bayesian-Cognitive-Modeling-in-Pymc3/blob/master/CaseStudies/NumberConceptDevelopment.ipynb#19.1-Knower-level-model-for-Give-N) to archive the same purpose.

+++

## PyMC3 implementation using matrix "trick"

Again, we try on some simulated data to make sure the implementation is correct.

```{code-cell} ipython3
maxwz = max([sum(w) for w in weights])
N = len(weights)
wmat2 = np.zeros((N, N))
amat2 = np.zeros((N, N), dtype="int32")
for i, a in enumerate(adj):
    amat2[i, a] = 1
    wmat2[i, a] = weights[i]

value = np.asarray(
    np.random.randn(
        N,
    ),
    dtype=theano.config.floatX,
)

print(np.sum(value * amat2, axis=1) / np.sum(wmat2, axis=1))


def mu_phi(value):
    N = len(weights)
    # Calculate mu based on average of neighbours
    mu = np.array([np.sum(weights[i] * value[adj[i]]) / Wplus[i] for i in range(N)])
    return mu


print(mu_phi(value))
```

Now create a new CAR distribution with the matrix multiplication instead of `theano.scan` to get the `mu`

```{code-cell} ipython3
class CAR2(distribution.Continuous):
    """
    Conditional Autoregressive (CAR) distribution

    Parameters
    ----------
    a : adjacency matrix
    w : weight matrix
    tau : precision at each location
    """

    def __init__(self, w, a, tau, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = a = tt.as_tensor_variable(a)
        self.w = w = tt.as_tensor_variable(w)
        self.tau = tau * tt.sum(w, axis=1)
        self.mode = 0.0

    def logp(self, x):
        tau = self.tau
        w = self.w
        a = self.a

        mu_w = tt.sum(x * a, axis=1) / tt.sum(w, axis=1)
        return tt.sum(continuous.Normal.dist(mu=mu_w, tau=tau).logp(x))
```

```{code-cell} ipython3
with pm.Model() as model2:
    # Vague prior on intercept
    beta0 = pm.Normal("beta0", mu=0.0, tau=1.0e-5)
    # Vague prior on covariate effect
    beta1 = pm.Normal("beta1", mu=0.0, tau=1.0e-5)

    # Random effects (hierarchial) prior
    tau_h = pm.Gamma("tau_h", alpha=3.2761, beta=1.81)
    # Spatial clustering prior
    tau_c = pm.Gamma("tau_c", alpha=1.0, beta=1.0)

    # Regional random effects
    theta = pm.Normal("theta", mu=0.0, tau=tau_h, shape=N)
    mu_phi = CAR2("mu_phi", w=wmat2, a=amat2, tau=tau_c, shape=N)

    # Zero-centre phi
    phi = pm.Deterministic("phi", mu_phi - tt.mean(mu_phi))

    # Mean model
    mu = pm.Deterministic("mu", tt.exp(logE + beta0 + beta1 * aff + theta + phi))

    # Likelihood
    Yi = pm.Poisson("Yi", mu=mu, observed=O)

    # Marginal SD of heterogeniety effects
    sd_h = pm.Deterministic("sd_h", tt.std(theta))
    # Marginal SD of clustering (spatial) effects
    sd_c = pm.Deterministic("sd_c", tt.std(phi))
    # Proportion sptial variance
    alpha = pm.Deterministic("alpha", sd_c / (sd_h + sd_c))

    infdata2 = pm.sample(
        1000,
        tune=500,
        cores=4,
        init="advi",
        target_accept=0.9,
        max_treedepth=15,
        return_inferencedata=True,
    )
```

**As you can see, it is appreciably faster using matrix multiplication.**

```{code-cell} ipython3
summary2 = az.summary(infdata2)
summary2[summary2["r_hat"] > 1.05]
```

```{code-cell} ipython3
az.plot_forest(
    infdata2,
    kind="ridgeplot",
    var_names=["phi"],
    combined=False,
    ridgeplot_overlap=3,
    ridgeplot_alpha=0.25,
    colors="white",
    figsize=(9, 7),
);
```

```{code-cell} ipython3
az.plot_trace(infdata2, var_names=["alpha", "sd_h", "sd_c"]);
```

```{code-cell} ipython3
az.plot_posterior(infdata2, var_names=["alpha"]);
```

## PyMC3 implementation using Matrix multiplication

There are almost always multiple ways to formulate a particular model. Some approaches work better than the others under different contexts (size of your dataset, properties of the sampler, etc). 

In this case, we can express the CAR prior as:  

$$\phi \sim \mathcal{N}(0, [D_\tau (I - \alpha B)]^{-1}).$$

You can find more details in the original [Stan case study](http://mc-stan.org/documentation/case-studies/mbjoseph-CARStan.html). You might come across similar constructs in Gaussian Process, which result in a zero-mean Gaussian distribution conditioned on a covariance function.

In the `Stan` Code, matrix D is generated in the model using a `transformed data{}` block:
```
transformed data{
  vector[n] zeros;
  matrix<lower = 0>[n, n] D;
  {
    vector[n] W_rowsums;
    for (i in 1:n) {
      W_rowsums[i] = sum(W[i, ]);
    }
    D = diag_matrix(W_rowsums);
  }
  zeros = rep_vector(0, n);
}
```
We can generate the same matrix quite easily:

```{code-cell} ipython3
X = np.hstack((np.ones((N, 1)), stats.zscore(aff, ddof=1)[:, None]))
W = wmat2
D = np.diag(W.sum(axis=1))
log_offset = logE[:, None]
```

Then in the `Stan` model:
```stan
model {
  phi ~ multi_normal_prec(zeros, tau * (D - alpha * W));
  ...
} 
```
since the precision matrix just generated by some matrix multiplication, we can do just that in `PyMC3`:

```{code-cell} ipython3
with pm.Model() as model3:
    # Vague prior on intercept and effect
    beta = pm.Normal("beta", mu=0.0, tau=1.0, shape=(2, 1))

    # Priors for spatial random effects
    tau = pm.Gamma("tau", alpha=2.0, beta=2.0)
    alpha = pm.Uniform("alpha", lower=0, upper=1)
    phi = pm.MvNormal("phi", mu=0, tau=tau * (D - alpha * W), shape=(1, N))

    # Mean model
    mu = pm.Deterministic("mu", tt.exp(tt.dot(X, beta) + phi.T + log_offset))

    # Likelihood
    Yi = pm.Poisson("Yi", mu=mu.ravel(), observed=O)

    infdata3 = pm.sample(1000, tune=2000, cores=4, target_accept=0.85, return_inferencedata=True)
```

```{code-cell} ipython3
az.plot_trace(infdata3, var_names=["alpha", "beta", "tau"]);
```

```{code-cell} ipython3
az.plot_posterior(infdata3, var_names=["alpha"]);
```

Notice that since the model parameterization is different than in the `WinBUGS` model, the `alpha` can't be interpreted in the same way.

+++

## PyMC3 implementation using Sparse Matrix

Note that in the node $\phi \sim \mathcal{N}(0, [D_\tau (I - \alpha B)]^{-1})$, we are computing the log-likelihood for a multivariate Gaussian distribution, which might not scale well in high-dimensions. We can take advantage of the fact that the covariance matrix here $[D_\tau (I - \alpha B)]^{-1}$ is **sparse**, and there are faster ways to compute its log-likelihood. 

For example, a more efficient sparse representation of the CAR in `Stan`:
```stan
functions {
  /**
  * Return the log probability of a proper conditional autoregressive (CAR) prior 
  * with a sparse representation for the adjacency matrix
  *
  * @param phi Vector containing the parameters with a CAR prior
  * @param tau Precision parameter for the CAR prior (real)
  * @param alpha Dependence (usually spatial) parameter for the CAR prior (real)
  * @param W_sparse Sparse representation of adjacency matrix (int array)
  * @param n Length of phi (int)
  * @param W_n Number of adjacent pairs (int)
  * @param D_sparse Number of neighbors for each location (vector)
  * @param lambda Eigenvalues of D^{-1/2}*W*D^{-1/2} (vector)
  *
  * @return Log probability density of CAR prior up to additive constant
  */
  real sparse_car_lpdf(vector phi, real tau, real alpha, 
    int[,] W_sparse, vector D_sparse, vector lambda, int n, int W_n) {
      row_vector[n] phit_D; // phi' * D
      row_vector[n] phit_W; // phi' * W
      vector[n] ldet_terms;
    
      phit_D = (phi .* D_sparse)';
      phit_W = rep_row_vector(0, n);
      for (i in 1:W_n) {
        phit_W[W_sparse[i, 1]] = phit_W[W_sparse[i, 1]] + phi[W_sparse[i, 2]];
        phit_W[W_sparse[i, 2]] = phit_W[W_sparse[i, 2]] + phi[W_sparse[i, 1]];
      }
    
      for (i in 1:n) ldet_terms[i] = log1m(alpha * lambda[i]);
      return 0.5 * (n * log(tau)
                    + sum(ldet_terms)
                    - tau * (phit_D * phi - alpha * (phit_W * phi)));
  }
}
```
with the data transformed in the model:
```stan
transformed data {
  int W_sparse[W_n, 2];   // adjacency pairs
  vector[n] D_sparse;     // diagonal of D (number of neighbors for each site)
  vector[n] lambda;       // eigenvalues of invsqrtD * W * invsqrtD
  
  { // generate sparse representation for W
  int counter;
  counter = 1;
  // loop over upper triangular part of W to identify neighbor pairs
    for (i in 1:(n - 1)) {
      for (j in (i + 1):n) {
        if (W[i, j] == 1) {
          W_sparse[counter, 1] = i;
          W_sparse[counter, 2] = j;
          counter = counter + 1;
        }
      }
    }
  }
  for (i in 1:n) D_sparse[i] = sum(W[i]);
  {
    vector[n] invsqrtD;  
    for (i in 1:n) {
      invsqrtD[i] = 1 / sqrt(D_sparse[i]);
    }
    lambda = eigenvalues_sym(quad_form(W, diag_matrix(invsqrtD)));
  }
}
```
and the likelihood:
```stan
model {
  phi ~ sparse_car(tau, alpha, W_sparse, D_sparse, lambda, n, W_n);
}
```

+++

This is quite a lot of code to digest, so my general approach is to compare the intermediate steps (whenever possible) with `Stan`. In this case, I will try to compute `tau, alpha, W_sparse, D_sparse, lambda, n, W_n` outside of the `Stan` model in `R` and compare with my own implementation.  

Below is a Sparse CAR implementation in `PyMC3` ([see also here](https://github.com/pymc-devs/pymc3/issues/2066#issuecomment-296397012)). Again, we try to avoid using any looping, as in `Stan`.

```{code-cell} ipython3
import scipy


class Sparse_CAR(distribution.Continuous):
    """
    Sparse Conditional Autoregressive (CAR) distribution

    Parameters
    ----------
    alpha : spatial smoothing term
    W : adjacency matrix
    tau : precision at each location
    """

    def __init__(self, alpha, W, tau, *args, **kwargs):
        self.alpha = alpha = tt.as_tensor_variable(alpha)
        self.tau = tau = tt.as_tensor_variable(tau)
        D = W.sum(axis=0)
        n, m = W.shape
        self.n = n
        self.median = self.mode = self.mean = 0
        super().__init__(*args, **kwargs)

        # eigenvalues of D^−1/2 * W * D^−1/2
        Dinv_sqrt = np.diag(1 / np.sqrt(D))
        DWD = np.matmul(np.matmul(Dinv_sqrt, W), Dinv_sqrt)
        self.lam = scipy.linalg.eigvalsh(DWD)

        # sparse representation of W
        w_sparse = scipy.sparse.csr_matrix(W)
        self.W = theano.sparse.as_sparse_variable(w_sparse)
        self.D = tt.as_tensor_variable(D)

        # Precision Matrix (inverse of Covariance matrix)
        # d_sparse = scipy.sparse.csr_matrix(np.diag(D))
        # self.D = theano.sparse.as_sparse_variable(d_sparse)
        # self.Phi = self.tau * (self.D - self.alpha*self.W)

    def logp(self, x):
        logtau = self.n * tt.log(tau)
        logdet = tt.log(1 - self.alpha * self.lam).sum()

        # tau * ((phi .* D_sparse)' * phi - alpha * (phit_W * phi))
        Wx = theano.sparse.dot(self.W, x)
        tau_dot_x = self.D * x.T - self.alpha * Wx.ravel()
        logquad = self.tau * tt.dot(x.ravel(), tau_dot_x.ravel())

        # logquad = tt.dot(x.T, theano.sparse.dot(self.Phi, x)).sum()
        return 0.5 * (logtau + logdet - logquad)
```

```{code-cell} ipython3
with pm.Model() as model4:
    # Vague prior on intercept and effect
    beta = pm.Normal("beta", mu=0.0, tau=1.0, shape=(2, 1))

    # Priors for spatial random effects
    tau = pm.Gamma("tau", alpha=2.0, beta=2.0)
    alpha = pm.Uniform("alpha", lower=0, upper=1)
    phi = Sparse_CAR("phi", alpha, W, tau, shape=(N, 1))

    # Mean model
    mu = pm.Deterministic("mu", tt.exp(tt.dot(X, beta) + phi + log_offset))

    # Likelihood
    Yi = pm.Poisson("Yi", mu=mu.ravel(), observed=O)

    infdata4 = pm.sample(1000, tune=2000, cores=4, target_accept=0.85, return_inferencedata=True)
```

```{code-cell} ipython3
az.plot_trace(infdata4, var_names=["alpha", "beta", "tau"]);
```

```{code-cell} ipython3
az.plot_posterior(infdata4, var_names=["alpha"])
```

As you can see above, the sparse representation returns the same estimates, while being much faster than any other implementation.

+++

## A few other warnings
In `Stan`, there is an option to write a `generated quantities` block for sample generation. Doing the similar in pymc3, however, is not recommended.  

Consider the following simple sample:

```python
# Data
x = np.array([1.1, 1.9, 2.3, 1.8])
n = len(x)

with pm.Model() as model1:
    # prior
    mu = pm.Normal('mu', mu=0, tau=.001)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    # observed
    xi = pm.Normal('xi', mu=mu, tau=1/(sigma**2), observed=x)
    # generation 
    p = pm.Deterministic('p', pm.math.sigmoid(mu))
    count = pm.Binomial('count', n=10, p=p, shape=10)
```

where we intended to use 

```python
count = pm.Binomial('count', n=10, p=p, shape=10)
```
to generate posterior prediction. However, if the new RV added to the model is a discrete variable it can cause weird turbulence to the trace. You can see [issue #1990](https://github.com/pymc-devs/pymc3/issues/1990) for related discussion.

+++

## Final remarks

In this notebook, most of the parameter conventions (e.g., using `tau` when defining a Normal distribution) and choice of priors are strictly matched with the original code in `Winbugs` or `Stan`. However, it is important to note that merely porting the code from one probabilistic programming language to the another is not necessarily the best practice. The aim is not just to run the code in `PyMC3`, but to make sure the model is appropriate so that it returns correct estimates, and runs efficiently (fast sampling). 

For example, as [@aseyboldt](https://github.com/aseyboldt) pointed out [here](https://github.com/pymc-devs/pymc3/pull/2080#issuecomment-297456574) and [here](https://github.com/pymc-devs/pymc3/issues/1924#issue-215496293), non-centered parametrizations are often a better choice than the centered parametrizations. In our case here, `phi` is following a zero-mean Normal distribution, thus it can be left out in the beginning and used to scale the values afterwards. Often, doing this can avoid correlations in the posterior (it will be slower in some cases, however).  

Another thing to keep in mind is that models can be sensitive to choices of prior distributions; for example, you can have a hard time using Normal variables with a large sd as prior. Gelman often recommends Cauchy or StudentT (*i.e.*, weakly-informative priors). More information on prior choice can be found on the [Stan wiki](https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations).  

There are always ways to improve  code. Since our computational graph with `pm.Model()` consist of `theano` objects, we can always do `print(VAR_TO_CHECK.tag.test_value)` right after the declaration or computation to check the shape. 
For example, in our last example, as suggested by [@aseyboldt](https://github.com/pymc-devs/pymc3/pull/2080#issuecomment-297456574) there seem to be a lot of correlation in the posterior. That probably slows down NUTS quite a bit. As a debugging tool and guide for reparametrization you can look at the singular value decomposition of the standardized samples from the trace – basically the eigenvalues of the correlation matrix. If the problem is high dimensional you can use stuff from `scipy.sparse.linalg` to only compute the largest singular value: 

```python
from scipy import linalg, sparse

vals = np.array([model.dict_to_array(v) for v in trace[1000:]]).T
vals[:] -= vals.mean(axis=1)[:, None]
vals[:] /= vals.std(axis=1)[:, None]

U, S, Vh = sparse.linalg.svds(vals, k=20)
```

Then look at `plt.plot(S)` to see if any principal components are obvious, and check which variables are contributing by looking at the singular vectors: `plt.plot(U[:, -1] ** 2)`. You can get the indices by looking at `model.bijection.ordering.vmap`.

Another great way to check the correlations in the posterior is to do a pairplot of the posterior (if your model doesn't contain too many parameters). You can see quite clearly if and where the the posterior parameters are correlated.

```{code-cell} ipython3
az.plot_pair(infdata1, var_names=["beta0", "beta1", "tau_h", "tau_c"], divergences=True);
```

```{code-cell} ipython3
az.plot_pair(infdata2, var_names=["beta0", "beta1", "tau_h", "tau_c"], divergences=True);
```

```{code-cell} ipython3
az.plot_pair(infdata3, var_names=["beta", "tau", "alpha"], divergences=True);
```

```{code-cell} ipython3
az.plot_pair(infdata4, var_names=["beta", "tau", "alpha"], divergences=True);
```

* Notebook Written by [Junpeng Lao](https://www.github.com/junpenglao/), inspired by `PyMC3` [issue#2022](https://github.com/pymc-devs/pymc3/issues/2022), [issue#2066](https://github.com/pymc-devs/pymc3/issues/2066) and [comments](https://github.com/pymc-devs/pymc3/issues/2066#issuecomment-296397012). I would like to thank [@denadai2](https://github.com/denadai2), [@aseyboldt](https://github.com/aseyboldt), and [@twiecki](https://github.com/twiecki) for the helpful discussion.

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w
```
