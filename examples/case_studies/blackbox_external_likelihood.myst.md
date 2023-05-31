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

(blackbox_external_likelihood)=
# Using a "black box" likelihood function (Cython)

:::{note}
This notebook in part of a set of two twin notebooks that perform the exact same task, this one
uses Cython whereas {ref}`this other one <blackbox_external_likelihood_numpy>` uses NumPy
:::

```{code-cell} ipython3
%load_ext Cython

import os
import platform

import arviz as az
import corner
import cython
import emcee
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt

print(f"Running on PyMC3 v{pm.__version__}")
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")
```

[PyMC3](https://docs.pymc.io/index.html) is a great tool for doing Bayesian inference and parameter estimation. It has a load of [in-built probability distributions](https://docs.pymc.io/api/distributions.html) that you can use to set up priors and likelihood functions for your particular model. You can even create your own [custom distributions](https://docs.pymc.io/prob_dists.html#custom-distributions).

However, this is not necessarily that simple if you have a model function, or probability distribution, that, for example, relies on an external code that you have little/no control over (and may even be, for example, wrapped `C` code rather than Python). This can be problematic went you need to pass parameters set as PyMC3 distributions to these external functions; your external function probably wants you to pass it floating point numbers rather than PyMC3 distributions!

```python
import pymc3 as pm
from external_module import my_external_func  # your external function!

# set up your model
with pm.Model():
    # your external function takes two parameters, a and b, with Uniform priors
    a = pm.Uniform('a', lower=0., upper=1.)
    b = pm.Uniform('b', lower=0., upper=1.)
    
    m = my_external_func(a, b)  # <--- this is not going to work!
```

Another issue is that if you want to be able to use the [gradient-based step samplers](https://docs.pymc.io/notebooks/getting_started.html#Gradient-based-sampling-methods) like [NUTS](https://docs.pymc.io/api/inference.html#module-pymc3.step_methods.hmc.nuts) and [Hamiltonian Monte Carlo (HMC)](https://docs.pymc.io/api/inference.html#hamiltonian-monte-carlo), then your model/likelihood needs a gradient to be defined. If you have a model that is defined as a set of Theano operators then this is no problem - internally it will be able to do automatic differentiation - but if your model is essentially a "black box" then you won't necessarily know what the gradients are.

Defining a model/likelihood that PyMC3 can use and that calls your "black box" function is possible, but it relies on creating a [custom Theano Op](https://docs.pymc.io/advanced_theano.html#writing-custom-theano-ops). This is, hopefully, a clear description of how to do this, including one way of writing a gradient function that could be generally applicable.

In the examples below, we create a very simple model and log-likelihood function in [Cython](http://cython.org/). Cython is used just as an example to show what you might need to do if calling external `C` codes, but you could in fact be using pure Python codes. The log-likelihood function used is actually just a [Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution), so defining this yourself is obviously overkill (and I'll compare it to doing the same thing purely with the pre-defined PyMC3 [Normal](https://docs.pymc.io/api/distributions/continuous.html#pymc3.distributions.continuous.Normal) distribution), but it should provide a simple to follow demonstration.

First, let's define a _super-complicated_&trade; model (a straight line!), which is parameterised by two variables (a gradient `m` and a y-intercept `c`) and calculated at a vector of points `x`. Here the model is defined in [Cython](http://cython.org/) and calls [GSL](https://www.gnu.org/software/gsl/) functions. This is just to show that you could be calling some other `C` library that you need. In this example, the model parameters are all packed into a list/array/tuple called `theta`.

Let's also define a _really-complicated_&trade; log-likelihood function (a Normal log-likelihood that ignores the normalisation), which takes in the list/array/tuple of model parameter values `theta`, the points at which to calculate the model `x`, the vector of "observed" data points `data`, and the standard deviation of the noise in the data `sigma`. This log-likelihood function calls the _super-complicated_&trade; model function.

```{code-cell} ipython3
%%cython -I/usr/include -L/usr/lib/x86_64-linux-gnu -lgsl -lgslcblas -lm

import cython

cimport cython

import numpy as np

cimport numpy as np

### STUFF FOR USING GSL (FEEL FREE TO IGNORE!) ###

# declare GSL vector structure and functions
cdef extern from "gsl/gsl_block.h":
    cdef struct gsl_block:
        size_t size
        double * data

cdef extern from "gsl/gsl_vector.h":
    cdef struct gsl_vector:
        size_t size
        size_t stride
        double * data
        gsl_block * block
        int owner

    ctypedef struct gsl_vector_view:
        gsl_vector vector

    int gsl_vector_scale (gsl_vector * a, const double x) nogil
    int gsl_vector_add_constant (gsl_vector * a, const double x) nogil
    gsl_vector_view gsl_vector_view_array (double * base, size_t n) nogil

###################################################


# define your super-complicated model that uses loads of external codes
cpdef my_model(theta, np.ndarray[np.float64_t, ndim=1] x):
    """
    A straight line!

    Note:
        This function could simply be:

            m, c = thetha
            return m*x + x

        but I've made it more complicated for demonstration purposes
    """
    m, c = theta  # unpack line gradient and y-intercept

    cdef size_t length = len(x)  # length of x

    cdef np.ndarray line = np.copy(x)  # make copy of x vector
    cdef gsl_vector_view lineview      # create a view of the vector
    lineview = gsl_vector_view_array(<double *>line.data, length) 

    # multiply x by m
    gsl_vector_scale(&lineview.vector, <double>m)

    # add c
    gsl_vector_add_constant(&lineview.vector, <double>c)

    # return the numpy array
    return line


# define your really-complicated likelihood function that uses loads of external codes
cpdef my_loglike(theta, np.ndarray[np.float64_t, ndim=1] x,
                 np.ndarray[np.float64_t, ndim=1] data, sigma):
    """
    A Gaussian log-likelihood function for a model with parameters given in theta
    """

    model = my_model(theta, x)

    return -(0.5/sigma**2)*np.sum((data - model)**2)
```

Now, as things are, if we wanted to sample from this log-likelihood function, using certain prior distributions for the model parameters (gradient and y-intercept) using PyMC3, we might try something like this (using a [PyMC3 DensityDist](https://docs.pymc.io/prob_dists.html#custom-distributions)):

```python
import pymc3 as pm

# create/read in our "data" (I'll show this in the real example below)
x = ...
sigma = ...
data = ...

with pm.Model():
    # set priors on model gradient and y-intercept
    m = pm.Uniform('m', lower=-10., upper=10.)
    c = pm.Uniform('c', lower=-10., upper=10.)

    # create custom distribution 
    pm.DensityDist('likelihood', my_loglike,
                   observed={'theta': (m, c), 'x': x, 'data': data, 'sigma': sigma})
    
    # sample from the distribution
    trace = pm.sample(1000)
```

But, this will give an error like:

```
ValueError: setting an array element with a sequence.
```

This is because `m` and `c` are Theano tensor-type objects.

So, what we actually need to do is create a [Theano Op](http://deeplearning.net/software/theano/extending/extending_theano.html). This will be a new class that wraps our log-likelihood function (or just our model function, if that is all that is required) into something that can take in Theano tensor objects, but internally can cast them as floating point values that can be passed to our log-likelihood function. We will do this below, initially without defining a [grad() method](http://deeplearning.net/software/theano/extending/op.html#grad) for the Op.

```{code-cell} ipython3
# define a theano Op for our likelihood function
class LogLike(tt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """

    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, data, x, sigma):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.x = x
        self.sigma = sigma

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.x, self.data, self.sigma)

        outputs[0][0] = np.array(logl)  # output the log-likelihood
```

Now, let's use this Op to repeat the example shown above. To do this let's create some data containing a straight line with additive Gaussian noise (with a mean of zero and a standard deviation of `sigma`). For simplicity we set [uniform](https://docs.pymc.io/api/distributions/continuous.html#pymc3.distributions.continuous.Uniform) prior distributions on the gradient and y-intercept. As we've not set the `grad()` method of the Op PyMC3 will not be able to use the gradient-based samplers, so will fall back to using the [Slice](https://docs.pymc.io/api/inference.html#module-pymc3.step_methods.slicer) sampler.

```{code-cell} ipython3
# set up our data
N = 10  # number of data points
sigma = 1.0  # standard deviation of noise
x = np.linspace(0.0, 9.0, N)

mtrue = 0.4  # true gradient
ctrue = 3.0  # true y-intercept

truemodel = my_model([mtrue, ctrue], x)

# make data
np.random.seed(716742)  # set random seed, so the data is reproducible each time
data = sigma * np.random.randn(N) + truemodel

ndraws = 3000  # number of draws from the distribution
nburn = 1000  # number of "burn-in points" (which we'll discard)

# create our Op
logl = LogLike(my_loglike, data, x, sigma)

# use PyMC3 to sampler from log-likelihood
with pm.Model():
    # uniform priors on m and c
    m = pm.Uniform("m", lower=-10.0, upper=10.0)
    c = pm.Uniform("c", lower=-10.0, upper=10.0)

    # convert m and c to a tensor vector
    theta = tt.as_tensor_variable([m, c])

    # use a DensityDist (use a lamdba function to "call" the Op)
    pm.DensityDist("likelihood", lambda v: logl(v), observed={"v": theta})

    trace = pm.sample(ndraws, tune=nburn, discard_tuned_samples=True)

# plot the traces
_ = az.plot_trace(trace, lines={"m": mtrue, "c": ctrue})

# put the chains in an array (for later!)
samples_pymc3 = np.vstack((trace["m"], trace["c"])).T
```

What if we wanted to use NUTS or HMC? If we knew the analytical derivatives of the model/likelihood function then we could add a [grad() method](http://deeplearning.net/software/theano/extending/op.html#grad) to the Op using that analytical form.

But, what if we don't know the analytical form. If our model/likelihood is purely Python and made up of standard maths operators and Numpy functions, then the [autograd](https://github.com/HIPS/autograd) module could potentially be used to find gradients (also, see [here](https://github.com/ActiveState/code/blob/master/recipes/Python/580610_Auto_differentiation/recipe-580610.py) for a nice Python example of automatic differentiation). But, if our model/likelihood truly is a "black box" then we can just use the good-old-fashioned [finite difference](https://en.wikipedia.org/wiki/Finite_difference) to find the gradients - this can be slow, especially if there are a large number of variables, or the model takes a long time to evaluate. Below, a function to find gradients has been defined that uses the finite difference (the central difference) - it uses an iterative method with successively smaller interval sizes to check that the gradient converges. But, you could do something far simpler and just use, for example, the SciPy [approx_fprime](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.approx_fprime.html) function. Here, the gradient function is defined in Cython for speed, but if the function it evaluates to find the gradients is the performance bottle neck then having this as a pure Python function may not make a significant speed difference.

```{code-cell} ipython3
%%cython

import cython

cimport cython

import numpy as np

cimport numpy as np

import warnings


def gradients(vals, func, releps=1e-3, abseps=None, mineps=1e-9, reltol=1e-3,
              epsscale=0.5):
    """
    Calculate the partial derivatives of a function at a set of values. The
    derivatives are calculated using the central difference, using an iterative
    method to check that the values converge as step size decreases.

    Parameters
    ----------
    vals: array_like
        A set of values, that are passed to a function, at which to calculate
        the gradient of that function
    func:
        A function that takes in an array of values.
    releps: float, array_like, 1e-3
        The initial relative step size for calculating the derivative.
    abseps: float, array_like, None
        The initial absolute step size for calculating the derivative.
        This overrides `releps` if set.
        `releps` is set then that is used.
    mineps: float, 1e-9
        The minimum relative step size at which to stop iterations if no
        convergence is achieved.
    epsscale: float, 0.5
        The factor by which releps if scaled in each iteration.

    Returns
    -------
    grads: array_like
        An array of gradients for each non-fixed value.
    """

    grads = np.zeros(len(vals))

    # maximum number of times the gradient can change sign
    flipflopmax = 10.

    # set steps
    if abseps is None:
        if isinstance(releps, float):
            eps = np.abs(vals)*releps
            eps[eps == 0.] = releps  # if any values are zero set eps to releps
            teps = releps*np.ones(len(vals))
        elif isinstance(releps, (list, np.ndarray)):
            if len(releps) != len(vals):
                raise ValueError("Problem with input relative step sizes")
            eps = np.multiply(np.abs(vals), releps)
            eps[eps == 0.] = np.array(releps)[eps == 0.]
            teps = releps
        else:
            raise RuntimeError("Relative step sizes are not a recognised type!")
    else:
        if isinstance(abseps, float):
            eps = abseps*np.ones(len(vals))
        elif isinstance(abseps, (list, np.ndarray)):
            if len(abseps) != len(vals):
                raise ValueError("Problem with input absolute step sizes")
            eps = np.array(abseps)
        else:
            raise RuntimeError("Absolute step sizes are not a recognised type!")
        teps = eps

    # for each value in vals calculate the gradient
    count = 0
    for i in range(len(vals)):
        # initial parameter diffs
        leps = eps[i]
        cureps = teps[i]

        flipflop = 0

        # get central finite difference
        fvals = np.copy(vals)
        bvals = np.copy(vals)

        # central difference
        fvals[i] += 0.5*leps  # change forwards distance to half eps
        bvals[i] -= 0.5*leps  # change backwards distance to half eps
        cdiff = (func(fvals)-func(bvals))/leps

        while 1:
            fvals[i] -= 0.5*leps  # remove old step
            bvals[i] += 0.5*leps

            # change the difference by a factor of two
            cureps *= epsscale
            if cureps < mineps or flipflop > flipflopmax:
                # if no convergence set flat derivative (TODO: check if there is a better thing to do instead)
                warnings.warn("Derivative calculation did not converge: setting flat derivative.")
                grads[count] = 0.
                break
            leps *= epsscale

            # central difference
            fvals[i] += 0.5*leps  # change forwards distance to half eps
            bvals[i] -= 0.5*leps  # change backwards distance to half eps
            cdiffnew = (func(fvals)-func(bvals))/leps

            if cdiffnew == cdiff:
                grads[count] = cdiff
                break

            # check whether previous diff and current diff are the same within reltol
            rat = (cdiff/cdiffnew)
            if np.isfinite(rat) and rat > 0.:
                # gradient has not changed sign
                if np.abs(1.-rat) < reltol:
                    grads[count] = cdiffnew
                    break
                else:
                    cdiff = cdiffnew
                    continue
            else:
                cdiff = cdiffnew
                flipflop += 1
                continue

        count += 1

    return grads
```

So, now we can just redefine our Op with a `grad()` method, right?

It's not quite so simple! The `grad()` method itself requires that its inputs are Theano tensor variables, whereas our `gradients` function above, like our `my_loglike` function, wants a list of floating point values. So, we need to define another Op that calculates the gradients. Below, I define a new version of the `LogLike` Op, called `LogLikeWithGrad` this time, that has a `grad()` method. This is followed by anothor Op called `LogLikeGrad` that, when called with a vector of Theano tensor variables, returns another vector of values that are the gradients (i.e., the [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)) of our log-likelihood function at those values. Note that the `grad()` method itself does not return the gradients directly, but instead returns the [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)-vector product (you can hopefully just copy what I've done and not worry about what this means too much!).

```{code-cell} ipython3
# define a theano Op for our likelihood function
class LogLikeWithGrad(tt.Op):

    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, data, x, sigma):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.x = x
        self.sigma = sigma

        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad(self.likelihood, self.data, self.x, self.sigma)

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.x, self.data, self.sigma)

        outputs[0][0] = np.array(logl)  # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        (theta,) = inputs  # our parameters
        return [g[0] * self.logpgrad(theta)]


class LogLikeGrad(tt.Op):

    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """

    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, loglike, data, x, sigma):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.x = x
        self.sigma = sigma

    def perform(self, node, inputs, outputs):
        (theta,) = inputs

        # define version of likelihood function to pass to derivative function
        def lnlike(values):
            return self.likelihood(values, self.x, self.data, self.sigma)

        # calculate gradients
        grads = gradients(theta, lnlike)

        outputs[0][0] = grads
```

Now, let's re-run PyMC3 with our new "grad"-ed Op. This time it will be able to automatically use NUTS.

```{code-cell} ipython3
# create our Op
logl = LogLikeWithGrad(my_loglike, data, x, sigma)

# use PyMC3 to sampler from log-likelihood
with pm.Model() as opmodel:
    # uniform priors on m and c
    m = pm.Uniform("m", lower=-10.0, upper=10.0)
    c = pm.Uniform("c", lower=-10.0, upper=10.0)

    # convert m and c to a tensor vector
    theta = tt.as_tensor_variable([m, c])

    # use a DensityDist
    pm.DensityDist("likelihood", lambda v: logl(v), observed={"v": theta})

    trace = pm.sample(ndraws, tune=nburn, discard_tuned_samples=True)

# plot the traces
_ = az.plot_trace(trace, lines={"m": mtrue, "c": ctrue})

# put the chains in an array (for later!)
samples_pymc3_2 = np.vstack((trace["m"], trace["c"])).T
```

Now, finally, just to check things actually worked as we might expect, let's do the same thing purely using PyMC3 distributions (because in this simple example we can!)

```{code-cell} ipython3
with pm.Model() as pymodel:
    # uniform priors on m and c
    m = pm.Uniform("m", lower=-10.0, upper=10.0)
    c = pm.Uniform("c", lower=-10.0, upper=10.0)

    # convert m and c to a tensor vector
    theta = tt.as_tensor_variable([m, c])

    # use a Normal distribution
    pm.Normal("likelihood", mu=(m * x + c), sigma=sigma, observed=data)

    trace = pm.sample(ndraws, tune=nburn, discard_tuned_samples=True)

# plot the traces
_ = az.plot_trace(trace, lines={"m": mtrue, "c": ctrue})

# put the chains in an array (for later!)
samples_pymc3_3 = np.vstack((trace["m"], trace["c"])).T
```

To check that they match let's plot all the examples together and also find the autocorrelation lengths.

```{code-cell} ipython3
import warnings

warnings.simplefilter(
    action="ignore", category=FutureWarning
)  # suppress emcee autocorr FutureWarning

matplotlib.rcParams["font.size"] = 18

hist2dkwargs = {
    "plot_datapoints": False,
    "plot_density": False,
    "levels": 1.0 - np.exp(-0.5 * np.arange(1.5, 2.1, 0.5) ** 2),
}  # roughly 1 and 2 sigma

colors = ["r", "g", "b"]
labels = ["Theanp Op (no grad)", "Theano Op (with grad)", "Pure PyMC3"]

for i, samples in enumerate([samples_pymc3, samples_pymc3_2, samples_pymc3_3]):
    # get maximum chain autocorrelartion length
    autocorrlen = int(np.max(emcee.autocorr.integrated_time(samples, c=3)))
    print("Auto-correlation length ({}): {}".format(labels[i], autocorrlen))

    if i == 0:
        fig = corner.corner(
            samples,
            labels=[r"$m$", r"$c$"],
            color=colors[i],
            hist_kwargs={"density": True},
            **hist2dkwargs,
            truths=[mtrue, ctrue],
        )
    else:
        corner.corner(
            samples, color=colors[i], hist_kwargs={"density": True}, fig=fig, **hist2dkwargs
        )

fig.set_size_inches(8, 8)
```

We can now check that the gradient Op works was we expect it to. First, just create and call the `LogLikeGrad` class, which should return the gradient directly (note that we have to create a [Theano function](http://deeplearning.net/software/theano/library/compile/function.html) to convert the output of the Op to an array). Secondly, we call the gradient from `LogLikeWithGrad` by using the [Theano tensor gradient](http://deeplearning.net/software/theano/library/gradient.html#theano.gradient.grad) function. Finally, we will check the gradient returned by the PyMC3 model for a Normal distribution, which should be the same as the log-likelihood function we defined. In all cases we evaluate the gradients at the true values of the model function (the straight line) that was created.

```{code-cell} ipython3
# test the gradient Op by direct call
theano.config.compute_test_value = "ignore"
theano.config.exception_verbosity = "high"

var = tt.dvector()
test_grad_op = LogLikeGrad(my_loglike, data, x, sigma)
test_grad_op_func = theano.function([var], test_grad_op(var))
grad_vals = test_grad_op_func([mtrue, ctrue])

print(f'Gradient returned by "LogLikeGrad": {grad_vals}')

# test the gradient called through LogLikeWithGrad
test_gradded_op = LogLikeWithGrad(my_loglike, data, x, sigma)
test_gradded_op_grad = tt.grad(test_gradded_op(var), var)
test_gradded_op_grad_func = theano.function([var], test_gradded_op_grad)
grad_vals_2 = test_gradded_op_grad_func([mtrue, ctrue])

print(f'Gradient returned by "LogLikeWithGrad": {grad_vals_2}')

# test the gradient that PyMC3 uses for the Normal log likelihood
test_model = pm.Model()
with test_model:
    m = pm.Uniform("m", lower=-10.0, upper=10.0)
    c = pm.Uniform("c", lower=-10.0, upper=10.0)

    pm.Normal("likelihood", mu=(m * x + c), sigma=sigma, observed=data)

    gradfunc = test_model.logp_dlogp_function([m, c], dtype=None)
    gradfunc.set_extra_values({"m_interval__": mtrue, "c_interval__": ctrue})
    grad_vals_pymc3 = gradfunc(np.array([mtrue, ctrue]))[1]  # get dlogp values

print(f'Gradient returned by PyMC3 "Normal" distribution: {grad_vals_pymc3}')
```

We can also do some [profiling](http://docs.pymc.io/notebooks/profiling.html) of the Op, as used within a PyMC3 Model, to check performance. First, we'll profile using the `LogLikeWithGrad` Op, and then doing the same thing purely using PyMC3 distributions.

```{code-cell} ipython3
# profile logpt using our Op
opmodel.profile(opmodel.logpt).summary()
```

```{code-cell} ipython3
# profile using our PyMC3 distribution
pymodel.profile(pymodel.logpt).summary()
```

## Authors

* Adapted from a blog post by [Matt Pitkin](http://mattpitkin.github.io/samplers-demo/pages/pymc3-blackbox-likelihood/) on August 27, 2018. That post was based on an example provided by [Jørgen Midtbø](https://github.com/jorgenem/).

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w
```
