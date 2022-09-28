#   Copyright 2022 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# coding: utf-8
"""
Experimental probability distributions for stochastic nodes in PyMC.

The imports from pymc are not fully replicated here: add imports as necessary.
"""

from typing import List, Tuple, Union

import aesara.tensor as at
import numpy as np
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.var import TensorVariable
from pymc.aesaraf import floatX
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import Continuous
from pymc.distributions.shape_utils import rv_size_is_none
from scipy import stats


class GenExtremeRV(RandomVariable):
    name: str = "Generalized Extreme Value"
    ndim_supp: int = 0
    ndims_params: List[int] = [0, 0, 0]
    dtype: str = "floatX"
    _print_name: Tuple[str, str] = ("Generalized Extreme Value", "\\operatorname{GEV}")

    def __call__(self, mu=0.0, sigma=1.0, xi=0.0, size=None, **kwargs) -> TensorVariable:
        return super().__call__(mu, sigma, xi, size=size, **kwargs)

    @classmethod
    def rng_fn(
        cls,
        rng: Union[np.random.RandomState, np.random.Generator],
        mu: np.ndarray,
        sigma: np.ndarray,
        xi: np.ndarray,
        size: Tuple[int, ...],
    ) -> np.ndarray:
        # Notice negative here, since remainder of GenExtreme is based on Coles parametrization
        return stats.genextreme.rvs(c=-xi, loc=mu, scale=sigma, random_state=rng, size=size)


gev = GenExtremeRV()


class GenExtreme(Continuous):
    r"""
        Univariate Generalized Extreme Value log-likelihood

    The cdf of this distribution is

    .. math::

       G(x \mid \mu, \sigma, \xi) = \exp\left[ -\left(1 + \xi z\right)^{-\frac{1}{\xi}} \right]

    where

    .. math::

        z = \frac{x - \mu}{\sigma}

    and is defined on the set:

    .. math::

        \left\{x: 1 + \xi\left(\frac{x-\mu}{\sigma}\right) > 0 \right\}.

    Note that this parametrization is per Coles (2001), and differs from that of
    Scipy in the sign of the shape parameter, :math:`\xi`.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(-10, 20, 200)
        mus = [0., 4., -1.]
        sigmas = [2., 2., 4.]
        xis = [-0.3, 0.0, 0.3]
        for mu, sigma, xi in zip(mus, sigmas, xis):
            pdf = st.genextreme.pdf(x, c=-xi, loc=mu, scale=sigma)
            plt.plot(x, pdf, label=rf'$\mu$ = {mu}, $\sigma$ = {sigma}, $\xi$={xi}')
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()


    ========  =========================================================================
    Support   * :math:`x \in [\mu - \sigma/\xi, +\infty]`, when :math:`\xi > 0`
              * :math:`x \in \mathbb{R}` when :math:`\xi = 0`
              * :math:`x \in [-\infty, \mu - \sigma/\xi]`, when :math:`\xi < 0`
    Mean      * :math:`\mu + \sigma(g_1 - 1)/\xi`, when :math:`\xi \neq 0, \xi < 1`
              * :math:`\mu + \sigma \gamma`, when :math:`\xi = 0`
              * :math:`\infty`, when :math:`\xi \geq 1`
                where :math:`\gamma` is the Euler-Mascheroni constant, and
                :math:`g_k = \Gamma (1-k\xi)`
    Variance  * :math:`\sigma^2 (g_2 - g_1^2)/\xi^2`, when :math:`\xi \neq 0, \xi < 0.5`
              * :math:`\frac{\pi^2}{6} \sigma^2`, when :math:`\xi = 0`
              * :math:`\infty`, when :math:`\xi \geq 0.5`
    ========  =========================================================================

    Parameters
    ----------
    mu: float
        Location parameter.
    sigma: float
        Scale parameter (sigma > 0).
    xi: float
        Shape parameter
    scipy: bool
        Whether or not to use the Scipy interpretation of the shape parameter
        (defaults to `False`).

    References
    ----------
    .. [Coles2001] Coles, S.G. (2001).
        An Introduction to the Statistical Modeling of Extreme Values
        Springer-Verlag, London

    """

    rv_op = gev

    @classmethod
    def dist(cls, mu=0, sigma=1, xi=0, scipy=False, **kwargs):
        # If SciPy, use its parametrization, otherwise convert to standard
        if scipy:
            xi = -xi
        mu = at.as_tensor_variable(floatX(mu))
        sigma = at.as_tensor_variable(floatX(sigma))
        xi = at.as_tensor_variable(floatX(xi))

        return super().dist([mu, sigma, xi], **kwargs)

    def logp(value, mu, sigma, xi):
        """
        Calculate log-probability of Generalized Extreme Value distribution
        at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        scaled = (value - mu) / sigma

        logp_expression = at.switch(
            at.isclose(xi, 0),
            -at.log(sigma) - scaled - at.exp(-scaled),
            -at.log(sigma)
            - ((xi + 1) / xi) * at.log1p(xi * scaled)
            - at.pow(1 + xi * scaled, -1 / xi),
        )

        logp = at.switch(at.gt(1 + xi * scaled, 0.0), logp_expression, -np.inf)

        return check_parameters(
            logp, sigma > 0, at.and_(xi > -1, xi < 1), msg="sigma > 0 or -1 < xi < 1"
        )

    def logcdf(value, mu, sigma, xi):
        """
        Compute the log of the cumulative distribution function for Generalized Extreme Value
        distribution at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or `TensorVariable`
            Value(s) for which log CDF is calculated. If the log CDF for
            multiple values are desired the values must be provided in a numpy
            array or `TensorVariable`.

        Returns
        -------
        TensorVariable
        """
        scaled = (value - mu) / sigma
        logc_expression = at.switch(
            at.isclose(xi, 0), -at.exp(-scaled), -at.pow(1 + xi * scaled, -1 / xi)
        )

        logc = at.switch(1 + xi * (value - mu) / sigma > 0, logc_expression, -np.inf)

        return check_parameters(
            logc, sigma > 0, at.and_(xi > -1, xi < 1), msg="sigma > 0 or -1 < xi < 1"
        )

    def moment(rv, size, mu, sigma, xi):
        r"""
        Using the mode, as the mean can be infinite when :math:`\xi > 1`
        """
        mode = at.switch(at.isclose(xi, 0), mu, mu + sigma * (at.pow(1 + xi, -xi) - 1) / xi)
        if not rv_size_is_none(size):
            mode = at.full(size, mode)
        return mode
