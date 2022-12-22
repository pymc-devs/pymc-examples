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


from typing import Dict

import numpy as np
import pymc as pm
from numpy.typing import ArrayLike

try:
    import dask.array
    import dask.dataframe
except ImportError:
    dask = None

try:
    import xhistogram.core
except ImportError:
    xhistogram = None


__all__ = ["quantile_histogram", "discrete_histogram", "histogram_approximation"]


def quantile_histogram(
    data: ArrayLike, n_quantiles=1000, zero_inflation=False
) -> Dict[str, ArrayLike]:
    if xhistogram is None:
        raise RuntimeError("quantile_histogram requires xhistogram package")

    if dask and isinstance(data, (dask.dataframe.Series, dask.dataframe.DataFrame)):
        data = data.to_dask_array(lengths=True)
    if zero_inflation:
        zeros = (data == 0).sum(0)
        mdata = np.ma.masked_where(data == 0, data)
        qdata = data[data > 0]
    else:
        mdata = data
        qdata = data.flatten()
    quantiles = np.percentile(qdata, np.linspace(0, 100, n_quantiles))
    if dask:
        (quantiles,) = dask.compute(quantiles)
    count, _ = xhistogram.core.histogram(mdata, bins=[quantiles], axis=0)
    count = count.transpose(count.ndim - 1, *range(count.ndim - 1))
    lower = quantiles[:-1]
    upper = quantiles[1:]

    if zero_inflation:
        count = np.concatenate([zeros[None], count])
        lower = np.concatenate([[0], lower])
        upper = np.concatenate([[0], upper])
    lower = lower.reshape(lower.shape + (1,) * (count.ndim - 1))
    upper = upper.reshape(upper.shape + (1,) * (count.ndim - 1))

    result = dict(
        lower=lower,
        upper=upper,
        mid=(lower + upper) / 2,
        count=count,
    )
    return result


def discrete_histogram(data: ArrayLike, min_count=None) -> Dict[str, ArrayLike]:
    if xhistogram is None:
        raise RuntimeError("discrete_histogram requires xhistogram package")

    if dask and isinstance(data, (dask.dataframe.Series, dask.dataframe.DataFrame)):
        data = data.to_dask_array(lengths=True)
    mid, count_uniq = np.unique(data, return_counts=True)
    if min_count is not None:
        mid = mid[count_uniq >= min_count]
        count_uniq = count_uniq[count_uniq >= min_count]
    bins = np.concatenate([mid, [mid.max() + 1]])
    if dask:
        mid, bins = dask.compute(mid, bins)
    count, _ = xhistogram.core.histogram(data, bins=[bins], axis=0)
    count = count.transpose(count.ndim - 1, *range(count.ndim - 1))
    mid = mid.reshape(mid.shape + (1,) * (count.ndim - 1))
    return dict(mid=mid, count=count)


def histogram_approximation(name, dist, *, observed, **h_kwargs):
    """Approximate a distribution with a histogram potential.

    Parameters
    ----------
    name : str
        Name for the Potential
    dist : pytensor.tensor.var.TensorVariable
        The output of pm.Distribution.dist()
    observed : ArrayLike
        Observed value to construct a histogram. Histogram is computed over 0th axis.
        Dask is supported.

    Returns
    -------
    pytensor.tensor.var.TensorVariable
        Potential

    Examples
    --------
    Discrete variables are reduced to unique repetitions (up to min_count)

    >>> import pymc as pm
    >>> import pymc_experimental as pmx
    >>> production = np.random.poisson([1, 2, 5], size=(1000, 3))
    >>> with pm.Model(coords=dict(plant=range(3))):
    ...     lam = pm.Exponential("lam", 1.0, dims="plant")
    ...     pot = pmx.distributions.histogram_approximation(
    ...         "pot", pm.Poisson.dist(lam), observed=production, min_count=2
    ...     )

    Continuous variables are discretized into n_quantiles

    >>> measurements = np.random.normal([1, 2, 3], [0.1, 0.4, 0.2], size=(10000, 3))
    >>> with pm.Model(coords=dict(tests=range(3))):
    ...     m = pm.Normal("m", dims="tests")
    ...     s = pm.LogNormal("s", dims="tests")
    ...     pot = pmx.distributions.histogram_approximation(
    ...         "pot", pm.Normal.dist(m, s),
    ...         observed=measurements, n_quantiles=50
    ...     )

    For special cases like Zero Inflation in Continuous variables there is a flag.
    The flag adds a separate bin for zeros

    >>> measurements = abs(measurements)
    >>> measurements[100:] = 0
    >>> with pm.Model(coords=dict(tests=range(3))):
    ...     m = pm.Normal("m", dims="tests")
    ...     s = pm.LogNormal("s", dims="tests")
    ...     pot = pmx.distributions.histogram_approximation(
    ...         "pot", pm.Normal.dist(m, s),
    ...         observed=measurements, n_quantiles=50, zero_inflation=True
    ...     )
    """
    if dask and isinstance(observed, (dask.dataframe.Series, dask.dataframe.DataFrame)):
        observed = observed.to_dask_array(lengths=True)
    if np.issubdtype(observed.dtype, np.integer):
        histogram = discrete_histogram(observed, **h_kwargs)
    else:
        histogram = quantile_histogram(observed, **h_kwargs)
    if dask is not None:
        (histogram,) = dask.compute(histogram)
    return pm.Potential(name, pm.logp(dist, histogram["mid"]) * histogram["count"])
