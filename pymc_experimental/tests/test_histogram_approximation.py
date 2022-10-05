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


import numpy as np
import pymc as pm
import pytest

import pymc_experimental as pmx


@pytest.mark.parametrize("use_dask", [True, False], ids="dask={}".format)
@pytest.mark.parametrize("zero_inflation", [True, False], ids="ZI={}".format)
@pytest.mark.parametrize("ndims", [1, 2], ids="ndims={}".format)
def test_histogram_init_cont(use_dask, zero_inflation, ndims):
    data = np.random.randn(*(10000, *(2,) * (ndims - 1)))
    if zero_inflation:
        data = abs(data)
        data[:100] = 0
    if use_dask:
        dask = pytest.importorskip("dask")
        dask_df = pytest.importorskip("dask.dataframe")
        data = dask_df.from_array(data)
    histogram = pmx.distributions.histogram_utils.quantile_histogram(
        data, n_quantiles=100, zero_inflation=zero_inflation
    )
    if use_dask:
        (histogram,) = dask.compute(histogram)
    assert isinstance(histogram, dict)
    assert isinstance(histogram["mid"], np.ndarray)
    assert np.issubdtype(histogram["mid"].dtype, np.floating)
    size = 99 + zero_inflation
    assert histogram["mid"].shape == (size,) + (1,) * len(data.shape[1:])
    assert histogram["lower"].shape == (size,) + (1,) * len(data.shape[1:])
    assert histogram["upper"].shape == (size,) + (1,) * len(data.shape[1:])
    assert histogram["count"].shape == (size,) + data.shape[1:]
    assert (histogram["count"].sum(0) == 10000).all()
    if zero_inflation:
        (histogram["count"][0] == 100).all()


@pytest.mark.parametrize("use_dask", [True, False], ids="dask={}".format)
@pytest.mark.parametrize("min_count", [None, 5], ids="min_count={}".format)
@pytest.mark.parametrize("ndims", [1, 2], ids="ndims={}".format)
def test_histogram_init_discrete(use_dask, min_count, ndims):
    data = np.random.randint(0, 100, size=(10000,) + (2,) * (ndims - 1))
    u, c = np.unique(data, return_counts=True)
    if use_dask:
        dask = pytest.importorskip("dask")
        dask_df = pytest.importorskip("dask.dataframe")
        data = dask_df.from_array(data)
    histogram = pmx.distributions.histogram_utils.discrete_histogram(data, min_count=min_count)
    if use_dask:
        (histogram,) = dask.compute(histogram)
    assert isinstance(histogram, dict)
    assert isinstance(histogram["mid"], np.ndarray)
    assert np.issubdtype(histogram["mid"].dtype, np.integer)
    if min_count is not None:
        size = int((c >= min_count).sum())
    else:
        size = len(u)
    assert histogram["mid"].shape == (size,) + (1,) * len(data.shape[1:])
    assert histogram["count"].shape == (size,) + data.shape[1:]
    if not min_count:
        assert (histogram["count"].sum(0) == 10000).all()


@pytest.mark.parametrize("use_dask", [True, False], ids="dask={}".format)
@pytest.mark.parametrize("ndims", [1, 2], ids="ndims={}".format)
def test_histogram_approx_cont(use_dask, ndims):
    data = np.random.randn(*(10000, *(2,) * (ndims - 1)))
    if use_dask:
        dask = pytest.importorskip("dask")
        dask_df = pytest.importorskip("dask.dataframe")
        data = dask_df.from_array(data)
    with pm.Model():
        m = pm.Normal("m")
        s = pm.HalfNormal("s", size=2 if ndims > 1 else 1)
        pot = pmx.distributions.histogram_utils.histogram_approximation(
            "histogram_potential", pm.Normal.dist(m, s), observed=data, n_quantiles=1000
        )
        trace = pm.sample(10, tune=0)  # very fast


@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.parametrize("ndims", [1, 2], ids="ndims={}".format)
def test_histogram_approx_discrete(use_dask, ndims):
    data = np.random.randint(0, 100, size=(10000, *(2,) * (ndims - 1)))
    if use_dask:
        dask = pytest.importorskip("dask")
        dask_df = pytest.importorskip("dask.dataframe")
        data = dask_df.from_array(data)
    with pm.Model():
        s = pm.Exponential("s", 1.0, size=2 if ndims > 1 else 1)
        pot = pmx.distributions.histogram_utils.histogram_approximation(
            "histogram_potential", pm.Poisson.dist(s), observed=data, min_count=10
        )
        trace = pm.sample(10, tune=0)  # very fast
