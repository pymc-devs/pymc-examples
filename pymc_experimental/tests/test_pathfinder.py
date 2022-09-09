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

import sys

import numpy as np
import pymc as pm
import pytest

import pymc_experimental as pmx


@pytest.mark.skipif(sys.platform == "win32", reason="JAX not supported on windows.")
def test_pathfinder():
    # Data of the Eight Schools Model
    J = 8
    y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

    with pm.Model() as model:

        mu = pm.Normal("mu", mu=0.0, sigma=10.0)
        tau = pm.HalfCauchy("tau", 5.0)

        theta = pm.Normal("theta", mu=0, sigma=1, shape=J)
        theta_1 = mu + tau * theta
        obs = pm.Normal("obs", mu=theta, sigma=sigma, shape=J, observed=y)

        idata = pmx.fit(method="pathfinder", iterations=100)

        assert idata is not None
        assert "theta" in idata.posterior._variables.keys()
        assert "tau" in idata.posterior._variables.keys()
        assert "mu" in idata.posterior._variables.keys()
        assert idata.posterior["mu"].shape == (1, 100)
        assert idata.posterior["tau"].shape == (1, 100)
        assert idata.posterior["theta"].shape == (1, 100, 8)
