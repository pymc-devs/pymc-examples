"""Injected code to the top of each notebook to mock long running code."""

import os
import numpy as np
import pymc as pm
import xarray as xr


def mock_sample(*args, **kwargs):
    if len(args) > 0:
        draws = args[0]
    else:
        draws = kwargs.get("draws", 1000)
    random_seed = kwargs.get("random_seed", None)
    rng = np.random.default_rng(random_seed)
    model = kwargs.get("model", None)
    chains = kwargs.get("chains", os.cpu_count())
    idata = pm.sample_prior_predictive(
        model=model,
        random_seed=random_seed,
        samples=draws,
    )
    n_chains = chains
    expanded_chains = xr.DataArray(
        np.ones(n_chains),
        coords={"chain": np.arange(n_chains)},
    )
    idata.add_groups(
        posterior=(idata.prior.mean("chain") * expanded_chains).transpose("chain", "draw", ...)
    )
    idata.posterior.attrs["sampling_time"] = 1.0

    if "prior" in idata:
        del idata.prior
    if "prior_predictive" in idata:
        del idata.prior_predictive

    # Create mock sample stats with diverging data
    if "sample_stats" not in idata:
        n_chains = chains
        n_draws = draws
        sample_stats = xr.Dataset(
            {
                "diverging": xr.DataArray(
                    np.zeros((n_chains, n_draws), dtype=int),
                    dims=("chain", "draw"),
                ),
                "energy": xr.DataArray(
                    rng.normal(loc=150, scale=2.5, size=(n_chains, n_draws)),
                    dims=("chain", "draw"),
                ),
                "tree_depth": xr.DataArray(
                    rng.choice([1, 2, 3], p=[0.01, 0.86, 0.13], size=(n_chains, n_draws)),
                    dims=("chain", "draw"),
                ),
                "acceptance_rate": xr.DataArray(
                    rng.beta(0.5, 0.5, size=(n_chains, n_draws)),
                    dims=("chain", "draw"),
                ),
                # Different sampler
                "accept": xr.DataArray(
                    rng.choice([0, 1], size=(n_chains, n_draws)),
                    dims=("chain", "draw"),
                ),
            }
        )
        idata.add_groups(sample_stats=sample_stats)

    return idata


pm.sample = mock_sample
pm.HalfFlat = pm.HalfNormal
pm.Flat = pm.Normal
