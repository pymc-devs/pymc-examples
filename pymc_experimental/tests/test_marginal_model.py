import itertools
from contextlib import suppress as does_not_warn

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest
from pymc import ImputationWarning, inputvars
from pymc.distributions import transforms
from pymc.logprob.abstract import _logprob
from pymc.util import UNSET
from scipy.special import logsumexp

from pymc_experimental.marginal_model import FiniteDiscreteMarginalRV, MarginalModel


@pytest.fixture
def disaster_model():
    # fmt: off
    disaster_data = pd.Series(
        [4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
         3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
         2, 2, 3, 4, 2, 1, 3, np.nan, 2, 1, 1, 1, 1, 3, 0, 0,
         1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
         0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
         3, 3, 1, np.nan, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]
    )
    # fmt: on
    years = np.arange(1851, 1962)

    with MarginalModel() as disaster_model:
        switchpoint = pm.DiscreteUniform("switchpoint", lower=years.min(), upper=years.max())
        early_rate = pm.Exponential("early_rate", 1.0, initval=3)
        late_rate = pm.Exponential("late_rate", 1.0, initval=1)
        rate = pm.math.switch(switchpoint >= years, early_rate, late_rate)
        with pytest.warns(ImputationWarning):
            disasters = pm.Poisson("disasters", rate, observed=disaster_data)

    return disaster_model, years


@pytest.mark.filterwarnings("error")
def test_marginalized_bernoulli_logp():
    """Test logp of IR TestFiniteMarginalDiscreteRV directly"""
    mu = pt.vector("mu")

    idx = pm.Bernoulli.dist(0.7, name="idx")
    y = pm.Normal.dist(mu=mu[idx], sigma=1.0, name="y")
    marginal_rv_node = FiniteDiscreteMarginalRV([mu], [idx, y], ndim_supp=None, n_updates=0,)(
        mu
    )[0].owner

    y_vv = y.clone()
    (logp,) = _logprob(
        marginal_rv_node.op,
        (y_vv,),
        *marginal_rv_node.inputs,
    )

    ref_logp = pm.logp(pm.NormalMixture.dist(w=[0.3, 0.7], mu=mu, sigma=1.0), y_vv)
    np.testing.assert_almost_equal(
        logp.eval({mu: [-1, 1], y_vv: 2}),
        ref_logp.eval({mu: [-1, 1], y_vv: 2}),
    )


@pytest.mark.filterwarnings("error")
def test_marginalized_basic():
    data = [2] * 5

    with MarginalModel() as m:
        sigma = pm.HalfNormal("sigma")
        idx = pm.Categorical("idx", p=[0.1, 0.3, 0.6])
        mu = pt.switch(
            pt.eq(idx, 0),
            -1.0,
            pt.switch(
                pt.eq(idx, 1),
                0.0,
                1.0,
            ),
        )
        y = pm.Normal("y", mu=mu, sigma=sigma)
        z = pm.Normal("z", y, observed=data)

    m.marginalize([idx])
    assert idx not in m.free_RVs
    assert [rv.name for rv in m.marginalized_rvs] == ["idx"]

    # Test logp
    with pm.Model() as m_ref:
        sigma = pm.HalfNormal("sigma")
        y = pm.NormalMixture("y", w=[0.1, 0.3, 0.6], mu=[-1, 0, 1], sigma=sigma)
        z = pm.Normal("z", y, observed=data)

    test_point = m_ref.initial_point()
    ref_logp = m_ref.compile_logp()(test_point)
    ref_dlogp = m_ref.compile_dlogp([m_ref["y"]])(test_point)

    # Assert we can marginalize and unmarginalize internally non-destructively
    for i in range(3):
        np.testing.assert_almost_equal(
            m.compile_logp()(test_point),
            ref_logp,
        )
        np.testing.assert_almost_equal(
            m.compile_dlogp([m["y"]])(test_point),
            ref_dlogp,
        )


@pytest.mark.filterwarnings("error")
def test_multiple_independent_marginalized_rvs():
    with MarginalModel() as m:
        sigma = pm.HalfNormal("sigma")
        idx1 = pm.Bernoulli("idx1", p=0.75)
        x = pm.Normal("x", mu=idx1, sigma=sigma)
        idx2 = pm.Bernoulli("idx2", p=0.75, shape=(5,))
        y = pm.Normal("y", mu=(idx2 * 2 - 1), sigma=sigma, shape=(5,))

    m.marginalize([idx1, idx2])
    m["x"].owner is not m["y"].owner
    _m = m.clone()._marginalize()
    _m["x"].owner is not _m["y"].owner

    with pm.Model() as m_ref:
        sigma = pm.HalfNormal("sigma")
        x = pm.NormalMixture("x", w=[0.25, 0.75], mu=[0, 1], sigma=sigma)
        y = pm.NormalMixture("y", w=[0.25, 0.75], mu=[-1, 1], sigma=sigma, shape=(5,))

    # Test logp
    test_point = m_ref.initial_point()
    x_logp, y_logp = m.compile_logp(vars=[m["x"], m["y"]], sum=False)(test_point)
    x_ref_log, y_ref_logp = m_ref.compile_logp(vars=[m_ref["x"], m_ref["y"]], sum=False)(test_point)
    np.testing.assert_array_almost_equal(x_logp, x_ref_log.sum())
    np.testing.assert_array_almost_equal(y_logp, y_ref_logp)


@pytest.mark.filterwarnings("error")
def test_multiple_dependent_marginalized_rvs():
    """Test that marginalization works when there is more than one dependent RV"""
    with MarginalModel() as m:
        sigma = pm.HalfNormal("sigma")
        idx = pm.Bernoulli("idx", p=0.75)
        x = pm.Normal("x", mu=idx, sigma=sigma)
        y = pm.Normal("y", mu=(idx * 2 - 1), sigma=sigma, shape=(5,))

    ref_logp_x_y_fn = m.compile_logp([idx, x, y])

    with pytest.warns(UserWarning, match="There are multiple dependent variables"):
        m.marginalize([idx])

    m["x"].owner is not m["y"].owner
    _m = m.clone()._marginalize()
    _m["x"].owner is _m["y"].owner

    tp = m.initial_point()
    ref_logp_x_y = logsumexp([ref_logp_x_y_fn({**tp, **{"idx": idx}}) for idx in (0, 1)])
    logp_x_y = m.compile_logp([x, y])(tp)
    np.testing.assert_array_almost_equal(logp_x_y, ref_logp_x_y)


@pytest.mark.filterwarnings("error")
def test_nested_marginalized_rvs():
    """Test that marginalization works when there are nested marginalized RVs"""

    with MarginalModel() as m:
        sigma = pm.HalfNormal("sigma")

        idx = pm.Bernoulli("idx", p=0.75)
        dep = pm.Normal("dep", mu=pt.switch(pt.eq(idx, 0), -1000, 1000), sigma=sigma)

        sub_idx = pm.Bernoulli("sub_idx", p=pt.switch(pt.eq(idx, 0), 0.15, 0.95), shape=(5,))
        sub_dep = pm.Normal("sub_dep", mu=dep + sub_idx * 100, sigma=sigma, shape=(5,))

    ref_logp_fn = m.compile_logp(vars=[idx, dep, sub_idx, sub_dep])

    with pytest.warns(UserWarning, match="There are multiple dependent variables"):
        m.marginalize([idx, sub_idx])

    assert set(m.marginalized_rvs) == {idx, sub_idx}

    # Test logp
    test_point = m.initial_point()
    test_point["dep"] = 1000
    test_point["sub_dep"] = np.full((5,), 1000 + 100)

    ref_logp = [
        ref_logp_fn({**test_point, **{"idx": idx, "sub_idx": np.array(sub_idxs)}})
        for idx in (0, 1)
        for sub_idxs in itertools.product((0, 1), repeat=5)
    ]
    logp = m.compile_logp(vars=[dep, sub_dep])(test_point)

    np.testing.assert_almost_equal(
        logp,
        logsumexp(ref_logp),
    )


@pytest.mark.filterwarnings("error")
def test_marginalized_change_point_model(disaster_model):
    m, years = disaster_model

    ip = m.initial_point()
    ip.pop("switchpoint")
    ref_logp_fn = m.compile_logp(
        [m["switchpoint"], m["disasters_observed"], m["disasters_missing"]]
    )
    ref_logp = logsumexp([ref_logp_fn({**ip, **{"switchpoint": year}}) for year in years])

    with pytest.warns(UserWarning, match="There are multiple dependent variables"):
        m.marginalize(m["switchpoint"])

    logp = m.compile_logp([m["disasters_observed"], m["disasters_missing"]])(ip)
    np.testing.assert_almost_equal(logp, ref_logp)


@pytest.mark.slow
@pytest.mark.filterwarnings("error")
def test_marginalized_change_point_model_sampling(disaster_model):
    m, _ = disaster_model

    rng = np.random.default_rng(211)

    with m:
        before_marg = pm.sample(chains=2, random_seed=rng).posterior.stack(sample=("draw", "chain"))

    with pytest.warns(UserWarning, match="There are multiple dependent variables"):
        m.marginalize([m["switchpoint"]])

    with m:
        after_marg = pm.sample(chains=2, random_seed=rng).posterior.stack(sample=("draw", "chain"))

    np.testing.assert_allclose(
        before_marg["early_rate"].mean(), after_marg["early_rate"].mean(), rtol=1e-2
    )
    np.testing.assert_allclose(
        before_marg["late_rate"].mean(), after_marg["late_rate"].mean(), rtol=1e-2
    )
    np.testing.assert_allclose(
        before_marg["disasters_missing"].mean(), after_marg["disasters_missing"].mean(), rtol=1e-2
    )


@pytest.mark.filterwarnings("error")
def test_not_supported_marginalized():
    """Marginalized graphs with non-Elemwise Operations are not supported as they
    would violate the batching logp assumption"""
    mu = pt.constant([-1, 1])

    # Allowed, as only elemwise operations connect idx to y
    with MarginalModel() as m:
        p = pm.Beta("p", 1, 1)
        idx = pm.Bernoulli("idx", p=p, size=2)
        y = pm.Normal("y", mu=pm.math.switch(idx, 0, 1))
        m.marginalize([idx])

    # ALlowed, as index operation does not connext idx to y
    with MarginalModel() as m:
        p = pm.Beta("p", 1, 1)
        idx = pm.Bernoulli("idx", p=p, size=2)
        y = pm.Normal("y", mu=pm.math.switch(idx, mu[0], mu[1]))
        m.marginalize([idx])

    # Not allowed, as index operation  connects idx to y
    with MarginalModel() as m:
        p = pm.Beta("p", 1, 1)
        idx = pm.Bernoulli("idx", p=p, size=2)
        # Not allowed
        y = pm.Normal("y", mu=mu[idx])
        with pytest.raises(NotImplementedError):
            m.marginalize(idx)

    # Not allowed, as index operation  connects idx to y, even though there is a
    # pure Elemwise connection between the two
    with MarginalModel() as m:
        p = pm.Beta("p", 1, 1)
        idx = pm.Bernoulli("idx", p=p, size=2)
        y = pm.Normal("y", mu=mu[idx] + idx)
        with pytest.raises(NotImplementedError):
            m.marginalize(idx)

    # Multivariate dependent RVs not supported
    with MarginalModel() as m:
        x = pm.Bernoulli("x", p=0.7)
        y = pm.Dirichlet("y", a=pm.math.switch(x, [1, 1, 1], [10, 10, 10]))
        with pytest.raises(
            NotImplementedError,
            match="Marginalization of withe dependent Multivariate RVs not implemented",
        ):
            m.marginalize(x)


@pytest.mark.filterwarnings("error")
def test_marginalized_deterministic_and_potential():
    with MarginalModel() as m:
        x = pm.Bernoulli("x", p=0.7)
        y = pm.Normal("y", x)
        z = pm.Normal("z", x)
        det = pm.Deterministic("det", y + z)
        pot = pm.Potential("pot", y + z + 1)

    with pytest.warns(UserWarning, match="There are multiple dependent variables"):
        m.marginalize([x])

    y_draw, z_draw, det_draw, pot_draw = pm.draw([y, z, det, pot], draws=5)
    np.testing.assert_almost_equal(y_draw + z_draw, det_draw)
    np.testing.assert_almost_equal(det_draw, pot_draw - 1)

    y_value = m.rvs_to_values[y]
    z_value = m.rvs_to_values[z]
    det_value, pot_value = m.replace_rvs_by_values([det, pot])
    assert set(inputvars([det_value, pot_value])) == {y_value, z_value}
    assert det_value.eval({y_value: 2, z_value: 5}) == 7
    assert pot_value.eval({y_value: 2, z_value: 5}) == 8


@pytest.mark.filterwarnings("error")
def test_not_supported_marginalized_deterministic_and_potential():
    with MarginalModel() as m:
        x = pm.Bernoulli("x", p=0.7)
        y = pm.Normal("y", x)
        det = pm.Deterministic("det", x + y)

    with pytest.raises(
        NotImplementedError, match="Cannot marginalize x due to dependent Deterministic det"
    ):
        m.marginalize([x])

    with MarginalModel() as m:
        x = pm.Bernoulli("x", p=0.7)
        y = pm.Normal("y", x)
        pot = pm.Potential("pot", x + y)

    with pytest.raises(
        NotImplementedError, match="Cannot marginalize x due to dependent Potential pot"
    ):
        m.marginalize([x])


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "transform, expected_warning",
    (
        (None, does_not_warn()),
        (UNSET, does_not_warn()),
        (transforms.log, does_not_warn()),
        (transforms.Chain([transforms.log, transforms.logodds]), does_not_warn()),
        (
            transforms.Interval(0, 1),
            pytest.warns(
                UserWarning, match="which depends on the marginalized idx may no longer work"
            ),
        ),
        (
            transforms.Chain([transforms.log, transforms.Interval(0, 1)]),
            pytest.warns(
                UserWarning, match="which depends on the marginalized idx may no longer work"
            ),
        ),
    ),
)
def test_marginalized_transforms(transform, expected_warning):
    w = [0.1, 0.3, 0.6]
    data = [0, 5, 10]
    initval = 0.5  # Value that will be negative on the unconstrained space

    with pm.Model() as m_ref:
        sigma = pm.Mixture(
            "sigma",
            w=w,
            comp_dists=pm.HalfNormal.dist([1, 2, 3]),
            initval=initval,
            transform=transform,
        )
        y = pm.Normal("y", 0, sigma, observed=data)

    with MarginalModel() as m:
        idx = pm.Categorical("idx", p=w)
        sigma = pm.HalfNormal(
            "sigma",
            pt.switch(
                pt.eq(idx, 0),
                1,
                pt.switch(
                    pt.eq(idx, 1),
                    2,
                    3,
                ),
            ),
            initval=initval,
            transform=transform,
        )
        y = pm.Normal("y", 0, sigma, observed=data)

    with expected_warning:
        m.marginalize([idx])

    ip = m.initial_point()
    if transform is not None:
        if transform is UNSET:
            transform_name = "log"
        else:
            transform_name = transform.name
        assert f"sigma_{transform_name}__" in ip
    np.testing.assert_allclose(m.compile_logp()(ip), m_ref.compile_logp()(ip))
