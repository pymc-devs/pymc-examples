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


import arviz as az
import numpy as np
import pymc as pm
import pytest
from pymc.distributions import transforms

import pymc_experimental as pmx


@pytest.mark.parametrize(
    "case",
    [
        (("a", dict(name="b")), dict(name="b", transform=None, dims=None)),
        (("a", None), dict(name="a", transform=None, dims=None)),
        (("a", transforms.log), dict(name="a", transform=transforms.log, dims=None)),
        (
            ("a", dict(transform=transforms.log)),
            dict(name="a", transform=transforms.log, dims=None),
        ),
        (("a", dict(name="b")), dict(name="b", transform=None, dims=None)),
        (("a", dict(name="b", dims="test")), dict(name="b", transform=None, dims="test")),
        (("a", ("test",)), dict(name="a", transform=None, dims=("test",))),
    ],
)
def test_parsing_arguments(case):
    inp, out = case
    test = pmx.utils.prior._arg_to_param_cfg(*inp)
    assert test == out


@pytest.fixture
def coords():
    return dict(test=range(3), simplex=range(4))


@pytest.fixture(
    params=[
        [
            ("t",),
            dict(
                a="d",
                b=dict(transform=transforms.log, dims=("test",)),
                c=dict(transform=transforms.simplex, dims=("simplex",)),
            ),
        ],
        [("t",), dict()],
    ]
)
def user_param_cfg(request):
    return request.param


@pytest.fixture
def param_cfg(user_param_cfg):
    return pmx.utils.prior._parse_args(user_param_cfg[0], **user_param_cfg[1])


@pytest.fixture
def transformed_data(param_cfg, coords):
    vars = dict()
    for k, cfg in param_cfg.items():
        if cfg["dims"] is not None:
            extra_dims = [len(coords[d]) for d in cfg["dims"]]
            if cfg["transform"] is not None:
                t = np.random.randn(*extra_dims)
                extra_dims = tuple(cfg["transform"].forward(t).shape.eval())
        else:
            extra_dims = []
        orig = np.random.randn(4, 100, *extra_dims)
        vars[k] = orig
    return vars


@pytest.fixture
def idata(transformed_data, param_cfg):
    vars = dict()
    for k, orig in transformed_data.items():
        cfg = param_cfg[k]
        if cfg["transform"] is not None:
            var = cfg["transform"].backward(orig).eval()
        else:
            var = orig
        assert not np.isnan(var).any()
        vars[k] = var
    return az.convert_to_inference_data(vars)


def test_idata_for_tests(idata, param_cfg):
    assert set(idata.posterior.keys()) == set(param_cfg)
    assert len(idata.posterior.coords["chain"]) == 4
    assert len(idata.posterior.coords["draw"]) == 100


def test_args_compose():
    cfg = pmx.utils.prior._parse_args(
        var_names=["a"],
        b=("test",),
        c=transforms.log,
        d="e",
        f=dict(dims="test"),
        g=dict(name="h", dims="test", transform=transforms.log),
    )
    assert cfg == dict(
        a=dict(name="a", dims=None, transform=None),
        b=dict(name="b", dims=("test",), transform=None),
        c=dict(name="c", dims=None, transform=transforms.log),
        d=dict(name="e", dims=None, transform=None),
        f=dict(name="f", dims="test", transform=None),
        g=dict(name="h", dims="test", transform=transforms.log),
    )


def test_transform_idata(transformed_data, idata, param_cfg):
    flat_info = pmx.utils.prior._flatten(idata, **param_cfg)
    expected_shape = 0
    for v in transformed_data.values():
        expected_shape += int(np.prod(v.shape[2:]))
    assert flat_info["data"].shape[1] == expected_shape
    assert len(flat_info["info"]) == len(param_cfg)
    assert "sinfo" in flat_info["info"][0]
    assert "vinfo" in flat_info["info"][0]


@pytest.fixture
def flat_info(idata, param_cfg):
    return pmx.utils.prior._flatten(idata, **param_cfg)


def test_mean_chol(flat_info):
    mean, chol = pmx.utils.prior._mean_chol(flat_info["data"])
    assert mean.shape == (flat_info["data"].shape[1],)
    assert chol.shape == (flat_info["data"].shape[1],) * 2


def test_mvn_prior_from_flat_info(flat_info, coords, param_cfg):
    with pm.Model(coords=coords) as model:
        priors = pmx.utils.prior._mvn_prior_from_flat_info("trace_prior_", flat_info)
        test_prior = pm.sample_prior_predictive(1)
    names = [p["name"] for p in param_cfg.values()]
    assert set(model.named_vars) == {"trace_prior_", *names}


def test_prior_from_idata(idata, user_param_cfg, coords, param_cfg):
    with pm.Model(coords=coords) as model:
        priors = pmx.utils.prior.prior_from_idata(
            idata, var_names=user_param_cfg[0], **user_param_cfg[1]
        )
        test_prior = pm.sample_prior_predictive(1)
    names = [p["name"] for p in param_cfg.values()]
    assert set(model.named_vars) == {"trace_prior_", *names}
