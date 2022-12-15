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


from typing import Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import arviz
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.logprob.transforms import RVTransform


class ParamCfg(TypedDict):
    name: str
    transform: Optional[RVTransform]
    dims: Optional[Union[str, Tuple[str]]]


class ShapeInfo(TypedDict):
    # shape might not match slice due to a transform
    shape: Tuple[int]  # transformed shape
    slice: slice


class VarInfo(TypedDict):
    sinfo: ShapeInfo
    vinfo: ParamCfg


class FlatInfo(TypedDict):
    data: np.ndarray
    info: List[VarInfo]


def _arg_to_param_cfg(key, value: Optional[Union[ParamCfg, RVTransform, str, Tuple]] = None):
    if value is None:
        cfg = ParamCfg(name=key, transform=None, dims=None)
    elif isinstance(value, Tuple):
        cfg = ParamCfg(name=key, transform=None, dims=value)
    elif isinstance(value, str):
        cfg = ParamCfg(name=value, transform=None, dims=None)
    elif isinstance(value, RVTransform):
        cfg = ParamCfg(name=key, transform=value, dims=None)
    else:
        cfg = value.copy()
        cfg.setdefault("name", key)
        cfg.setdefault("transform", None)
        cfg.setdefault("dims", None)
    return cfg


def _parse_args(
    var_names: Sequence[str], **kwargs: Union[ParamCfg, RVTransform, str, Tuple]
) -> Dict[str, ParamCfg]:
    results = dict()
    for var in var_names:
        results[var] = _arg_to_param_cfg(var)
    for key, val in kwargs.items():
        results[key] = _arg_to_param_cfg(key, val)
    return results


def _flatten(idata: arviz.InferenceData, **kwargs: ParamCfg) -> FlatInfo:
    posterior = idata.posterior
    vars = list()
    info = list()
    begin = 0
    for key, cfg in kwargs.items():
        data = (
            posterior[key]
            # combine all draws from all chains
            .stack(__sample__=["chain", "draw"])
            # move sample dim to the first position
            # no matter where it was before
            .transpose("__sample__", ...)
            # we need numpy data for all the rest functionality
            .values
        )
        # omitting __sample__
        # we need shape in the untransformed space
        if cfg["transform"] is not None:
            # some transforms need original shape
            data = cfg["transform"].forward(data).eval()
        shape = data.shape[1:]
        # now we can get rid of shape
        data = data.reshape(data.shape[0], -1)
        end = begin + data.shape[1]
        vars.append(data)
        sinfo = dict(shape=shape, slice=slice(begin, end))
        info.append(dict(sinfo=sinfo, vinfo=cfg))
        begin = end
    return dict(data=np.concatenate(vars, axis=-1), info=info)


def _mean_chol(flat_array: np.ndarray):
    mean = flat_array.mean(0)
    cov = np.cov(flat_array, rowvar=False)
    cov = np.atleast_2d(cov)
    chol = np.linalg.cholesky(cov)
    return mean, chol


def _mvn_prior_from_flat_info(name, flat_info: FlatInfo):
    mean, chol = _mean_chol(flat_info["data"])
    base_dist = pm.Normal(name, np.zeros_like(mean))
    interim = mean + chol @ base_dist
    result = dict()
    for var_info in flat_info["info"]:
        sinfo = var_info["sinfo"]
        vinfo = var_info["vinfo"]
        var = interim[sinfo["slice"]].reshape(sinfo["shape"])
        if vinfo["transform"] is not None:
            var = vinfo["transform"].backward(var)
        var = pm.Deterministic(vinfo["name"], var, dims=vinfo["dims"])
        result[vinfo["name"]] = var
    return result


def prior_from_idata(
    idata: arviz.InferenceData,
    name="trace_prior_",
    *,
    var_names: Sequence[str],
    **kwargs: Union[ParamCfg, RVTransform, str, Tuple]
) -> Dict[str, pt.TensorVariable]:
    """
    Create a prior from posterior using MvNormal approximation.

    The approximation uses MvNormal distribution.
    Keep in mind that this function will only work well for unimodal
    posteriors and will fail when complicated interactions happen.

    Moreover, if a retrieved variable is constrained, you
    should specify a transform for the variable, e.g.
    ``pymc.distributions.transforms.log`` for standard
    deviation posterior.

    Parameters
    ----------
    idata: arviz.InferenceData
        Inference data with posterior group
    var_names: Sequence[str]
        names of variables to take as is from the posterior
    kwargs: Union[ParamCfg, RVTransform, str, Tuple]
        names of variables with additional configuration, see more in Examples

    Examples
    --------
    >>> import pymc as pm
    >>> import pymc.distributions.transforms as transforms
    >>> import numpy as np
    >>> with pm.Model(coords=dict(test=range(4), options=range(3))) as model1:
    ...     a = pm.Normal("a")
    ...     b = pm.Normal("b", dims="test")
    ...     c = pm.HalfNormal("c")
    ...     d = pm.Normal("d")
    ...     e = pm.Normal("e")
    ...     f = pm.Dirichlet("f", np.ones(3), dims="options")
    ...     trace = pm.sample(progressbar=False)

    You can reuse the posterior in the new model.

    >>> with pm.Model(coords=dict(test=range(4), options=range(3))) as model2:
    ...     priors = prior_from_idata(
    ...         trace,                  # the old trace (posterior)
    ...         var_names=["a", "d"],   # take variables as is
    ...
    ...         e="new_e",              # assign new name "new_e" for a variable
    ...                                 # similar to dict(name="new_e")
    ...
    ...         b=("test", ),           # set a dim to "test"
    ...                                 # similar to dict(dims=("test", ))
    ...
    ...         c=transforms.log,       # apply log transform to a positive variable
    ...                                 # similar to dict(transform=transforms.log)
    ...
    ...                                 # set a name, assign a dim and apply simplex transform
    ...         f=dict(name="new_f", dims="options", transform=transforms.simplex)
    ...     )
    ...     trace1 = pm.sample_prior_predictive(100)
    """
    param_cfg = _parse_args(var_names=var_names, **kwargs)
    flat_info = _flatten(idata, **param_cfg)
    return _mvn_prior_from_flat_info(name, flat_info)
