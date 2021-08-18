<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> cb0c201 (latest ZeroSumNormal code, pymc3 v3, random seed for sampling)
from typing import List

try:
    import aesara.tensor as aet
except ImportError:
    import theano.tensor as aet

<<<<<<< HEAD
import numpy as np
import pymc3 as pm
from scipy import stats
from pymc3.distributions.distribution import generate_samples, draw_values

def extend_axis_aet(array, axis):
    n = array.shape[axis] + 1
    sum_vals = array.sum(axis, keepdims=True)
    norm = sum_vals / (np.sqrt(n) + n)
    fill_val = norm - sum_vals / np.sqrt(n)
    
    out = aet.concatenate([array, fill_val.astype(str(array.dtype))], axis=axis)
    return out - norm.astype(str(array.dtype))


def extend_axis_rev_aet(array: np.ndarray, axis: int):
    if axis < 0:
        axis = axis % array.ndim
    assert axis >= 0 and axis < array.ndim

    n = array.shape[axis]
    last = aet.take(array, [-1], axis=axis)
    
    sum_vals = -last * np.sqrt(n)
    norm = sum_vals / (np.sqrt(n) + n)
    slice_before = (slice(None, None),) * axis
    return array[slice_before + (slice(None, -1),)] + norm.astype(str(array.dtype))


def extend_axis(array, axis):
    n = array.shape[axis] + 1
    sum_vals = array.sum(axis, keepdims=True)
    norm = sum_vals / (np.sqrt(n) + n)
    fill_val = norm - sum_vals / np.sqrt(n)
    
    out = np.concatenate([array, fill_val.astype(str(array.dtype))], axis=axis)
    return out - norm.astype(str(array.dtype))


def extend_axis_rev(array, axis):
    n = array.shape[axis]
    last = np.take(array, [-1], axis=axis)
    
    sum_vals = -last * np.sqrt(n)
    norm = sum_vals / (np.sqrt(n) + n)
    slice_before = (slice(None, None),) * len(array.shape[:axis])
    return array[slice_before + (slice(None, -1),)] + norm.astype(str(array.dtype))


class ZeroSumTransform(pm.distributions.transforms.Transform):
    name = "zerosum"
    
    _active_dims: List[int]
    
    def __init__(self, active_dims):
        self._active_dims = active_dims
    
    def forward(self, x):
        for axis in self._active_dims:
            x = extend_axis_rev_aet(x, axis=axis)
        return x
    
    def forward_val(self, x, point=None):
        for axis in self._active_dims:
            x = extend_axis_rev(x, axis=axis)
        return x
    
    def backward(self, z):
        z = aet.as_tensor_variable(z)
        for axis in self._active_dims:
            z = extend_axis_aet(z, axis=axis)
        return z
    
    def jacobian_det(self, x):
        return aet.constant(0.)
    
    
class ZeroSumNormal(pm.Continuous):
    def __init__(self, sigma=1, *, active_dims=None, active_axes=None, **kwargs):
        shape = kwargs.get("shape", ())
        dims = kwargs.get("dims", None)
        if isinstance(shape, int):
            shape = (shape,)
        
        if isinstance(dims, str):
            dims = (dims,)

        self.mu = self.median = self.mode = aet.zeros(shape)
        self.sigma = aet.as_tensor_variable(sigma)
        
        if active_dims is None and active_axes is None:
            if shape:
                active_axes = (-1,)
            else:
                active_axes = ()
        
        if isinstance(active_axes, int):
            active_axes = (active_axes,)
        
        if isinstance(active_dims, str):
            active_dims = (active_dims,)
        
        if active_axes is not None and active_dims is not None:
            raise ValueError("Only one of active_axes and active_dims can be specified.")
        
        if active_dims is not None:
            model = pm.modelcontext(None)
            print(model.RV_dims)
            if dims is None:
                raise ValueError("active_dims can only be used with the dims kwargs.")
            active_axes = []
            for dim in active_dims:
                active_axes.append(dims.index(dim))
        
        super().__init__(**kwargs, transform=ZeroSumTransform(active_axes))

    def logp(self, x):
        return pm.Normal.dist(sigma=self.sigma).logp(x)
    
    @staticmethod
    def _random(scale, size):
        samples = stats.norm.rvs(loc=0, scale=scale, size=size)
        return samples - np.mean(samples, axis=-1, keepdims=True)
    
    def random(self, point=None, size=None):
        sigma, = draw_values([self.sigma], point=point, size=size)
        return generate_samples(self._random, scale=sigma, dist_shape=self.shape, size=size)

    def _distr_parameters_for_repr(self):
        return ["sigma"]

    def logcdf(self, value):
        raise NotImplementedError()
=======
import pymc3 as pm
=======
>>>>>>> cb0c201 (latest ZeroSumNormal code, pymc3 v3, random seed for sampling)
import numpy as np
import pymc3 as pm
from scipy import stats
from pymc3.distributions.distribution import generate_samples, draw_values

def extend_axis_aet(array, axis):
    n = array.shape[axis] + 1
    sum_vals = array.sum(axis, keepdims=True)
    norm = sum_vals / (np.sqrt(n) + n)
    fill_val = norm - sum_vals / np.sqrt(n)
    
    out = aet.concatenate([array, fill_val.astype(str(array.dtype))], axis=axis)
    return out - norm.astype(str(array.dtype))


def extend_axis_rev_aet(array: np.ndarray, axis: int):
    if axis < 0:
        axis = axis % array.ndim
    assert axis >= 0 and axis < array.ndim

    n = array.shape[axis]
    last = aet.take(array, [-1], axis=axis)
    
    sum_vals = -last * np.sqrt(n)
    norm = sum_vals / (np.sqrt(n) + n)
    slice_before = (slice(None, None),) * axis
    return array[slice_before + (slice(None, -1),)] + norm.astype(str(array.dtype))


def extend_axis(array, axis):
    n = array.shape[axis] + 1
    sum_vals = array.sum(axis, keepdims=True)
    norm = sum_vals / (np.sqrt(n) + n)
    fill_val = norm - sum_vals / np.sqrt(n)
    
    out = np.concatenate([array, fill_val.astype(str(array.dtype))], axis=axis)
    return out - norm.astype(str(array.dtype))


<<<<<<< HEAD
def make_sum_zero_hh(N: int) -> np.ndarray:
    """
    Build a householder transformation matrix that maps e_1 to a vector of all 1s.
    """
    e_1 = np.zeros(N)
    e_1[0] = 1
    a = np.ones(N)
    a /= np.sqrt(a @ a)
    v = a + e_1
    v /= np.sqrt(v @ v)
    return np.eye(N) - 2 * np.outer(v, v)
>>>>>>> 2da3052 (ZeroSumNormal: initial commit)
=======
def extend_axis_rev(array, axis):
    n = array.shape[axis]
    last = np.take(array, [-1], axis=axis)
    
    sum_vals = -last * np.sqrt(n)
    norm = sum_vals / (np.sqrt(n) + n)
    slice_before = (slice(None, None),) * len(array.shape[:axis])
    return array[slice_before + (slice(None, -1),)] + norm.astype(str(array.dtype))


class ZeroSumTransform(pm.distributions.transforms.Transform):
    name = "zerosum"
    
    _active_dims: List[int]
    
    def __init__(self, active_dims):
        self._active_dims = active_dims
    
    def forward(self, x):
        for axis in self._active_dims:
            x = extend_axis_rev_aet(x, axis=axis)
        return x
    
    def forward_val(self, x, point=None):
        for axis in self._active_dims:
            x = extend_axis_rev(x, axis=axis)
        return x
    
    def backward(self, z):
        z = aet.as_tensor_variable(z)
        for axis in self._active_dims:
            z = extend_axis_aet(z, axis=axis)
        return z
    
    def jacobian_det(self, x):
        return aet.constant(0.)
    
    
class ZeroSumNormal(pm.Continuous):
    def __init__(self, sigma=1, *, active_dims=None, active_axes=None, **kwargs):
        shape = kwargs.get("shape", ())
        dims = kwargs.get("dims", None)
        if isinstance(shape, int):
            shape = (shape,)
        
        if isinstance(dims, str):
            dims = (dims,)

        self.mu = self.median = self.mode = aet.zeros(shape)
        self.sigma = aet.as_tensor_variable(sigma)
        
        if active_dims is None and active_axes is None:
            if shape:
                active_axes = (-1,)
            else:
                active_axes = ()
        
        if isinstance(active_axes, int):
            active_axes = (active_axes,)
        
        if isinstance(active_dims, str):
            active_dims = (active_dims,)
        
        if active_axes is not None and active_dims is not None:
            raise ValueError("Only one of active_axes and active_dims can be specified.")
        
        if active_dims is not None:
            model = pm.modelcontext(None)
            print(model.RV_dims)
            if dims is None:
                raise ValueError("active_dims can only be used with the dims kwargs.")
            active_axes = []
            for dim in active_dims:
                active_axes.append(dims.index(dim))
        
        super().__init__(**kwargs, transform=ZeroSumTransform(active_axes))

    def logp(self, x):
        return pm.Normal.dist(sigma=self.sigma).logp(x)
    
    @staticmethod
    def _random(scale, size):
        samples = stats.norm.rvs(loc=0, scale=scale, size=size)
        return samples - np.mean(samples, axis=-1, keepdims=True)
    
    def random(self, point=None, size=None):
        sigma, = draw_values([self.sigma], point=point, size=size)
        return generate_samples(self._random, scale=sigma, dist_shape=self.shape, size=size)

    def _distr_parameters_for_repr(self):
        return ["sigma"]

    def logcdf(self, value):
        raise NotImplementedError()
>>>>>>> cb0c201 (latest ZeroSumNormal code, pymc3 v3, random seed for sampling)
