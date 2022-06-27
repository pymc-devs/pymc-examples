import aesara
import pymc_experimental as pmx
import aesara.tensor as at
import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sparse", [True, False])
def test_spline_construction(dtype, sparse):
    x = np.linspace(0, 1, 20, dtype=dtype)
    np_out = pmx.utils.spline.numpy_bspline_basis(x, 10, 3)
    assert np_out.shape == (20, 10)
    assert np_out.dtype == dtype
    spline_op = pmx.utils.spline.BSplineBasis(sparse=sparse)
    out = spline_op(x, at.constant(10), at.constant(3))
    if not sparse:
        assert isinstance(out.type, at.TensorType)
    else:
        assert isinstance(out.type, aesara.sparse.SparseTensorType)
    B = out.eval()
    if not sparse:
        np.testing.assert_allclose(B, np_out)
    else:
        np.testing.assert_allclose(B.todense(), np_out)
    assert B.shape == (20, 10)


@pytest.mark.parametrize("shape", [(100,), (100, 5)])
@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("points", [dict(n=1001), dict(eval_points=np.linspace(0, 1, 1001))])
def test_interpolation_api(shape, sparse, points):
    x = np.random.randn(*shape)
    yt = pmx.utils.spline.bspline_interpolation(x, **points, sparse=sparse)
    y = yt.eval()
    assert y.shape == (1001, *shape[1:])


@pytest.mark.parametrize(
    "params",
    [
        (dict(sparse="foo", n=100, degree=1), TypeError, "sparse should be True or False"),
        (dict(n=100, degree=0.5), TypeError, "degree should be integer"),
        (
            dict(n=100, eval_points=np.linspace(0, 1), degree=1),
            ValueError,
            "Please provide one of n or eval_points",
        ),
        (
            dict(degree=1),
            ValueError,
            "Please provide one of n or eval_points",
        ),
    ],
)
def test_bad_calls(params):
    kw, E, err = params
    x = np.random.randn(10)
    with pytest.raises(E, match=err):
        pmx.utils.spline.bspline_interpolation(x, **kw)
