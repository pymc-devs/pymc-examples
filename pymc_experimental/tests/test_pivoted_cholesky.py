# try:
#     import gpytorch
#     import torch
# except ImportError as e:
#     # print(
#     #     f"Please install Pytorch and GPyTorch to use this pivoted Cholesky implementation. Error {e}"
#     # )
#     pass
# import numpy as np
#
# import pymc_experimental as pmx
#
#
# def test_match_gpytorch_linearcg_output():
#     N = 10
#     rank = 5
#     np.random.seed(1234)  # nans with seed 1234
#     K = np.random.randn(N, N)
#     K = K @ K.T + N * np.eye(N)
#     K_torch = torch.from_numpy(K)
#
#     L_gpt = gpytorch.pivoted_cholesky(K_torch, rank=rank, error_tol=1e-3)
#     L_np, _ = pmx.utils.pivoted_cholesky(K, max_iter=rank, error_tol=1e-3)
#     assert np.allclose(L_gpt, L_np.T)
