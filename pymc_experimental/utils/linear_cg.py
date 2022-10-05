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

EVAL_CG_TOLERANCE = 0.01
CG_TOLERANCE = 1


def masked_fill(vector, mask, fill_value):
    masked_vector = np.ma.array(vector, mask=mask)
    vector = masked_vector.filled(fill_value=fill_value)
    return vector


def linear_cg_updates(
    result, alpha, residual_inner_prod, eps, beta, residual, precond_residual, curr_conjugate_vec
):

    # Everything inside _jit_linear_cg_updates
    result = result + alpha * curr_conjugate_vec
    beta = np.copy(residual_inner_prod)

    residual_inner_prod = residual.T @ precond_residual

    # safe division
    is_zero = beta < eps
    beta = masked_fill(beta, mask=is_zero, fill_value=1)

    beta = residual_inner_prod / beta
    beta = masked_fill(beta, mask=is_zero, fill_value=0)
    curr_conjugate_vec = beta * curr_conjugate_vec + precond_residual
    return (
        result,
        alpha,
        residual_inner_prod,
        eps,
        beta,
        residual,
        precond_residual,
        curr_conjugate_vec,
    )


def linear_cg(
    mat: np.matrix,
    rhs,
    n_tridiag=0,
    tolerance=None,
    eps=1e-10,
    stop_updating_after=1e-10,
    max_iter=1000,
    max_tridiag_iter=20,
    initial_guess=None,
    preconditioner=None,
    terminate_cg_by_size=False,
    use_eval_tolerange=False,
):

    if initial_guess is None:
        initial_guess = np.zeros_like(rhs)

    if preconditioner is None:
        preconditioner = lambda x: x
        precond = False
    else:
        precond = True

    if tolerance is None:
        if use_eval_tolerance:
            tolerance = EVAL_CG_TOLERANCE
        else:
            tolerance = CG_TOLERANCE

    # If we are running m CG iterations, we obviously can't get more than m Lanczos coefficients
    if max_tridiag_iter > max_iter:
        raise RuntimeError(
            "Getting a tridiagonalization larger than the number of CG iterations run is not possible!"
        )

    is_vector = len(rhs.shape) == 1
    if is_vector:
        rhs = rhs[:, np.newaxis]

    num_rows = rhs.size
    n_iter = min(max_iter, num_rows) if terminate_cg_by_size else max_iter
    n_tridiag_iter = min(max_tridiag_iter, num_rows)

    # norm of rhs for convergence tests
    rhs_norm = np.linalg.norm(rhs, 2)
    # make almost-zero norms be 1 (so we don't get divide-by-zero errors)
    rhs_is_zero = rhs_norm < eps
    rhs_norm = masked_fill(rhs_norm, mask=rhs_is_zero, fill_value=1)

    # lets normalize rhs
    rhs = rhs / rhs_norm

    # residuals
    residual = rhs - mat @ initial_guess
    batch_shape = residual.shape[:-2]

    result = np.copy(initial_guess)

    if not np.allclose(residual, residual):
        raise RuntimeError("NaNs encountered when trying to perform matrix-vector multiplication")

    # sometimes we are lucky and preconditioner solves the system right away
    # check for convergence
    residual_norm = np.linalg.norm(residual, 2)
    has_converged = residual_norm < stop_updating_after

    if has_converged.all() and not n_tridiag:
        n_iter = 0  # skip iterations
    else:
        precond_residual = preconditioner(residual)

        curr_conjugate_vec = precond_residual
        residual_inner_prod = residual.T @ precond_residual

        # define storage matrices
        mul_storage = np.zeros_like(residual)
        alpha = np.zeros((*batch_shape, 1, rhs.shape[-1]))
        beta = np.zeros_like(alpha)
        is_zero = np.zeros((*batch_shape, 1, rhs.shape[-1]))

    # Define tridiagonal matrices if applicable
    if n_tridiag:
        t_mat = np.zeros((n_tridiag_iter, n_tridiag_iter, *batch_shape, n_tridiag))
        alpha_tridiag_is_zero = np.zeros(*batch_shape, n_tridiag)
        alpha_reciprocal = np.zeros(*batch_shape, n_tridiag)
        prev_alpha_reciprocal = np.zeros_like(alpha_reciprocal)
        prev_beta = np.zeros_like(alpha_reciprocal)

    update_tridiag = True
    last_tridiag_iter = 0

    # it is possible that we don't reach tolerance even after all the iterations are over
    tolerance_reached = False

    # start iteration
    for k in range(n_iter):
        mvms = mat @ curr_conjugate_vec
        if precond:
            alpha = curr_conjugate_vec @ mvms  # scalar

            # safe division
            is_zero = alpha < eps
            alpha = masked_fill(alpha, mask=is_zero, fill_value=1)
            alpha = residual_inner_prod / alpha
            alpha = masked_fill(alpha, mask=is_zero, fill_value=0)

            # cancel out updates by setting directions which have converged to zero
            alpha = masked_fill(alpha, mask=has_converged, fill_value=0)
            residual = residual - alpha * mvms

            # update precond_residual
            precond_residual = preconditioner(residual)

            # Everything inside _jit_linear_cg_updates
            (
                result,
                alpha,
                residual_inner_prod,
                eps,
                beta,
                residual,
                precond_residual,
                curr_conjugate_vec,
            ) = linear_cg_updates(
                result,
                alpha,
                residual_inner_prod,
                eps,
                beta,
                residual,
                precond_residual,
                curr_conjugate_vec,
            )

        else:
            # everything inside _jit_linear_cg_updates_no_precond
            alpha = curr_conjugate_vec.T @ mvms

            # safe division
            is_zero = alpha < eps
            alpha = masked_fill(alpha, mask=is_zero, fill_value=1)
            alpha = residual_inner_prod / alpha
            alpha = masked_fill(alpha, is_zero, fill_value=0)

            alpha = masked_fill(alpha, has_converged, fill_value=0)  # <- I'm here
            residual = residual - alpha * mvms
            precond_residual = np.copy(residual)

            (
                result,
                alpha,
                residual_inner_prod,
                eps,
                beta,
                residual,
                precond_residual,
                curr_conjugate_vec,
            ) = linear_cg_updates(
                result,
                alpha,
                residual_inner_prod,
                eps,
                beta,
                residual,
                precond_residual,
                curr_conjugate_vec,
            )

        residual_norm = np.linalg.norm(residual, 2)
        residual_norm = masked_fill(residual_norm, mask=rhs_is_zero, fill_value=0)
        has_converged = residual_norm < stop_updating_after

        if (
            k >= min(10, max_iter - 1)
            and bool(residual_norm.mean() < tolerance)
            and not (n_tridiag and k < min(n_tridiag_iter, max_iter - 1))
        ):
            tolerance_reached = True
            break

        # Update tridiagonal matrices, if applicable
        if n_tridiag and k < n_tridiag_iter and update_tridiag:
            alpha_tridiag = np.copy(alpha)
            beta_tridiag = np.copy(beta)

            alpha_tridiag_is_zero = alpha_tridiag == 0
            alpha_tridiag = masked_fill(alpha_tridiag, mask=alpha_tridiag_is_zero, fill_value=1)
            alpha_reciprocal = 1 / alpha_tridiag
            alpha_tridiag = masked_fill(alpha_tridiag, mask=alpha_tridiag_is_zero, fill_value=0)

            if k == 0:
                t_mat[k, k] = alpha_reciprocal
            else:
                t_mat[k, k] += np.squeeze(alpha_reciprocal + prev_beta * prev_alpha_reciprocal)
                t_mat[k, k - 1] = np.sqrt(prev_beta) * prev_alpha_reciprocal
                t_mat[k - 1, k] = np.copy(t_mat[k, k - 1])

                if t_mat[k - 1, k].max() < 1e-6:
                    update_tridiag = False

            last_tridiag_iter = k

            prev_alpha_reciprocal = np.copy(alpha_reciprocal)
            prev_beta = np.copy(beta_tridiag)

    # Un-normalize
    result = result * rhs_norm
    if not tolerance_reached and n_iter > 0:
        raise RuntimeError(
            "CG terminated in {} iterations with average residual norm {}"
            " which is larger than the tolerance of {} specified by"
            " gpytorch.settings.cg_tolerance."
            " If performance is affected, consider raising the maximum number of CG iterations by running code in"
            " a gpytorch.settings.max_cg_iterations(value) context.".format(
                k + 1, residual_norm.mean(), tolerance
            )
        )

    if n_tridiag:
        t_mat = t_mat[: last_tridiag_iter + 1, : last_tridiag_iter + 1]
        return result, t_mat.transpose(-1, *range(2, 2 + len(batch_shape)), 0, 1)
    else:
        # We set the estimated Lanczos tri-diagonal matrices to be identity so that
        # the subsequent eigen decomposition https://arxiv.org/pdf/1809.11165.pdf (eq.S7)
        # would work fine.
        # t_mat = np.zeros((n_tridiag_iter, n_tridiag_iter, *batch_shape, n_tridiag))
        # Note that after transpose the last two dimensions are dimensions 0 and 1 of the matrix above
        # Which are the same values i.e. n_tridiag_iter
        # So we generate identity matrices of size n_tridiag_iter and repeat them [n_iter, *range(2, 2+len(batch_shape))] times
        # TODO: for same input, n_tridiag = True and n_tridiag = False must produce t_mat with same shape (with assumed n_tridiag=1)
        n_tridiag = 1
        eye = np.eye(n_tridiag_iter)
        t_mat_eye = np.tile(eye, [n_tridiag] + [1] * (len(batch_shape) + 2))
        return result, t_mat_eye
