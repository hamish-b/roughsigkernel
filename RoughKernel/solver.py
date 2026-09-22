import jax.numpy as jnp
import roughpy_jax as rpj
import jax
from roughpy_jax.algebra import  as_free_tensor
from functools import partial
from .utils import (ft_pairs, 
                   sigs_over_intervals, 
                   upper_tri_to_symmetric, 
                   eval_adj, 
                   add_tensor_scalar,
                   get, get_, set_,
                   )


class RoughKernel:
    """
    Solves the signature kernel PDE (Algorithm 5.1 in the rough signature
    kernel PDE paper) for a batch of paths.
    """

    def __init__(self, n):
        """
        n - truncation depth for the log-signature / signature
        """
        self.n = n

    @staticmethod
    @jax.jit
    def initialise_PDE(X_SPTs_zero, Y_SPTs_zero, tensor_basis):

        L, M, _ = X_SPTs_zero.shape # L is no. of intervals, M is the no. of pairs
                                    # we assume L is the same for both X and Y. It would not be
                                    # too difficult to edit this code to allow for distinct
                                    # L_X and L_Y

        K = jnp.zeros((L + 1, L + 1, M), dtype=jnp.float32) 
        K = K.at[0, :, :].set(1)
        K = K.at[:, 0, :].set(1)

        phi = rpj.FreeTensor.zero(basis=tensor_basis, batch_dims=(L+1, L+1, M,))
        psi = rpj.FreeTensor.zero(basis=tensor_basis, batch_dims=(L+1, L+1, M,))

        for i in range(1, L + 1):
            phi = set_(phi, i, 0, get_(X_SPTs_zero, i - 1))

        for j in range(1, L + 1):
            psi = set_(psi, 0, j, get_(Y_SPTs_zero, j - 1))

        return phi, psi, K

    @staticmethod
    @jax.jit
    def compute_phi(xi, xti, phi01, psi01, K00):
        phi11 = phi01 + xti.__mul__(K00) \
            + rpj.ft_mul(phi01, xti) \
            + add_tensor_scalar(
                as_free_tensor(rpj.ft_adjoint_left_mul(psi01, xi)),
                -rpj.tensor_pairing(psi01, xti))
        return phi11

    @staticmethod
    @jax.jit
    def compute_psi(yj, ytj, phi10, psi10, K00):
        psi11 = psi10 + ytj.__mul__(K00) \
            + rpj.ft_mul(psi10, ytj) \
            + add_tensor_scalar(
                as_free_tensor(rpj.ft_adjoint_left_mul(phi10, yj)),
                -rpj.tensor_pairing(phi10, ytj))
        return psi11

    @staticmethod
    @jax.jit
    def compute_K(xi, yj, phi00, phi01, phi10, phi11, psi00, psi01, psi10, psi11, K00, K01, K10):
        eval_adj_ = eval_adj(phi00, psi00, xi, yj)
        next_eval_adj = eval_adj(phi11, psi11, xi, yj)
        temp_2 = eval_adj(phi01, psi01, xi, yj)
        temp_3 = eval_adj(phi10, psi10, xi, yj)

        G = rpj.tensor_pairing(xi, yj)
        f_1 = K00 * G + eval_adj_
        f_2 = K01 * G + temp_2
        f_3 = K10 * G + temp_3

        u_p = K10 + K01 - K00 + f_1
        f_p = u_p * G + next_eval_adj

        K11 = K10 + K01 - K00 + (1. / 4) * (f_1 + f_2 + f_3 + f_p)
        return K11

#-------------------------------------------------------------
# Implements algorithm 5.1 to compute the PDE
#-------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0, 4))
    def partition_compute(self, phi, psi, K, L, xlsps, ylsps, xlspts, ylspts):

        def outer_body(i, carry):
            phi, psi, K = carry

            def inner_body(j, carry):
                phi, psi, K = carry
                xi, yj = get_(xlsps, i), get_(ylsps, j)
                xti, ytj = get_(xlspts, i), get_(ylspts, j)

                phi00, phi01, phi10 = get(phi, i, j), get(phi, i, j + 1), get(phi, i + 1, j)
                psi00, psi01, psi10 = get(psi, i, j), get(psi, i, j + 1), get(psi, i + 1, j)
                K00, K01, K10 = K[i, j], K[i, j + 1], K[i + 1, j]

                phi11 = self.compute_phi(xi, xti, phi01, psi01, K00)
                psi11 = self.compute_psi(yj, ytj, phi10, psi10, K00)
                K11 = self.compute_K(xi, yj, phi00, phi01, phi10, phi11,
                                        psi00, psi01, psi10, psi11, K00, K01, K10)

                phi = set_(phi, i + 1, j + 1, phi11)
                psi = set_(psi, i + 1, j + 1, psi11)
                K = K.at[i + 1, j + 1].set(K11)
                return (phi, psi, K)

            return jax.lax.fori_loop(0, L, inner_body, (phi, psi, K)) # change this L to N?

        phi, psi, K = jax.lax.fori_loop(0, L, outer_body, (phi, psi, K))
        return K   

    # ------------------------------------------------------------------
    # Top-level driver: depends on self.n / self.R (via make_Lie), so
    # it's an instance method.
    # ------------------------------------------------------------------
    
    def solve_PDE(self, intervals, X, Y, pair_batch_size=4096):

        L = len(intervals) 

        X_LIS, Y_LIS, Tensor_Basis = X, Y, X.group_basis
        B1, B2 = X.batch_dims[0], Y.batch_dims[0]

        # list containing every pair (x_i, y_j) s.t. x in X, y in Y. if X = Y, only need B*(B+1)/2
        if X == Y:
            pairs = jnp.stack(jnp.triu_indices(B1), axis=1)
            same = True
        else:
            pairs = jnp.array([(i, j) for i in range(B1) for j in range(B2)])
            same = False

        X_LSPs, X_LSPTs, X_SPTs_zero = sigs_over_intervals(X_LIS, intervals, self.n)
        Y_LSPs, Y_LSPTs, Y_SPTs_zero = sigs_over_intervals(Y_LIS, intervals, self.n)

        results = []

        # Process P pairs at a time
        for start in range(0, len(pairs), pair_batch_size):

            pair_batch = pairs[start:start + pair_batch_size]

            xspts_zero = ft_pairs(X_SPTs_zero, pair_batch, 0, Tensor_Basis)
            yspts_zero = ft_pairs(Y_SPTs_zero, pair_batch, 1, Tensor_Basis)
            xlsps = ft_pairs(X_LSPs, pair_batch, 0, Tensor_Basis)
            ylsps = ft_pairs(Y_LSPs, pair_batch, 1, Tensor_Basis)
            xlspts = ft_pairs(X_LSPTs, pair_batch, 0, Tensor_Basis)
            ylspts = ft_pairs(Y_LSPTs, pair_batch, 1, Tensor_Basis)

            phi_init, psi_init, K_init = self.initialise_PDE(
                X_SPTs_zero=xspts_zero,
                Y_SPTs_zero=yspts_zero,
                tensor_basis=Tensor_Basis,
            )

            K = self.partition_compute(
                phi=phi_init,
                psi=psi_init,
                K=K_init,
                L=L,
                xlsps=xlsps,
                ylsps=ylsps,
                xlspts=xlspts,
                ylspts=ylspts,
            )

            results.append(K[-1, -1])

        # Combine all pair results
        K_final = jnp.concatenate(results, axis=0)

        if same:
            Gram = upper_tri_to_symmetric(
                K_final, pairs, B1
            )
        else:
            Gram = K_final.reshape(B1, B2)

        return Gram
