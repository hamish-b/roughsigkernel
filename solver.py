import jax.numpy as jnp
import roughpy_jax as rpj
import jax
from roughpy_jax.streams import LieIncrementStream
from roughpy_jax.algebra import  as_free_tensor
from functools import partial
from jax.tree_util import tree_map
from utils import (ft_pairs, 
                   sigs_over_intervals, 
                   upper_tri_to_symmetric, 
                   eval_adj, 
                   add_tensor_scalar)


class RoughKernel:
    """
    Solves the signature kernel PDE (Algorithm 5.1 in the rough signature
    kernel PDE paper) for a batch of paths.
    """

    def __init__(self, n, R):
        """
        n - truncation depth for the log-signature / signature
        R - resolution used when building the Lie increment stream
        """
        self.n = n
        self.R = R

    # ------------------------------------------------------------------
    # Setup: building the Lie increment stream depends on self.n / self.R,
    # so this is an instance method.
    # ------------------------------------------------------------------
    def make_Lie(self, data, times):
        W = len(data[0][0])
        Lie_Basis = rpj.LieBasis(depth=self.n, width=W)
        data_Lie = LieIncrementStream.from_increments(
            timestamps=times,
            data=data,
            input_data_basis=None,
            resolution=self.R,
            lie_basis=Lie_Basis
        )
        return data_Lie

    @staticmethod
    def initialise_PDE(X_SPTs_zero, Y_SPTs_zero, tensor_basis):
        '''
        Inputs:
        X_SPTs_zero - a length L tuple of FreeTensors of shape (M, D(n, W)) where M = M*(B+1)/2
        where B is the batch size and D(n, W) is the dimension of the tensor basis with respect to
        depth n and width W. Each of these tensors are the truncated (to n-1) signatures over
        each interval in the partition with the additional change that the initial '1' in each
        signature is changed to a '0'.

        Y_SPTs_zero - sim.

        tensor_basis - Tensor basis parameterised by n and W.

        Outputs:
        phi - (L+1) by (L+1) list of tensors each with batch size M
        psi - as above
        K - (L+1, L+1, M) shaped jnp array which gives the initial conditions for each pairwise kernel
        function (the M necessary to represent each pair)
        '''
        L = len(X_SPTs_zero)
        M, _ = X_SPTs_zero[0].shape # M is the no. of pairs

        K = jnp.zeros((L + 1, L + 1, M), dtype=jnp.float32)
        # Since the paper gives K[0, v] as the inner product of Z_0^x and Z_v^y (and analogously
        # for K[u, 0]), we assume Z_0^x = 1 = Z_0^y where 1 = (1, 0, 0, ...) in the signature sense
        K = K.at[0, :, :].set(1)
        K = K.at[:, 0, :].set(1)

        zero_tensor = rpj.FreeTensor.zero(basis=tensor_basis, batch_dims=(M,))
        phi = [[zero_tensor for _ in range(L + 1)] for _ in range(L + 1)]
        psi = [[zero_tensor for _ in range(L + 1)] for _ in range(L + 1)]

        for i in range(1, L + 1):
            phi[i][0] = X_SPTs_zero[i - 1]
        for j in range(1, L + 1):
            psi[0][j] = Y_SPTs_zero[j - 1]

        phi = rpj.FreeTensor(phi, tensor_basis)
        psi = rpj.FreeTensor(psi, tensor_basis)

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

    # ------------------------------------------------------------------
    # This one is jitted, and since it's a bound method `self` becomes
    # positional argument 0 — so it must be added to static_argnums,
    # shifting every other static index up by one from the free-function
    # version. `self` is safe to mark static here because RoughKernel
    # instances are hashed by identity and n/R don't change after init.
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnums=(0, 4,))
    def partition_compute(self, phi, psi, K, L, xlsps, ylsps, xlspts, ylspts):
        
        def get(tree, i, j):
            return tree_map(lambda x: x[i, j], tree)

        def set_(tree, i, j, val):
            return tree_map(lambda x, v: x.at[i, j].set(v), tree, val)
        
        for i in range(L):
            for j in range(L):
                xi = xlsps[i]
                yj = ylsps[j]
                xti = xlspts[i]
                ytj = ylspts[j]
                phi00, phi01, phi10 = get(phi, i, j), get(phi, i, j + 1), get(phi, i + 1, j)
                psi00, psi01, psi10 = get(psi, i, j), get(psi, i, j + 1), get(psi, i + 1, j)
                K00, K01, K10 = K[i][j], K[i][j + 1], K[i + 1][j]

                phi11 = self.compute_phi(xi, xti, phi01, psi01, K00)
                psi11 = self.compute_psi(yj, ytj, phi10, psi10, K00)
                K11 = self.compute_K(xi, yj, phi00, phi01, phi10, phi11,
                                      psi00, psi01, psi10, psi11, K00, K01, K10)

                phi = set_(phi, i + 1, j + 1, phi11)
                psi = set_(psi, i + 1, j + 1, psi11)
                K = K.at[i + 1, j + 1].set(K11)
        return K

# better way with JAX if can get working, having issues with xlsps
    @partial(jax.jit, static_argnums=(0, 4))
    def partition_compute_mod(self, phi, psi, K, L, xlsps, ylsps, xlspts, ylspts):

        def get(tree, i, j):
            return tree_map(lambda x: x[i, j], tree)

        def set_(tree, i, j, val):
            return tree_map(lambda x, v: x.at[i, j].set(v), tree, val)

        def outer_body(i, carry):
            phi, psi, K = carry

            def inner_body(j, carry):
                phi, psi, K = carry
                xi, yj = xlsps[i], ylsps[j]
                xti, ytj = xlspts[i], ylspts[j]

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

            return jax.lax.fori_loop(0, L, inner_body, (phi, psi, K))

        phi, psi, K = jax.lax.fori_loop(0, L, outer_body, (phi, psi, K))
        return phi, psi, K   

    # ------------------------------------------------------------------
    # Top-level driver: depends on self.n / self.R (via make_Lie), so
    # it's an instance method.
    # ------------------------------------------------------------------
    
    def solve_PDE(self, intervals, X, Y, is_Lie, times_X = None, times_Y = None):

        L = len(intervals)

        if is_Lie:
            X_Lie, Y_Lie, Tensor_Basis = X, Y, X.group_basis
            B1, B2 = X.batch_dims, Y.batch_dims
        else:
            X_Lie, Y_Lie = self.make_Lie(X, times_X), self.make_Lie(Y, times_Y)
            B1, B2 = len(X), len(Y)
            Tensor_Basis = X_Lie.group_basis

        # list containing every pair (x_i, y_j) s.t. x in X, y in Y. if X = Y, only need B*(B+1)/2
        if X == Y:
            pairs = jnp.stack(jnp.triu_indices(B1), axis=1)
            same = True
        else:
            pairs = jnp.array([(i, j) for i in range(B1) for j in range(B2)])
            same = False

        X_LSPs, X_LSPTs, X_SPTs_zero = sigs_over_intervals(X_Lie, intervals, self.n)
        Y_LSPs, Y_LSPTs, Y_SPTs_zero = sigs_over_intervals(Y_Lie, intervals, self.n)

        xspts_zero = ft_pairs(X_SPTs_zero, pairs, 0, Tensor_Basis)
        yspts_zero = ft_pairs(Y_SPTs_zero, pairs, 1, Tensor_Basis)
        xlsps = ft_pairs(X_LSPs, pairs, 0, Tensor_Basis)
        ylsps = ft_pairs(Y_LSPs, pairs, 1, Tensor_Basis)
        xlspts = ft_pairs(X_LSPTs, pairs, 0, Tensor_Basis)
        ylspts = ft_pairs(Y_LSPTs, pairs, 1, Tensor_Basis)

        phi_init, psi_init, K_init = self.initialise_PDE(
            X_SPTs_zero=xspts_zero,
            Y_SPTs_zero=yspts_zero,
            tensor_basis=Tensor_Basis
        )

        K = self.partition_compute(
            phi=phi_init,
            psi=psi_init,
            K=K_init,
            L=L,
            xlsps=xlsps,
            ylsps=ylsps,
            xlspts=xlspts,
            ylspts=ylspts
        )

        if same:
            Gram = upper_tri_to_symmetric(K[-1, -1], pairs, B1)
        else:
            Gram = K[-1, -1].reshape(B1, B2)

        return Gram