import jax.numpy as jnp
import roughpy_jax as rpj
import numpy as np
import jax
from roughpy_jax.streams import LieIncrementStream
from roughpy_jax.streams.lie_increment_stream import _zero_lie
from roughpy_jax.intervals import IntervalType, Partition
from roughpy_jax.streams.piecewise_abelian_stream import to_piecewise_abelian_stream
from roughpy_jax.algebra import to_signature, antipode, to_log_signature, lie_to_tensor, as_free_tensor, _remove_unit_term
from roughpy_jax.dense_algebra import get_batch_shape, _algebra_scalar_multiply, broadcast_to_batch_shape
from utils import ft_pairs, sigs_over_intervals, upper_tri_to_symmetric

#--------------------------------------------------------------------------------
# Some functions which improve simplicity of code
#--------------------------------------------------------------------------------

def make_Lie(data, times, n, R):

    W = len(data[0][0])
    Lie_Basis = rpj.LieBasis(depth = n, width = W)
    Tensor_Basis = rpj.to_tensor_basis(Lie_Basis)
    data_Lie = LieIncrementStream.from_increments(
            timestamps=times,
            data=data,
            input_data_basis=None,
            resolution=R,
            lie_basis=Lie_Basis
        )
    return data_Lie, Tensor_Basis
    
def eval_adj(phi, psi, x, y):

    r_x_y = rpj.ft_adjoint_right_mul(x, y)  
    r_y_x = rpj.ft_adjoint_right_mul(y, x)

    return rpj.tensor_pairing(phi, r_x_y) + rpj.tensor_pairing(psi, r_y_x)

def add_tensor_scalar(a, s): # since multiplicative identity in T^n(V) is (1, 0, 0, 0,...)
    result = a.__array__().copy()
    result[0, 0] += s[0]
    return rpj.FreeTensor(result, a.basis)

#--------------------------------------------------------------------------------------
# function which sets up the initial conditions of the PDE
#--------------------------------------------------------------------------------------

def initialise_PDE(X_SPTs_zero, Y_SPTs_zero, tensor_basis):
    '''
    Inputs:
    X_SPTs_zero - a length L tuple of FreeTensors of shape (M, D(n, W)) where M = M*(B+1)/2 
    where B is the batch size and D(n, W) is the dimension of the tensor basis with respect to
    depth n and width W. Each of these tensors are the truncated (to n-1) signatures over 
    each interval in the partition with the additional change that the initial '1' in each 
    signature is changed to a '0'.

    Y_SPTs_zero - sim.

    Tensor_Basis - Tensor basis parameterised by n and W.

    Outputs:
    phi - (L+1) by (L+1) list of tensors each with batch size M
    psi - as above
    K - (L+1, L+1, M) shaped jnp array which gives the initial conditions for each pairwise kernel
    function (the M necessary to represent each pair)
    '''

    L = len(X_SPTs_zero)
    M, _ = X_SPTs_zero[0].shape

    K = jnp.zeros((L+1, L+1, M), dtype=jnp.float32) 
    # Since the paper gives K[0, v] as the inner product of Z_0^x and Z_v^y and analogous for K[u, 0], I assume we are setting Z_0^x = 1 = Z_0^y where 1 = (1,0,0,0,...) 
    # is in the signature sense
    
    K = K.at[0, :, :].set(1)
    K = K.at[:, 0, :].set(1)

    zero_tensor = rpj.FreeTensor.zero(basis=tensor_basis, batch_dims=(M,))
    phi = [[zero_tensor]*(L+1)]*(L+1)  
    psi = [[zero_tensor]*(L+1)]*(L+1)
    phi = jnp.asarray(phi)
    psi = jnp.asarray(psi)

    # there is some sort of bug that requires phi to be jnp array instead of a list of lists of tensors
    # the tensor version does not correctly update
    for i in range(1, L+1):
        phi = phi.at[i, 0].set(X_SPTs_zero[i-1].__array__()) 

    for j in range(1, L+1):
        psi = psi.at[0, j].set(Y_SPTs_zero[j-1].__array__()) 

    return phi, psi, K

#---------------------------------------------------------------------------------
# functions for computing algorithm 5.1 in rough signature kernel PDE paper
#---------------------------------------------------------------------------------
# rpj.ft_mul is correct
def compute_phi(xi, xti, phi01, psi01, K00):
    phi11 = phi01 + xti.__mul__(K00)\
            + rpj.ft_mul(phi01,xti)\
            + add_tensor_scalar(as_free_tensor(rpj.ft_adjoint_left_mul(psi01, xi)), 
                                - rpj.tensor_pairing(psi01, xti))

    return phi11

def compute_psi(yj, ytj, phi10, psi10, K00):
    psi11 = psi10 + ytj.__mul__(K00)\
            + rpj.ft_mul(psi10, ytj)\
            + add_tensor_scalar(as_free_tensor(rpj.ft_adjoint_left_mul(phi10, yj)), 
                                - rpj.tensor_pairing(phi10, ytj))
    return psi11

def compute_K(xi, yj, phi00, phi01, phi10, phi11, psi00, psi01, psi10, psi11, K00, K01, K10, tensor_basis):

    eval_adj_ = eval_adj(phi00, psi00, xi, yj)
    next_eval_adj = eval_adj(phi11, psi11, xi, yj) # maybe these can be done with built in rpj methods
    temp_2 =  eval_adj(phi01, psi01, xi, yj)
    temp_3 = eval_adj(phi10, psi10, xi, yj)

    G = rpj.tensor_pairing(xi, yj)
    f_1 = K00 * G + eval_adj_
    f_2 = K01 * G + temp_2 
    f_3 = K10 * G + temp_3 

    u_p = K10 + K01 - K00 + f_1
    f_p = u_p * G + next_eval_adj

    K11 = K10 + K01 - K00 + (1./4)*(f_1 + f_2 + f_3 + f_p)
    return K11

def partition_compute(phi, psi, K, L, xlsps, ylsps, xlspts, ylspts, tensor_basis):

    for i in range(L):
        for j in range(L):
            xi = xlsps[i]
            yj = ylsps[j]
            xti = xlspts[i]
            ytj = ylspts[j]
            phi00, phi01, phi10 = rpj.FreeTensor(phi[i][j], tensor_basis), rpj.FreeTensor(phi[i][j+1], tensor_basis), rpj.FreeTensor(phi[i+1][j], tensor_basis)
            psi00, psi01, psi10 = rpj.FreeTensor(psi[i][j], tensor_basis), rpj.FreeTensor(psi[i][j+1], tensor_basis), rpj.FreeTensor(psi[i+1][j], tensor_basis)
            K00, K01, K10 = K[i][j], K[i][j+1], K[i+1][j]
            phi11 = compute_phi(xi, xti, phi01, psi01, K00).__array__()
            psi11 = compute_psi(yj, ytj, phi10, psi10, K00).__array__()
            phi = phi.at[i+1, j+1].set(phi11)
            psi = psi.at[i+1, j+1].set(psi11)
            K11 = compute_K(xi, yj, phi00, phi01, phi10, rpj.FreeTensor(phi11, tensor_basis), psi00, psi01, psi10, rpj.FreeTensor(psi11, tensor_basis), K00, K01, K10, tensor_basis)
            K = K.at[i+1, j+1].set(K11)

    return K    
#-------------------------------------------------------------------------
# combines previous functions to solve the PDE and return the kernel matrix
#-------------------------------------------------------------------------

def solve_PDE(data, times, intervals, n, R, *, testing=False):

    B = len(data)
    L = len(intervals)

    data_Lie, Tensor_Basis = make_Lie(data, times, n, R)

    pairs = jnp.stack(jnp.triu_indices(B), axis=1)
    X_LSPs, X_LSPTs, X_SPTs_zero = sigs_over_intervals(data_Lie, intervals, n)

    xspts_zero = ft_pairs(X_SPTs_zero, pairs, 0, Tensor_Basis)
    yspts_zero = ft_pairs(X_SPTs_zero, pairs, 1, Tensor_Basis)
    xlsps = ft_pairs(X_LSPs, pairs, 0, Tensor_Basis)
    ylsps = ft_pairs(X_LSPs, pairs, 1, Tensor_Basis)
    xlspts = ft_pairs(X_LSPTs, pairs, 0, Tensor_Basis)
    ylspts = ft_pairs(X_LSPTs, pairs, 1, Tensor_Basis)

    phi_init, psi_init, K_init = initialise_PDE(X_SPTs_zero=xspts_zero, 
                                 Y_SPTs_zero=yspts_zero,
                                 tensor_basis=Tensor_Basis)

    K = partition_compute(phi=phi_init,
                          psi=psi_init,
                          K=K_init,
                          L=L,
                          xlsps=xlsps,
                          ylsps=ylsps,
                          xlspts=xlspts,
                          ylspts=ylspts,
                          tensor_basis=Tensor_Basis
    )
    return K