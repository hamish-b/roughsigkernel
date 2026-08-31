import jax.numpy as jnp
import roughpy_jax as rpj
import numpy as np
from roughpy_jax.streams import LieIncrementStream
from roughpy_jax.streams.lie_increment_stream import _zero_lie
from roughpy_jax.intervals import IntervalType, Partition
from roughpy_jax.algebra import to_signature, antipode, to_log_signature, lie_to_tensor, as_free_tensor
from roughpy_jax.dense_algebra import get_batch_shape, _algebra_scalar_multiply, broadcast_to_batch_shape
from utils import one_to_zero, trunc


def make_Lie_single(data, times, n, R, W):

    Lie_Basis = rpj.LieBasis(depth = n, width = W)
    Tensor_Basis = rpj.to_tensor_basis(Lie_Basis)

    X_Lie = LieIncrementStream.from_increments(timestamps=jnp.array(times),
                                            data=data,
                                            resolution=R,
                                            input_data_basis=None,
                                            lie_basis=Lie_Basis)
    return X_Lie, Tensor_Basis




def sigs_unif_intervals(X_Lie, Y_Lie, L, M):
    '''
    L - no. of uniform intervals to calculate the signature (and log-signature) of X over
    M - ... Y ...
    '''

    endpoints_X = jnp.linspace(0, 1, L + 1, dtype=jnp.float32).tolist()
    partition_X = Partition(endpoints_X, IntervalType.ClOpen)

    endpoints_Y = jnp.linspace(0, 1, M + 1, dtype=jnp.float32).tolist()
    partition_Y = Partition(endpoints_Y, IntervalType.ClOpen)


    X_LSP = tuple(lie_to_tensor(X_Lie.log_signature(interval)) for interval in partition_X.to_intervals())
    Y_LSP = tuple(lie_to_tensor(Y_Lie.log_signature(interval)) for interval in partition_Y.to_intervals())

    X_SP = tuple(rpj.ft_exp(x_ls, out_basis=x_ls.basis) for x_ls in X_LSP) 
    Y_SP = tuple(rpj.ft_exp(y_ls, out_basis=y_ls.basis) for y_ls in Y_LSP)

    return X_SP, Y_SP, X_LSP, Y_LSP

def initialise_PDE_pairwise(X_SP, Y_SP, n, Tensor_Basis):

    L = len(X_SP)
    M = len(Y_SP)
    
    # K represents our target function f in the original PDE (Algorithm 5.1)
    K = np.zeros((L+1, M+1), dtype=np.float32) 

    # phi and psi follow notation from Algorithm 5.1 - batch_dims set to (1,) for now

    zero_tensor = rpj.FreeTensor.zero(basis=Tensor_Basis, batch_dims=(1,))
    phi = [[zero_tensor]*(M+1)]*(L+1)    
    psi = [[zero_tensor]*(M+1)]*(L+1)

    # Since the paper gives K[0, v] as the inner product of Z_0^x and Z_v^y and analogous for K[u, 0], I assume we are setting Z_0^x = 1 = Z_0^y where 1 = (1,0,0,0,...) 
    # is in the signature sense

    K[0,:] = 1. 
    K[:,0] = 1.

    # truncate signature and replace initial 1 with 0

    X_SPT = trunc(X_SP, n, n-1)
    Y_SPT = trunc(Y_SP, n, n-1)

    X_SPT_zero = one_to_zero(X_SPT, Tensor_Basis)
    Y_SPT_zero = one_to_zero(Y_SPT, Tensor_Basis)
    
    for i in range(1, L + 1):
        phi[i][0] = X_SPT_zero[i-1]
        
    for j in range(1, M + 1):
        psi[0][j] = Y_SPT_zero[j-1]        
    return K, phi, psi    

def inner_prod(X,Y): # this function is created so we can get scalars instead of arrays as outputs (but probably slower so should fix later)
    return np.sum(X.__array__()*Y.__array__())  

def right_adj(A,C, tensor_basis):

    ant_A = antipode(A)
    ant_C = antipode(C)
    
    left_adj = rpj.ft_adjoint_left_mul(ant_A, ant_C)

    left_adj = rpj.FreeTensor(left_adj, basis=tensor_basis)  
    
    adj_A_C = antipode(left_adj)
    
    return adj_A_C

    
def eval_adj(phi, psi, x, y, tensor_basis):
    
    r_x_y = right_adj(x, y, tensor_basis)  
    r_y_x = right_adj(y, x, tensor_basis)

    return inner_prod(phi, r_x_y) + inner_prod(psi, r_y_x)

def add_tensor_scalar(a, s): # since multiplicative identity in T^n(V) is (1, 0, 0, 0,...)
    result = a.__array__().copy()
    result[0, 0] += s[0]
    return rpj.FreeTensor(result, a.basis)
    


def algorithm_solve(X_SP, Y_SP, X_LSP, Y_LSP, X_LSPT, Y_LSPT, K, phi, psi, tensor_basis):

    L = len(X_LSP)
    M = len(Y_LSP)

    for i in range(L):
        for j in range(M):
                
            # phi rpj.FreeTensor(phi[i][j+1].__array__().__mul__(X_LSPT[i]), tensor_basis)
            # has replaced the incorrect rpj.ft_mul(phi[i][j+1], X_LSPT[i]). Not changed in main solver yet
            phi[i+1][j+1] =  phi[i][j+1] + X_LSPT[i].__mul__(K[i,j])\
                            + rpj.FreeTensor(phi[i][j+1].__array__().__mul__(X_LSPT[i]), tensor_basis)\
                            + add_tensor_scalar(as_free_tensor(rpj.ft_adjoint_left_mul(psi[i][j+1], X_LSP[i])), 
                                                -rpj.tensor_pairing(psi[i][j+1], X_LSPT[i]))

            # psi (analogous for psi)
            psi[i+1][j+1] =  psi[i+1][j] + Y_LSPT[j].__mul__(K[i,j])\
                            + rpj.FreeTensor(psi[i+1][j].__array__().__mul__(Y_LSPT[j]), tensor_basis)\
                            + add_tensor_scalar(as_free_tensor(rpj.ft_adjoint_left_mul(phi[i+1][j], Y_LSP[j])), 
                                                -rpj.tensor_pairing(phi[i+1][j], Y_LSPT[j]))

            eval_adj_ = eval_adj(phi[i][j], psi[i][j], X_LSP[i], Y_LSP[j], tensor_basis)
                

            # the kernel equation
            next_eval_adj = eval_adj(phi[i+1][j+1], psi[i+1][j+1], X_LSP[i], Y_LSP[j], tensor_basis)
            temp_2 =  eval_adj(phi[i][j+1], psi[i][j+1], X_LSP[i], Y_LSP[j], tensor_basis)
            temp_3 = eval_adj(phi[i+1][j], psi[i+1][j], X_LSP[i], Y_LSP[j], tensor_basis)
    
            G = rpj.tensor_pairing(X_LSP[i],Y_LSP[j])
            f_1 = K[i,j] * G + eval_adj_
            f_2 = K[i,j+1] * G + temp_2 
            f_3 = K[i+1,j] * G + temp_3 

            u_p = K[i+1,j] + K[i,j+1] - K[i,j] + f_1
            f_p = u_p * G + next_eval_adj


            K[i+1,j+1] =  K[i+1,j] + K[i,j+1] - K[i,j] + (1./4)*(f_1 + f_2 + f_3 + f_p)
    return K, phi, psi

def solve_PDE_pairwise(X_Lie, Y_Lie, L, M, n, tensor_basis, give_intermediate = False):

    X_SP, Y_SP, X_LSP, Y_LSP = sigs_unif_intervals(X_Lie=X_Lie, Y_Lie=Y_Lie, L=L, M=M)
    
    K_init, phi_init, psi_init = initialise_PDE_pairwise(X_SP=X_SP, Y_SP=Y_SP, n=n, Tensor_Basis=tensor_basis)

    X_LSPT = trunc(X_LSP, n, n-1) 
    Y_LSPT = trunc(Y_LSP, n, n-1)

    K, phi, psi = algorithm_solve(X_SP=X_SP, Y_SP=Y_SP, X_LSP=X_LSP, Y_LSP=Y_LSP, X_LSPT=X_LSPT,
                                  Y_LSPT=Y_LSPT, K=K_init, phi=phi_init, psi=psi_init, tensor_basis=tensor_basis)
    if give_intermediate:
        return K, phi, psi
    else: 
        return K[-1, -1]