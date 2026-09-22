import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import roughpy_jax as rpj
from roughpy_jax.intervals import IntervalType, Partition
from roughpy_jax.algebra import lie_to_tensor, _remove_unit_term
from roughpy_jax.dense_algebra import identity_like, _algebra_scalar_multiply
from roughpy_jax.streams import LieIncrementStream
from jax import random

#---------------------------------------------------------------------------
# Helper functions to convert a normal time-series into a LieIncrementStream
#---------------------------------------------------------------------------

def make_incremental(times, data):
    '''
    Makes the data incremental (which is the format LieIncrementStream.from_increments()) needs.

    Inputs:
    times - jnp.array of shape (B, N) where B is batch size, N is the length of the time-series
    data - jnp.array of shape (B, N, W) where W is the dimension the time-series takes values in

    Outputs:
    times_del - jnp.array of shape (B, N-1) that has the first element of every time series removed
    data_inc - jnp.array of shape (B, N-1, W) that contains the increments corresponding to the data
    '''
    times_del = times[:, 1:]
    data_inc = jnp.diff(data, axis=1) 

    return times_del, data_inc

def to_list_format(times, data):
    """
    Converts times and data into a list format needed for LieIncrementStream.from_increments()

    Inputs:
    times - jnp.array of shape (B, N), B batch size, N length of time-series
    data - jnp.array of shape (B, N, W), W dimension of vector space the time-series takes values in

    Outputs
    times - length-B list of length-N lists of floats
    data - length-B list of (N, W) shaped jnp.arrays
    """
    times_list = [list(map(float, times[b])) for b in range(times.shape[0])]
    data_list = [data[b] for b in range(data.shape[0])]
    return times_list, data_list

def make_LIS(times, data, n, W, R, incremental = False, input_basis = None):
    """
    Converts time-series data into a LieIncrementStream object

    Inputs:
    times - jnp.array of shape (B, N) where B is batch size, N is the length of the time-series
    data - jnp.array of shape (B, N, W) where W is the dimension the time-series takes values in
    n - level to truncate the signature (from now on referred to as depth)
    R - resolution of the LieIncrementStream, there will be 2^(R) contiguous dyadic intervals 
        of size 2^(-R)and the LieIncrementStream takes the log-signature over each of these. 
        It also stores, the value of the log-signature for every larger dyadic interval. i.e.
        if R = 2, then the LIS will store the values of log-signatures over [0, 1/4],...,[3/4, 1]
        as well as [0, 1/2], [1/2, 1] and [0, 1].
    incremental - if data is already incremental don't need to apply make_incremental()
    input_basis - if data has an input basis use this

    Outputs:
    data_LIS - data in LieIncrementStream form
    """
    if not incremental:
        times, data = make_incremental(times, data)

    times, data = to_list_format(times, data)

    Lie_Basis = rpj.LieBasis(depth = n, width = W)

    data_LIS = LieIncrementStream.from_increments(
            timestamps=times,
            data=data,
            input_data_basis=input_basis,
            resolution=R,
            lie_basis=Lie_Basis
        )
    return data_LIS

#--------------------------------------------------------------------------
# Helper functions to manipulate certain rpj objects
#--------------------------------------------------------------------------

def uniform_intervals(interval_count):
    """
    Creates `interval_count` uniform intervals from 0 to 1.
    """
    endpoints = jnp.linspace(0, 1, interval_count + 1, dtype=jnp.float32).tolist()
    partition = Partition(endpoints, IntervalType.ClOpen)
    return partition.to_intervals()

def trunc(batch_tensor, old_depth, new_depth):
    return batch_tensor.change_depth(new_depth).change_depth(old_depth)

def sigs_over_intervals(X_LIS, intervals, n):
    '''
    calculates log sigs, truncated log-sigs, and signature (with zero instead of 1 in first element)
    over each interval in intervals. Outputs three tuples of length len(intervals).
    '''

    X_LSP_tuple = tuple(lie_to_tensor(X_LIS.log_signature(interval)) for interval in intervals)
    X_LSPs = rpj.FreeTensor(X_LSP_tuple, X_LSP_tuple[0].basis)
    X_SPs = rpj.ft_exp(X_LSPs, out_basis=X_LSPs.basis)
    X_LSPTs = trunc(X_LSPs, n, n-1)
    X_SPTs = trunc(X_SPs, n, n-1)
    X_SPTs_zero = _remove_unit_term(X_SPTs)

    return X_LSPs, X_LSPTs, X_SPTs_zero

def ft_pairs(batch_tensor, pairs, order, tensor_basis):
    return rpj.FreeTensor(batch_tensor.data[:, pairs[:, order], :], tensor_basis)

#-------------------------------------------------------------------------------
# Helper functions for updating jnp arrays
#-------------------------------------------------------------------------------

def get(tree, i, j):
    return tree_map(lambda x: x[i, j], tree)

def get_(tree, i):
    return tree_map(lambda x: x[i], tree)
        
def set_(tree, i, j, val):
    return tree_map(lambda x, v: x.at[i, j].set(v), tree, val)

#-------------------------------------------------------------------------------
# Helper functions for simplifying code in RoughKernel methods
#-------------------------------------------------------------------------------

def eval_adj(phi, psi, x, y):

    r_x_y = rpj.ft_adjoint_right_mul(x, y)  
    r_y_x = rpj.ft_adjoint_right_mul(y, x)

    return rpj.tensor_pairing(phi, r_x_y) + rpj.tensor_pairing(psi, r_y_x)

def add_tensor_scalar(a, s): 
    '''
    Given a batch of size B containing tensors (a_1, a_2, a_3, ...) and a scalar (s) returns a
    batch of modified tensors like (a_1 + s, a_2, a_3, ...) 
    '''
    result = a.data.at[:, 0].add(s)
    return rpj.FreeTensor(result, a.basis)

# creates a symmetric BxB matrix from a length B*(B+1)/2 array
def upper_tri_to_symmetric(array, pairs, B):
    A = jnp.zeros((B, B), dtype=array.dtype)
    for k in range(len(pairs)):
        i, j = pairs[k]
        A = A.at[i, j].set(array[k])
        A = A.at[j, i].set(array[k])
    return A 

def perturb_streams(streams, message, lie_basis, stream_index, row_index):

    message_Lie = rpj.Lie(message, lie_basis)

    grid0, grid1 = jnp.ix_(stream_index, row_index)

    pieces_to_perturb = streams[grid0, grid1]
    pieces_to_perturb_Lie = rpj.Lie(pieces_to_perturb, lie_basis)
    perturbed_pieces = rpj.algebra.cbh(pieces_to_perturb_Lie, message_Lie).data

    perturbed_streams = streams.at[grid0, grid1].set(perturbed_pieces)

    return perturbed_streams


    
