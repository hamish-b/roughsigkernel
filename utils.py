import jax
import jax.numpy as jnp
import roughpy_jax as rpj
from roughpy_jax.intervals import IntervalType, Partition
from roughpy_jax.algebra import lie_to_tensor, _remove_unit_term
from roughpy_jax.dense_algebra import identity_like, _algebra_scalar_multiply
from roughpy_jax.streams import LieIncrementStream
from jax import random

#----------------------------------------------------------------------------
# helper function for generating data (made by gen AI)
#----------------------------------------------------------------------------

def generate_sinusoidal_timeseries(key, B, N, W, n_harmonics=3, freq_range=(1.0, 5.0)):
    """
    Generate batched sinusoidal-like time-series data using JAX.

    Args:
        key: jax.random.PRNGKey for reproducibility.
        B: int, batch size (number of independent time series).
        N: int, number of timestamps/datapoints per series.
        W: int, dimensionality of the vector space at each timestamp.
        n_harmonics: int, number of sine components summed per dimension
            (more harmonics -> richer/more complex shapes).
        freq_range: tuple (min_freq, max_freq), range for random frequencies.

    Returns:
        times: jnp.ndarray of shape (B, N), timestamps in [0, 1], sorted per batch.
        data: jnp.ndarray of shape (B, N, W), values bounded in [-1, 1].
    """
    key_times, key_freq, key_phase, key_weights = random.split(key, 4)

    # --- Timestamps: random in [0, 1], sorted within each batch ---
    raw_times = random.uniform(key_times, shape=(B, N))
    times = jnp.sort(raw_times, axis=1)  # (B, N)

    # --- Random harmonic parameters, per batch/dim/harmonic ---
    freqs = random.uniform(
        key_freq, shape=(B, W, n_harmonics),
        minval=freq_range[0], maxval=freq_range[1]
    )  # (B, W, H)

    phases = random.uniform(
        key_phase, shape=(B, W, n_harmonics),
        minval=0.0, maxval=2 * jnp.pi
    )  # (B, W, H)

    # Random positive weights for each harmonic, normalized to sum to 1
    # so the weighted sum of sines stays within [-1, 1].
    raw_weights = random.uniform(key_weights, shape=(B, W, n_harmonics))
    weights = raw_weights / jnp.sum(raw_weights, axis=-1, keepdims=True)  # (B, W, H)

    def single_series(t, freq, phase, weight):
        # t: (N,) ; freq, phase, weight: (W, H)
        # broadcast t -> (N, 1, 1) against (W, H) params
        angles = 2 * jnp.pi * freq[None, :, :] * t[:, None, None] + phase[None, :, :]
        sines = jnp.sin(angles)  # (N, W, H)
        weighted_sum = jnp.sum(sines * weight[None, :, :], axis=-1)  # (N, W)
        return weighted_sum

    data = jax.vmap(single_series, in_axes=(0, 0, 0, 0))(times, freqs, phases, weights)
    # data: (B, N, W), guaranteed in [-1, 1] since weights sum to 1 and |sin| <= 1

    return times, data


def to_list_format(times, data):
    """
    Convert the batched jnp arrays into the exact list structure requested:
      - times: length-B list of length-N lists of floats
      - data: length-B list of (N, W) jnp arrays
    """
    times_list = [list(map(float, times[b])) for b in range(times.shape[0])]
    data_list = [data[b] for b in range(data.shape[0])]
    return times_list, data_list

# given a times and data (which together represent a time-series), returns incremented data and times without the first timestamp 

def make_incremental(times, data):
    '''
    times - jnp.array of shape (B, N)
    data - jnp.array of shape (B, N, D)
    '''
    times_del = times[:, 1:]
    data_inc = jnp.diff(data, axis=1) 

    return times_del, data_inc

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

#--------------------------------------------------------------------------
# Helper functions to manipulate certain rpj objects
#--------------------------------------------------------------------------

# creates `interval_count` uniform intervals from 0 to 1

def uniform_intervals(interval_count):
    endpoints = jnp.linspace(0, 1, interval_count + 1, dtype=jnp.float32).tolist()
    partition = Partition(endpoints, IntervalType.ClOpen)
    return partition.to_intervals()

# truncates from old_depth to new_depth, padding the difference with zeroes (.change_depth(old_depth)) 
# which allows for calculations with original depth tensors

def trunc(X_LSP, old_depth, new_depth):
    return tuple(x.change_depth(new_depth).change_depth(old_depth) for x in X_LSP)

 
def ft_pairs(tuple_of_arrays, pairs, order, tensor_basis):
    '''
    helper function which changes the batch size of each tensor in the input so that there are
    len(pairs[:, order]) elements in each batch (in this case representing the the first or second 
    elements out of the B*(B+1)/2 pairs)
    '''
    return tuple(rpj.FreeTensor(jnp.asarray(array)[pairs[:, order]], tensor_basis) for array in tuple_of_arrays) 


def sigs_over_intervals(X_Lie, intervals, n):
    '''
    calculates log sigs, truncated log-sigs, and signature (with zero instead of 1 in first element)
    over each interval in intervals. Outputs three tuples of length len(intervals).
    '''
    X_LSPs = tuple(lie_to_tensor(X_Lie.log_signature(interval)) for interval in intervals)
    X_SPs = tuple(rpj.ft_exp(x_ls, out_basis=x_ls.basis) for x_ls in X_LSPs)
    X_LSPTs = trunc(X_LSPs, n, n-1)
    X_SPTs = trunc(X_SPs, n, n-1)
    X_SPTs_zero = tuple(_remove_unit_term(x) for x in X_SPTs)
    return X_LSPs, X_LSPTs, X_SPTs_zero

#-------------------------------------------------------------------------------
# Helper functions for simplifying code in RoughKernel methods
#-------------------------------------------------------------------------------

def eval_adj(phi, psi, x, y):

    r_x_y = rpj.ft_adjoint_right_mul(x, y)  
    r_y_x = rpj.ft_adjoint_right_mul(y, x)

    return rpj.tensor_pairing(phi, r_x_y) + rpj.tensor_pairing(psi, r_y_x)

def add_tensor_scalar(a, s): 
    '''
    Given a tensor (a_1, a_2, a_3, ...) and a scalar (s) returns
    (a_1 + s, a_2, a_3, ...)
    '''
    #result = _algebra_scalar_multiply(identity_like(a), s)
    #return result
    result = a.__array__().copy() 
    result[0, 0] += s[0]
    return rpj.FreeTensor(result, a.basis)

# creates a symmetric BxB matrix from a length B*(B+1)/2 array
def upper_tri_to_symmetric(x, B):
    i, j = jnp.triu_indices(B)
    A = jnp.zeros((B, B), dtype=x.dtype)
    A = A.at[i, j].set(x)
    A = A.at[j, i].set(x)
    return A 

    
