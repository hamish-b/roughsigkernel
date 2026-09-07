import jax
import jax.numpy as jnp
import roughpy_jax as rpj
from roughpy_jax.streams import LieIncrementStream
from solver import RoughKernel
from utils import uniform_intervals, perturb_streams

 
key = jax.random.PRNGKey(0)
key_B, key_sporadic = jax.random.split(key, 2)
 
N_STREAMS_TOTAL = 2000     # total streams available (as in your example)
FEATURE_DIM = 10           # dimension of each row vector (matches Lie_Basis dim)
N_ROWS_PERTURBED = 50      # "50 rows of each stream" for A and B
N_STREAMS_PERTURBED = 50
L = 75 # no. of intervals

streams = jnp.load('../BrownianStreamData/ensembles-10k/ensemble-0.npy')

streams = streams[:N_STREAMS_TOTAL]
Lie_Basis = rpj.LieBasis(width = 4, depth = 2)
# ---------------------------------------------------------------------
# Message A: identical construction to the original, just idx_A -> 50 rows
# ---------------------------------------------------------------------
message_A = jnp.ones((N_STREAMS_PERTURBED, N_ROWS_PERTURBED, FEATURE_DIM), dtype=jnp.float32) * 1e-1
idx_A = jnp.arange(N_ROWS_PERTURBED)          # first 50 rows of each stream
sidx_A = jnp.arange(N_STREAMS_PERTURBED)                        # same 25 streams as before
 
streams_pert_A = perturb_streams(streams, message_A, Lie_Basis, sidx_A, idx_A)
 
# ---------------------------------------------------------------------
# Message B: last 6 of the 10 feature components are Uniform(0,1),
# first 4 components left at 0. Same rows/streams as A for comparability.
# ---------------------------------------------------------------------
idx_B = jnp.arange(N_ROWS_PERTURBED)
sidx_B = jnp.arange(N_STREAMS_PERTURBED)
 
last6_vals = jax.random.uniform(
    key_B, shape=(N_STREAMS_PERTURBED, N_ROWS_PERTURBED, 6), minval=0.0, maxval=1.0, dtype=jnp.float64
)
message_B = jnp.concatenate(
    [jnp.zeros((N_STREAMS_PERTURBED, N_ROWS_PERTURBED, FEATURE_DIM - 6), dtype=jnp.float64), last6_vals],
    axis=-1,
)
 
streams_pert_B = perturb_streams(streams, message_B, Lie_Basis, sidx_B, idx_B)

 # ---------------------------------------------------------------------
# Message C: same values as A, sporadic rows instead of first 50
# ---------------------------------------------------------------------
idx_sporadic = jax.random.choice(
    key_sporadic, 2048, shape=(N_ROWS_PERTURBED,), replace=False
)
idx_sporadic = jnp.sort(idx_sporadic)
message_C = jnp.ones((N_STREAMS_PERTURBED, N_ROWS_PERTURBED, FEATURE_DIM), dtype=jnp.float64) * 1e-1
idx_C = idx_sporadic
sidx_C = jnp.arange(N_STREAMS_PERTURBED)
 
streams_pert_C = perturb_streams(streams, message_C, Lie_Basis, sidx_C, idx_C)
 
# ---------------------------------------------------------------------
# Message D: same values as B, same sporadic rows as C
# ---------------------------------------------------------------------
idx_D = idx_sporadic
sidx_D = sidx_C
 
last6_vals_D = jax.random.uniform(
    jax.random.fold_in(key_B, 1), shape=(N_STREAMS_PERTURBED, N_ROWS_PERTURBED, 6),
    minval=0.0, maxval=1.0, dtype=jnp.float64,
)
message_D = jnp.concatenate(
    [jnp.zeros((N_STREAMS_PERTURBED, N_ROWS_PERTURBED, FEATURE_DIM - 6), dtype=jnp.float64),
     last6_vals_D],
    axis=-1,
)
 
streams_pert_D = perturb_streams(streams, message_D, Lie_Basis, sidx_D, idx_D)

# ---------------------------------------------------------------------
# Shared setup (identical to your original code)
# ---------------------------------------------------------------------
times = jnp.arange(0, 2048, dtype=jnp.float32) / 2048
times_list = [times for _ in range(N_STREAMS_TOTAL)]
 
intervals = uniform_intervals(L)
rk = RoughKernel(n=2, R=10)
 
 
def build_lis(streams_pert):
    streams_list = [streams_pert[b] for b in range(N_STREAMS_TOTAL)]
    return LieIncrementStream.from_increments(
        timestamps=times_list,
        data=streams_list,
        input_data_basis=Lie_Basis,
        lie_basis=Lie_Basis,
        resolution=10,
    )
 
 
streams_LIS_A = build_lis(streams_pert_A)
streams_LIS_B = build_lis(streams_pert_B)
streams_LIS_C = build_lis(streams_pert_C)
streams_LIS_D = build_lis(streams_pert_D)
 
# ---------------------------------------------------------------------
# Gram matrices
# ---------------------------------------------------------------------
K_A = rk.solve_PDE(intervals, streams_LIS_A, streams_LIS_A, is_Lie=True)
jnp.save('K_A_2_5_k.npy', K_A)
K_B = rk.solve_PDE(intervals, streams_LIS_B, streams_LIS_B, is_Lie=True)
jnp.save('K_B_2_5_k.npy', K_B)
K_C = rk.solve_PDE(intervals, streams_LIS_C, streams_LIS_C, is_Lie=True)
jnp.save('K_C_2_5_k.npy', K_C)
K_D = rk.solve_PDE(intervals, streams_LIS_D, streams_LIS_D, is_Lie=True)
jnp.save('K_D_2_5_k.npy', K_D)
