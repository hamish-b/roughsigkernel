import roughpy_jax as rpj
import jax.numpy as jnp

def perturb_streams(streams, message, lie_basis, stream_index, row_index):

    message_Lie = rpj.Lie(message, lie_basis)

    grid0, grid1 = jnp.ix_(stream_index, row_index)

    pieces_to_perturb = streams[grid0, grid1]
    pieces_to_perturb_Lie = rpj.Lie(pieces_to_perturb, lie_basis)
    perturbed_pieces = rpj.algebra.cbh(pieces_to_perturb_Lie, message_Lie).data

    perturbed_streams = streams.at[grid0, grid1].set(perturbed_pieces)

    return perturbed_streams