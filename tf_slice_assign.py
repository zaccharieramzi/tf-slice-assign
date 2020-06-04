import numpy as np
# NOTE: numpy is imported for argsorting. We might not use it but then lose in
# code clarity (and a bit in speed but negligible).
import tensorflow as tf


def slice_assign(sliced_tensor, assigned_tensor, *slice_args):
    """Assign a tensor to the slice of another tensor.

    No broadcast is performed.

    Args:
        - sliced_tensor (tf.Tensor): the tensor whose slice you want changed.
        - assigned_tensor (tf.Tensor): the tensor which you want assigned.
        - *slice_args (str or slice): the slices arguments. Can be ':', '...'
        or slice.
    """
    n_dims = len(tf.shape(sliced_tensor))
    # parsing the slice specifications
    n_slices = len(slice_args)
    dims_to_index = []
    corresponding_ranges = []
    three_dots = False
    for i_dim, slice_spec in enumerate(slice_args):
        if isinstance(slice_spec, str):
            if slice_spec == ':':
                continue
            elif slice_spec == '...':
                three_dots = True
            else:
                raise ValueError('Slices must be :, ..., or slice object.')
        else:
            start, stop, step = slice_spec.start, slice_spec.stop, slice_spec.step
            if step is None:
                step = 1
            corresponding_range = tf.range(start, stop, step)
            if three_dots:
                dims_to_index.append(i_dim + (n_dims - n_slices))
            else:
                dims_to_index.append(i_dim)
            corresponding_ranges.append(corresponding_range)
    dims_left_out = [
        i_dim for i_dim in range(n_dims) if i_dim not in dims_to_index
    ]
    scatted_nd_perm = dims_to_index + dims_left_out
    inverse_scatter_nd_perm = list(np.argsort(scatted_nd_perm))
    # reshaping the tensors
    sliced_tensor_reshaped = tf.transpose(sliced_tensor, perm=scatted_nd_perm)
    assigned_tensor_reshaped = tf.transpose(assigned_tensor, perm=scatted_nd_perm)

    # creating the indices
    mesh_ranges = tf.meshgrid(*corresponding_ranges, indexing='xy')
    update_indices = tf.stack([
        tf.reshape(slicing_range, (-1))
        for slicing_range in mesh_ranges
    ], axis=-1)

    # finalisation
    sliced_tensor_reshaped = tf.tensor_scatter_nd_update(
        tensor=sliced_tensor_reshaped,
        indices=update_indices,
        updates=assigned_tensor_reshaped,
    )
    sliced_tensor_updated = tf.transpose(
        sliced_tensor_reshaped,
        perm=inverse_scatter_nd_perm,
    )
    return sliced_tensor_updated
