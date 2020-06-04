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
            corresponding_range = tf.range(start, stop, step)
            if three_dots:
                dims_to_index.append(i_dim + (n_dims - n_slices))
            else:
                dims_to_index.append(i_dim)
            corresponding_ranges.append(corresponding_range)
    sliced_tensor_reshaped = sliced_tensor
    update_indices = []
    assigned_tensor_reshaped = assigned_tensor
    sliced_tensor_reshaped = tf.tensor_scatter_nd_update(
        tensor=sliced_tensor_reshaped,
        indices=update_indices,
        updates=assigned_tensor_reshaped,
    )
    sliced_tensor_updated = sliced_tensor_reshaped
    return sliced_tensor_updated
