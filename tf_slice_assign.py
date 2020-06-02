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
    dims_to_index = []
    for i_dim, slice in enumerate(slice_args):
        pass
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
