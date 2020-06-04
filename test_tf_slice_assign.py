import numpy as np
import pytest
import tensorflow as tf

from tf_slice_assign import slice_assign


def handle_slice_arg(slice_arg):
    if isinstance(slice_arg, str):
        if slice_arg == ':':
            return slice(None)
        elif slice_arg == '...':
            return ...
    else:
        return slice_arg

@pytest.mark.parametrize('tensor_shape, slice_args', [
    ((8,), ['...']),
])
def test_slice_assign(tensor_shape, slice_args):
    original_tensor = np.random.normal(size=tensor_shape)
    numpy_slices = [handle_slice_arg(slice_arg) for slice_arg in slice_args]
    original_tensor_slice = original_tensor[numpy_slices]
    assigned_tensor =  np.random.normal(size=original_tensor_slice.shape)
    expected_result = original_tensor[:]
    expected_result[numpy_slices] = assigned_tensor
    result = slice_assign(
        tf.constant(original_tensor),
        tf.constant(assigned_tensor),
        *slice_args,
    )
    tf_tester = tf.test.TestCase()
    tf_tester.assertAllEqual(expected_result, result)
