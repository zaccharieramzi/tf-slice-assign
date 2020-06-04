import tensorflow as tf

from tf_slice_assign import slice_assign


sad_checkerboard = tf.zeros((64, 64), dtype=tf.uint8)
checkerboard_filling = tf.ones((32, 32), dtype=tf.uint8)
happier_checker_board = slice_assign(sad_checkerboard, checkerboard_filling, slice(None, None, 2), slice(None, None, 2))
happy_checker_board = slice_assign(happier_checker_board, checkerboard_filling, slice(1, None, 2), slice(1, None, 2))
print('Sad checkerboard was', sad_checkerboard.numpy(), sep='\n')
print('Happy checkerboard now is', happy_checker_board.numpy(), sep='\n')
