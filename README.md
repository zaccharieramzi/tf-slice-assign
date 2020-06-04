# tf-slice-assign

[![Build Status](https://travis-ci.com/zaccharieramzi/tf-slice-assign.svg?token=wHL4tmyGD3TP6bSo6Mdh&branch=master)](https://travis-ci.com/zaccharieramzi/tf-slice-assign)

A tool for assignment to a slice in TensorFlow.

In TensorFlow, as opposed to Pytorch, it is currently impossible to assign to
the slice of a tensor in a range of different settings.
To mitigate this issue, `tf-slice-assign` introduces a single function that
allows to do exactly this using [`tensor_scatter_nd_update`](https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update).


## List of GitHub issues and StackOverflow questions regarding TensorFlow slice assignment
In the following table, I am trying to give the reasons as to why no mitigation
for the current problem exists.

| Link                                                                                                              | Status                                                                                                                             |
|-------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| [SO](https://stackoverflow.com/questions/62092147/how-to-efficiently-assign-to-a-slice-of-a-tensor-in-tensorflow) | No answer for dynamically shaped input                                                                                             |
| [GH](https://github.com/tensorflow/tensorflow/issues/36559#issue-561880519)                                       | Question is about `tf.Variable`                                                                                                    |
| [SO](https://stackoverflow.com/questions/39157723/how-to-do-slice-assignment-in-tensorflow)                       | Answers for `tf.Variable` or using `tensor_scatter_update` in a non-adaptable way                                                  |
| [GH](https://github.com/tensorflow/tensorflow/issues/33131#issue-503809713)                                       | Suggestion to use `tensor_scatter_nd_update`                                                                                       |
| [GH](https://github.com/tensorflow/tensorflow/issues/14132#issue-270037738)                                       | An answer suggest creating a mask, but a mask can actually be as difficult to create as the indices for `tensor_scatter_nd_update` |
