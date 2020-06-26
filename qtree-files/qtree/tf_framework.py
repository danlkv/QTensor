"""
This file implements Tensorflow framework of the
simulator. It's main use is in conjunction with the :py:mod:`optimizer`
module, and example programs are listed in :py:mod:`simulator` module.
"""

import numpy as np
import tensorflow as tf
import qtree.operators as ops
import qtree.optimizer as opt
import qtree.utils as utils
import qtree.system_defs as defs


def get_sliced_tf_buckets(buckets, slice_dict):
    """
    Takes buckets and returns their Tensorflow counterparts, where
    all data attributes of tensors are filled with Tensorflow
    placeholders. 

    Parameters
    ----------
    buckets : list of list
              buckets as returned by :py:meth:`read_buckets`y
              and :py:meth:`reorder_buckets`.
    slice_dict : dict
              dictionary of {variable : slice} pairs

    Returns
    -------
    tf_buckets : list of lists
               Buckets having Tensorflow tensors in place of Tensor.data
               attribute
    placeholder_dict: dict
               dictionary of the form {placeholder: data_key}.
    """
    # import pdb
    # pdb.set_trace()

    placeholder_dict = {}
    # Create tf buckets from  buckets
    tf_buckets = []
    for bucket in buckets:
        tf_bucket = []
        for tensor in bucket:
            # Save the reference to placeholder in the dictionary
            placeholder = tf.stop_gradient(
                tf.placeholder(defs.TF_ARRAY_TYPE,
                               tensor.shape, name=tensor.name)
                )
            placeholder_dict[placeholder] = tensor.data_key

            # sort tensor dimensions
            transpose_order = np.argsort(list(map(int, tensor.indices)))
            data = tf.transpose(placeholder, transpose_order)

            # transpose indices
            indices_sorted = [tensor.indices[pp] for pp
                              in transpose_order]

            # slice tensor
            slice_bounds = []
            indices_sliced = []
            for idx in indices_sorted:
                if idx in slice_dict:
                    # insert slice variables into the placeholder dict
                    slice_start = tf.stop_gradient(
                        tf.placeholder(
                            tf.int32,
                            name=idx.name + '_start')
                        )
                    slice_stop = tf.stop_gradient(
                        tf.placeholder(
                            tf.int32,
                            name=idx.name + '_stop')
                        )
                    placeholder_dict[slice_start] = (idx, 'start')
                    placeholder_dict[slice_stop] = (idx, 'stop')
                    slice_bounds.append(slice(slice_start, slice_stop))

                    # update the size of tensor variables
                    indices_sliced.append(idx.copy(
                        size=slice_dict[idx].stop-slice_dict[idx].start))
                else:
                    slice_bounds.append(slice(None))
                    indices_sliced.append(idx)

            data = data[tuple(slice_bounds)]
            # Create new tensor with a placeholder for data
            new_tensor = tensor.copy(
                indices=indices_sliced,
                data=data)

            tf_bucket.append(new_tensor)

        tf_buckets.append(tf_bucket)

    return tf_buckets, placeholder_dict


def assign_tensor_placeholders(placeholder_dict, data_dict):
    """
    Builds feed dictionary for Tensorflow from the placeholder
    dictionary, which holds placeholders of all gates in the circuit,
    and a global data dictionary.

    Parameters
    ----------
    placeholder_dict : dict
           Dictionary of {tensorflow.placeholder : data_key} pairs
    data_dict : dict
           Dictionary of {data_key : np.array} pairs

    Returns
    -------
    feed_dict : dict
          Dictionary to feed in Tensorflow session
    """
    feed_dict = {}

    # Try to fill all fixed gates placeholders
    for placeholder, data_key in placeholder_dict.items():
        try:
            feed_dict[placeholder] = data_dict[data_key]
        except KeyError:
            pass

    return feed_dict


def assign_variable_placeholders(placeholder_dict, slice_dict):
    """
    Builds feed dictionary for Tensorflow from the placeholder
    dictionary which holds information about variables
    and variable slice information

    Parameters
    ----------
    placeholder_dict : dict
           Dictionary of {tensorflow.placeholder : data_key} pairs
    slice_dict : dict
           Dictionary of {variable : slice} pairs

    Returns
    -------
    feed_dict : dict
          Dictionary to feed in Tensorflow session
    """
    feed_dict = {}
    # Try to fill all variables with placeholders
    for placeholder, data_key in placeholder_dict.items():
        var, slice_end = data_key
        try:
            feed_dict[placeholder] = getattr(slice_dict[var], slice_end)
        except KeyError:
            pass

    return feed_dict


def slice_tf_buckets(tf_buckets, old_pdict, idx_parallel):
    """
    Takes (symbolic) slices of the Tensorflow buckets
    over the variables in idx_parallel. Updates the placeholder
    dictionary.

    Parameters
    ----------
    tf_buckets : list of lists
              Buckets containing Tensorflow tensors and variables
    old_pdict : dict
              Placeholder dictionary
    idx_parallel : list
              Indices to parallelize over

    Returns
    -------
    sliced_buckets : list of lists
              buckets with (symbolically) sliced gates
    pdict : dict
              updated placeholder dictionary
    """
    # import pdb
    # pdb.set_trace()

    pdict = {key: val for key, val in old_pdict.items()}
    # Define slice variables
    slice_var_dict = {'q_{}'.format(var):
                      tf.stop_gradient(
                          tf.placeholder(dtype=tf.int32,
                                         shape=[],
                                         name='q_{}'.format(var))
                      )
                      for var in idx_parallel}
    pdict.update(slice_var_dict)

    # Create tf buckets from unordered buckets
    sliced_buckets = []
    for bucket in tf_buckets:
        sliced_bucket = []
        for tensor, variables in bucket:
            slice_bounds = []
            new_shape = []
            for var in variables:
                if var in idx_parallel:
                    slice_bounds.append(slice_var_dict[f'q_{var}'])
                    new_shape.append(1)
                else:
                    slice_bounds.append(slice(None))
                    new_shape.append(2)
            sliced_bucket.append(
                (tf.reshape(tensor[tuple(slice_bounds)], new_shape),
                 variables)
            )
        sliced_buckets.append(sliced_bucket)

    return sliced_buckets, pdict


def run_tf_session(tf_variable, feed_dict):
    """
    Run Tensorflow session and get variable value

    Parameters
    ----------
    tf_variable : tensorflow.Tensor
               variable to evaluate
    feed_dict : dict
               dictionary with placeholder values
    Returns
    -------
    res : numpy.array
               result of the calculation
    """
    # Configure tensorflow for single threaded execution
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )

    with tf.Session(config=session_conf) as sess:
        res = sess.run(tf_variable, feed_dict=feed_dict)

    return res


def process_bucket_tf(bucket):
    """
    Process bucket in the bucket elimination algorithm.
    We multiply all tensors in the bucket and sum over the
    variable which the bucket corresponds to. This way the
    variable of the bucket is removed from the expression.

    Parameters
    ----------
    bucket : list
           List containing tuples of tensors (gates) with their indices.

    Returns
    -------
    tensor : optimizer.Tensor
           wrapper tensor object holding the resulting computational graph
    """
    result_data = bucket[0].data
    result_indices = bucket[0].indices

    for tensor in bucket[1:]:
        expr = utils.get_einsum_expr(list(map(int, result_indices)),
                                     list(map(int, tensor.indices)))

        result_data = tf.einsum(expr, result_data, tensor.data)
        # Merge and sort indices and shapes
        result_indices = tuple(sorted(
            set(result_indices + tensor.indices),
            key=int))

    if len(result_indices) > 0:
        first_index, *result_indices = result_indices
        tag = first_index.identity
    else:
        tag = 'f'
        result_indices = []

    # reduce
    result = opt.Tensor(f'E{tag}', result_indices,
                        data=tf.reduce_sum(result_data, axis=0))
    return result


def eval_tf_buckets(buckets, feed_dict):
    """
    This is a test function which substitutes actual numpy tensors
    in place of buckets

    Parameters
    ----------
    buckets : list of lists
              holds Tensors with tensorflow placeholders in place of data
    feed_dict : dict
              dictionary of {placeholder : numpy.array} pairs
    Returns
    -------
    np_buckets : list of lists
              buckets with Tensors where data are numpy arrays
    """

    np_buckets = []
    for bucket in buckets:
        np_bucket = []
        for tensor in bucket:
            data = run_tf_session(tensor.data, feed_dict)
            np_bucket.append(tensor.copy(data=data))
        np_buckets.append(np_bucket)

    return np_buckets
