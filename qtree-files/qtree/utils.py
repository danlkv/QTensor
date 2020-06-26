"""
This module implements different utility functions
which don't definitely fit somewhere else. It also serves
for dependency disentanglement purposes.
"""
import numpy as np


def unravel_index(value, dimensions):
    """
    Python analog of the numpy.unravel_index
    Supports more than 32 dimensions
    """
    unravel_size = np.prod(dimensions)
    val = value
    coords = [None, ] * len(dimensions)

    for ii in range(len(dimensions) - 1, -1, -1):
        if val < 0:
            raise ValueError('Index is out of bounds for array with size'
                             ' {}'.format(unravel_size))
        tmp = val // dimensions[ii]
        coords[ii] = val % dimensions[ii]
        val = tmp
    return tuple(coords)


def slice_from_bits(value, vars_to_slice):
    """
    Generates a 1x1x1x1x..x1 slice (a single entry) of a set of
    variables. The order of variables is given by
    the input list order and their ranges are retrieved from variable
    objects. The integer is decomposed into a multiindex
    The width of the value is the length of the variable list

    value: int
           Linear index in the size(v1) .. size(vN) array
    vars_to_slice: list of Var
           Variables defining the multidimensional array
    """

    dimensions = [var.size for var in vars_to_slice]
    multiindex = unravel_index(value, dimensions)

    return {var: slice(at, at+1) for var, at
            in zip(vars_to_slice, multiindex)}


def slice_values_generator(vars_to_slice, offset, comm_size):
    """
    Generates dictionaries containing consequtive slices for
    each variable. Slices are generated according to the order
    of variables in var_parallel, in the big endian order (last
    element in var_parallel changes fastest).

    Parameters
    ----------
    vars_to_slice : list
            variables to parallelize over
    offset : int
            offset to start from. Usually equals to rank
    comm_size : int
            Step size in the task array. Usually the number
            of parallel workers

    Yields
    ------
    slice_dict : dict
            dictionary of {var_parallel : value} pairs
    """
    dimensions = [var.size for var in vars_to_slice]
    total_tasks = np.prod(dimensions)

    # iterate over all possible values of variables idx_parallel
    for pos in range(offset, total_tasks, comm_size):
        multiindex = list(unravel_index(pos, dimensions))

        yield {var_parallel: slice(at, at+1)
               for var_parallel, at
               in zip(vars_to_slice, multiindex)}


def num_to_alpha(integer):
    """
    Transform integer to [a-z], [A-Z]

    Parameters
    ----------
    integer : int
        Integer to transform

    Returns
    -------
    a : str
        alpha-numeric representation of the integer
    """
    ascii = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if integer < 52:
        return ascii[integer]
    else:
        raise ValueError('Too large index for einsum')


def num_to_alnum(integer):
    """
    Transform integer to [a-z], [a0-z0]-[a9-z9]

    Parameters
    ----------
    integer : int
        Integer to transform

    Returns
    -------
    a : str
        alpha-numeric representation of the integer
    """
    ascii_lowercase = 'abcdefghijklmnopqrstuvwxyz'
    if integer < 26:
        return ascii_lowercase[integer]
    else:
        return ascii_lowercase[integer % 25 - 1] + str(integer // 25)


def get_einsum_expr(idx1, idx2):
    """
    Takes two tuples of indices and returns an einsum expression
    to evaluate the sum over repeating indices

    Parameters
    ----------
    idx1 : list-like
          indices of the first argument
    idx2 : list-like
          indices of the second argument

    Returns
    -------
    expr : str
          Einsum command to sum over indices repeating in idx1
          and idx2.
    """
    result_indices = sorted(list(set(idx1 + idx2)))
    # remap indices to reduce their order, as einsum does not like
    # large numbers
    idx_to_least_idx = {old_idx: new_idx for new_idx, old_idx
                        in enumerate(result_indices)}

    str1 = ''.join(num_to_alpha(idx_to_least_idx[ii]) for ii in idx1)
    str2 = ''.join(num_to_alpha(idx_to_least_idx[ii]) for ii in idx2)
    str3 = ''.join(num_to_alpha(idx_to_least_idx[ii]) for ii in result_indices)
    return str1 + ',' + str2 + '->' + str3


def sequential_profile_decorator(filename=None):
    """
    Profiles execution of a function and writes stats to
    the specified file
    """
    import cProfile

    def prof_decorator(f):
        def wrap_f(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            result = f(*args, **kwargs)
            pr.disable()

            if filename is None:
                pr.print_stats()
            else:
                filename_r = filename
                pr.dump_stats(filename_r)

            return result
        return wrap_f
    return prof_decorator


def mpi_profile_decorator(comm, filename=None):
    """
    Profiles execution of MPI processes and writes stats to
    separate files
    """
    import cProfile

    def prof_decorator(f):
        def wrap_f(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            result = f(*args, **kwargs)
            pr.disable()

            if filename is None:
                pr.print_stats()
            else:
                filename_r = filename + ".{}".format(comm.rank)
                pr.dump_stats(filename_r)

            return result
        return wrap_f
    return prof_decorator
