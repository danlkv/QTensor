import itertools

def bucket_elimination(buckets, process_bucket_fn,
                       n_var_nosum=0):
    """
    Algorithm to evaluate a contraction of a large number of tensors.
    The variables to contract over are assigned ``buckets`` which
    hold tensors having respective variables. The algorithm
    proceeds through contracting one variable at a time, thus we eliminate
    buckets one by one.

    Parameters
    ----------
    buckets : list of lists
    process_bucket_fn : function
              function that will process this kind of buckets
    n_var_nosum : int, optional
              number of variables that have to be left in the
              result. Expected at the end of bucket list
    Returns
    -------
    result : numpy.array
    """
    # import pdb
    # pdb.set_trace()
    n_var_contract = len(buckets) - n_var_nosum

    result = None
    for n in range(n_var_contract):
        bucket = buckets[n]
        if len(bucket) > 0:
            tensor = process_bucket_fn(bucket)
            #-- Memory management
            buckets[n] = []
            #--

            if len(tensor.indices) > 0:
                # tensor is not scalar.
                # Move it to appropriate bucket
                first_index = int(tensor.indices[-1])
                buckets[first_index].append(tensor)
            else:   # tensor is scalar
                if result is not None:
                    result *= tensor
                else:
                    result = tensor
        # free up space, the tensors are no longer needed
        buckets[n] = []

    # form a single list of the rest if any
    rest = list(itertools.chain.from_iterable(buckets[n_var_contract:]))
    if len(rest) > 0:
        # only multiply tensors
        tensor = process_bucket_fn(rest, no_sum=True)
        if result is not None:
            result *= tensor
        else:
            result = tensor
    return result

