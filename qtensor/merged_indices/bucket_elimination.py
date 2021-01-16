import itertools

def bucket_elimination(buckets, ibunch, process_bucket_fn,
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
    ibunch : list of lists of indices to contract.
    process_bucket_fn : function
    function that will process buckets, takes list of indices to contract + buckets
    n_var_nosum : int, optional
              number of variables that have to be left in the
              result. Expected at the end of bucket list
    Returns
    -------
    result : numpy.array
    """
    n_var_contract = len(buckets) - n_var_nosum
    assert len(ibunch) == len(buckets), "Buckets length should be same as ibunch length"

    result = None
    for ixs, bucket in zip(ibunch, buckets[:n_var_contract]):
        if len(bucket) > 0:
            tensor = process_bucket_fn(ixs, bucket)
            if len(tensor.indices) > 0:
                # tensor is not scalar.
                # Move it to appropriate bucket
                smallest_ix = min([int(x) for x in tensor.indices])
                appended = False
                for j, ixs in enumerate(ibunch):
                    if smallest_ix in map(int, ixs):
                        buckets[j].append(tensor)
                        appended = True
                if not appended:
                    raise Exception('Algorithmic error, investigate.')
            else:   # tensor is scalar
                if result is not None:
                    result *= tensor
                else:
                    result = tensor

    # form a single list of the rest if any
    rest = list(itertools.chain.from_iterable(buckets[n_var_contract:]))
    if len(rest) > 0:
        # only multiply tensors
        tensor = process_bucket_fn([], rest, no_sum=True)
        if result is not None:
            result *= tensor
        else:
            result = tensor
    return result

