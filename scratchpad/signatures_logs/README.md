# Log of operations performed

Three levels of logs are provided:

 - Signatures of contractions only `log_localhost_signatures_N'
 - Signatures with buckets `log_with_buckets_N`
 - Full log `log_full_N`

Where `N` is the size of the task


### Signature format:
`list, list -> list`

for example:

```T_ijk * T_klj = [1,2,3], [3,4,2] -> [1,2,3,4]```

### Bucket format:

`NAME( v_i, .... )` where `v_i` are unique indices of the tensor with id `i`

For example:

``` T_ijkl = E982(v_999,v_1007,v_1012,v_1014) ```



