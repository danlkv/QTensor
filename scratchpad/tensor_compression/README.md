## Structure

- `test.py` - script used to generate tensors
- `tensor*.bin` - binary files with tensor data


## Running `test.py`

In case you want to generate more tensors, install `qtensor` as described in
README.md of qtensor github repository, then run test.py.

Parameters for configuring tensor sizes is `requested_sizes` variable. Also can
configure SEED global variable which will give a different graph for QAOA. Also
can change values in `get_gb` function.

If you need a large tensor, may need to change `degree` and `p` values to a
larger value.

## Tensor format

filename format: `tensor_dims-D_dtype-T.bin`

* `dims-D` - D is number of dimensions. Size of each dimension is 2. Number of
  elements in tensor is 2^D.
* `dtype-T` - T is datatype. complex128 takes 16 bytes per element and
  complex64 takes 8 bytes.

Large tensors will have `complex128` dtype, but I still included `complex64`
for reference and it may be useful to look at it.

Tensors are stored in flat format, complex numbers are stored in pairs (real,
imaginary) parts. So a 1-dimensional 2-sized array would be [real1, imag1,
real2, imag2].
