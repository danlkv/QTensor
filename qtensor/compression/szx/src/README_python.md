# Using the Python Wrapper for QC Compression
### Steps to Build:
1. Clone the repository, switch to threshold_integrate branch

2. Change directory to "SZx/szx/src/"

3. Run the following NVCC command:
nvcc --shared --compiler-options '-fPIC' -I ../include/ -I $CUDA_SAMPLES_PATH -o cuszx_wrapper.so *.cu *.c

    - $CUDA_SAMPLES_PATH should be the path to the include/ directory of CUDA's samples

### Using the Python API:
**def cuszx_device_compress(oriData, outSize, absErrBound, nbEle, blockSize,threshold)**
- Parameters:
    - oriData: CUPY array to be compressed, should be flattened to 1-D
    - outSize: CTypes size_t pointer, will store the resulting compressed data size in bytes
    - absErrBound: Float, the relative-to-value-range error bound for compression
    - nbEle: Integer, number of data elements
    - blockSize: Integer, cuSZx runtime parameter (recommended value = 256)
    - threshold: Float, the relative-to-value-range threshold for compression
- Returns:
    - o_bytes: GPU device pointer to compressed bytes
    - outSize: See 'Parameters'

**def cuszx_device_decompress(nbEle, cmpBytes)**
- Parameters:
    - nbEle: Integer, number of data elements
    - cmpBytes: GPU device pointer to compressed bytes
- Returns:
    - newData: GPU float pointer (CTypes) to decompressed data