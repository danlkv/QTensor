# Building newSZ

1. Clone the NVCOMP repository from https://github.com/NVIDIA/nvcomp.git

2. Change to 'branch-2.2' branch. (`git checkout branch-2.2`)

3. Follow build instructions in NVCOMP repository (you can ignore -DNVCOMP_EXTS_ROOT flag)

4. Copy shared library `nvcomp/build/lib/libnvcomp.so` to current directory (`qtensor/compression/newsz/`)

5. Run the following command: `nvcc --shared --compiler-options '-fPIC' -lnvcomp -o libnewsz_wrapper.so *.cu --library-path=<PATH_TO_NVCOMP_LIB> --library=nvcomp -I/PATH_TO_NVCOMP/nvcomp/build/include/`

# Running newSZ

- Specify --compress=newsz when running main.py
