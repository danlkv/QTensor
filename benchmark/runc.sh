ifort  -g -I$MKLROOT/include  -m64 ./curve.f90  -Wl,--start-group $MKLROOT/lib/intel64/libmkl_gf_lp64.a $MKLROOT/lib/intel64/libmkl_intel_thread.a $MKLROOT/lib/intel64/libmkl_core.a -Wl,--end-group   -L$MKLROOT/../compilers_and_libraries_2020.1.217/linux/compiler/lib/intel64_lin/ -liomp5 -lpthread  -ldl -DUSE_MM_MALLOC -DSTANDALONE -DOPENQU_HAVE_MPI -DMKL_ILP64

