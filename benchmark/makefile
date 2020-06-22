CC = ifort


GROUP = ${MKLROOT}/lib/intel64/libmkl_gf_lp64.a 
GROUP += ${MKLROOT}/lib/intel64/libmkl_intel_thread.a 
GROUP += ${MKLROOT}/lib/intel64/libmkl_core.a

LIB = ${MKLROOT}/../compilers_and_libraries_2020.1.217/linux/compiler/lib/intel64_lin/

OPTS = -DUSE_MM_MALLOC -DSTANDALONE -DOPENQU_HAVE_MPI -DMKL_ILP64

clean:
	-rm -rf ./build

# -liomp5 -ldl 
build: clean
	mkdir -p ./build
	${CC} -m64 -mkl -pthread -static-intel ./src/curve.f90  -o ./build/curve.out


VARS:=cycles,instructions$\
,cache-misses,page-faults$\
,L1-dcache-loads,L1-dcache-load-misses$\
,L1-dcache-prefetches,L1-dcache-prefetch-misses
stat: clean build
	perf stat -B -e $(subst : ,:, $(VARS)) ./build/curve.out $(power)

