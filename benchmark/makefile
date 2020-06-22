
ifndef MKLROOT
	MKLROOT = /home/cameron/intel/mkl
endif

# HOST = $(shell hostname)

# CC=gfortran
CC = ifort

INC = ${MKLROOT}/include

GROUP = ${MKLROOT}/lib/intel64/libmkl_gf_lp64.a 
GROUP += ${MKLROOT}/lib/intel64/libmkl_intel_thread.a 
GROUP += ${MKLROOT}/lib/intel64/libmkl_core.a

LIB = ${MKLROOT}/../compilers_and_libraries_2020.1.217/linux/compiler/lib/intel64_lin/

OPTS = -DUSE_MM_MALLOC -DSTANDALONE -DOPENQU_HAVE_MPI -DMKL_ILP64

clean:
	-rm -rf ./build

build:
	mkdir -p ./build
	${CC} -I${INC} -m64 -mkl ./src/curve.f90 -Wl,--start-group ${GROUP} -Wl,--end-group -L${LIB} -liomp5 -lpthread -ldl ${OPTS} -o ./build/curve.out


VARS:=cycles,instructions$\
,cache-misses,page-faults$\
,L1-dcache-loads,L1-dcache-load-misses$\
,L1-dcache-prefetches,L1-dcache-prefetch-misses
stat: clean build
	perf stat -B -e $(subst : ,:, $(VARS)) ./build/curve.out $(power)

