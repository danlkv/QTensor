CC = ifort

OPTS = -DUSE_MM_MALLOC -DSTANDALONE -DOPENQU_HAVE_MPI -DMKL_ILP64

clean:
	-rm -rf ./build

FLAGS = -m64 -mkl -pthread -static-intel -liomp5 -ldl -parallel

build: clean
	mkdir -p ./build
	${CC} ${FLAGS} -O0 ./src/curve.f90 -o ./build/curve.out

debug: clean
	${CC} ${FLAGS} -g -O0 -traceback -debug all ./src/curve.f90 -o ./build/curve.out

opt: clean
	mkdir -p ./build
	${CC} ${FLAGS} -O3 ./src/curve.f90 -o ./build/curve.out

VARS:=cycles,instructions$\
,cache-misses,page-faults$\
,L1-dcache-loads,L1-dcache-load-misses$\
,L1-dcache-prefetches,L1-dcache-prefetch-misses
stat: build
	perf stat -B -e $(subst : ,:, $(VARS)) ./build/curve.out ${POWER}

