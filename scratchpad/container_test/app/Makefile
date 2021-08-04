# Generated automatically from Makefile.in by configure.
ALL: pi
SHELL = /bin/sh
DIRS = 
pi: pi.c
	mpicc -o pi pi.c -lm 
profile.alog: pi.c
	mpicc -o pi.log -mpilog pi.c -lm
	mpirun -np 4 pi.log
	/bin/mv pi.log_profile.log profile.alog
clean:
	/bin/rm -f pi pi.o pi.log
	#for dir in $(DIRS) ; do \
	#    ( cd $$dir ; make clean ) ; done
