program     Curve

USE mkl_service

integer         n, p, batch
integer         idx, jdx, power, maxiter
double precision            tmp
double precision            start, finish
double precision            mi, median, mean, ma
double precision            alpha, beta
parameter       (alpha=1.0, beta=0.0)
parameter       (power=12)
integer*8 maxn
parameter       (maxn = 2 ** power)
parameter       (maxiter=100)
double precision A,B,C
pointer (Aptr,A(maxn,*)), (Bptr,B(maxn,*)), (Cptr,C(maxn,*))
double precision            marks(maxiter)

external        DGEMM

! Set number of iterations in internal loop
batch = 1000

Aptr = mkl_calloc(maxn * maxn, 8, 64)
Bptr = mkl_calloc(maxn * maxn, 8, 64)
Cptr = mkl_calloc(maxn * maxn, 8, 64)


! Iterate over different matrix sizes
do p=3,power
    ! Set new matrix sizes
    n = 2 ** p

    ! Allocate square matrices

    ! Randomly initialize values
    call random_number(A(n,n))
    call random_number(B(n,n))
    call random_number(C(n,n))

    ! Try to load matrices in cache
    call DGEMM('N','N',n,n,n,alpha,A, ,B,n,beta,C,n)
    
    ! Get `maxiter` benchmark samples
    do idx=1,maxiter

        ! Time `batch` evaluations
        call cpu_time(start)
        do jdx=1,batch
            call DGEMM('N','N',n,n,n,alpha,A,  n,  B,  n,  beta,C,  n  )
        end do
        call cpu_time(finish)

        ! Average time over batch size
        marks(idx) = (finish - start) / batch

        ! If timing takes too long, reduce batch size if possible
        if (finish - start .gt. 1) then
            batch = batch / 2 + 1
        end if
    end do

    ! Poor selection sort
    do idx=1,maxiter
        do jdx=idx,maxbatch
            if (marks(idx).gt.marks(jdx)) then
                tmp = marks(idx)
                marks(idx) = marks(jdx)
                marks(jdx) = tmp
            end if
        end do
    end do

    ! Compute and print min, mean, median, max
    mi = marks(1)
    if (mod(maxiter, 2) == 0) then
        median = (marks(maxiter / 2) + marks(maxiter / 2 + 1)) / 2.0
    else
        median = marks(maxiter / 2 + 1)
    end if
    mean = sum(marks) / maxiter
    ma = marks(maxiter)

    print*
    print '(i6, 10(",", F15.12))', n, mi, median, mean,  ma

end do 

! Deallocate matrices
call mkl_free(Aptr)
call mkl_free(Bptr)
call mkl_free(Cptr)

stop

end

