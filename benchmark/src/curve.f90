program Curve

use f95_precision, only: wp => dp
use blas95, only: gemm
use mkl_service

implicit none

integer*8 :: n
integer*8, parameter :: maxn = 4104
integer*8 :: p, p_lo, p_up
character(len=3) :: p_str
integer :: idx, iter, jdx, batch
real(wp), parameter :: alpha = 1, beta = 1
real(wp) :: start, finish, tmp

double precision            mi, median, mean, ma
parameter       (iter=100)
double precision A,B,C
pointer (Aptr,A(maxn,*)), (Bptr,B(maxn,*)), (Cptr,C(maxn,*))
real(wp) :: marks(iter)
character(len=3) power_str
integer stat
integer power_arg

! parse the power num
call get_command_argument(1, p_str)
read(p_str, *, iostat=stat) p_lo
call get_command_argument(2, p_str)
read(p_str, *, iostat=stat) p_up

! print* , power_str
! read(power_str, *, iostat=stat) power_arg
! print* , power_arg

! Set number of iterations in internal loop
batch = 1000


Aptr = mkl_calloc(maxn * maxn, 8, 64)
Bptr = mkl_calloc(maxn * maxn, 8, 64)
Cptr = mkl_calloc(maxn * maxn, 8, 64)

! Iterate over different matrix sizes
do p=p_lo,p_up
    ! Set new matrix sizes
    n = 2 ** p

    ! Allocate square matrices

    ! Randomly initialize values
    call random_number(A(n,n))
    call random_number(B(n,n))
    call random_number(C(n,n))

    ! Try to load matrices in cache
    call DGEMM('N','N',n,n,n,alpha,A,maxn,B,maxn,beta,C,maxn)
    
    ! Get `maxiter` benchmark samples
    do idx=1,iter

        ! Time `batch` evaluations
        call cpu_time(start)
        do jdx=1,batch
            call DGEMM('N','N',n,n,n,alpha,A,maxn,B,maxn,beta,C,maxn)
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
    do idx=1,iter
        do jdx=idx,iter
            if (marks(idx).gt.marks(jdx)) then
                tmp = marks(idx)
                marks(idx) = marks(jdx)
                marks(jdx) = tmp
            end if
        end do
    end do

    ! Compute and print min, mean, median, max
    mi = marks(1)
    if (mod(iter, 2) == 0) then
        median = (marks(iter / 2) + marks(iter / 2 + 1)) / 2.0
    else
        median = marks(iter / 2 + 1)
    end if
    mean = sum(marks) / iter
    ma = marks(iter)

    print '(i6, 10(",", F15.12))', n, mi, median, mean,  ma

end do 

! Deallocate matrices
call mkl_free(Aptr)
call mkl_free(Bptr)
call mkl_free(Cptr)

stop

end

