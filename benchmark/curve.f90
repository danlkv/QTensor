program   Test1

integer         n, p
integer         idx, jdx, power, maxiter
double precision            tmp
double precision            start, finish
double precision            mi, median, mean, ma
double precision            alpha, beta
parameter       (alpha=1.0, beta=0.0)
parameter       (power=13)
parameter       (maxn = 2 ** power)
parameter       (maxiter=100)
double precision, allocatable :: A(:,:), B(:,:), C(:,:)
double precision            marks(maxiter)

external        DGEMM

batch = 1000
do p=3,power
    n = 2 ** p
    allocate(A(n,n))
    allocate(B(n,n))
    allocate(C(n,n))

    call random_number(A(n,n))
    call random_number(B(n,n))
    call random_number(C(n,n))


    call DGEMM('N','N',n,n,n,alpha,A,n,B,n,beta,C,n)
    do idx=1,maxiter
        call cpu_time(start)
        do jdx=1, batch
            call DGEMM('N','N',n,n,n,alpha,A,  n,  B,  n,  beta,C,  n  )
        end do
        call cpu_time(finish)
        marks(idx) = (finish - start) / batch
        if (finish - start .gt. 1) then
            batch = batch / 2 + 1
        end if
    end do

    do idx=1,maxiter
        do jdx=idx,maxbatch
            if (marks(idx).gt.marks(jdx)) then
                tmp = marks(idx)
                marks(idx) = marks(jdx)
                marks(jdx) = tmp
            end if
        end do
    end do

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

    deallocate(A)
    deallocate(B)
    deallocate(C)
end do 

stop
end

