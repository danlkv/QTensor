using LinearAlgebra
using BenchmarkTools

function run(n)
    nops = 2 * n ^ 3
    A, B, C = rand(n, n), rand(n, n), rand(n, n)
    b = @benchmark mul!($C, $A, $B)
    return nops / mean(b.times)
end

function bench()
    BLAS.set_num_threads(56)
    for i in 4096:-256:512
        println(i, ", ", run(i))
    end
    for i in 512:-32:64
        println(i, ", ", run(i))
    end
    for i in 64:-1:16
        println(i, ", ", run(i))
    end
end

