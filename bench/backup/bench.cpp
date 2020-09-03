/*******************************************************************************
* Copyright 2020 Intel Corporation.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation the
* rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
* sell copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*******************************************************************************/

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "mkl.h"

#include "timer.h"

enum Trans {do_trans, no_trans};
typedef long dim_t;

static const int MIN_NLOOPS = 20;
static const double MIN_TIME = 1;
static const double F_MIN = -1.0;
static const double F_MAX =  1.0;

template <typename T>
static void fill_array(T *matrix, dim_t nrows, dim_t ncols, dim_t ld)
{
    for (dim_t j = 0; j < ncols; j++) {
        for (dim_t i = 0; i < nrows; i++) {
            double scale = (double) random() / RAND_MAX;
            matrix[i + j * ld] = (T) (F_MIN + scale * (F_MAX - F_MIN));
        }
    }
}

template <typename T>
void gemm(enum Trans ta, enum Trans tb, dim_t m, dim_t n, dim_t k, T alpha,
        const T *a, dim_t lda, const T *b, dim_t ldb, T beta, T *c, dim_t ldc)
{
    if (sizeof(T) == 8) {
        cblas_dgemm(CblasColMajor,
                ta == no_trans ? CblasNoTrans : CblasTrans,
                tb == no_trans ? CblasNoTrans : CblasTrans,
                m, n, k, alpha,
               (double *) a, lda,
               (double *) b, ldb, beta,
               (double *) c, ldc);
    } else {
        cblas_sgemm(CblasColMajor,
                ta == no_trans ? CblasNoTrans : CblasTrans,
                tb == no_trans ? CblasNoTrans : CblasTrans,
                m, n, k, alpha,
               (float *) a, lda,
               (float *) b, ldb, beta,
               (float *) c, ldc);
    }
}

template <typename T>
void bench(enum Trans ta, enum Trans tb, dim_t m, dim_t n, dim_t k,
        double *perf_max, double *perf_avg)
{
    dim_t a_ncols = ta == no_trans ? k : m;
    dim_t a_nrows = ta == no_trans ? m : k;
    dim_t b_ncols = tb == no_trans ? n : k;
    dim_t b_nrows = tb == no_trans ? k : n;

    // Using tight leading dimensions.
    dim_t lda = a_nrows;
    dim_t ldb = b_nrows;
    dim_t ldc = m;

    T *a = (T *) malloc(sizeof(*a) * lda * a_ncols);
    T *b = (T *) malloc(sizeof(*b) * ldb * b_ncols);
    T *c = (T *) malloc(sizeof(*c) * ldc * n);

    fill_array(a, a_nrows, a_ncols, lda);
    fill_array(b, b_nrows, b_ncols, ldb);
    fill_array(c, m, n, ldc);

    T alpha = 1.0;
    T beta  = 0.0;
    // Warm-up
    gemm<T>(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

    int count_loops_ref = 0;
    double time = 0.0;
    double ctime = 0.0;
    double time_min = HUGE_VAL;
    double start_time;
    tic(&start_time);
    do {
        gemm<T>(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

        ctime = toc(&start_time);
        time += ctime;
        time_min = fmin(ctime, time_min);

        count_loops_ref++;
    } while (count_loops_ref < MIN_NLOOPS || time < MIN_TIME);

    long long nops = 2LL * m * n * k;
    double time_avg = time / count_loops_ref;
    *perf_avg = (double) nops / time_avg * 1e-9;
    *perf_max = (double) nops / time_min * 1e-9;

    free(a);
    free(b);
    free(c);
}

template <typename T>
void run_size(enum Trans ta, enum Trans tb, dim_t m, dim_t n, dim_t k)
{
    double perf_avg, perf_max;
    bench<T>(ta, tb, m, n, k, &perf_max, &perf_avg);
    printf("%s : %c%c : m = %4ld, n = %4ld, k = %4ld : "
            "perf_avg = %12.6f [GFlop/s], perf_max = %12.6f [GFlop/s]\n",
            sizeof(T) == 8 ? "dgemm" : "sgemm",
            ta == no_trans ? 'N' : 'T',
            tb == no_trans ? 'N' : 'T',
            m, n, k,
            perf_avg, perf_max);
    return;
}

int main(void)
{
    // Modify problem size here if needed.
    //               transa    transb    M     N     K
    run_size<float> (no_trans, no_trans, 4000, 4000, 4000);
    run_size<float> (no_trans, do_trans, 3000, 3000, 3000);
    run_size<float> (do_trans, no_trans, 2000, 2000, 2000);
    run_size<float> (do_trans, do_trans, 1000, 1000, 1000);
    run_size<double>(no_trans, no_trans, 4000, 4000, 4000);
    run_size<double>(no_trans, do_trans, 3000, 3000, 3000);
    run_size<double>(do_trans, no_trans, 2000, 2000, 2000);
    run_size<double>(do_trans, do_trans, 1000, 1000, 1000);

    return EXIT_SUCCESS;
}
