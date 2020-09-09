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
#include "time.h"

enum Trans {do_trans, no_trans};
typedef MKL_INT dim_t;

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

static inline int ld_fix(int m, int size_elt)
{
    int ld, offset = 64 / size_elt;
    ld = (m + offset - 1) / offset * offset;
    ld = (((ld * size_elt) % 256) == 0) ? ld + offset : ld;
    return ld; 
}

template <typename T>
void bench(enum Trans ta, enum Trans tb, dim_t n, dim_t k,
        double *perf_max, double *perf_avg)
{
    dim_t a_ncols = ta == no_trans ? n : n;
    dim_t a_nrows = ta == no_trans ? n : n;
    dim_t b_ncols = tb == no_trans ? n : n;
    dim_t b_nrows = tb == no_trans ? n : n;

    // Using tight leading dimensions.
    dim_t lda = ld_fix(a_nrows, sizeof(T));
    dim_t ldb = ld_fix(b_nrows, sizeof(T));
    dim_t ldc = ld_fix(n, sizeof(T));

    double *numm;
    double num;

    const double **a_batch = (const double **) malloc(sizeof(numm) * k);
    const double **b_batch = (const double **) malloc(sizeof(numm) * k);
    const double **c_batch = (const double **) malloc(sizeof(numm) * k);
    int i;

    for (i = 0; i < k; i++)
    {
        double *a = (double *) malloc(sizeof(num) * lda * a_ncols);
        double *b = (double *) malloc(sizeof(num) * ldb * b_ncols);
        double *c = (double *) malloc(sizeof(num) * ldc * n);

        fill_array(a, a_nrows, a_ncols, lda);
        fill_array(b, b_nrows, b_ncols, ldb);
        fill_array(c, n, n, ldc);

        a_batch[i] = a;
        b_batch[i] = b;
        c_batch[i] = c;
    }

    const CBLAS_LAYOUT layout = CblasColMajor;
    const CBLAS_TRANSPOSE trans[1] = { CblasNoTrans };
    const MKL_INT m_array[1] = { n };
    const MKL_INT n_array[1] = { n };
    const MKL_INT k_array[1] = { n };
    const double alpha_array[1] = { 1.0 };
    const double **a_array = a_batch;
    const MKL_INT lda_array[1] = { lda };
    const double **b_array = b_batch;
    const MKL_INT ldb_array[1] = { ldb };
    const double beta_array[1] = { 0.0 };
    double **c_array = c_batch;
    const MKL_INT ldc_array[1] = { ldc };
    const MKL_INT group_count = 1;
    const MKL_INT group_size[1] = { k };


    // Warm-up
    cblas_dgemm_batch(layout, trans, trans, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size); // Segfault


    int count_loops_ref = 0;
    double time = 0.0;
    double ctime = 0.0;
    double time_min = HUGE_VAL;
    struct timespec start_time;
    tic(&start_time);
    do {
        cblas_dgemm_batch(layout, trans, trans, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size); // Segfault

        ctime = toc(&start_time);
        time += ctime;
        time_min = fmin(ctime, time_min);

        count_loops_ref++;
    } while (count_loops_ref < MIN_NLOOPS || time < MIN_TIME);

    long long nops = 2LL * n * n * n;
    nops *= k;
    double time_avg = time / count_loops_ref;
    *perf_avg = (double) nops / time_avg * 1e-9;
    *perf_max = (double) nops / time_min * 1e-9;

    for (i = 0; i < k; i++)
    {
        free(a_batch[i]);
        free(b_batch[i]);
        free(c_batch[i]);
    }
    free(a_batch);
    free(b_batch);
    free(c_batch);
}

template <typename T>
void run_size(enum Trans ta, enum Trans tb, dim_t n, dim_t k)
{
    double perf_avg, perf_max;
    bench<T>(ta, tb, n, k, &perf_max, &perf_avg);
    printf("%4ld, %4ld, %12.6f, %12.6f\n", n, k, perf_avg, perf_max);
    return;
}

int main(void)
{
    // Modify problem size here if needed.
    //               transa    transb    M     N     K
    int i, k;

    for (k = 10000; k >= 10; k /= 10)
        for (i = 256; i >= 2; i /= 2)
            run_size<double>(no_trans, no_trans, i, k);

    return EXIT_SUCCESS;
}

