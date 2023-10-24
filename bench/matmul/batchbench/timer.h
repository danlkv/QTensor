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

#ifndef TIMER_H
#define TIMER_H

#include "mkl.h"
#include "time.h"

// Get current time passed as input.
static inline void tic(struct timespec *start)
{
    clock_gettime(CLOCK_MONOTONIC, &*start);// dsecnd();
}

// Return the elapsed time in seconds since last tic or toc.
static inline double toc(struct timespec *start_time)
{
    struct timespec end_time;
    clock_gettime(CLOCK_MONOTONIC, &end_time); // dsecnd();
    double elapsed_time = 1e9 * (end_time.tv_sec - start_time->tv_sec) + end_time.tv_nsec - start_time->tv_nsec;

    // Setting current time.
    *start_time = end_time;

    return elapsed_time / 1e9;
}

#endif // TIMER_H
