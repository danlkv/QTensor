#ifndef CUSZX_ENTRY_H
#define CUSZX_ENTRY_H

#include <stdio.h>
#include "cuszx_float.h"
#include "cuszxd_float.h"

#define GPU

extern "C" unsigned char* cuSZx_fast_compress_args_unpredictable_blocked_float(float *oriData, size_t *outSize, float absErrBound, size_t nbEle, int blockSize, float threshold);

extern "C" void cuSZx_fast_decompress_args_unpredictable_blocked_float(float** newData, size_t nbEle, unsigned char* cmpBytes);

extern "C" unsigned char* device_ptr_cuSZx_compress_float(float *oriData, size_t *outSize, float absErrBound, size_t nbEle, int blockSize, float threshold);

extern "C" float* device_ptr_cuSZx_decompress_float(size_t nbEle, unsigned char* cmpBytes);

#endif /* ----- #ifndef CUSZX_ENTRY_H  ----- */
