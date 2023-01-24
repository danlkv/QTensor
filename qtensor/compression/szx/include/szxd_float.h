/**
 *  @file szxd_float.h
 *  @author Sheng Di
 *  @date Feb, 2022
 *  @brief Header file for the szd_float.c.
 *  (C) 2016 by Mathematics and Computer Science (MCS), Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#ifndef _SZXD_Float_H
#define _SZXD_Float_H

#ifdef __cplusplus
extern "C" {
#endif

void SZ_fast_decompress_args_with_prediction_float(float** newData, float* pred, size_t r5, size_t r4, size_t r3, size_t r2, size_t r1, unsigned char* cmpBytes, size_t cmpSize);
int SZ_fast_decompress_args_unpredictable_one_block_float(float* newData, size_t blockSize, unsigned char* cmpBytes);
void SZ_fast_decompress_args_unpredictable_blocked_float(float** newData, size_t nbEle, unsigned char* cmpBytes);
void SZ_fast_decompress_args_unpredictable_blocked_randomaccess_float(float** newData, size_t nbEle, unsigned char* cmpBytes);
void SZ_fast_decompress_args_unpredictable_blocked_randomaccess_float_openmp(float** newData, size_t nbEle, unsigned char* cmpBytes);

void SZ_fast_decompress_args_unpredictable_float(float** newData, size_t r5, size_t r4, size_t r3, size_t r2, size_t r1, unsigned char* cmpBytes, 
size_t cmpSize);

#ifdef __cplusplus
}
#endif

#endif /* ----- #ifndef _SZXD_Float_H  ----- */
