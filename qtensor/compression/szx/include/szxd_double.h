/**
 *  @file szxd_double.h
 *  @author Sheng Di
 *  @date Feb, 2022
 *  @brief Header file for the szd_double.c.
 *  (C) 2016 by Mathematics and Computer Science (MCS), Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#ifndef _SZXD_Double_H
#define _SZXD_Double_H

#ifdef __cplusplus
extern "C" {
#endif

int SZ_fast_decompress_args_unpredictable_one_block_double(double* newData, size_t blockSize, unsigned char* cmpBytes);
void SZ_fast_decompress_args_unpredictable_blocked_double(double** newData, size_t nbEle, unsigned char* cmpBytes);
void SZ_fast_decompress_args_unpredictable_blocked_randomaccess_double(double** newData, size_t nbEle, unsigned char* cmpBytes);
void SZ_fast_decompress_args_unpredictable_blocked_randomaccess_double_openmp(double** newData, size_t nbEle, unsigned char* cmpBytes);

void SZ_fast_decompress_args_unpredictable_double(double** newData, size_t r5, size_t r4, size_t r3, size_t r2, size_t r1, unsigned char* cmpBytes, 
size_t cmpSize);

#ifdef __cplusplus
}
#endif

#endif /* ----- #ifndef _SZXD_Double_H  ----- */
