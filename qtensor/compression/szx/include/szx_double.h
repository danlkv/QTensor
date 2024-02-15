/**
 *  @file szx_double.h
 *  @author Sheng Di
 *  @date July, 2017
 *  @brief Header file for the sz_double.c.
 *  (C) 2016 by Mathematics and Computer Science (MCS), Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include <szx_float.h>

#ifndef _SZ_Double_H
#define _SZ_Double_H

#ifdef __cplusplus
extern "C" {
#endif

void SZ_fast_compress_args_unpredictable_one_block_double(double *oriData, size_t nbEle, float absErrBound,
                                                                unsigned char *outputBytes, int *outSize,
                                                                unsigned char *leadNumberArray_int, float medianValue,
                                                                float radius);
                                                                
size_t computeStateMedianRadius_double(double *oriData, size_t nbEle, float absErrBound, int blockSize,
                                      unsigned char *stateArray, float *medianArray, float *radiusArray) ;
                                      
void max_min_double(double *x, int n, double *tmp_max, double *tmp_min);

void simd_max_min_double(double *x, int n, double *tmp_max, double *tmp_min);

void computeStateMedianRadius_double2(double *oriData, size_t nbEle, float absErrBound,
                                     unsigned char *state, float *median, float *radius) ;
                                     
unsigned char *
SZ_fast_compress_args_unpredictable_blocked_double(double *oriData, size_t *outSize, float absErrBound, size_t nbEle,
                                                  int blockSize) ;
                                                  
unsigned char *
SZ_fast_compress_args_unpredictable_blocked_randomaccess_double_openmp(double *oriData, size_t *outSize, 
									float absErrBound, size_t nbEle, int blockSize) ;
                                                               
                                                               
unsigned char *
SZ_fast_compress_args_unpredictable_blocked_randomaccess_double(double *oriData, size_t *outSize, 
								float absErrBound, size_t nbEle, int blockSize) ;
    
unsigned char *
SZ_fast_compress_args_unpredictable_double(double *data, size_t *outSize, float absErrBound, size_t r5, size_t r4,
                                          size_t r3, size_t r2, size_t r1, float mValue, float radius);
                                          
unsigned char *SZ_skip_compress_double(double *data, size_t dataLength, size_t *outSize) ;

void computeReqLength_double(float realPrecision, short radExpo, int *reqLength, float *medianValue) ;



#ifdef __cplusplus
}
#endif

#endif /* ----- #ifndef _SZ_Double_H  ----- */

