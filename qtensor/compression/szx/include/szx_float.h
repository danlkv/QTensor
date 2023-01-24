/**
 *  @file sz_float.h
 *  @author Sheng Di
 *  @date July, 2017
 *  @brief Header file for the sz_float.c.
 *  (C) 2016 by Mathematics and Computer Science (MCS), Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#ifndef _SZ_Float_H
#define _SZ_Float_H

#ifdef __cplusplus
extern "C" {
#endif

unsigned char * SZ_fast_compress_args_with_prediction_float(float *pred, float *data, size_t *outSize, float absErrBound, size_t r5,
                                            size_t r4, size_t r3, size_t r2, size_t r1, float medianValue, float radius);

void SZ_fast_compress_args_unpredictable_one_block_float(float *oriData, size_t nbEle, float absErrBound,
                                                                unsigned char *outputBytes, int *outSize,
                                                                unsigned char *leadNumberArray_int, float medianValue,
                                                                float radius);
                                                                
size_t computeStateMedianRadius_float(float *oriData, size_t nbEle, float absErrBound, int blockSize,
                                      unsigned char *stateArray, float *medianArray, float *radiusArray) ;
                                      
void max_min_float(float *x, int n, float *tmp_max, float *tmp_min);

void simd_max_min_float(float *x, int n, float *tmp_max, float *tmp_min);

void computeStateMedianRadius_float2(float *oriData, size_t nbEle, float absErrBound,
                                     unsigned char *state, float *median, float *radius) ;
                                     
unsigned char *
SZ_fast_compress_args_unpredictable_blocked_float(float *oriData, size_t *outSize, float absErrBound, size_t nbEle,
                                                  int blockSize) ;
                                                  
unsigned char *
SZ_fast_compress_args_unpredictable_blocked_randomaccess_float_openmp(float *oriData, size_t *outSize, float absErrBound,
                                                               size_t nbEle, int blockSize) ;
                                                               
                                                               
unsigned char *
SZ_fast_compress_args_unpredictable_blocked_randomaccess_float(float *oriData, size_t *outSize, float absErrBound,
    size_t nbEle, int blockSize) ;
    
unsigned char *
SZ_fast_compress_args_unpredictable_float(float *data, size_t *outSize, float absErrBound, size_t r5, size_t r4,
                                          size_t r3, size_t r2, size_t r1, float mValue, float radius);
                                          
unsigned char *SZ_skip_compress_float(float *data, size_t dataLength, size_t *outSize) ;

void computeReqLength_float(double realPrecision, short radExpo, int *reqLength, float *medianValue) ;



#ifdef __cplusplus
}
#endif

#endif /* ----- #ifndef _SZ_Float_H  ----- */

