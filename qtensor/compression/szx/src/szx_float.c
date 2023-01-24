/**
 *  @file szx_float.c
 *  @author Sheng Di, Kai Zhao
 *  @date Aug, 2022
 *  @brief SZ_Init, Compression and Decompression functions
 *  (C) 2022 by Mathematics and Computer Science (MCS), Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */


#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "szx.h"
#include "szx_float.h"
#include "szx_BytesToolkit.h"
#include "szx_TypeManager.h"
#include <assert.h>

#ifdef _OPENMP
#include "omp.h"
#endif

#if defined(__AVX__) || defined(__AVX2__)  || defined(__AVX512F__)
#include <immintrin.h>
#endif

unsigned char *
SZ_fast_compress_args_with_prediction_float(float *pred, float *data, size_t *outSize, float absErrBound, size_t r5,
                                            size_t r4, size_t r3, size_t r2, size_t r1, float medianValue,
                                            float radius) {
    size_t dataLength = computeDataLength(r5, r4, r3, r2, r1);
    float *delta = (float *) malloc(sizeof(float) * dataLength);
    size_t i = 0;
    for (i = 0; i < dataLength; i++)
        delta[i] = data[i] - pred[i];
    unsigned char *output = SZ_fast_compress_args_unpredictable_float(delta, outSize, absErrBound, r5, r4, r3, r2, r1,
                                                                      medianValue, radius);
    return output;
}

inline void SZ_fast_compress_args_unpredictable_one_block_float(float *oriData, size_t nbEle, float absErrBound,
                                                                unsigned char *outputBytes, int *outSize,
                                                                unsigned char *leadNumberArray_int, float medianValue,
                                                                float radius) {
    size_t totalSize = 0, i = 0;

    int reqLength;

    //compute median, value range, and radius

    short radExpo = getExponent_float(radius);
    computeReqLength_float(absErrBound, radExpo, &reqLength, &medianValue);

    int reqBytesLength = reqLength / 8;
    int resiBitsLength = reqLength % 8;
    int rightShiftBits = 0;

    size_t leadNumberArray_size = nbEle % 4 == 0 ? nbEle / 4 : nbEle / 4 + 1;

    register lfloat lfBuf_pre;
    register lfloat lfBuf_cur;
    lfBuf_pre.ivalue = 0;

    unsigned char *leadNumberArray = outputBytes + 1 + sizeof(float);

    unsigned char *exactMidbyteArray = leadNumberArray + leadNumberArray_size;

    if (resiBitsLength != 0) {
        rightShiftBits = 8 - resiBitsLength;
        reqBytesLength++;
    }

    register unsigned char leadingNum = 0;
    size_t residualMidBytes_size = 0;
    if (sysEndianType == LITTLE_ENDIAN_SYSTEM) {
        if (reqBytesLength == 2) {
            for (i = 0; i < nbEle; i++) {
                leadingNum = 0;
                lfBuf_cur.value = oriData[i] - medianValue;

                lfBuf_cur.ivalue = lfBuf_cur.ivalue >> rightShiftBits;

                lfBuf_pre.ivalue = lfBuf_cur.ivalue ^ lfBuf_pre.ivalue;

                if (lfBuf_pre.ivalue >> 8 == 0)
                    leadingNum = 3;
                else if (lfBuf_pre.ivalue >> 16 == 0)
                    leadingNum = 2;
                else if (lfBuf_pre.ivalue >> 24 == 0)
                    leadingNum = 1;

                leadNumberArray_int[i] = leadingNum;

                if (leadingNum == 0) {
                    exactMidbyteArray[residualMidBytes_size] = lfBuf_cur.byte[2];
                    exactMidbyteArray[residualMidBytes_size + 1] = lfBuf_cur.byte[3];
                    residualMidBytes_size += 2;
                } else if (leadingNum == 1) {
                    exactMidbyteArray[residualMidBytes_size] = lfBuf_cur.byte[2];
                    residualMidBytes_size++;
                }

                lfBuf_pre = lfBuf_cur;
            }
        } else if (reqBytesLength == 3) {
            for (i = 0; i < nbEle; i++) {
                leadingNum = 0;
                lfBuf_cur.value = oriData[i] - medianValue;

                lfBuf_cur.ivalue = lfBuf_cur.ivalue >> rightShiftBits;

                lfBuf_pre.ivalue = lfBuf_cur.ivalue ^ lfBuf_pre.ivalue;

                if (lfBuf_pre.ivalue >> 8 == 0)
                    leadingNum = 3;
                else if (lfBuf_pre.ivalue >> 16 == 0)
                    leadingNum = 2;
                else if (lfBuf_pre.ivalue >> 24 == 0)
                    leadingNum = 1;

                leadNumberArray_int[i] = leadingNum;

                if (leadingNum == 0) {
                    exactMidbyteArray[residualMidBytes_size] = lfBuf_cur.byte[1];
                    exactMidbyteArray[residualMidBytes_size + 1] = lfBuf_cur.byte[2];
                    exactMidbyteArray[residualMidBytes_size + 2] = lfBuf_cur.byte[3];
                    residualMidBytes_size += 3;
                } else if (leadingNum == 1) {
                    exactMidbyteArray[residualMidBytes_size] = lfBuf_cur.byte[1];
                    exactMidbyteArray[residualMidBytes_size + 1] = lfBuf_cur.byte[2];
                    residualMidBytes_size += 2;
                } else if (leadingNum == 2) {
                    exactMidbyteArray[residualMidBytes_size] = lfBuf_cur.byte[1];
                    residualMidBytes_size++;
                }

                lfBuf_pre = lfBuf_cur;
            }
        } else if (reqBytesLength == 1) {
            for (i = 0; i < nbEle; i++) {
                leadingNum = 0;
                lfBuf_cur.value = oriData[i] - medianValue;

                lfBuf_cur.ivalue = lfBuf_cur.ivalue >> rightShiftBits;

                lfBuf_pre.ivalue = lfBuf_cur.ivalue ^ lfBuf_pre.ivalue;

                if (lfBuf_pre.ivalue >> 8 == 0)
                    leadingNum = 3;
                else if (lfBuf_pre.ivalue >> 16 == 0)
                    leadingNum = 2;
                else if (lfBuf_pre.ivalue >> 24 == 0)
                    leadingNum = 1;

                leadNumberArray_int[i] = leadingNum;

                if (leadingNum == 0) {
                    exactMidbyteArray[residualMidBytes_size] = lfBuf_cur.byte[3];
                    residualMidBytes_size++;
                }

                lfBuf_pre = lfBuf_cur;
            }
        } else //reqBytesLength == 4
        {
            for (i = 0; i < nbEle; i++) {
                leadingNum = 0;
                lfBuf_cur.value = oriData[i] - medianValue;

                lfBuf_cur.ivalue = lfBuf_cur.ivalue >> rightShiftBits;

                lfBuf_pre.ivalue = lfBuf_cur.ivalue ^ lfBuf_pre.ivalue;

                if (lfBuf_pre.ivalue >> 8 == 0)
                    leadingNum = 3;
                else if (lfBuf_pre.ivalue >> 16 == 0)
                    leadingNum = 2;
                else if (lfBuf_pre.ivalue >> 24 == 0)
                    leadingNum = 1;

                leadNumberArray_int[i] = leadingNum;

                if (leadingNum == 0) {
                    exactMidbyteArray[residualMidBytes_size] = lfBuf_cur.byte[0];
                    exactMidbyteArray[residualMidBytes_size + 1] = lfBuf_cur.byte[1];
                    exactMidbyteArray[residualMidBytes_size + 2] = lfBuf_cur.byte[2];
                    exactMidbyteArray[residualMidBytes_size + 3] = lfBuf_cur.byte[3];
                    residualMidBytes_size += 4;
                } else if (leadingNum == 1) {
                    exactMidbyteArray[residualMidBytes_size] = lfBuf_cur.byte[0];
                    exactMidbyteArray[residualMidBytes_size + 1] = lfBuf_cur.byte[1];
                    exactMidbyteArray[residualMidBytes_size + 2] = lfBuf_cur.byte[2];
                    residualMidBytes_size += 3;
                } else if (leadingNum == 2) {
                    exactMidbyteArray[residualMidBytes_size] = lfBuf_cur.byte[0];
                    exactMidbyteArray[residualMidBytes_size + 1] = lfBuf_cur.byte[1];
                    residualMidBytes_size += 2;
                } else //leadingNum == 3
                {
                    exactMidbyteArray[residualMidBytes_size] = lfBuf_cur.byte[0];
                    residualMidBytes_size++;
                }

                lfBuf_pre = lfBuf_cur;
            }
        }

        convertIntArray2ByteArray_fast_2b_args(leadNumberArray_int, nbEle, leadNumberArray);
        int k = 0;

        unsigned char reqLengthB = (unsigned char) reqLength;
        outputBytes[k] = reqLengthB;
        k++;
        floatToBytes(&(outputBytes[k]), medianValue);
        k += sizeof(float);
        //sizeToBytes(&(outputBytes[k]), leadNumberArray_size);
        //outputBytes[k] = leadNumberArray_size;  //leadNumberArray_size can be calculated based on block size (=blockSize/4)

        totalSize = 1 + sizeof(float) + leadNumberArray_size + residualMidBytes_size;
    } else {

    }

    *outSize = totalSize;

}

size_t computeStateMedianRadius_float(float *oriData, size_t nbEle, float absErrBound, int blockSize,
                                      unsigned char *stateArray, float *medianArray, float *radiusArray) {
    size_t nbConstantBlocks = 0;
    size_t i = 0, j = 0;
    size_t nbBlocks = nbEle / blockSize;
    size_t offset = 0;

    for (i = 0; i < nbBlocks; i++) {
        float min = oriData[offset];
        float max = oriData[offset];
        for (j = 1; j < blockSize; j++) {
            float v = oriData[offset + j];
            if (min > v)
                min = v;
            else if (max < v)
                max = v;
        }
        float valueRange = max - min;
        float radius = valueRange / 2;
        float medianValue = min + radius;

        if (radius <= absErrBound) {
            stateArray[i] = 0;
            nbConstantBlocks++;
        } else
            stateArray[i] = 1;

        stateArray[i] = radius <= absErrBound ? 0 : 1;
        medianArray[i] = medianValue;
        radiusArray[i] = radius;
        offset += blockSize;
    }

    int remainCount = nbEle % blockSize;
    if (remainCount != 0) {
        float min = oriData[offset];
        float max = oriData[offset];
        for (j = 1; j < remainCount; j++) {
            float v = oriData[offset + j];
            if (min > v)
                min = v;
            else if (max < v)
                max = v;
        }
        float valueRange = max - min;
        float radius = valueRange / 2;
        float medianValue = min + radius;
        if (radius <= absErrBound) {
            stateArray[i] = 0;
            nbConstantBlocks++;
        } else
            stateArray[i] = 1;
        medianArray[i] = medianValue;
        radiusArray[i] = radius;
    }
    return nbConstantBlocks;
}


void max_min_float(float *x, int n, float *tmp_max, float *tmp_min) {
    for (size_t i = 0; i < n; i++) {
        if (x[i] > *tmp_max) {
            *tmp_max = x[i];
        }
        if (x[i] < *tmp_min) {
            *tmp_min = x[i];
        }
    }
}

void simd_max_min_float(float *x, int n, float *tmp_max, float *tmp_min) {
    *tmp_max = x[0];
    *tmp_min = x[0];
#ifdef  __AVX512F__
    //    printf("use avx512, n=%d \n", n);
    int n16 = n & -16, i = 0, j=0;
    if (n > 16) {
        float *ptr_x = x;
        __m512 max1 = _mm512_loadu_ps(ptr_x);
//        __m512 max2 = _mm512_loadu_ps(ptr_x + 16);
        __m512 min1 = max1;
//        __m512 min2 = max2;
        __m512 tmp1;
//        __m512 tmp2;
        for (; i < n16; i += 16) {
            tmp1 = _mm512_loadu_ps(ptr_x);
            max1 = _mm512_max_ps(tmp1, max1);
            min1 = _mm512_min_ps(tmp1, min1);
//            tmp2 = _mm512_loadu_ps(ptr_x+16);
//            max2 = _mm512_max_ps(tmp2, max2);
//            min2 = _mm512_min_ps(tmp2, min2);
            ptr_x += 16;
        }
//        max1 = _mm512_max_ps(max1, max2);
//        min1 = _mm512_min_ps(min1, min2);
          __m256 max256 = _mm256_max_ps(_mm512_extractf32x8_ps(max1,0), _mm512_extractf32x8_ps(max1,1));
          __m128 max128 = _mm_max_ps(_mm256_extractf128_ps(max256,0), _mm256_extractf128_ps(max256,1));
          __m256 min256 = _mm256_min_ps(_mm512_extractf32x8_ps(min1,0), _mm512_extractf32x8_ps(min1,1));
          __m128 min128 = _mm_min_ps(_mm256_extractf128_ps(min256,0), _mm256_extractf128_ps(min256,1));
          for (j=0;j<4;j++){
            *tmp_max = *tmp_max < max128[j] ? max128[j] : *tmp_max;
            *tmp_min = *tmp_min > min128[j] ? min128[j] : *tmp_min;
          }

        if ( i < n ) {
            max_min_float(ptr_x, n - i, tmp_max, tmp_min);
        }
    } else {
        max_min_float(x, n, tmp_max, tmp_min);
    }
#elif __AVX2__
//        printf("use avx2, n=%d \n", n);
    //    fflush(stdout);
    int n16 = n & -16, i = 0;
    if (n > 16) {
        float *ptr_x = x;
        __m256 max1 = _mm256_loadu_ps(ptr_x);
        __m256 max2 = _mm256_loadu_ps(ptr_x + 8);
        __m256 min1 = max1;
        __m256 min2 = max2;
        for (; i < n16; i += 16) {
            max1 = _mm256_max_ps(_mm256_loadu_ps(ptr_x), max1);
            min1 = _mm256_min_ps(_mm256_loadu_ps(ptr_x), min1);
            max2 = _mm256_max_ps(_mm256_loadu_ps(ptr_x + 8), max2);
            min2 = _mm256_min_ps(_mm256_loadu_ps(ptr_x + 8), min2);
            ptr_x += 16;
        }
//        printf("%d %d %d\n", n, n16, i);
//        exit(0);
        max1 = _mm256_max_ps(max1, max2);
        min1 = _mm256_min_ps(min1, min2);
        for (int j = 0; j < 8; j++) {
            *tmp_max = *tmp_max < max1[j] ? max1[j] : *tmp_max;
            *tmp_min = *tmp_min > min1[j] ? min1[j] : *tmp_min;
        }
        if ( i < n ) {
            max_min_float(ptr_x, n - i, tmp_max, tmp_min);
        }
    } else {
        max_min_float(x, n, tmp_max, tmp_min);
    }
#else
    max_min_float(x, n, tmp_max, tmp_min);
#endif
}

void computeStateMedianRadius_float2(float *oriData, size_t nbEle, float absErrBound,
                                     unsigned char *state, float *median, float *radius) {
     float min = oriData[0];
     float max = oriData[0];
     simd_max_min_float(oriData, nbEle, &max, &min);

    float valueRange = max - min;
    *radius = valueRange / 2;
    *median = min + *radius;

    if (*radius <= absErrBound) {
        *state = 0;
    } else {
        *state = 1;
    }
}


unsigned char *
SZ_fast_compress_args_unpredictable_blocked_float(float *oriData, size_t *outSize, float absErrBound, size_t nbEle,
                                                  int blockSize) {
    float *op = oriData;

    *outSize = 0;
    size_t maxPreservedBufferSize =
            sizeof(float) * nbEle; //assume that the compressed data size would not exceed the original size
    unsigned char *outputBytes = (unsigned char *) malloc(maxPreservedBufferSize);
    memset(outputBytes, 0, maxPreservedBufferSize);
    unsigned char *leadNumberArray_int = (unsigned char *) malloc(blockSize * sizeof(int));

    size_t i = 0;
    int oSize = 0;

    size_t nbBlocks = nbEle / blockSize;
    size_t remainCount = nbEle % blockSize;
    size_t stateNBBytes =
            remainCount == 0 ? (nbBlocks % 8 == 0 ? nbBlocks / 8 : nbBlocks / 8 + 1) : ((nbBlocks + 1) % 8 == 0 ?
                                                                                        (nbBlocks + 1) / 8 :
                                                                                        (nbBlocks + 1) / 8 + 1);
    size_t actualNBBlocks = remainCount == 0 ? nbBlocks : nbBlocks + 1;

    unsigned char *stateArray = (unsigned char *) malloc(actualNBBlocks);
    float *medianArray = (float *) malloc(actualNBBlocks * sizeof(float));
    float *radiusArray = (float *) malloc(actualNBBlocks * sizeof(float));

    size_t nbConstantBlocks = computeStateMedianRadius_float(oriData, nbEle, absErrBound, blockSize, stateArray,
                                                             medianArray, radiusArray);

    unsigned char *r = outputBytes; // + sizeof(size_t) + stateNBBytes;
    r[0] = SZx_VER_MAJOR;
    r[1] = SZx_VER_MINOR;
    r[2] = 1;
    r[3] = 0; // indicates this is not a random access version
    r[4] = (unsigned char) blockSize;
    r = r + 5; //1 byte
    sizeToBytes(r, nbConstantBlocks);
    r += sizeof(size_t); //r is the starting address of 'stateNBBytes'

    unsigned char *p = r + stateNBBytes; //p is the starting address of constant median values.
    unsigned char *q =
            p + sizeof(float) * nbConstantBlocks; //q is the starting address of the non-constant data sblocks
    //3: versions, 1: metadata: state, 1: metadata: blockSize, sizeof(size_t): nbConstantBlocks, ....
    *outSize += (3 + 1 + 1 + sizeof(size_t) + stateNBBytes + sizeof(float) * nbConstantBlocks);

    //printf("nbConstantBlocks = %zu, percent = %f\n", nbConstantBlocks, 1.0f*(nbConstantBlocks*blockSize)/nbEle);
    for (i = 0; i < nbBlocks; i++, op += blockSize) {
        if (stateArray[i]) {
            SZ_fast_compress_args_unpredictable_one_block_float(op, blockSize, absErrBound, q, &oSize,
                                                                leadNumberArray_int, medianArray[i], radiusArray[i]);
            q += oSize;
            *outSize += oSize;
        } else {
            floatToBytes(p, medianArray[i]);
            p += sizeof(float);
        }
    }

    if (remainCount != 0) {
        if (stateArray[i]) {
            SZ_fast_compress_args_unpredictable_one_block_float(op, remainCount, absErrBound, q, &oSize,
                                                                leadNumberArray_int, medianArray[i], radiusArray[i]);
            *outSize += oSize;
        } else {
            floatToBytes(p, medianArray[i]);
        }

    }

    convertIntArray2ByteArray_fast_1b_args(stateArray, actualNBBlocks, r);
	
    free(stateArray);
    free(medianArray);	
    free(radiusArray);
    free(leadNumberArray_int);

    return outputBytes;
}

unsigned char *
SZ_fast_compress_args_unpredictable_blocked_randomaccess_float_openmp(float *oriData, size_t *outSize, float absErrBound,
                                                               size_t nbEle, int blockSize) {
#ifdef _OPENMP
    printf("use openmp\n");

#ifdef __AVX512F__
    printf("use avx512\n");
#elif __AVX2__
    printf("use avx2\n");
#else
#endif
    printf("blockSize = %d\n",blockSize);
    sz_cost_start();
    float *op = oriData;

    size_t i = 0;
    size_t nbBlocks = nbEle / blockSize;
    size_t remainCount = nbEle % blockSize;
    size_t actualNBBlocks = remainCount == 0 ? nbBlocks : nbBlocks + 1;
    size_t stateNBBytes = (actualNBBlocks % 8 == 0 ? actualNBBlocks / 8 : actualNBBlocks / 8 + 1);

    unsigned char *stateArray = (unsigned char *) malloc(actualNBBlocks);
    float *medianArray = (float *) malloc(actualNBBlocks * sizeof(float));

    size_t nbNonConstantBlocks = 0;

    unsigned char *tmp_q = (unsigned char *) malloc(blockSize * sizeof(float) * actualNBBlocks);
    int *outSizes = (int *) malloc(actualNBBlocks * sizeof(int));
    size_t *outSizesAccumlate = (size_t *) malloc(actualNBBlocks * sizeof(size_t));
    int *nbNonConstantBlockAccumlate = (int *) malloc(actualNBBlocks * sizeof(int));

    (*outSize) = 0;
    size_t maxPreservedBufferSize =
    sizeof(float) * nbEle; //assume that the compressed data size would not exceed the original size
    unsigned char *outputBytes = (unsigned char *) malloc(maxPreservedBufferSize);
    memset(outputBytes, 0, maxPreservedBufferSize);
    unsigned char *r = outputBytes; // + sizeof(size_t) + stateNBBytes;
    r[0] = SZx_VER_MAJOR;
    r[1] = SZx_VER_MINOR;
    r[2] = 1;
    r[3] = 1; //support random access decompression
    r = r + 4; //4 byte

    int nbThreads = 1;
    unsigned char *leadNumberArray_int;
    size_t z0[200],z1[200];

    size_t nbConstantBlocks;
    unsigned char *R, *p, *q;
    float *pf;
    uint16_t *O;

#pragma omp parallel
{
#pragma omp single
{
    nbThreads = omp_get_num_threads();
    printf("nbThreads = %d\n", nbThreads);
    assert(nbThreads<200);
    leadNumberArray_int = (unsigned char *) malloc(blockSize * sizeof(int) * nbThreads);

    sz_cost_end_msg("sequential-1 malloc");
    sz_cost_start();
}
#pragma omp for reduction(+:nbNonConstantBlocks) schedule(static)
    for (i = 0; i < nbBlocks; i++) {
        float radius;
        computeStateMedianRadius_float2(op + i * blockSize, blockSize, absErrBound, stateArray + i, medianArray + i,
                                        &radius);
        if (stateArray[i]) {
            SZ_fast_compress_args_unpredictable_one_block_float(op + i * blockSize, blockSize, absErrBound,
                                                                tmp_q + i * blockSize * sizeof(float), outSizes + i,
                                                                leadNumberArray_int +
                                                                omp_get_thread_num() * blockSize * sizeof(int),
                                                                medianArray[i], radius);
            outSizesAccumlate[i]=outSizes[i];
            nbNonConstantBlocks += 1;
        }else{
            outSizes[i]=0;
            outSizesAccumlate[i]=0;
        }
    }
#pragma omp single
{
    sz_cost_end_msg("parallel-1 compress");
//    exit(0);
    if (remainCount != 0) {
        i = nbBlocks;
        float radius;
        computeStateMedianRadius_float2(op + i * blockSize, remainCount, absErrBound, stateArray + i, medianArray + i,
                                        &radius);
        if (stateArray[i]) {
            SZ_fast_compress_args_unpredictable_one_block_float(op + i * blockSize, remainCount, absErrBound,
                                                                tmp_q + i * blockSize * sizeof(float), outSizes + i,
                                                                leadNumberArray_int, medianArray[i], radius);
            outSizesAccumlate[i] = outSizes[i];
            nbNonConstantBlocks += 1;
        }else{
            outSizesAccumlate[i] = 0;
            outSizes[i]=0;
        }
    }

    nbConstantBlocks = actualNBBlocks - nbNonConstantBlocks;

    sizeToBytes(r, blockSize);
    r += sizeof(size_t);
    sizeToBytes(r, nbConstantBlocks);
    r += sizeof(size_t);
    O = (uint16_t*) r; //o is the starting address of 'block-size array'
    R = r + nbNonConstantBlocks * sizeof(uint16_t); //R is the starting address of the state array
    p = R + stateNBBytes; //p is the starting address of constant median values.
    pf = (float *) p;
    q = p + sizeof(float) * nbConstantBlocks; //q is the starting address of the non-constant data sblocks
    // unsigned char *q0 = q;
    // printf("%lu %lu %lu %lu\n",r-outputBytes, R-outputBytes, p-outputBytes, q-outputBytes);
    // 3: versions, 1: metadata: state, 1: metadata: blockSize, sizeof(size_t): nbConstantBlocks, ....
    *outSize = q - outputBytes;

    sz_cost_start();

}
    int tid = omp_get_thread_num();
    int lo = tid * actualNBBlocks / nbThreads;
    int hi = (tid + 1) * actualNBBlocks / nbThreads;
    int b;
    nbNonConstantBlockAccumlate[lo]=stateArray[lo];
    for (b = lo+1; b < hi; b++){
        outSizesAccumlate[b] = outSizesAccumlate[b] + outSizesAccumlate[b-1];
    }
    for (b = lo+1; b < hi; b++){
        nbNonConstantBlockAccumlate[b]=stateArray[b]+nbNonConstantBlockAccumlate[b-1];
    }
    z0[tid] = outSizesAccumlate[hi-1];
    z1[tid] = nbNonConstantBlockAccumlate[hi-1];
    size_t offset0=0, offset1=0;
#pragma omp barrier
    for (int j = 0; j < tid; j++) {
        offset0+=z0[j];
        offset1+=z1[j];
    }
    for (b = lo; b < hi; b++){
        outSizesAccumlate[b] = outSizesAccumlate[b] + offset0;
        nbNonConstantBlockAccumlate[b] = nbNonConstantBlockAccumlate[b] + offset1;
    }
#pragma omp single
{
    sz_cost_end_msg("parallel-2 prefix sum");
    sz_cost_start();
};
#pragma omp for schedule(static)
    for (i = 0; i < actualNBBlocks; i++) {
        if (stateArray[i]) {
            memcpy(q+outSizesAccumlate[i]-outSizes[i], tmp_q + i * blockSize * sizeof(float), outSizes[i]);
            O[nbNonConstantBlockAccumlate[i]-1]=outSizes[i];
        } else {
            pf[i-nbNonConstantBlockAccumlate[i]]=medianArray[i];
        }
    }
#pragma omp single
{
    sz_cost_end_msg("parallel-3 memcpy");
    sz_cost_start();

    *outSize += outSizesAccumlate[actualNBBlocks-1];

    convertIntArray2ByteArray_fast_1b_args(stateArray, actualNBBlocks, R);
    sz_cost_end_msg("sequential-2 int2byte");
    sz_cost_start();
    free(nbNonConstantBlockAccumlate);
    free(outSizesAccumlate);
    free(leadNumberArray_int);
    free(tmp_q);
    free(medianArray);
    free(stateArray);
    free(outSizes);
    sz_cost_end_msg("sequential-3 free");
    printf("blocksize = %d, actualNBBlocks = %lu\n", blockSize, actualNBBlocks);
    printf("nbConstantBlocks = %zu, percent = %f\n", nbConstantBlocks, 1.0f * (nbConstantBlocks * blockSize) / nbEle);
    printf("CR = %.3f, nbEle = %lu \n", nbEle*4.0/(*outSize), nbEle);
}
}
    return outputBytes;
#else
    return NULL;
#endif
}

unsigned char *
SZ_fast_compress_args_unpredictable_blocked_randomaccess_float(float *oriData, size_t *outSize, float absErrBound,
    size_t nbEle, int blockSize) {
    float *op = oriData;

    *outSize = 0;
    size_t maxPreservedBufferSize =
            sizeof(float) * nbEle; //assume that the compressed data size would not exceed the original size
    unsigned char *outputBytes = (unsigned char *) malloc(maxPreservedBufferSize);
    memset(outputBytes, 0, maxPreservedBufferSize);
    unsigned char *leadNumberArray_int = (unsigned char *) malloc(blockSize * sizeof(int));

    size_t i = 0;
    int oSize = 0;

    size_t nbBlocks = nbEle / blockSize;
    size_t remainCount = nbEle % blockSize;
    size_t actualNBBlocks = remainCount == 0 ? nbBlocks : nbBlocks + 1;

    size_t stateNBBytes = (actualNBBlocks % 8 == 0 ? actualNBBlocks / 8 : actualNBBlocks / 8 + 1);

    unsigned char *stateArray = (unsigned char *) malloc(actualNBBlocks);
    float *medianArray = (float *) malloc(actualNBBlocks * sizeof(float));
    float *radiusArray = (float *) malloc(actualNBBlocks * sizeof(float));

    size_t nbConstantBlocks = computeStateMedianRadius_float(oriData, nbEle, absErrBound, blockSize, stateArray,
                                                             medianArray, radiusArray);

    size_t nbNonConstantBlocks = actualNBBlocks - nbConstantBlocks;

    unsigned char *r = outputBytes; // + sizeof(size_t) + stateNBBytes;
    r[0] = SZx_VER_MAJOR;
    r[1] = SZx_VER_MINOR;
    r[2] = 1;
    r[3] = 1; //support random access decompression
    r = r + 4; //1 byte

    sizeToBytes(r, blockSize);
    r += sizeof(size_t);
    sizeToBytes(r, nbConstantBlocks);
    r += sizeof(size_t); //r is the starting address of 'block-size array'
    uint16_t *O=(uint16_t*)r;
    unsigned char *R = r + nbNonConstantBlocks*sizeof(uint16_t); //R is the starting address of the state array
    unsigned char *p = R + stateNBBytes; //p is the starting address of constant median values.
    unsigned char *q =
            p + sizeof(float) * nbConstantBlocks; //q is the starting address of the non-constant data sblocks
    //3: versions, 1: metadata: state, 1: metadata: blockSize, sizeof(size_t): nbConstantBlocks, ....
    *outSize = q-outputBytes;

    size_t nonConstantBlockID = 0;
    //printf("nbConstantBlocks = %zu, percent = %f\n", nbConstantBlocks, 1.0f*(nbConstantBlocks*blockSize)/nbEle);
    for (i = 0; i < nbBlocks; i++, op += blockSize) {
        if (stateArray[i]) {
            SZ_fast_compress_args_unpredictable_one_block_float(op, blockSize, absErrBound, q, &oSize,
                                                                leadNumberArray_int, medianArray[i], radiusArray[i]);
            q += oSize;
            *outSize += oSize;
            O[nonConstantBlockID++] = oSize;
        } else {
            floatToBytes(p, medianArray[i]);
            p += sizeof(float);
        }
    }

    if (remainCount != 0) {
        if (stateArray[i]) {
            SZ_fast_compress_args_unpredictable_one_block_float(op, remainCount, absErrBound, q, &oSize,
                                                                leadNumberArray_int, medianArray[i], radiusArray[i]);
            *outSize += oSize;
            O[nonConstantBlockID] = oSize;
        } else {
            floatToBytes(p, medianArray[i]);
        }

    }

    convertIntArray2ByteArray_fast_1b_args(stateArray, actualNBBlocks, R);

    free(leadNumberArray_int);

    return outputBytes;

}


unsigned char *
SZ_fast_compress_args_unpredictable_float(float *data, size_t *outSize, float absErrBound, size_t r5, size_t r4,
                                          size_t r3, size_t r2, size_t r1, float mValue, float radius) {
    size_t totalSize = 0;
    float medianValue = mValue;

    size_t dataLength = computeDataLength(r5, r4, r3, r2, r1);

    size_t maxPreservedBufferSize =
            sizeof(float) * dataLength; //assume that the compressed data size would not exceed the original size

    unsigned char *outputBytes = (unsigned char *) malloc(maxPreservedBufferSize);
    memset(outputBytes, 0, maxPreservedBufferSize);
    unsigned char *r = outputBytes; // + sizeof(size_t) + stateNBBytes;
    r[0] = SZx_VER_MAJOR;
    r[1] = SZx_VER_MINOR;
    r[2] = 1; //SZx_VER_SUPERFAST
    r[3] = 0; //support random access decompression

//	sz_cost_start();
    size_t i;
    int reqLength;
    short radExpo = getExponent_float(radius);

    computeReqLength_float(absErrBound, radExpo, &reqLength, &medianValue);

    int reqBytesLength = reqLength / 8;
    int resiBitsLength = reqLength % 8;
    int rightShiftBits = 0;

    size_t leadNumberArray_size = dataLength % 4 == 0 ? dataLength / 4 : dataLength / 4 + 1;

    register lfloat lfBuf_pre;
    register lfloat lfBuf_cur;
    lfBuf_pre.ivalue = 0;

    unsigned char *leadNumberArray = outputBytes + 4 + 1 + sizeof(float) + sizeof(size_t);

    unsigned char *exactMidbyteArray = leadNumberArray + leadNumberArray_size;

    if (resiBitsLength != 0) {
        rightShiftBits = 8 - resiBitsLength;
        reqBytesLength++;
    }

    register unsigned char leadingNum = 0;

    unsigned char *leadNumberArray_int = (unsigned char *) malloc(dataLength);

    size_t residualMidBytes_size = 0;
    if (sysEndianType == LITTLE_ENDIAN_SYSTEM) {
        if (reqBytesLength == 3) {
            for (i = 0; i < dataLength; i++) {
                leadingNum = 0;
                lfBuf_cur.value = data[i] - medianValue;

                lfBuf_cur.ivalue = lfBuf_cur.ivalue >> rightShiftBits;

                lfBuf_pre.ivalue = lfBuf_cur.ivalue ^ lfBuf_pre.ivalue;

                if (lfBuf_pre.ivalue >> 8 == 0)
                    leadingNum = 3;
                else if (lfBuf_pre.ivalue >> 16 == 0)
                    leadingNum = 2;
                else if (lfBuf_pre.ivalue >> 24 == 0)
                    leadingNum = 1;

                leadNumberArray_int[i] = leadingNum;

                if (leadingNum == 0) {
                    exactMidbyteArray[residualMidBytes_size] = lfBuf_cur.byte[1];
                    exactMidbyteArray[residualMidBytes_size + 1] = lfBuf_cur.byte[2];
                    exactMidbyteArray[residualMidBytes_size + 2] = lfBuf_cur.byte[3];
                    residualMidBytes_size += 3;
                } else if (leadingNum == 1) {
                    exactMidbyteArray[residualMidBytes_size] = lfBuf_cur.byte[1];
                    exactMidbyteArray[residualMidBytes_size + 1] = lfBuf_cur.byte[2];
                    residualMidBytes_size += 2;
                } else if (leadingNum == 2) {
                    exactMidbyteArray[residualMidBytes_size] = lfBuf_cur.byte[1];
                    residualMidBytes_size++;
                }

                lfBuf_pre = lfBuf_cur;
            }
        } else if (reqBytesLength == 2) {
            for (i = 0; i < dataLength; i++) {

                leadingNum = 0;
                lfBuf_cur.value = data[i] - medianValue;

                lfBuf_cur.ivalue = lfBuf_cur.ivalue >> rightShiftBits;

                lfBuf_pre.ivalue = lfBuf_cur.ivalue ^ lfBuf_pre.ivalue;

                if (lfBuf_pre.ivalue >> 8 == 0)
                    leadingNum = 3;
                else if (lfBuf_pre.ivalue >> 16 == 0)
                    leadingNum = 2;
                else if (lfBuf_pre.ivalue >> 24 == 0)
                    leadingNum = 1;

                leadNumberArray_int[i] = leadingNum;

                if (leadingNum == 0) {
                    exactMidbyteArray[residualMidBytes_size] = lfBuf_cur.byte[2];
                    exactMidbyteArray[residualMidBytes_size + 1] = lfBuf_cur.byte[3];
                    residualMidBytes_size += 2;
                } else if (leadingNum == 1) {
                    exactMidbyteArray[residualMidBytes_size] = lfBuf_cur.byte[2];
                    residualMidBytes_size++;
                }

                lfBuf_pre = lfBuf_cur;
            }
        } else if (reqBytesLength == 1) {
            for (i = 0; i < dataLength; i++) {
                leadingNum = 0;
                lfBuf_cur.value = data[i] - medianValue;

                lfBuf_cur.ivalue = lfBuf_cur.ivalue >> rightShiftBits;

                lfBuf_pre.ivalue = lfBuf_cur.ivalue ^ lfBuf_pre.ivalue;

                if (lfBuf_pre.ivalue >> 8 == 0)
                    leadingNum = 3;
                else if (lfBuf_pre.ivalue >> 16 == 0)
                    leadingNum = 2;
                else if (lfBuf_pre.ivalue >> 24 == 0)
                    leadingNum = 1;

                leadNumberArray_int[i] = leadingNum;

                if (leadingNum == 0) {
                    exactMidbyteArray[residualMidBytes_size] = lfBuf_cur.byte[3];
                    residualMidBytes_size++;
                }

                lfBuf_pre = lfBuf_cur;
            }
        }else //reqBytesLength == 4
        {
            for (i = 0; i < dataLength; i++) {
                leadingNum = 0;
                lfBuf_cur.value = data[i] - medianValue;

                lfBuf_cur.ivalue = lfBuf_cur.ivalue >> rightShiftBits;

                lfBuf_pre.ivalue = lfBuf_cur.ivalue ^ lfBuf_pre.ivalue;

                if (lfBuf_pre.ivalue >> 8 == 0)
                    leadingNum = 3;
                else if (lfBuf_pre.ivalue >> 16 == 0)
                    leadingNum = 2;
                else if (lfBuf_pre.ivalue >> 24 == 0)
                    leadingNum = 1;

                leadNumberArray_int[i] = leadingNum;

                if (leadingNum == 0) {
                    exactMidbyteArray[residualMidBytes_size] = lfBuf_cur.byte[0];
                    exactMidbyteArray[residualMidBytes_size + 1] = lfBuf_cur.byte[1];
                    exactMidbyteArray[residualMidBytes_size + 2] = lfBuf_cur.byte[2];
                    exactMidbyteArray[residualMidBytes_size + 3] = lfBuf_cur.byte[3];
                    residualMidBytes_size += 4;
                } else if (leadingNum == 1) {
                    exactMidbyteArray[residualMidBytes_size] = lfBuf_cur.byte[0];
                    exactMidbyteArray[residualMidBytes_size + 1] = lfBuf_cur.byte[1];
                    exactMidbyteArray[residualMidBytes_size + 2] = lfBuf_cur.byte[2];
                    residualMidBytes_size += 3;
                } else if (leadingNum == 2) {
                    exactMidbyteArray[residualMidBytes_size] = lfBuf_cur.byte[0];
                    exactMidbyteArray[residualMidBytes_size + 1] = lfBuf_cur.byte[1];
                    residualMidBytes_size += 2;
                } else //leadingNum == 3
                {
                    exactMidbyteArray[residualMidBytes_size] = lfBuf_cur.byte[0];
                    residualMidBytes_size++;
                }

                lfBuf_pre = lfBuf_cur;
            }
        }

        convertIntArray2ByteArray_fast_2b_args(leadNumberArray_int, dataLength, leadNumberArray);

        int k = 4;

        unsigned char reqLengthB = (unsigned char) reqLength;
        outputBytes[k] = reqLengthB;
        k++;
        floatToBytes(&(outputBytes[k]), medianValue);
        k += sizeof(float);
        sizeToBytes(&(outputBytes[k]), leadNumberArray_size);

        totalSize = 4 + 1 + sizeof(float) + sizeof(size_t) + leadNumberArray_size + residualMidBytes_size;
    } else {

    }

    *outSize = totalSize;

    free(leadNumberArray_int);
//	sz_cost_end();
//	printf("compression time = %f\n", sz_totalCost);

    return outputBytes;
}

unsigned char *SZ_skip_compress_float(float *data, size_t dataLength, size_t *outSize) {
    *outSize = dataLength * sizeof(float);
    unsigned char *out = (unsigned char *) malloc(dataLength * sizeof(float));
    memcpy(out, data, dataLength * sizeof(float));
    return out;
}

inline void computeReqLength_float(double realPrecision, short radExpo, int *reqLength, float *medianValue) {
    short reqExpo = getPrecisionReqLength_double(realPrecision);
    *reqLength = 9 + radExpo - reqExpo + 1; //radExpo-reqExpo == reqMantiLength
    if (*reqLength < 9)
        *reqLength = 9;
    if (*reqLength > 32) {
        *reqLength = 32;
        *medianValue = 0;
    }
}
