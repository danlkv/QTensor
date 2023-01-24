/**
 *  @file szx_dataCompression.h
 *  @author Sheng Di
 *  @date July, 2022
 *  @brief Header file for the dataCompression.c.
 *  (C) 2022 by Mathematics and Computer Science (MCS), Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#ifndef _SZX_DataCompression_H
#define _SZX_DataCompression_H

#ifdef __cplusplus
extern "C" {
#endif

#include "szx.h"
#include <stdio.h>
#include <stdbool.h>

#define computeMinMax(data) \
        for(i=1;i<size;i++)\
        {\
                data_ = data[i];\
                if(min>data_)\
                        min = data_;\
                else if(max<data_)\
                        max = data_;\
        }\


//dataCompression.c
int computeByteSizePerIntValue(long valueRangeSize);
long computeRangeSize_int(void* oriData, int dataType, size_t size, int64_t* valueRangeSize);
double computeRangeSize_double(double* oriData, size_t size, double* valueRangeSize, double* medianValue);
float computeRangeSize_float(float* oriData, size_t size, float* valueRangeSize, float* medianValue);

double min_d(double a, double b);
double max_d(double a, double b);
float min_f(float a, float b);
float max_f(float a, float b);
double getRealPrecision_double(double valueRangeSize, int errBoundMode, double absErrBound, double relBoundRatio, int *status);
double getRealPrecision_float(float valueRangeSize, int errBoundMode, double absErrBound, double relBoundRatio, int *status);
double getRealPrecision_int(long valueRangeSize, int errBoundMode, double absErrBound, double relBoundRatio, int *status);
void symTransform_8bytes(unsigned char data[8]);
void symTransform_2bytes(unsigned char data[2]);
void symTransform_4bytes(unsigned char data[4]);

void compressInt8Value(int8_t tgtValue, int8_t minValue, int byteSize, unsigned char* bytes);
void compressInt16Value(int16_t tgtValue, int16_t minValue, int byteSize, unsigned char* bytes);
void compressInt32Value(int32_t tgtValue, int32_t minValue, int byteSize, unsigned char* bytes);
void compressInt64Value(int64_t tgtValue, int64_t minValue, int byteSize, unsigned char* bytes);

void compressUInt8Value(uint8_t tgtValue, uint8_t minValue, int byteSize, unsigned char* bytes);
void compressUInt16Value(uint16_t tgtValue, uint16_t minValue, int byteSize, unsigned char* bytes);
void compressUInt32Value(uint32_t tgtValue, uint32_t minValue, int byteSize, unsigned char* bytes);
void compressUInt64Value(uint64_t tgtValue, uint64_t minValue, int byteSize, unsigned char* bytes);
    
int compIdenticalLeadingBytesCount_double(unsigned char* preBytes, unsigned char* curBytes);
int compIdenticalLeadingBytesCount_float(unsigned char* preBytes, unsigned char* curBytes);

#ifdef __cplusplus
}
#endif

#endif /* ----- #ifndef _SZX_DataCompression_H  ----- */

