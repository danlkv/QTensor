/**
 *  @file szx.h
 *  @author Sheng Di
 *  @date April, 2022
 *  @brief Header file for the whole compressor.
 *  (C) 2022 by Mathematics and Computer Science (MCS), Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#ifndef _SZX_H
#define _SZX_H

#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>      /* For gettimeofday(), in microseconds */
#include <time.h>          /* For time(), in seconds */
#include "szx_float.h"
#include "szx_rw.h"
#include "szx_utility.h"

#ifdef _WIN32
#define PATH_SEPARATOR ';'
#else
#define PATH_SEPARATOR ':'
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "szx_defines.h"
#include "szx_double.h"
#include "szxd_double.h"
#include "szx_float.h"
#include "szxd_float.h"
#include "szx_TypeManager.h"

typedef union lint16
{
	unsigned short usvalue;
	short svalue;
	unsigned char byte[2];
} lint16;

typedef union lint32
{
	int ivalue;
	unsigned int uivalue;
	unsigned char byte[4];
} lint32;

typedef union lint64
{
	long lvalue;
	unsigned long ulvalue;
	unsigned char byte[8];
} lint64;

typedef union ldouble
{
    double value;
    unsigned long lvalue;
    unsigned char byte[8];
} ldouble;

typedef union lfloat
{
    float value;
    unsigned int ivalue;
    unsigned char byte[4];
} lfloat;


extern int versionNumber[4];

//-------------------key global variables--------------
extern int dataEndianType; //*endian type of the data read from disk
extern int sysEndianType; //*sysEndianType is actually set automatically.

int computeDimension(size_t r5, size_t r4, size_t r3, size_t r2, size_t r1);
size_t computeDataLength(size_t r5, size_t r4, size_t r3, size_t r2, size_t r1);
int filterDimension(size_t r5, size_t r4, size_t r3, size_t r2, size_t r1, size_t* correctedDimension);
unsigned char* SZ_fast_compress_args(int fastMode, int dataType, void *data, size_t *outSize, int errBoundMode, float absErrBound,
float relBoundRatio, size_t r5, size_t r4, size_t r3, size_t r2, size_t r1);
void* SZ_fast_decompress_pred(int dataType, float* preData, unsigned char *curBytes, size_t byteLength, size_t r5, size_t r4, size_t r3, size_t r2, size_t r1);
void* SZ_fast_decompress(int fastMode, int dataType, unsigned char *bytes, size_t byteLength, size_t r5, size_t r4, size_t r3, size_t r2, size_t r1);

#ifdef __cplusplus
}
#endif

#endif /* ----- #ifndef _SZX_H  ----- */
