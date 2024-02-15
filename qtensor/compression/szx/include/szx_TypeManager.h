/**
 *  @file TypeManager.h
 *  @author Sheng Di
 *  @date July, 2017
 *  @brief Header file for the TypeManager.c.
 *  (C) 2016 by Mathematics and Computer Science (MCS), Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#ifndef _SZX_TypeManager_H
#define _SZX_TypeManager_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdint.h>

size_t convertIntArray2ByteArray_fast_1b_args(unsigned char* intArray, size_t intArrayLength, unsigned char *result);
size_t convertIntArray2ByteArray_fast_1b(unsigned char* intArray, size_t intArrayLength, unsigned char **result);
size_t convertIntArray2ByteArray_fast_1b_to_result(unsigned char* intArray, size_t intArrayLength, unsigned char *result);
void convertByteArray2IntArray_fast_1b_args(size_t intArrayLength, unsigned char* byteArray, size_t byteArrayLength, unsigned char* intArray);
void convertByteArray2IntArray_fast_1b(size_t intArrayLength, unsigned char* byteArray, size_t byteArrayLength, unsigned char **intArray);
size_t convertIntArray2ByteArray_fast_2b_args(unsigned char* timeStepType, size_t timeStepTypeLength, unsigned char *result);
size_t convertIntArray2ByteArray_fast_2b(unsigned char* timeStepType, size_t timeStepTypeLength, unsigned char **result);
void convertByteArray2IntArray_fast_2b(size_t stepLength, unsigned char* byteArray, size_t byteArrayLength, unsigned char **intArray);
int getLeftMovingSteps(size_t k, unsigned char resiBitLength);

#ifdef __cplusplus
}
#endif

#endif /* ----- #ifndef _SZX_TypeManager_H  ----- */

