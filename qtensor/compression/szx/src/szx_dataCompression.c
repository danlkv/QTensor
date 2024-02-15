/**
 *  @file double_compression.c
 *  @author Sheng Di, Dingwen Tao, Xin Liang, Xiangyu Zou, Tao Lu, Wen Xia, Xuan Wang, Weizhe Zhang
 *  @date April, 2016
 *  @brief Compression Technique for double array
 *  (C) 2016 by Mathematics and Computer Science (MCS), Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "szx.h"
#include "szx_dataCompression.h"
#include "szx_BytesToolkit.h"

int computeByteSizePerIntValue(long valueRangeSize)
{
	if(valueRangeSize<=256)
		return 1;
	else if(valueRangeSize<=65536)
		return 2;
	else if(valueRangeSize<=4294967296) //2^32
		return 4;
	else
		return 8;
}

long computeRangeSize_int(void* oriData, int dataType, size_t size, int64_t* valueRangeSize)
{
	size_t i = 0;
	long max = 0, min = 0;

	if(dataType==SZ_UINT8)
	{
		unsigned char* data = (unsigned char*)oriData;
		unsigned char data_; 
		min = data[0], max = min;
		computeMinMax(data);
	}
	else if(dataType == SZ_INT8)
	{
		char* data = (char*)oriData;
		char data_;
		min = data[0], max = min;
		computeMinMax(data);
	}
	else if(dataType == SZ_UINT16)
	{
		unsigned short* data = (unsigned short*)oriData;
		unsigned short data_; 
		min = data[0], max = min;
		computeMinMax(data);
	}
	else if(dataType == SZ_INT16)
	{ 
		short* data = (short*)oriData;
		short data_; 
		min = data[0], max = min;
		computeMinMax(data);
	}
	else if(dataType == SZ_UINT32)
	{
		unsigned int* data = (unsigned int*)oriData;
		unsigned int data_; 
		min = data[0], max = min;
		computeMinMax(data);
	}
	else if(dataType == SZ_INT32)
	{
		int* data = (int*)oriData;
		int data_; 
		min = data[0], max = min;
		computeMinMax(data);
	}
	else if(dataType == SZ_UINT64)
	{
		unsigned long* data = (unsigned long*)oriData;
		unsigned long data_; 
		min = data[0], max = min;
		computeMinMax(data);
	}
	else if(dataType == SZ_INT64)
	{
		long* data = (long *)oriData;
		long data_; 
		min = data[0], max = min;
		computeMinMax(data);
	}

	*valueRangeSize = max - min;
	return min;	
}

float computeRangeSize_float(float* oriData, size_t size, float* valueRangeSize, float* medianValue)
{
	size_t i = 0;
	float min = oriData[0];
	float max = min;
	for(i=1;i<size;i++)
	{
		float data = oriData[i];
		if(min>data)
			min = data;
		else if(max<data)
			max = data;
	}

	*valueRangeSize = max - min;
	*medianValue = min + *valueRangeSize/2;
	return min;
}

double computeRangeSize_double(double* oriData, size_t size, double* valueRangeSize, double* medianValue)
{
	size_t i = 0;
	double min = oriData[0];
	double max = min;
	for(i=1;i<size;i++)
	{
		double data = oriData[i];
		if(min>data)
			min = data;
		else if(max<data)
			max = data;
	}
	
	*valueRangeSize = max - min;
	*medianValue = min + *valueRangeSize/2;
	return min;
}

double min_d(double a, double b)
{
	if(a<b)
		return a;
	else
		return b;
}

double max_d(double a, double b)
{
	if(a>b)
		return a;
	else
		return b;
}

float min_f(float a, float b)
{
	if(a<b)
		return a;
	else
		return b;
}

float max_f(float a, float b)
{
	if(a>b)
		return a;
	else
		return b;
}

double getRealPrecision_double(double valueRangeSize, int errBoundMode, double absErrBound, double relBoundRatio, int *status)
{
	int state = SZ_SCES;
	double precision = 0;
	if(errBoundMode==ABS||errBoundMode==ABS_OR_PW_REL||errBoundMode==ABS_AND_PW_REL)
		precision = absErrBound; 
	else if(errBoundMode==REL||errBoundMode==REL_OR_PW_REL||errBoundMode==REL_AND_PW_REL)
		precision = relBoundRatio*valueRangeSize;
	else if(errBoundMode==ABS_AND_REL)
		precision = min_d(absErrBound, relBoundRatio*valueRangeSize);
	else if(errBoundMode==ABS_OR_REL)
		precision = max_d(absErrBound, relBoundRatio*valueRangeSize);
	else if(errBoundMode==PW_REL)
		precision = 0;
	else
	{
		printf("Error: error-bound-mode is incorrect!\n");
		state = SZ_BERR;
	}
	*status = state;
	return precision;
}

double getRealPrecision_float(float valueRangeSize, int errBoundMode, double absErrBound, double relBoundRatio, int *status)
{
	int state = SZ_SCES;
	double precision = 0;
	if(errBoundMode==ABS||errBoundMode==ABS_OR_PW_REL||errBoundMode==ABS_AND_PW_REL)
		precision = absErrBound; 
	else if(errBoundMode==REL||errBoundMode==REL_OR_PW_REL||errBoundMode==REL_AND_PW_REL)
		precision = relBoundRatio*valueRangeSize;
	else if(errBoundMode==ABS_AND_REL)
		precision = min_f(absErrBound, relBoundRatio*valueRangeSize);
	else if(errBoundMode==ABS_OR_REL)
		precision = max_f(absErrBound, relBoundRatio*valueRangeSize);
	else if(errBoundMode==PW_REL)
		precision = 0;
	else
	{
		printf("Error: error-bound-mode is incorrect!\n");
		state = SZ_BERR;
	}
	*status = state;
	return precision;
}

double getRealPrecision_int(long valueRangeSize, int errBoundMode, double absErrBound, double relBoundRatio, int *status)
{
	int state = SZ_SCES;
	double precision = 0;
	if(errBoundMode==ABS||errBoundMode==ABS_OR_PW_REL||errBoundMode==ABS_AND_PW_REL)
		precision = absErrBound; 
	else if(errBoundMode==REL||errBoundMode==REL_OR_PW_REL||errBoundMode==REL_AND_PW_REL)
		precision = relBoundRatio*valueRangeSize;
	else if(errBoundMode==ABS_AND_REL)
		precision = min_f(absErrBound, relBoundRatio*valueRangeSize);
	else if(errBoundMode==ABS_OR_REL)
		precision = max_f(absErrBound, relBoundRatio*valueRangeSize);
	else if(errBoundMode==PW_REL)
		precision = -1;
	else
	{
		printf("Error: error-bound-mode is incorrect!\n");
		state = SZ_BERR;
	}
	*status = state;
	return precision;
}

inline void symTransform_8bytes(unsigned char data[8])
{
	unsigned char tmp = data[0];
	data[0] = data[7];
	data[7] = tmp;

	tmp = data[1];
	data[1] = data[6];
	data[6] = tmp;
	
	tmp = data[2];
	data[2] = data[5];
	data[5] = tmp;
	
	tmp = data[3];
	data[3] = data[4];
	data[4] = tmp;
}

inline void symTransform_2bytes(unsigned char data[2])
{
	unsigned char tmp = data[0];
	data[0] = data[1];
	data[1] = tmp;
}

inline void symTransform_4bytes(unsigned char data[4])
{
	unsigned char tmp = data[0];
	data[0] = data[3];
	data[3] = tmp;

	tmp = data[1];
	data[1] = data[2];
	data[2] = tmp;
}

inline void compressInt8Value(int8_t tgtValue, int8_t minValue, int byteSize, unsigned char* bytes)
{
	uint8_t data = tgtValue - minValue;
	memcpy(bytes, &data, byteSize); //byteSize==1
}

inline void compressInt16Value(int16_t tgtValue, int16_t minValue, int byteSize, unsigned char* bytes)
{
	uint16_t data = tgtValue - minValue;
	unsigned char tmpBytes[2];
	int16ToBytes_bigEndian(tmpBytes, data);
	memcpy(bytes, tmpBytes + 2 - byteSize, byteSize);
}

inline void compressInt32Value(int32_t tgtValue, int32_t minValue, int byteSize, unsigned char* bytes)
{
	uint32_t data = tgtValue - minValue;
	unsigned char tmpBytes[4];
	int32ToBytes_bigEndian(tmpBytes, data);
	memcpy(bytes, tmpBytes + 4 - byteSize, byteSize);
}

inline void compressInt64Value(int64_t tgtValue, int64_t minValue, int byteSize, unsigned char* bytes)
{
	uint64_t data = tgtValue - minValue;
	unsigned char tmpBytes[8];
	int64ToBytes_bigEndian(tmpBytes, data);
	memcpy(bytes, tmpBytes + 8 - byteSize, byteSize);
}

inline void compressUInt8Value(uint8_t tgtValue, uint8_t minValue, int byteSize, unsigned char* bytes)
{
	uint8_t data = tgtValue - minValue;
	memcpy(bytes, &data, byteSize); //byteSize==1
}

inline void compressUInt16Value(uint16_t tgtValue, uint16_t minValue, int byteSize, unsigned char* bytes)
{
	uint16_t data = tgtValue - minValue;
	unsigned char tmpBytes[2];
	int16ToBytes_bigEndian(tmpBytes, data);
	memcpy(bytes, tmpBytes + 2 - byteSize, byteSize);
}

inline void compressUInt32Value(uint32_t tgtValue, uint32_t minValue, int byteSize, unsigned char* bytes)
{
	uint32_t data = tgtValue - minValue;
	unsigned char tmpBytes[4];
	int32ToBytes_bigEndian(tmpBytes, data);
	memcpy(bytes, tmpBytes + 4 - byteSize, byteSize);
}

inline void compressUInt64Value(uint64_t tgtValue, uint64_t minValue, int byteSize, unsigned char* bytes)
{
	uint64_t data = tgtValue - minValue;
	unsigned char tmpBytes[8];
	int64ToBytes_bigEndian(tmpBytes, data);
	memcpy(bytes, tmpBytes + 8 - byteSize, byteSize);
}

int compIdenticalLeadingBytesCount_double(unsigned char* preBytes, unsigned char* curBytes)
{
	int i, n = 0;
	for(i=0;i<8;i++)
		if(preBytes[i]==curBytes[i])
			n++;
		else
			break;
	if(n>3) n = 3;
	return n;
}


inline int compIdenticalLeadingBytesCount_float(unsigned char* preBytes, unsigned char* curBytes)
{
	int i, n = 0;
	for(i=0;i<4;i++)
		if(preBytes[i]==curBytes[i])
			n++;
		else
			break;
	if(n>3) n = 3;
	return n;
}
