/**
 *  @file TypeManager.c
 *  @author Sheng Di
 *  @date May, 2016
 *  @brief TypeManager is used to manage the type array: parsing of the bytes and other types in between.
 *  (C) 2016 by Mathematics and Computer Science (MCS), Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "szx.h"

size_t convertIntArray2ByteArray_fast_1b_args(unsigned char* intArray, size_t intArrayLength, unsigned char *result)
{
	size_t byteLength = 0;
	size_t i, j; 
	if(intArrayLength%8==0)
		byteLength = intArrayLength/8;
	else
		byteLength = intArrayLength/8+1;

	size_t n = 0;
	int tmp, type;
	for(i = 0;i<byteLength;i++)
	{
		tmp = 0;
		for(j = 0;j<8&&n<intArrayLength;j++)
		{
			type = intArray[n];
			//if(type == 1)
			tmp = (tmp | (type << (7-j)));
			n++;
		}
    	result[i] = (unsigned char)tmp;
	}
	return byteLength;
}

size_t convertIntArray2ByteArray_fast_1b(unsigned char* intArray, size_t intArrayLength, unsigned char **result)
{
	size_t byteLength = 0;
	size_t i, j; 
	if(intArrayLength%8==0)
		byteLength = intArrayLength/8;
	else
		byteLength = intArrayLength/8+1;
		
	if(byteLength>0)
		*result = (unsigned char*)malloc(byteLength*sizeof(unsigned char));
	else
		*result = NULL;
	size_t n = 0;
	int tmp, type;
	for(i = 0;i<byteLength;i++)
	{
		tmp = 0;
		for(j = 0;j<8&&n<intArrayLength;j++)
		{
			type = intArray[n];
			if(type == 1)
				tmp = (tmp | (1 << (7-j)));
			n++;
		}
    	(*result)[i] = (unsigned char)tmp;
	}
	return byteLength;
}

size_t convertIntArray2ByteArray_fast_1b_to_result(unsigned char* intArray, size_t intArrayLength, unsigned char *result)
{
	size_t byteLength = 0;
	size_t i, j; 
	if(intArrayLength%8==0)
		byteLength = intArrayLength/8;
	else
		byteLength = intArrayLength/8+1;
		
	size_t n = 0;
	int tmp, type;
	for(i = 0;i<byteLength;i++)
	{
		tmp = 0;
		for(j = 0;j<8&&n<intArrayLength;j++)
		{
			type = intArray[n];
			if(type == 1)
				tmp = (tmp | (1 << (7-j)));
			n++;
		}
    	result[i] = (unsigned char)tmp;
	}
	return byteLength;
}

void convertByteArray2IntArray_fast_1b_args(size_t intArrayLength, unsigned char* byteArray, size_t byteArrayLength, unsigned char* intArray)
{   
	size_t n = 0, i;
	int tmp;
	for (i = 0; i < byteArrayLength-1; i++) 
	{
		tmp = byteArray[i];
		intArray[n++] = (tmp & 0x80) >> 7;
		intArray[n++] = (tmp & 0x40) >> 6;
		intArray[n++] = (tmp & 0x20) >> 5;
		intArray[n++] = (tmp & 0x10) >> 4;
		intArray[n++] = (tmp & 0x08) >> 3;
		intArray[n++] = (tmp & 0x04) >> 2;
		intArray[n++] = (tmp & 0x02) >> 1;
		intArray[n++] = (tmp & 0x01) >> 0;		
	}
	
	tmp = byteArray[i];	
	if(n == intArrayLength)
		return;
	intArray[n++] = (tmp & 0x80) >> 7;
	if(n == intArrayLength)
		return;	
	intArray[n++] = (tmp & 0x40) >> 6;
	if(n == intArrayLength)
		return;	
	intArray[n++] = (tmp & 0x20) >> 5;
	if(n == intArrayLength)
		return;
	intArray[n++] = (tmp & 0x10) >> 4;
	if(n == intArrayLength)
		return;	
	intArray[n++] = (tmp & 0x08) >> 3;
	if(n == intArrayLength)
		return;	
	intArray[n++] = (tmp & 0x04) >> 2;
	if(n == intArrayLength)
		return;	
	intArray[n++] = (tmp & 0x02) >> 1;
	if(n == intArrayLength)
		return;	
	intArray[n++] = (tmp & 0x01) >> 0;	
}

void convertByteArray2IntArray_fast_1b(size_t intArrayLength, unsigned char* byteArray, size_t byteArrayLength, unsigned char **intArray)	
{
    if(intArrayLength > byteArrayLength*8)
    {
    	printf("Error: intArrayLength > byteArrayLength*8\n");
    	printf("intArrayLength=%zu, byteArrayLength = %zu", intArrayLength, byteArrayLength);
    	exit(0);
    }
	if(intArrayLength>0)
		*intArray = (unsigned char*)malloc(intArrayLength*sizeof(unsigned char));
	else
		*intArray = NULL;    
    
	size_t n = 0, i;
	int tmp;
	for (i = 0; i < byteArrayLength-1; i++) 
	{
		tmp = byteArray[i];
		(*intArray)[n++] = (tmp & 0x80) >> 7;
		(*intArray)[n++] = (tmp & 0x40) >> 6;
		(*intArray)[n++] = (tmp & 0x20) >> 5;
		(*intArray)[n++] = (tmp & 0x10) >> 4;
		(*intArray)[n++] = (tmp & 0x08) >> 3;
		(*intArray)[n++] = (tmp & 0x04) >> 2;
		(*intArray)[n++] = (tmp & 0x02) >> 1;
		(*intArray)[n++] = (tmp & 0x01) >> 0;		
	}
	
	tmp = byteArray[i];	
	if(n == intArrayLength)
		return;
	(*intArray)[n++] = (tmp & 0x80) >> 7;
	if(n == intArrayLength)
		return;	
	(*intArray)[n++] = (tmp & 0x40) >> 6;
	if(n == intArrayLength)
		return;	
	(*intArray)[n++] = (tmp & 0x20) >> 5;
	if(n == intArrayLength)
		return;
	(*intArray)[n++] = (tmp & 0x10) >> 4;
	if(n == intArrayLength)
		return;	
	(*intArray)[n++] = (tmp & 0x08) >> 3;
	if(n == intArrayLength)
		return;	
	(*intArray)[n++] = (tmp & 0x04) >> 2;
	if(n == intArrayLength)
		return;	
	(*intArray)[n++] = (tmp & 0x02) >> 1;
	if(n == intArrayLength)
		return;	
	(*intArray)[n++] = (tmp & 0x01) >> 0;		
}


inline size_t convertIntArray2ByteArray_fast_2b_args(unsigned char* timeStepType, size_t timeStepTypeLength, unsigned char *result)
{
	register unsigned char tmp = 0;
	size_t i, j = 0, byteLength = 0;
	if(timeStepTypeLength%4==0)
		byteLength = timeStepTypeLength*2/8;
	else
		byteLength = timeStepTypeLength*2/8+1;
	size_t n = 0;
	if(timeStepTypeLength%4==0)
	{
		for(i = 0;i<byteLength;i++)
		{
			tmp = 0;

			tmp |= timeStepType[n++] << 6;
			tmp |= timeStepType[n++] << 4;
			tmp |= timeStepType[n++] << 2;
			tmp |= timeStepType[n++];

		/*	for(j = 0;j<4;j++) 
			{
				unsigned char type = timeStepType[n++];
				tmp = tmp | type << (6-(j<<1));
			}*/

			result[i] = tmp;
		}		
	}
	else
	{
		size_t byteLength_ = byteLength - 1;
		for(i = 0;i<byteLength_;i++)
		{
			tmp = 0;

			tmp |= timeStepType[n++] << 6;
			tmp |= timeStepType[n++] << 4;
			tmp |= timeStepType[n++] << 2;
			tmp |= timeStepType[n++];	

		/*	for(j = 0;j<4;j++)
			{
				unsigned char type = timeStepType[n++];
				tmp = tmp | type << (6-(j<<1));
			}*/

			result[i] = tmp;
		}
		tmp = 0;
        int mod4 = timeStepTypeLength%4;
        for(j=0;j<mod4;j++)
		{
			unsigned char type = timeStepType[n++];
			tmp = tmp | type << (6-(j<<1));			
		}
		result[i] = tmp;
	}

/*	//The original version (the slowest version)
 * for(i = 0;i<byteLength;i++)
	{
		tmp = 0;

		for(j = 0;j<4&&n<timeStepTypeLength;j++)
		{
			unsigned char type = timeStepType[n++];
			tmp = tmp | type << (6-(j<<1));
		}

		result[i] = tmp;
	}
*/
	return byteLength;
}

/**
 * little endian
 * [01|10|11|00|....]-->[01|10|11|00][....]
 * @param timeStepType
 * @return
 */
size_t convertIntArray2ByteArray_fast_2b(unsigned char* timeStepType, size_t timeStepTypeLength, unsigned char **result)
{
	size_t i, j, byteLength = 0;
	if(timeStepTypeLength%4==0)
		byteLength = timeStepTypeLength*2/8;
	else
		byteLength = timeStepTypeLength*2/8+1;
	if(byteLength>0)
		*result = (unsigned char*)malloc(byteLength*sizeof(unsigned char));
	else
		*result = NULL;
	size_t n = 0;
	for(i = 0;i<byteLength;i++)
	{
		int tmp = 0;
		for(j = 0;j<4&&n<timeStepTypeLength;j++)
		{
			int type = timeStepType[n];
			switch(type)
			{
			case 0: 
				
				break;
			case 1:
				tmp = (tmp | (1 << (6-j*2)));
				break;
			case 2:
				tmp = (tmp | (2 << (6-j*2)));
				break;
			case 3:
				tmp = (tmp | (3 << (6-j*2)));
				break;
			default:
				printf("Error: wrong timestep type...: type[%zu]=%d\n", n, type);
				exit(0);
			}
			n++;
		}
		(*result)[i] = (unsigned char)tmp;
	}
	return byteLength;
}

void convertByteArray2IntArray_fast_2b(size_t stepLength, unsigned char* byteArray, size_t byteArrayLength, unsigned char **intArray)
{
	if(stepLength > byteArrayLength*4)
	{
		printf("Error: stepLength > byteArray.length*4\n");
		printf("stepLength=%zu, byteArray.length=%zu\n", stepLength, byteArrayLength);
		exit(0);
	}
	if(stepLength>0)
		*intArray = (unsigned char*)malloc(stepLength*sizeof(unsigned char));
	else
		*intArray = NULL;
	size_t i, n = 0;

	int mod4 = stepLength%4;
	if(mod4==0)
	{
		for (i = 0; i < byteArrayLength; i++) {
			unsigned char tmp = byteArray[i];
			(*intArray)[n++] = (tmp & 0xC0) >> 6;
			(*intArray)[n++] = (tmp & 0x30) >> 4;
			(*intArray)[n++] = (tmp & 0x0C) >> 2;
			(*intArray)[n++] = tmp & 0x03;
		}	
	}
	else
	{
		size_t t = byteArrayLength - mod4;
		for (i = 0; i < t; i++) {
			unsigned char tmp = byteArray[i];
			(*intArray)[n++] = (tmp & 0xC0) >> 6;
			(*intArray)[n++] = (tmp & 0x30) >> 4;
			(*intArray)[n++] = (tmp & 0x0C) >> 2;
			(*intArray)[n++] = tmp & 0x03;
		}
		unsigned char tmp = byteArray[i];				
		switch(mod4)
		{
		case 1:
			(*intArray)[n++] = (tmp & 0xC0) >> 6;
			break;
		case 2:
			(*intArray)[n++] = (tmp & 0xC0) >> 6;
			(*intArray)[n++] = (tmp & 0x30) >> 4;			
			break;
		case 3:	
			(*intArray)[n++] = (tmp & 0xC0) >> 6;
			(*intArray)[n++] = (tmp & 0x30) >> 4;
			(*intArray)[n++] = (tmp & 0x0C) >> 2;		
			break;
		}
	}
}


inline int getLeftMovingSteps(size_t k, unsigned char resiBitLength)
{
	return 8 - k%8 - resiBitLength;
}


