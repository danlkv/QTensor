/**
 *  @file szxd_double.c
 *  @author Sheng Di, Kai Zhao
 *  @date Feb, 2022
 *  @brief 
 *  (C) 2022 by Mathematics and Computer Science (MCS), Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include <stdlib.h> 
#include <stdio.h>
#include <string.h>
#include "szxd_double.h"
#include "szx.h"
#include "szx_BytesToolkit.h"
#include "szx_TypeManager.h"
#ifdef _OPENMP
#include "omp.h"
#endif

int SZ_fast_decompress_args_unpredictable_one_block_double(double* newData, size_t blockSize, unsigned char* cmpBytes)
{
	int cmpSize = 0;
	size_t nbEle = blockSize;
	
	register double medianValue;
	size_t leadNumArray_size = nbEle%4==0?nbEle/4:nbEle/4+1;
	
	size_t k = 0;
	int reqLength = (int)cmpBytes[k];
	k++;
	medianValue = (double)bytesToFloat(&(cmpBytes[k]));
	k+=sizeof(float);
	
	unsigned char* leadNumArray = &(cmpBytes[k]);
	k += leadNumArray_size;
	unsigned char* residualMidBytes = &(cmpBytes[k]);	
	unsigned char* q = residualMidBytes;
		
	cmpSize = k;	
		
	size_t i = 0, j = 0;
	k = 0;
	
	register ldouble lfBuf_pre;
	register ldouble lfBuf_cur;
	
	lfBuf_pre.lvalue = 0;

	int reqBytesLength, resiBitsLength; 
	register unsigned char leadingNum;

	reqBytesLength = reqLength/8;
	resiBitsLength = reqLength%8;
	int rightShiftBits = 0;
	
	if(resiBitsLength!=0)
	{
		rightShiftBits = 8 - resiBitsLength;
		reqBytesLength ++;
	}
	
	//sz_cost_start();
	if(sysEndianType==LITTLE_ENDIAN_SYSTEM)
	{
		//reqBytesLength must not be equal to 1 for double data
		if(reqBytesLength == 3)
		{
			for(i=0;i < nbEle;i++)
			{
				lfBuf_cur.value = 0;
				
				j = (i >> 2); //i/4
				k = (i & 0x03) << 1; //(i%4)*2
				leadingNum = (leadNumArray[j] >> (6 - k)) & 0x03;
				
				if(leadingNum == 1)
				{	
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];
					lfBuf_cur.byte[5] = q[0];
					lfBuf_cur.byte[6] = q[1];				
					q += 2;
				}
				else if(leadingNum == 2)
				{
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];
					lfBuf_cur.byte[5] = q[0];									
					q += 1;
				}
				else if(leadingNum == 3)
				{
					lfBuf_cur.byte[5] = lfBuf_pre.byte[5];
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];				
				}
				else //==0
				{
					lfBuf_cur.byte[5] = q[0];
					lfBuf_cur.byte[6] = q[1];					
					lfBuf_cur.byte[7] = q[2];					
					q += 3;
				}

				lfBuf_cur.lvalue = lfBuf_cur.lvalue << rightShiftBits;
				newData[i] = lfBuf_cur.value + medianValue;
				lfBuf_cur.lvalue = lfBuf_cur.lvalue >> rightShiftBits;
				
				lfBuf_pre = lfBuf_cur;
			}
		}
		else if(reqBytesLength == 2)
		{
			for(i=0;i < nbEle;i++)
			{
				lfBuf_cur.value = 0;
				
				j = (i >> 2); //i/4
				k = (i & 0x03) << 1; //(i%4)*2
				leadingNum = (leadNumArray[j] >> (6 - k)) & 0x03;
	
				if(leadingNum == 1)
				{	
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];
					lfBuf_cur.byte[6] = q[0];			
					q += 1;	
				}
				else if(leadingNum >= 2)
				{
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];									
				}
				else //==0
				{
					lfBuf_cur.byte[6] = q[0];					
					lfBuf_cur.byte[7] = q[1];					
					q += 2;
				}
				
				lfBuf_cur.lvalue = lfBuf_cur.lvalue << rightShiftBits;
				newData[i] = lfBuf_cur.value + medianValue;
				lfBuf_cur.lvalue = lfBuf_cur.lvalue >> rightShiftBits;
				
				lfBuf_pre = lfBuf_cur;
			}					
		}
		else if(reqBytesLength == 4)
		{
			for(i=0;i < nbEle;i++)
			{
				lfBuf_cur.value = 0;
				
				j = (i >> 2); //i/4
				k = (i & 0x03) << 1; //(i%4)*2
				leadingNum = (leadNumArray[j] >> (6 - k)) & 0x03;
				
				if(leadingNum == 1)
				{	
					lfBuf_cur.byte[4] = q[0];
					lfBuf_cur.byte[5] = q[1];
					lfBuf_cur.byte[6] = q[2];				
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];					
					q += 3;
				}
				else if(leadingNum == 2)
				{
					lfBuf_cur.byte[4] = q[0];									
					lfBuf_cur.byte[5] = q[1];									
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];					
					q += 2;
				}
				else if(leadingNum == 3)
				{
					lfBuf_cur.byte[4] = q[0];									
					lfBuf_cur.byte[5] = lfBuf_pre.byte[5];
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];	
					q += 1;				
				}
				else //==0
				{
					lfBuf_cur.byte[4] = q[0];
					lfBuf_cur.byte[5] = q[1];
					lfBuf_cur.byte[6] = q[2];					
					lfBuf_cur.byte[7] = q[3];					
					q += 4;
				}

				lfBuf_cur.lvalue = lfBuf_cur.lvalue << rightShiftBits;
				newData[i] = lfBuf_cur.value + medianValue;
				lfBuf_cur.lvalue = lfBuf_cur.lvalue >> rightShiftBits;
				
				lfBuf_pre = lfBuf_cur;			
			}
		} 		
		else if(reqBytesLength == 5)
		{
			for(i=0;i < nbEle;i++)
			{
				lfBuf_cur.value = 0;
				
				j = (i >> 2); //i/4
				k = (i & 0x03) << 1; //(i%4)*2
				leadingNum = (leadNumArray[j] >> (6 - k)) & 0x03;
				
				if(leadingNum == 1)
				{	
					lfBuf_cur.byte[3] = q[0];
					lfBuf_cur.byte[4] = q[1];
					lfBuf_cur.byte[5] = q[2];
					lfBuf_cur.byte[6] = q[3];				
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];					
					q += 4;
				}
				else if(leadingNum == 2)
				{
					lfBuf_cur.byte[3] = q[0];
					lfBuf_cur.byte[4] = q[1];									
					lfBuf_cur.byte[5] = q[2];									
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];					
					q += 3;
				}
				else if(leadingNum == 3)
				{
					lfBuf_cur.byte[3] = q[0];			
					lfBuf_cur.byte[4] = q[1];									
					lfBuf_cur.byte[5] = lfBuf_pre.byte[5];
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];	
					q += 2;				
				}
				else //==0
				{
					lfBuf_cur.byte[3] = q[0];
					lfBuf_cur.byte[4] = q[1];
					lfBuf_cur.byte[5] = q[2];					
					lfBuf_cur.byte[6] = q[3];		
					lfBuf_cur.byte[7] = q[3];				
					q += 5;
				}

				lfBuf_cur.lvalue = lfBuf_cur.lvalue << rightShiftBits;
				newData[i] = lfBuf_cur.value + medianValue;
				lfBuf_cur.lvalue = lfBuf_cur.lvalue >> rightShiftBits;
				
				lfBuf_pre = lfBuf_cur;			
			}
		}
		else if(reqBytesLength == 6)
		{
			for(i=0;i < nbEle;i++)
			{
				lfBuf_cur.value = 0;
				
				j = (i >> 2); //i/4
				k = (i & 0x03) << 1; //(i%4)*2
				leadingNum = (leadNumArray[j] >> (6 - k)) & 0x03;
				
				if(leadingNum == 1)
				{	
					lfBuf_cur.byte[2] = q[0];					
					lfBuf_cur.byte[3] = q[1];
					lfBuf_cur.byte[4] = q[2];
					lfBuf_cur.byte[5] = q[3];
					lfBuf_cur.byte[6] = q[4];				
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];					
					q += 5;
				}
				else if(leadingNum == 2)
				{
					lfBuf_cur.byte[2] = q[0];					
					lfBuf_cur.byte[3] = q[1];
					lfBuf_cur.byte[4] = q[2];									
					lfBuf_cur.byte[5] = q[3];									
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];					
					q += 4;
				}
				else if(leadingNum == 3)
				{
					lfBuf_cur.byte[2] = q[0];			
					lfBuf_cur.byte[3] = q[1];						
					lfBuf_cur.byte[4] = q[2];									
					lfBuf_cur.byte[5] = lfBuf_pre.byte[5];
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];	
					q += 3;				
				}
				else //==0
				{
					lfBuf_cur.byte[2] = q[0];						
					lfBuf_cur.byte[3] = q[1];
					lfBuf_cur.byte[4] = q[2];
					lfBuf_cur.byte[5] = q[3];					
					lfBuf_cur.byte[6] = q[4];		
					lfBuf_cur.byte[7] = q[5];				
					q += 6;
				}

				lfBuf_cur.lvalue = lfBuf_cur.lvalue << rightShiftBits;
				newData[i] = lfBuf_cur.value + medianValue;
				lfBuf_cur.lvalue = lfBuf_cur.lvalue >> rightShiftBits;
				
				lfBuf_pre = lfBuf_cur;			
			}
		}
		else if(reqBytesLength == 7)
		{
			for(i=0;i < nbEle;i++)
			{
				lfBuf_cur.value = 0;
				
				j = (i >> 2); //i/4
				k = (i & 0x03) << 1; //(i%4)*2
				leadingNum = (leadNumArray[j] >> (6 - k)) & 0x03;
				
				if(leadingNum == 1)
				{	
					lfBuf_cur.byte[1] = q[0];		
					lfBuf_cur.byte[2] = q[1];					
					lfBuf_cur.byte[3] = q[2];
					lfBuf_cur.byte[4] = q[3];
					lfBuf_cur.byte[5] = q[4];
					lfBuf_cur.byte[6] = q[5];				
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];					
					q += 6;
				}
				else if(leadingNum == 2)
				{
					lfBuf_cur.byte[1] = q[0];							
					lfBuf_cur.byte[2] = q[1];					
					lfBuf_cur.byte[3] = q[2];
					lfBuf_cur.byte[4] = q[3];									
					lfBuf_cur.byte[5] = q[4];									
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];					
					q += 5;
				}
				else if(leadingNum == 3)
				{
					lfBuf_cur.byte[1] = q[0];			
					lfBuf_cur.byte[2] = q[1];					
					lfBuf_cur.byte[3] = q[2];						
					lfBuf_cur.byte[4] = q[3];									
					lfBuf_cur.byte[5] = lfBuf_pre.byte[5];
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];	
					q += 4;				
				}
				else //==0
				{
					lfBuf_cur.byte[1] = q[0];					
					lfBuf_cur.byte[2] = q[1];						
					lfBuf_cur.byte[3] = q[2];
					lfBuf_cur.byte[4] = q[3];
					lfBuf_cur.byte[5] = q[4];					
					lfBuf_cur.byte[6] = q[5];		
					lfBuf_cur.byte[7] = q[6];				
					q += 7;
				}

				lfBuf_cur.lvalue = lfBuf_cur.lvalue << rightShiftBits;
				newData[i] = lfBuf_cur.value + medianValue;
				lfBuf_cur.lvalue = lfBuf_cur.lvalue >> rightShiftBits;
				
				lfBuf_pre = lfBuf_cur;			
			}
		}		
		else //reqBytesLength == 8
		{
			for(i=0;i < nbEle;i++)
			{
				lfBuf_cur.value = 0;
				
				j = (i >> 2); //i/4
				k = (i & 0x03) << 1; //(i%4)*2
				leadingNum = (leadNumArray[j] >> (6 - k)) & 0x03;
				
				if(leadingNum == 1)
				{
					lfBuf_cur.byte[0] = q[0];								
					lfBuf_cur.byte[1] = q[1];		
					lfBuf_cur.byte[2] = q[2];					
					lfBuf_cur.byte[3] = q[3];
					lfBuf_cur.byte[4] = q[4];
					lfBuf_cur.byte[5] = q[5];
					lfBuf_cur.byte[6] = q[6];				
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];					
					q += 7;
				}
				else if(leadingNum == 2)
				{
					lfBuf_cur.byte[0] = q[0];						
					lfBuf_cur.byte[1] = q[1];							
					lfBuf_cur.byte[2] = q[2];					
					lfBuf_cur.byte[3] = q[3];
					lfBuf_cur.byte[4] = q[4];									
					lfBuf_cur.byte[5] = q[5];									
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];					
					q += 6;
				}
				else if(leadingNum == 3)
				{
					lfBuf_cur.byte[0] = q[0];						
					lfBuf_cur.byte[1] = q[1];			
					lfBuf_cur.byte[2] = q[2];					
					lfBuf_cur.byte[3] = q[3];						
					lfBuf_cur.byte[4] = q[4];									
					lfBuf_cur.byte[5] = lfBuf_pre.byte[5];
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];	
					q += 5;				
				}
				else //==0
				{
					lfBuf_cur.byte[0] = q[0];						
					lfBuf_cur.byte[1] = q[1];					
					lfBuf_cur.byte[2] = q[2];						
					lfBuf_cur.byte[3] = q[3];
					lfBuf_cur.byte[4] = q[4];
					lfBuf_cur.byte[5] = q[5];					
					lfBuf_cur.byte[6] = q[6];		
					lfBuf_cur.byte[7] = q[7];				
					q += 8;
				}

				lfBuf_cur.lvalue = lfBuf_cur.lvalue << rightShiftBits;
				newData[i] = lfBuf_cur.value + medianValue;
				lfBuf_cur.lvalue = lfBuf_cur.lvalue >> rightShiftBits;
				
				lfBuf_pre = lfBuf_cur;			
			}
		}				
	}
	else
	{
		
	}
	
	cmpSize += (q - residualMidBytes); //add the number of residualMidBytes
	return cmpSize;
}


void SZ_fast_decompress_args_unpredictable_blocked_double(double** newData, size_t nbEle, unsigned char* cmpBytes)
{
	*newData = (double*)malloc(sizeof(double)*nbEle);

	unsigned char* r = cmpBytes;
	r += 4;
	int blockSize = r[0];  //get block size
	r++;
	size_t nbConstantBlocks = bytesToLong_bigEndian(r); //get number of constant blocks
	r += sizeof(size_t);

	size_t nbBlocks = nbEle/blockSize;
	size_t remainCount = nbEle%blockSize;
	size_t stateNBBytes = remainCount == 0 ? (nbBlocks%8==0?nbBlocks/8:nbBlocks/8+1) : ((nbBlocks+1)%8==0? (nbBlocks+1)/8:(nbBlocks+1)/8+1);
	size_t actualNBBlocks = remainCount==0 ? nbBlocks : nbBlocks+1;
	unsigned char* stateArray = (unsigned char*)malloc(actualNBBlocks);
	float* constantMedianArray = (float*)malloc(nbConstantBlocks*sizeof(float));

	convertByteArray2IntArray_fast_1b_args(actualNBBlocks, r, stateNBBytes, stateArray); //get the stateArray

	unsigned char* p = r + stateNBBytes; //p is the starting address of constant median values.

	size_t i = 0, j = 0, k = 0; //k is used to keep track of constant block index
	for(i = 0;i < nbConstantBlocks;i++, j+=4) //get the median values for constant-value blocks
		constantMedianArray[i] = bytesToFloat(p+j);

	unsigned char* q = p + sizeof(float)*nbConstantBlocks; //q is the starting address of the non-constant data blocks
	double* op = *newData;

	for(i=0;i<nbBlocks;i++, op += blockSize)
	{
		unsigned char state = stateArray[i];
		if(state) //non-constant block
		{
			int cmpSize = SZ_fast_decompress_args_unpredictable_one_block_double(op, blockSize, q);
			q += cmpSize;
		}
		else //constant block
		{
			float medianValue = constantMedianArray[k];
			for(j=0;j<blockSize;j++)
				op[j] = medianValue;
			p += sizeof(float);
			k ++;
		}
	}

	if(remainCount)
	{
		unsigned char state = stateArray[i];
		if(state) //non-constant block
		{
			SZ_fast_decompress_args_unpredictable_one_block_double(op, remainCount, q);
		}
		else //constant block
		{
			float medianValue = constantMedianArray[k];
			for(j=0;j<remainCount;j++)
				op[j] = medianValue;
		}
	}

	free(stateArray);
	free(constantMedianArray);
}

void SZ_fast_decompress_args_unpredictable_blocked_randomaccess_double_openmp(double** newData, size_t nbEle, unsigned char* cmpBytes) {

	*newData = (double *) malloc(sizeof(double) * nbEle);
	sz_cost_start();
	unsigned char *r = cmpBytes;
	r += 4; //skip version information
	int blockSize = bytesToLong_bigEndian(r);  //get block size
    r += sizeof(size_t);
	size_t nbConstantBlocks = bytesToLong_bigEndian(r); //get number of constant blocks
	r += sizeof(size_t);

	size_t nbBlocks = nbEle / blockSize;
	size_t remainCount = nbEle % blockSize;
	size_t stateNBBytes =
			remainCount == 0 ? (nbBlocks % 8 == 0 ? nbBlocks / 8 : nbBlocks / 8 + 1) : ((nbBlocks + 1) % 8 == 0 ?
																						(nbBlocks + 1) / 8 :
																						(nbBlocks + 1) / 8 + 1);
	size_t actualNBBlocks = remainCount == 0 ? nbBlocks : nbBlocks + 1;

	size_t nbNonConstantBlocks = actualNBBlocks - nbConstantBlocks;

	unsigned char *stateArray = (unsigned char *) malloc(actualNBBlocks);
//	double *constantMedianArray = (double *) malloc(nbConstantBlocks * sizeof(double));
    unsigned char **qarray = (unsigned char **) malloc(actualNBBlocks * sizeof(unsigned char *));
    float *parray = (float *) malloc(actualNBBlocks * sizeof(float));

    int16_t* O = (int16_t*) r;
    unsigned char *R = r + nbNonConstantBlocks*sizeof(uint16_t); //block-size information
    unsigned char *p = R + stateNBBytes; //p is the starting address of constant median values.
    float *constantMedianArray = (float *) p;
    unsigned char *q = p + sizeof(float) * nbConstantBlocks; //q is the starting address of the non-constant data blocks
    double *op = *newData;

	size_t nonConstantBlockID = 0, constantBlockID = 0;
//    sz_cost_end_msg("sequential-1 malloc");

//    sz_cost_start();
    size_t i = 0;// k = 0; //k is used to keep track of constant block index
//    for (i = 0; i < nbConstantBlocks; i++, k += 4) //get the median values for constant-value blocks
//        constantMedianArray[i] = bytesToFloat(p + k);

    convertByteArray2IntArray_fast_1b_args(actualNBBlocks, R, stateNBBytes, stateArray); //get the stateArray
//    sz_cost_end_msg("sequential-2 byte to int");

//    sz_cost_start();
    for (i = 0; i < actualNBBlocks; i++) {
		if (stateArray[i]) {
			qarray[i] = q;

			q += O[nonConstantBlockID++];
		} else {
			parray[i] = constantMedianArray[constantBlockID++];
		}
	}

//   sz_cost_end_msg("sequential-3 sum");
//	sz_cost_start();
#pragma omp parallel for schedule(static)
	for (i = 0; i < nbBlocks; i++) {
		if (stateArray[i]) {//non-constant block
			SZ_fast_decompress_args_unpredictable_one_block_double(op + i * blockSize, blockSize, qarray[i]);
		} else {//constant block
			for (int j = 0; j < blockSize; j++)
				op[i * blockSize + j] = parray[i];
		}
	}
//	sz_cost_end_msg("parallel-1");

//	sz_cost_start();
	if (remainCount) {
        i = nbBlocks;
        if (stateArray[i]) { //non-constant block
			SZ_fast_decompress_args_unpredictable_one_block_double(op + i * blockSize, remainCount, qarray[i]);
		} else {//constant block
			for (int j = 0; j < remainCount; j++)
				op[i * blockSize + j] = parray[i];
		}
	}

	free(parray);
	free(qarray);
	free(stateArray);
//	free(constantMedianArray);
//	sz_cost_end_msg("sequence-3 free");
}


void SZ_fast_decompress_args_unpredictable_blocked_randomaccess_double(double** newData, size_t nbEle, unsigned char* cmpBytes){
	*newData = (double*)malloc(sizeof(double)*nbEle);
	
	unsigned char* r = cmpBytes;
	r+=4; //skip version information
    int blockSize = bytesToLong_bigEndian(r);  //get block size
    r += sizeof(size_t);
	size_t nbConstantBlocks = bytesToLong_bigEndian(r); //get number of constant blocks
	r += sizeof(size_t);
		
	size_t nbBlocks = nbEle/blockSize;
	size_t remainCount = nbEle%blockSize;
	size_t stateNBBytes = remainCount == 0 ? (nbBlocks%8==0?nbBlocks/8:nbBlocks/8+1) : ((nbBlocks+1)%8==0? (nbBlocks+1)/8:(nbBlocks+1)/8+1);
	size_t actualNBBlocks = remainCount==0 ? nbBlocks : nbBlocks+1;
	
	size_t nbNonConstantBlocks = actualNBBlocks - nbConstantBlocks;
	

	unsigned char* stateArray = (unsigned char*)malloc(actualNBBlocks);
	float* constantMedianArray = (float*)malloc(nbConstantBlocks*sizeof(float));

    int16_t* O = (int16_t*) r;
    unsigned char* R = r+ nbNonConstantBlocks*sizeof(uint16_t); //block-size information

    convertByteArray2IntArray_fast_1b_args(actualNBBlocks, R, stateNBBytes, stateArray); //get the stateArray
	
	unsigned char* p = R + stateNBBytes; //p is the starting address of constant median values.
	
	size_t i = 0, j = 0, k = 0; //k is used to keep track of constant block index
	for(i = 0;i < nbConstantBlocks;i++, j+=4) //get the median values for constant-value blocks
		constantMedianArray[i] = bytesToFloat(p+j);

	unsigned char* q = p + sizeof(float)*nbConstantBlocks; //q is the starting address of the non-constant data blocks
	double* op = *newData;

	size_t nonConstantBlockID=0;
	for(i=0;i<nbBlocks;i++, op += blockSize)
	{
		unsigned char state = stateArray[i];
		if(state) //non-constant block
		{
            SZ_fast_decompress_args_unpredictable_one_block_double(op, blockSize, q);
            q += O[nonConstantBlockID];
            nonConstantBlockID++;
		}
		else //constant block
		{
			float medianValue = constantMedianArray[k];			
			for(j=0;j<blockSize;j++)
				op[j] = medianValue;
			p += sizeof(float);
			k ++;
		}
	}

	if(remainCount)
	{
		unsigned char state = stateArray[i];
		if(state) //non-constant block
		{
			SZ_fast_decompress_args_unpredictable_one_block_double(op, remainCount, q);	
		}
		else //constant block
		{
			float medianValue = constantMedianArray[k];				
			for(j=0;j<remainCount;j++)
				op[j] = medianValue;
		}		
	}
	
	free(stateArray);
	free(constantMedianArray);
}

void SZ_fast_decompress_args_unpredictable_double(double** newData, size_t r5, size_t r4, size_t r3, size_t r2, size_t r1, unsigned char* cmpBytes,
size_t cmpSize)
{
	size_t nbEle = computeDataLength(r5, r4, r3, r2, r1);
	*newData = (double*)malloc(sizeof(double)*nbEle);	
	
	register double medianValue;
	size_t leadNumArray_size;

    unsigned char *r = cmpBytes;
    r += 4; //skip version information

	size_t k = 0;
	int reqLength = (int)r[k];
	k++;
	medianValue = (double)bytesToFloat(&(r[k]));
	k+=sizeof(float);
	leadNumArray_size = bytesToSize(&(r[k]));
	k+=sizeof(size_t);
	
	unsigned char* leadNumArray = &(r[k]);
	k += leadNumArray_size;
	unsigned char* residualMidBytes = &(r[k]);
	unsigned char* q = residualMidBytes;
		
	size_t i = 0, j = 0;
	k = 0;
	
	register ldouble lfBuf_pre;
	register ldouble lfBuf_cur;
	
	lfBuf_pre.lvalue = 0;

	int reqBytesLength, resiBitsLength; 
	register unsigned char leadingNum;

	reqBytesLength = reqLength/8;
	resiBitsLength = reqLength%8;
	int rightShiftBits = 0;
	
	if(resiBitsLength!=0)
	{
		rightShiftBits = 8 - resiBitsLength;
		reqBytesLength ++;
	}
	
	//sz_cost_start();
	if(sysEndianType==LITTLE_ENDIAN_SYSTEM) {

		//reqBytesLength must not be equal to 1 for double data
		if(reqBytesLength == 3)
		{
			for(i=0;i < nbEle;i++)
			{
				lfBuf_cur.value = 0;
				
				j = (i >> 2); //i/4
				k = (i & 0x03) << 1; //(i%4)*2
				leadingNum = (leadNumArray[j] >> (6 - k)) & 0x03;
				
				if(leadingNum == 1)
				{	
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];
					lfBuf_cur.byte[5] = q[0];
					lfBuf_cur.byte[6] = q[1];				
					q += 2;
				}
				else if(leadingNum == 2)
				{
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];
					lfBuf_cur.byte[5] = q[0];									
					q += 1;
				}
				else if(leadingNum == 3)
				{
					lfBuf_cur.byte[5] = lfBuf_pre.byte[5];
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];				
				}
				else //==0
				{
					lfBuf_cur.byte[5] = q[0];
					lfBuf_cur.byte[6] = q[1];					
					lfBuf_cur.byte[7] = q[2];					
					q += 3;
				}

				lfBuf_cur.lvalue = lfBuf_cur.lvalue << rightShiftBits;
				(*newData)[i] = lfBuf_cur.value + medianValue;
				lfBuf_cur.lvalue = lfBuf_cur.lvalue >> rightShiftBits;
				
				lfBuf_pre = lfBuf_cur;
			}
		}
		else if(reqBytesLength == 2)
		{
			for(i=0;i < nbEle;i++)
			{
				lfBuf_cur.value = 0;
				
				j = (i >> 2); //i/4
				k = (i & 0x03) << 1; //(i%4)*2
				leadingNum = (leadNumArray[j] >> (6 - k)) & 0x03;
	
				if(leadingNum == 1)
				{	
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];
					lfBuf_cur.byte[6] = q[0];			
					q += 1;	
				}
				else if(leadingNum >= 2)
				{
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];									
				}
				else //==0
				{
					lfBuf_cur.byte[6] = q[0];					
					lfBuf_cur.byte[7] = q[1];					
					q += 2;
				}
				
				lfBuf_cur.lvalue = lfBuf_cur.lvalue << rightShiftBits;
				(*newData)[i] = lfBuf_cur.value + medianValue;
				lfBuf_cur.lvalue = lfBuf_cur.lvalue >> rightShiftBits;
				
				lfBuf_pre = lfBuf_cur;
			}					
		}
		else if(reqBytesLength == 4)
		{
			for(i=0;i < nbEle;i++)
			{
				lfBuf_cur.value = 0;
				
				j = (i >> 2); //i/4
				k = (i & 0x03) << 1; //(i%4)*2
				leadingNum = (leadNumArray[j] >> (6 - k)) & 0x03;
				
				if(leadingNum == 1)
				{	
					lfBuf_cur.byte[4] = q[0];
					lfBuf_cur.byte[5] = q[1];
					lfBuf_cur.byte[6] = q[2];				
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];					
					q += 3;
				}
				else if(leadingNum == 2)
				{
					lfBuf_cur.byte[4] = q[0];									
					lfBuf_cur.byte[5] = q[1];									
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];					
					q += 2;
				}
				else if(leadingNum == 3)
				{
					lfBuf_cur.byte[4] = q[0];									
					lfBuf_cur.byte[5] = lfBuf_pre.byte[5];
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];	
					q += 1;				
				}
				else //==0
				{
					lfBuf_cur.byte[4] = q[0];
					lfBuf_cur.byte[5] = q[1];
					lfBuf_cur.byte[6] = q[2];					
					lfBuf_cur.byte[7] = q[3];					
					q += 4;
				}

				lfBuf_cur.lvalue = lfBuf_cur.lvalue << rightShiftBits;
				(*newData)[i] = lfBuf_cur.value + medianValue;
				lfBuf_cur.lvalue = lfBuf_cur.lvalue >> rightShiftBits;
				
				lfBuf_pre = lfBuf_cur;			
			}
		} 		
		else if(reqBytesLength == 5)
		{
			for(i=0;i < nbEle;i++)
			{
				lfBuf_cur.value = 0;
				
				j = (i >> 2); //i/4
				k = (i & 0x03) << 1; //(i%4)*2
				leadingNum = (leadNumArray[j] >> (6 - k)) & 0x03;
				
				if(leadingNum == 1)
				{	
					lfBuf_cur.byte[3] = q[0];
					lfBuf_cur.byte[4] = q[1];
					lfBuf_cur.byte[5] = q[2];
					lfBuf_cur.byte[6] = q[3];				
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];					
					q += 4;
				}
				else if(leadingNum == 2)
				{
					lfBuf_cur.byte[3] = q[0];
					lfBuf_cur.byte[4] = q[1];									
					lfBuf_cur.byte[5] = q[2];									
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];					
					q += 3;
				}
				else if(leadingNum == 3)
				{
					lfBuf_cur.byte[3] = q[0];			
					lfBuf_cur.byte[4] = q[1];									
					lfBuf_cur.byte[5] = lfBuf_pre.byte[5];
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];	
					q += 2;				
				}
				else //==0
				{
					lfBuf_cur.byte[3] = q[0];
					lfBuf_cur.byte[4] = q[1];
					lfBuf_cur.byte[5] = q[2];					
					lfBuf_cur.byte[6] = q[3];		
					lfBuf_cur.byte[7] = q[3];				
					q += 5;
				}

				lfBuf_cur.lvalue = lfBuf_cur.lvalue << rightShiftBits;
				(*newData)[i] = lfBuf_cur.value + medianValue;
				lfBuf_cur.lvalue = lfBuf_cur.lvalue >> rightShiftBits;
				
				lfBuf_pre = lfBuf_cur;			
			}
		}
		else if(reqBytesLength == 6)
		{
			for(i=0;i < nbEle;i++)
			{
				lfBuf_cur.value = 0;
				
				j = (i >> 2); //i/4
				k = (i & 0x03) << 1; //(i%4)*2
				leadingNum = (leadNumArray[j] >> (6 - k)) & 0x03;
				
				if(leadingNum == 1)
				{	
					lfBuf_cur.byte[2] = q[0];					
					lfBuf_cur.byte[3] = q[1];
					lfBuf_cur.byte[4] = q[2];
					lfBuf_cur.byte[5] = q[3];
					lfBuf_cur.byte[6] = q[4];				
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];					
					q += 5;
				}
				else if(leadingNum == 2)
				{
					lfBuf_cur.byte[2] = q[0];					
					lfBuf_cur.byte[3] = q[1];
					lfBuf_cur.byte[4] = q[2];									
					lfBuf_cur.byte[5] = q[3];									
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];					
					q += 4;
				}
				else if(leadingNum == 3)
				{
					lfBuf_cur.byte[2] = q[0];			
					lfBuf_cur.byte[3] = q[1];						
					lfBuf_cur.byte[4] = q[2];									
					lfBuf_cur.byte[5] = lfBuf_pre.byte[5];
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];	
					q += 3;				
				}
				else //==0
				{
					lfBuf_cur.byte[2] = q[0];						
					lfBuf_cur.byte[3] = q[1];
					lfBuf_cur.byte[4] = q[2];
					lfBuf_cur.byte[5] = q[3];					
					lfBuf_cur.byte[6] = q[4];		
					lfBuf_cur.byte[7] = q[5];				
					q += 6;
				}

				lfBuf_cur.lvalue = lfBuf_cur.lvalue << rightShiftBits;
				(*newData)[i] = lfBuf_cur.value + medianValue;
				lfBuf_cur.lvalue = lfBuf_cur.lvalue >> rightShiftBits;
				
				lfBuf_pre = lfBuf_cur;			
			}
		}
		else if(reqBytesLength == 7)
		{
			for(i=0;i < nbEle;i++)
			{
				lfBuf_cur.value = 0;
				
				j = (i >> 2); //i/4
				k = (i & 0x03) << 1; //(i%4)*2
				leadingNum = (leadNumArray[j] >> (6 - k)) & 0x03;
				
				if(leadingNum == 1)
				{	
					lfBuf_cur.byte[1] = q[0];		
					lfBuf_cur.byte[2] = q[1];					
					lfBuf_cur.byte[3] = q[2];
					lfBuf_cur.byte[4] = q[3];
					lfBuf_cur.byte[5] = q[4];
					lfBuf_cur.byte[6] = q[5];				
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];					
					q += 6;
				}
				else if(leadingNum == 2)
				{
					lfBuf_cur.byte[1] = q[0];							
					lfBuf_cur.byte[2] = q[1];					
					lfBuf_cur.byte[3] = q[2];
					lfBuf_cur.byte[4] = q[3];									
					lfBuf_cur.byte[5] = q[4];									
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];					
					q += 5;
				}
				else if(leadingNum == 3)
				{
					lfBuf_cur.byte[1] = q[0];			
					lfBuf_cur.byte[2] = q[1];					
					lfBuf_cur.byte[3] = q[2];						
					lfBuf_cur.byte[4] = q[3];									
					lfBuf_cur.byte[5] = lfBuf_pre.byte[5];
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];	
					q += 4;				
				}
				else //==0
				{
					lfBuf_cur.byte[1] = q[0];					
					lfBuf_cur.byte[2] = q[1];						
					lfBuf_cur.byte[3] = q[2];
					lfBuf_cur.byte[4] = q[3];
					lfBuf_cur.byte[5] = q[4];					
					lfBuf_cur.byte[6] = q[5];		
					lfBuf_cur.byte[7] = q[6];				
					q += 7;
				}

				lfBuf_cur.lvalue = lfBuf_cur.lvalue << rightShiftBits;
				(*newData)[i] = lfBuf_cur.value + medianValue;
				lfBuf_cur.lvalue = lfBuf_cur.lvalue >> rightShiftBits;
				
				lfBuf_pre = lfBuf_cur;			
			}
		}		
		else //reqBytesLength == 8
		{
			for(i=0;i < nbEle;i++)
			{
				lfBuf_cur.value = 0;
				
				j = (i >> 2); //i/4
				k = (i & 0x03) << 1; //(i%4)*2
				leadingNum = (leadNumArray[j] >> (6 - k)) & 0x03;
				
				if(leadingNum == 1)
				{
					lfBuf_cur.byte[0] = q[0];								
					lfBuf_cur.byte[1] = q[1];		
					lfBuf_cur.byte[2] = q[2];					
					lfBuf_cur.byte[3] = q[3];
					lfBuf_cur.byte[4] = q[4];
					lfBuf_cur.byte[5] = q[5];
					lfBuf_cur.byte[6] = q[6];				
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];					
					q += 7;
				}
				else if(leadingNum == 2)
				{
					lfBuf_cur.byte[0] = q[0];						
					lfBuf_cur.byte[1] = q[1];							
					lfBuf_cur.byte[2] = q[2];					
					lfBuf_cur.byte[3] = q[3];
					lfBuf_cur.byte[4] = q[4];									
					lfBuf_cur.byte[5] = q[5];									
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];					
					q += 6;
				}
				else if(leadingNum == 3)
				{
					lfBuf_cur.byte[0] = q[0];						
					lfBuf_cur.byte[1] = q[1];			
					lfBuf_cur.byte[2] = q[2];					
					lfBuf_cur.byte[3] = q[3];						
					lfBuf_cur.byte[4] = q[4];									
					lfBuf_cur.byte[5] = lfBuf_pre.byte[5];
					lfBuf_cur.byte[6] = lfBuf_pre.byte[6];
					lfBuf_cur.byte[7] = lfBuf_pre.byte[7];	
					q += 5;				
				}
				else //==0
				{
					lfBuf_cur.byte[0] = q[0];						
					lfBuf_cur.byte[1] = q[1];					
					lfBuf_cur.byte[2] = q[2];						
					lfBuf_cur.byte[3] = q[3];
					lfBuf_cur.byte[4] = q[4];
					lfBuf_cur.byte[5] = q[5];					
					lfBuf_cur.byte[6] = q[6];		
					lfBuf_cur.byte[7] = q[7];				
					q += 8;
				}

				lfBuf_cur.lvalue = lfBuf_cur.lvalue << rightShiftBits;
				(*newData)[i] = lfBuf_cur.value + medianValue;
				lfBuf_cur.lvalue = lfBuf_cur.lvalue >> rightShiftBits;
				
				lfBuf_pre = lfBuf_cur;			
			}
		}



    }
	
	//sz_cost_end();
	//printf("totalCost = %f\n", sz_totalCost);
	//free(leadNum);
	
}
