#include <stdio.h>
#include <math.h>
#include "cuszx_float.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define MAX_BLK_SIZE 256

__device__ uint32_t num_state2;
__device__ uint64_t total_sig;

__device__
void gridReduction_cg(double *results) 
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int bid = blockIdx.x;

    if (bid==0){
        double data = results[tidy*gridDim.x+tidx];

        for (int i=(tidx+blockDim.x); i<gridDim.x; i+=blockDim.x){
            if (tidy<2) data = min(data, results[tidy*gridDim.x+i]);
            else if (tidy<4) data = max(data, results[tidy*gridDim.x+i]);
            else data += results[tidy*gridDim.x+i];
        }
        __syncthreads();                  

        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
            if (tidy<2) data = min(data, __shfl_xor_sync(FULL_MASK, data, offset));
            else if (tidy<4) data = max(data, __shfl_xor_sync(FULL_MASK, data, offset));
            else data += __shfl_down_sync(FULL_MASK, data, offset);
        }

        if (tidx==0) results[tidy*gridDim.x] = data;
    }
}

__device__ void _IntArray2ByteArray(int leadingNum, int mbase, unsigned char* meta)
{
    leadingNum = leadingNum << (3-threadIdx.x%4)*2;
    for (int i = 1; i < 4; i *= 2) {
        unsigned int mask = 0xffffffff;
        leadingNum |= __shfl_down_sync(mask, leadingNum, i);
    }

    if (threadIdx.x%4==0)
        meta[mbase+threadIdx.y*8+threadIdx.x/4] = (unsigned char)leadingNum;
    __syncthreads();


}

__device__ int _compute_reqLength(int redius, int absErrBound)
{
    int radExpo = (redius & 0x7F800000) >> 23;
    radExpo -= 127;
    int reqExpo = (absErrBound & 0x7F800000) >> 23;
    reqExpo -= 127;
    return 9+radExpo-reqExpo+1;
}

__device__ int _shfl_scan(int lznum, int *sums)
{
    // Below is the basic structure of using a shfl instruction
    // for a scan.
    // Record "value" as a variable - we accumulate it along the way
    int value = lznum;

    // Now accumulate in log steps up the chain
    // compute sums, with another thread's value who is
    // distance delta away (i).  Note
    // those threads where the thread 'i' away would have
    // been out of bounds of the warp are unaffected.  This
    // creates the scan sum.

#pragma unroll
    for (int i = 1; i <= warpSize; i *= 2) {
        unsigned int mask = 0xffffffff;
        int n = __shfl_up_sync(mask, value, i);

        if (threadIdx.x >= i) value += n;
                      
    }

    // value now holds the scan value for the individual thread
    // next sum the largest values for each warp

    // write the sum of the warp to smem
    if (threadIdx.x == warpSize - 1) {
        sums[threadIdx.y] = value;
    }
    __syncthreads();

    //
    // scan sum the warp sums
    // the same shfl scan operation, but performed on warp sums
    //
    if (threadIdx.y == 0 && threadIdx.x < blockDim.y) {
        int warp_sum = sums[threadIdx.x];

        int mask = (1 << blockDim.y) - 1;
        for (int i = 1; i <= blockDim.y; i *= 2) {
            //int n = __shfl_up_sync(mask, warp_sum, i, blockDim.y);
            int n = __shfl_up_sync(mask, warp_sum, i);
            if (threadIdx.x >= i) warp_sum += n;
        }

        sums[threadIdx.x] = warp_sum;
    }
    __syncthreads();

    // perform a uniform add across warps in the block
    // read neighbouring warp's sum and add it to threads value
    int blockSum = 0;
    if (threadIdx.y > 0) {
        blockSum = sums[threadIdx.y - 1];
    }
    value += blockSum;

    return value;
}

__device__ void _compute_oneBlock(unsigned long bbase, int mbase, int obase, int reqLength, float *value, int *ivalue, uchar4 *cvalue, int *sums, unsigned char *meta, short *offsets, unsigned char *midBytes)
{
	int reqBytesLength;
	int rightShiftBits;


	if (reqLength%8 != 0)
	{
		reqBytesLength = reqLength/8+1;		
		rightShiftBits = 8 - reqLength%8;
    }else{
		reqBytesLength = reqLength/8;		
		rightShiftBits = 0;
    }

    int cur_ivalue = (ivalue[threadIdx.y*blockDim.x+threadIdx.x] >> rightShiftBits) & ((1<<(32-rightShiftBits))-1);
    ivalue[threadIdx.y*blockDim.x+threadIdx.x] = cur_ivalue;
    __syncthreads();                  

    int pre_ivalue = 0;
    if (threadIdx.x!=0 || threadIdx.y!=0) pre_ivalue = ivalue[threadIdx.y*blockDim.x+threadIdx.x-1];
    pre_ivalue = cur_ivalue ^ pre_ivalue;
    __syncthreads();                  

    int leadingNum = 0;
    if (reqBytesLength == 2)
    {
        if (pre_ivalue >> 16 == 0) leadingNum = 2;
        else if (pre_ivalue >> 24 == 0) leadingNum = 1;
    }else if (reqBytesLength == 3)
    {
        if (pre_ivalue >> 8 == 0) leadingNum = 3;
        else if (pre_ivalue >> 16 == 0) leadingNum = 2;
        else if (pre_ivalue >> 24 == 0) leadingNum = 1;
    }else if (reqBytesLength == 1)
    {
        if (pre_ivalue >> 24 == 0) leadingNum = 1;

    }else if (reqBytesLength == 4)
    {
        if (pre_ivalue == 0) leadingNum = 4;
        else if (pre_ivalue >> 8 == 0) leadingNum = 3;
        else if (pre_ivalue >> 16 == 0) leadingNum = 2;
        else if (pre_ivalue >> 24 == 0) leadingNum = 1;
    }
    //midBytes[bbase+threadIdx.y*blockDim.x+threadIdx.x] = leadingNum; 

    int midByte_size = reqBytesLength - leadingNum;
    int midByte_sum = _shfl_scan(midByte_size, sums);
    uchar4 cur_cvalue = cvalue[threadIdx.y*blockDim.x+threadIdx.x];
    if (reqBytesLength == 2)
    {
        if (midByte_size == 1){
            midBytes[bbase+midByte_sum-1] = cur_cvalue.z; 
        }else if (midByte_size == 2){
            midBytes[bbase+midByte_sum-1] = cur_cvalue.w; 
            midBytes[bbase+midByte_sum-2] = cur_cvalue.z;
        }
    }else if (reqBytesLength == 3)
    {
        if (midByte_size == 1){
            midBytes[bbase+midByte_sum-1] = cur_cvalue.y; 
        }else if (midByte_size == 2){
            midBytes[bbase+midByte_sum-1] = cur_cvalue.z; 
            midBytes[bbase+midByte_sum-2] = cur_cvalue.y; 
        }else if (midByte_size == 3){
            midBytes[bbase+midByte_sum-1] = cur_cvalue.w; 
            midBytes[bbase+midByte_sum-2] = cur_cvalue.z; 
            midBytes[bbase+midByte_sum-3] = cur_cvalue.y; 
        }
    }else if (reqBytesLength == 1)
    {
        if (midByte_size == 1)
            midBytes[bbase+midByte_sum-1] = cur_cvalue.w; 
    }else if (reqBytesLength == 4)
    {
        if (midByte_size == 1){
            midBytes[bbase+midByte_sum-1] = cur_cvalue.x; 
        }else if (midByte_size == 2){
            midBytes[bbase+midByte_sum-1] = cur_cvalue.y; 
            midBytes[bbase+midByte_sum-2] = cur_cvalue.x; 
        }else if (midByte_size == 3){
            midBytes[bbase+midByte_sum-1] = cur_cvalue.z; 
            midBytes[bbase+midByte_sum-2] = cur_cvalue.y; 
            midBytes[bbase+midByte_sum-3] = cur_cvalue.x; 
        }else if (midByte_size == 4){
            midBytes[bbase+midByte_sum-1] = cur_cvalue.w; 
            midBytes[bbase+midByte_sum-2] = cur_cvalue.z; 
            midBytes[bbase+midByte_sum-3] = cur_cvalue.y; 
            midBytes[bbase+midByte_sum-4] = cur_cvalue.x; 
        }
    }

    if (threadIdx.x==0 && threadIdx.y==0) meta[mbase] = (unsigned char)reqLength;
    if (threadIdx.x==blockDim.x-1 && threadIdx.y==blockDim.y-1) offsets[obase] = (short)midByte_sum;
    _IntArray2ByteArray(leadingNum, mbase+1, meta);

}

__global__ void apply_threshold(float *data, float threshold, size_t length){
    
    if(threadIdx.x == 0 && blockIdx.x == 0){
	printf("tid threshold: %f\n", threshold);
    }

    for (unsigned long tid = threadIdx.x+blockDim.x*blockIdx.x; tid < length; tid+=blockDim.x*gridDim.x)
    {
        if (fabs(data[tid]) <= threshold)
        {
            data[tid] = 0.0;
        }
    }
}

__global__ void compress_float(float *oriData, unsigned char *meta, short *offsets, unsigned char *midBytes, float absErrBound, int bs, size_t nb, size_t mSize, float sparsity_level, uint32_t *blk_idx, uint8_t *blk_subidx,float *blk_vals, float threshold, uint8_t *blk_sig) 
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int bid = blockIdx.x;

    float data, radius, medianValue;
    unsigned mask;
    unsigned char state;
    extern __shared__ float shared[];

    __shared__ float block_vals[MAX_BLK_SIZE];
    __shared__ uint8_t block_idxs[MAX_BLK_SIZE];
    __shared__ int num_sig;
    __shared__ int index;
    float* value = shared;
    int* ivalue = (int*)shared;
    uchar4* cvalue = (uchar4*)shared;
    int* sums = &ivalue[bs];

    //if(threadIdx.x == 0 && blockIdx.x == 0){
//	printf("tid threshold: %f\n", threshold);
  //  }

    for (unsigned long b=bid; b<nb; b+=gridDim.x){
        if (tidx ==0 && tidy ==0)
        {
            num_sig = 0;
        }
        __syncthreads();


        for (size_t i = b*bs+(tidx + blockDim.x*tidy); i < b*bs +bs; i+=blockDim.x*blockDim.y)
        {
            // fabs(data[tid]) <= threshold
            float old = oriData[i];
	    if (fabs(oriData[i]) > threshold)
            {
                int idx = atomicAdd(&num_sig, 1);
                block_vals[idx] = oriData[i];
                block_idxs[idx] = (uint8_t) (0xff & (i - (b*bs)));
            }else{
                oriData[i] = 0.0;
            }
            //if(fabs(old) > threshold && oriData[i] ==0.0){
		//printf("something wrong\n");
	    //}
        }
        __syncthreads();

        data = oriData[b*bs+tidy*warpSize+tidx];
        float Min = data;
        float Max = data;

        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
            Min = min(Min, __shfl_xor_sync(FULL_MASK, Min, offset));
            Max = max(Max, __shfl_xor_sync(FULL_MASK, Max, offset));
        }
        if (tidx==0){
            value[tidy] = Min;
            value[blockDim.y+tidy] = Max;
        }
        __syncthreads();                  

        if (tidy==0){
            if (tidx < blockDim.y){
                Min = value[tidx];
                Max = value[blockDim.y+tidx];
            }

            mask = __ballot_sync(FULL_MASK, tidx < blockDim.y);
            for (int offset = blockDim.y/2; offset > 0; offset /= 2) 
            {
                Min = min(Min, __shfl_xor_sync(mask, Min, offset));
                Max = max(Max, __shfl_xor_sync(mask, Max, offset));
            }
            
            if (tidx==0){
                radius = (Max - Min)/2;
                value[0] = radius;
                value[1] = Min + radius;
                value[2] = absErrBound;
            }
        }
        __syncthreads();                  

        radius = value[0];
        medianValue = value[1];

        if (num_sig==0)
        {
            state = 1; // All zeros
        }else if( num_sig > 0 && radius <= absErrBound){
            state = 0; // Constant block with non zeros
        } else if( ((float) num_sig/(float)bs) <= sparsity_level && num_sig > 0){
            state = 2; // Do grouping, store as-is with bitmap/index
        } else{
            state = 3; // Do normal non-constant block
        }
        

        // state = radius <= absErrBound ? 0 : 1;
        if (tidx==0){
	    
            meta[b] = state;
            meta[nb+b*mSize] = cvalue[1].x;
            meta[nb+b*mSize+1] = cvalue[1].y;
            meta[nb+b*mSize+2] = cvalue[1].z;
            meta[nb+b*mSize+3] = cvalue[1].w;
        } 
        __syncthreads();                  
        int tid = tidx + tidy*blockDim.x;
        //if(tid == 0) printf("s %d %d\n", b, (int)state);
	if (state==2)
        {
            int idx = 0;
            if (tidx ==0 && tidy == 0)
            {
		//printf("level: %f\n", ((float)num_sig/(float)bs));
                idx = atomicAdd(&num_state2, (uint32_t)num_sig);
                blk_idx[b] = idx;    // Store the index of where this block has values and indices within block
                blk_sig[b] = (uint8_t) 0xff & num_sig;
            	index = idx;
	    }
            __syncthreads();
	    idx = index;
            for (int i = tid; i < num_sig; i+=blockDim.x*blockDim.y)
            {
                blk_vals[idx+i] = block_vals[i];   // Store the value of the significant data point in the block
                blk_subidx[idx+i] = block_idxs[i]; // Store the byte value of index within block of significant data point
                //printf("blk %f %f , ind %d\n", block_vals[i], block_idxs[i], idx);
	    }
            
        }
        

        if (state==3){
            int reqLength = _compute_reqLength(ivalue[0], ivalue[2]);
            __syncthreads();                  
            value[tidy*blockDim.x+tidx] = data - medianValue;
            __syncthreads();                  
            _compute_oneBlock(b*bs*sizeof(float), nb+b*mSize+4, b, reqLength, value, ivalue, cvalue, sums, meta, offsets, midBytes);
        }

    }

}

__global__ void get_numsig(uint64_t *num_sig){
    *num_sig = (uint64_t)num_state2;
}
