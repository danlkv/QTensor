#include <stdio.h>
#include <math.h>
#include "cuszxd_float.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ int _deshfl_scan(int lznum, int *sums)
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

__device__ int _compareByte(int pre, int cur, int reqBytesLength)
{
    if (reqBytesLength == 2)
    {
        if ((pre&0x0000ff00) > (cur&0x0000ff00)){
            cur &= 0x000000ff;
            cur |= (pre & 0x0000ff00);
        }
        if ((pre&0x000000ff) > (cur&0x000000ff)){
            cur &= 0x0000ff00;
            cur |= (pre & 0x000000ff);
        }
    }else if (reqBytesLength == 3)
    {
        if ((pre&0x00ff0000) > (cur&0x00ff0000)){
            cur &= 0x0000ffff;
            cur |= (pre & 0x00ff0000);
        }
        if ((pre&0x0000ff00) > (cur&0x0000ff00)){
            cur &= 0x00ff00ff;
            cur |= (pre & 0x0000ff00);
        }
        if ((pre&0x000000ff) > (cur&0x000000ff)){
            cur &= 0x00ffff00;
            cur |= (pre & 0x000000ff);
        }
    }else if (reqBytesLength == 1)
    {
        if (pre > cur)
            cur = pre;
    }else if (reqBytesLength == 4)
    {
        if ((pre&0xff000000) > (cur&0xff000000)){
            cur &= 0x00ffffff;
            cur |= (pre & 0xff000000);
        }
        if ((pre&0x00ff0000) > (cur&0x00ff0000)){
            cur &= 0xff00ffff;
            cur |= (pre & 0x00ff0000);
        }
        if ((pre&0x0000ff00) > (cur&0x0000ff00)){
            cur &= 0xffff00ff;
            cur |= (pre & 0x0000ff00);
        }
        if ((pre&0x000000ff) > (cur&0x000000ff)){
            cur &= 0xffffff00;
            cur |= (pre & 0x000000ff);
        }
    }
    return cur;
}

__device__ int _retrieve_leading(int pos, int reqBytesLength, int* sums)
{
#pragma unroll
    for (int i = 1; i <= warpSize; i *= 2) {
        unsigned int mask = 0xffffffff;
        int n = __shfl_up_sync(mask, pos, i);
        if (threadIdx.x >= i)
            pos = _compareByte(n, pos, reqBytesLength);
    }

    if (threadIdx.x == warpSize - 1)
        sums[threadIdx.y] = pos;
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x < blockDim.y) {
        int warp_pos = sums[threadIdx.x];

        int mask = (1 << blockDim.y) - 1;
        for (int i = 1; i <= blockDim.y; i *= 2) {
            int n = __shfl_up_sync(mask, warp_pos, i);
            if (threadIdx.x >= i)
                warp_pos = _compareByte(n, warp_pos, reqBytesLength);
        }

        sums[threadIdx.x] = warp_pos;
    }
    __syncthreads();

    if (threadIdx.y > 0) {
        int block_pos = sums[threadIdx.y - 1];
        pos = _compareByte(block_pos, pos, reqBytesLength);
    }

    return pos;
}

#define MAX_BLK_SIZE 256

__global__ void decompress_state2(float *out, unsigned char* stateArray, uint32_t *blk_idx, float *blk_vals, uint8_t *blk_subidx,uint32_t blockSize, uint8_t *blk_sig){
    int bid = blockIdx.x;
    uint8_t state = stateArray[bid];

    __shared__ float block_vals[MAX_BLK_SIZE];
    __shared__ uint8_t block_subidx[MAX_BLK_SIZE];
    // __shared__ char idx_taken[MAX_BLK_SIZE];
    __shared__ float s_out[MAX_BLK_SIZE];
    __shared__ int sig_count;
    if (state != 2)
    {
        return;
    }

    int local_sig = blk_sig[bid];
    int idx = blk_idx[bid];
    
    for (size_t i = threadIdx.x; i < local_sig; i+=blockDim.x)
    {
        block_vals[i] = blk_vals[idx+i];
        block_subidx[i]=blk_subidx[idx+i];
        // idx_taken[block_subidx[i]] = 1;
        atomicAdd(&sig_count, 1);
        
    }
    
    __syncthreads();
    
    for (size_t i = threadIdx.x; i < blockSize; i+=blockDim.x)
    {
        s_out[i] = 0.0;
    }

    __syncthreads();
    for (size_t i = threadIdx.x; i < local_sig; i+=blockDim.x)
    {
        s_out[block_subidx[i]] = block_vals[i];
    }
    __syncthreads();
    for (size_t i = threadIdx.x; i < blockSize; i+=blockDim.x)
    {
        out[bid*blockSize+i] = s_out[i];
    }
}

__global__ void decompress_float(unsigned char *data, int bs, size_t nc, size_t mSize) 
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tid = tidy*warpSize+tidx;
    int bid = blockIdx.x;

    float medianValue;
    unsigned char leadingNum;
    extern __shared__ float shared[];
    float* value = shared;
    int* ivalue = (int*)shared;
    uchar4* c4value = (uchar4*)shared;
    unsigned char* cvalue = (unsigned char*)shared;
    int* sums = &ivalue[bs];
    int reqLength;
    float* fbytes = (float*)data;
	int reqBytesLength;
	int rightShiftBits;


    bool bi = false;
    for (int b=bid; b<nc; b+=gridDim.x){
        bi = false;
        if (b==26192) bi=true;
        value[tid] = fbytes[b*bs+tid];
        __syncthreads();                  
        medianValue = value[0];
        reqLength = (int)cvalue[4];
        if (reqLength%8 != 0)
        {
            reqBytesLength = reqLength/8+1;		
            rightShiftBits = 8 - reqLength%8;
        }else{
            reqBytesLength = reqLength/8;		
            rightShiftBits = 0;
        }
        leadingNum = cvalue[5+(tid>>2)];
        leadingNum = (leadingNum >> (6-((tid&0x03)<<1))) & 0x03;
        int midByte_size = reqBytesLength - leadingNum;
        int midByte_sum = _deshfl_scan(midByte_size, sums);

        uchar4 tmp;
        tmp.x = 0;
        tmp.y = 0;
        tmp.z = 0;
        tmp.w = 0;
        int pos = 0;
        if (reqBytesLength == 2)
        {
            if (midByte_size == 1){
                tmp.z = cvalue[mSize+midByte_sum-1]; 
                pos |= tid<<8;
            }else if (midByte_size == 2){
                tmp.w = cvalue[mSize+midByte_sum-1]; 
                tmp.z = cvalue[mSize+midByte_sum-2];
                pos |= tid;
                pos |= tid<<8;
            }
        }else if (reqBytesLength == 3)
        {
            if (midByte_size == 1){
                tmp.y = cvalue[mSize+midByte_sum-1]; 
                pos |= tid<<16;
            }else if (midByte_size == 2){
                tmp.z = cvalue[mSize+midByte_sum-1]; 
                tmp.y = cvalue[mSize+midByte_sum-2]; 
                pos |= tid<<8;
                pos |= tid<<16;
            }else if (midByte_size == 3){
                tmp.w = cvalue[mSize+midByte_sum-1]; 
                tmp.z = cvalue[mSize+midByte_sum-2]; 
                tmp.y = cvalue[mSize+midByte_sum-3]; 
                pos |= tid;
                pos |= tid<<8;
                pos |= tid<<16;
            }
        }else if (reqBytesLength == 1)
        {
            if (midByte_size == 1)
                tmp.w = cvalue[mSize+midByte_sum-1]; 
                pos |= tid;
        }else if (reqBytesLength == 4)
        {
            if (midByte_size == 1){
                tmp.x = cvalue[mSize+midByte_sum-1]; 
                pos |= tid<<24;
            }else if (midByte_size == 2){
                tmp.y = cvalue[mSize+midByte_sum-1]; 
                tmp.x = cvalue[mSize+midByte_sum-2]; 
                pos |= tid<<16;
                pos |= tid<<24;
            }else if (midByte_size == 3){
                tmp.z = cvalue[mSize+midByte_sum-1]; 
                tmp.y = cvalue[mSize+midByte_sum-2]; 
                tmp.x = cvalue[mSize+midByte_sum-3]; 
                pos |= tid<<8;
                pos |= tid<<16;
                pos |= tid<<24;
            }else if (midByte_size == 4){
                tmp.w = cvalue[mSize+midByte_sum-1]; 
                tmp.z = cvalue[mSize+midByte_sum-2]; 
                tmp.y = cvalue[mSize+midByte_sum-3]; 
                tmp.x = cvalue[mSize+midByte_sum-4]; 
                pos |= tid;
                pos |= tid<<8;
                pos |= tid<<16;
                pos |= tid<<24;
            }
        }
        __syncthreads();                  
        c4value[tid] = tmp;

        pos = _retrieve_leading(pos, reqBytesLength, sums);

        if (leadingNum == 2){
            tmp.w = c4value[pos&0xff].w; 
            tmp.z = c4value[(pos>>8)&0xff].z;
        }else if (leadingNum == 3){
            tmp.w = c4value[pos&0xff].w; 
            tmp.z = c4value[(pos>>8)&0xff].z;
            tmp.y = c4value[(pos>>16)&0xff].y; 
        }else if (leadingNum == 1){
            tmp.w = c4value[pos&0xff].w; 
        }else if (leadingNum == 4){
            tmp.w = c4value[pos&0xff].w; 
            tmp.z = c4value[(pos>>8)&0xff].z;
            tmp.y = c4value[(pos>>16)&0xff].y; 
            tmp.x = c4value[pos>>24].x; 
        }
        c4value[tid] = tmp;
        __syncthreads();                  
        ivalue[tid] = ivalue[tid] << rightShiftBits;

        fbytes[b*bs+tid] = value[tid] + medianValue;
    }
}
