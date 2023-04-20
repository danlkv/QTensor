#include "cuszx_entry.h"
#include "szx_defines.h"
#include "szx_BytesToolkit.h"
#include "szx_TypeManager.h"
#include "timingGPU.h"
#include "szx.h"
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <cub/cub.cuh>

#define SPARSITY_LEVEL 0.25
#define BLOCKS 40
#define THREADS_PER_BLOCK 256

TimingGPU timer_GPU;
void bin(unsigned n)
{
    unsigned i;
    for (i = 1 << 31; i > 0; i = i / 2)
        (n & i) ? printf("1") : printf("0");
}

__host__ __device__ size_t convert_state_to_out(unsigned char* meta, size_t length, unsigned char *result){
    size_t out_length;

    if(length%4==0)
		out_length = length/4;
	else
		out_length = length/4+1;

    for (size_t i = 0; i < out_length; i++)
    {
        uint8_t tmp = 0;

        for (size_t j = 0; j < 4; j++)
        {
            if (i*4 + j < length)
            {
                tmp |= (0x03 & meta[i*4+j]) << 2*j;
            }
            
        }
        result[i] = tmp;
    }
    return out_length;
}

__global__ void convert_state_to_out_kernel(unsigned char* meta, size_t length, unsigned char *result, size_t out_length){
    

    for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < out_length; i += blockDim.x*gridDim.x){
        uint8_t tmp = 0;

        for (size_t j = 0; j < 4; j++)
        {
            if (i*4 + j < length)
            {
                tmp |= (0x03 & meta[i*4+j]) << 2*j;
            }
            
        }
        result[i] = tmp;
    }
}

__global__ void convert_out_to_state_kernel(size_t nbBlocks, unsigned char* cmp, unsigned char* out_state, size_t state_length, int *num_state2blks, int *ncBlocks){
    for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < state_length; i += blockDim.x*gridDim.x){
        for (size_t j = 0; j < 4; j++)
        {
            if (4*i + j < nbBlocks)
            {
                out_state[4*i + j]= (cmp[i] >> 2*j) & 0x03;
                if (out_state[4*i+j] == 2)
                {
                    atomicAdd(num_state2blks, 1);
                }else if(out_state[4*i+j]==3){
                    atomicAdd(ncBlocks, 1);
                }
                
            }
            
        }
    }
}

// nbBlocks, r, stateNBBytes, stateArray
__host__ __device__ size_t convert_out_to_state(size_t nbBlocks, unsigned char* cmp, unsigned char* out_state){
    size_t state_length;
    if(nbBlocks%4==0)
		state_length = nbBlocks/4;
	else
		state_length = nbBlocks/4+1;

    for (size_t i = 0; i < state_length; i++)
    {
        for (size_t j = 0; j < 4; j++)
        {
            if (4*i + j < nbBlocks)
            {
                out_state[4*i + j]= (cmp[i] >> 2*j) & 0x03;
            }
            
        }
    }
    return nbBlocks;
}

__host__ __device__ size_t convert_block2_to_out(unsigned char *result, uint32_t numBlocks, uint64_t num_sig, uint32_t *blk_idx, float *blk_vals, uint8_t *blk_subidx, uint8_t *blk_sig){
    size_t out_length = 0;
    
    memcpy(result, blk_idx, numBlocks*sizeof(uint32_t));
    out_length += numBlocks*4;
    memcpy(result+out_length, blk_vals, num_sig*sizeof(float));
    out_length += num_sig*sizeof(float);
    memcpy(result+out_length, blk_subidx, num_sig*sizeof(uint8_t));
    out_length += num_sig*sizeof(uint8_t);
    memcpy(result+out_length, blk_sig, numBlocks*sizeof(uint8_t));
    out_length+= numBlocks*sizeof(uint8_t);

    return out_length;
}

__global__ void convert_block2_to_out_kernel(unsigned char *result, uint32_t numBlocks, uint64_t num_sig, uint32_t *blk_idx, float *blk_vals, uint8_t *blk_subidx, uint8_t *blk_sig){
    
    size_t out_length = 0;
    unsigned char *tmp_result = result;
    for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < numBlocks; i += blockDim.x*gridDim.x){
        uint32_t local_blkidx = blk_idx[i];
        tmp_result[4*i] = (local_blkidx) & 0xff;
        tmp_result[4*i+1] = (local_blkidx >> (8*1)) & 0xff;
        tmp_result[4*i+2] = (local_blkidx >> (8*2)) & 0xff;
        tmp_result[4*i+3] = (local_blkidx >> (8*3)) & 0xff;
    }
    // memcpy(result, blk_idx, numBlocks*sizeof(uint32_t));
    out_length += numBlocks*4;
    tmp_result = result+out_length;
    
    for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < num_sig; i += blockDim.x*gridDim.x){
        float value = blk_vals[i];
	    memcpy(&tmp_result[4*i], &value, sizeof(float));
	//unsigned char *v = ()
        //tmp_result[(int)4*i] = (unsigned char)((value) & 0xff);
        //tmp_result[(int)4*i+1] = (unsigned char)((value >> (8*1)) & 0xff);
        //tmp_result[(int)4*i+2] = (unsigned char)((value >> (8*2)) & 0xff);
        //tmp_result[(int)4*i+3] = (unsigned char)((value >> (8*3)) & 0xff);
    }
    // memcpy(result+out_length, blk_vals, num_sig*sizeof(float));
    out_length += num_sig*sizeof(float);
    tmp_result = result+out_length;
    
    for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < num_sig; i += blockDim.x*gridDim.x){
        tmp_result[i] = blk_subidx[i];
        
    }

    out_length += num_sig*sizeof(uint8_t);
    tmp_result = result+out_length;
    
    for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < numBlocks; i += blockDim.x*gridDim.x){
        tmp_result[i] = blk_sig[i];
        
    }
    out_length+= numBlocks*sizeof(uint8_t);

    // return out_length;
}

__global__ void convert_out_to_block2_kernel(unsigned char *in_cmp, uint32_t numBlocks, uint64_t num_sig, uint32_t *blk_idx, float *blk_vals, uint8_t *blk_subidx, uint8_t *blk_sig){
    size_t out_length = 0;
    
    unsigned char *tmp_result = in_cmp;
    for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < numBlocks; i += blockDim.x*gridDim.x){
        
        uint32_t local_blkidx = (tmp_result[4*i] & 0xff) | ((tmp_result[4*i+1] & 0xff) << (8*1)) 
                                | ((tmp_result[4*i+2] & 0xff) << (8*2)) | ((tmp_result[4*i+3] & 0xff) << (8*3));
        blk_idx[i] = local_blkidx;
    }
    // memcpy(result, blk_idx, numBlocks*sizeof(uint32_t));
    out_length += numBlocks*4;
    tmp_result = in_cmp+out_length;
    
    for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < num_sig; i += blockDim.x*gridDim.x){
        float value = 0.0;
        memcpy(&value, &tmp_result[4*i], sizeof(float));
        blk_vals[i] = value;
	    
	//unsigned char *v = ()
        //tmp_result[(int)4*i] = (unsigned char)((value) & 0xff);
        //tmp_result[(int)4*i+1] = (unsigned char)((value >> (8*1)) & 0xff);
        //tmp_result[(int)4*i+2] = (unsigned char)((value >> (8*2)) & 0xff);
        //tmp_result[(int)4*i+3] = (unsigned char)((value >> (8*3)) & 0xff);
    }
    // memcpy(result+out_length, blk_vals, num_sig*sizeof(float));
    out_length += num_sig*sizeof(float);
    tmp_result = in_cmp+out_length;
    
    for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < num_sig; i += blockDim.x*gridDim.x){
        blk_subidx[i] = tmp_result[i];
        
    }

    out_length += num_sig*sizeof(uint8_t);
    tmp_result = in_cmp+out_length;
    
    for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < numBlocks; i += blockDim.x*gridDim.x){
        blk_sig[i] = tmp_result[i];
        
    }
    out_length+= numBlocks*sizeof(uint8_t);
}

__host__ __device__ size_t convert_out_to_block2(unsigned char *in_cmp, uint32_t numBlocks, uint64_t num_sig, uint32_t *blk_idx, float *blk_vals, uint8_t *blk_subidx, uint8_t *blk_sig){
    size_t out_length = 0;
    memcpy(blk_idx, in_cmp, numBlocks*sizeof(uint32_t));
    out_length += numBlocks*4;
    memcpy(blk_vals, in_cmp+out_length,num_sig*sizeof(float));
    out_length += num_sig*sizeof(float);
    memcpy(blk_subidx, in_cmp+out_length, num_sig*sizeof(uint8_t));
    out_length += num_sig*sizeof(uint8_t);
    memcpy(blk_sig, in_cmp+out_length, numBlocks*sizeof(uint8_t));
    out_length += numBlocks*sizeof(uint8_t);
//    printf("outlength: %d\n",out_length);
    return out_length;
}

int _post_proc(float *oriData, unsigned char *meta, short *offsets, unsigned char *midBytes, unsigned char *outBytes, size_t nbEle, int blockSize, uint64_t num_sig, uint32_t *blk_idx, float *blk_vals, uint8_t *blk_subidx, uint8_t *blk_sig)
{
    int out_size = 0;

    size_t nbConstantBlocks = 0;
    size_t nbBlocks = nbEle/blockSize;
    size_t ncBytes = blockSize/4;
    size_t mSize = sizeof(float)+1+ncBytes; //Number of bytes for each data block's metadata.
    out_size += 5+sizeof(size_t)+sizeof(float)*nbBlocks;
    if (nbBlocks%8==0)
        out_size += nbBlocks/8;
    else
        out_size += nbBlocks/8+1;
    int s0 = 0;
    int s1 = 0;
    int s2 = 0;
    int s3 = 0;
    for (int i=0; i<nbBlocks; i++){
        if (meta[i]==0 || meta[i]==1 || meta[i] == 2) nbConstantBlocks++;
        else out_size += 1+(blockSize/4)+offsets[i];
    
    	if(meta[i]==0) s0++;
    	if(meta[i]==1) s1++;
    	if(meta[i]==2) s2++;
    	if(meta[i]==3) s3++;
    }
//    printf("%d %d %d %d\n", s0, s1, s2, s3);
    out_size += (nbBlocks-nbConstantBlocks)*sizeof(short)+(nbEle%blockSize)*sizeof(float);

    //outBytes = (unsigned char*)malloc(out_size);
  //  printf("accessing outbytes now...\n");
	unsigned char* r = outBytes;
    unsigned char* r_old = outBytes; 
	r[0] = SZx_VER_MAJOR;
	r[1] = SZx_VER_MINOR;
	r[2] = 1;
	r[3] = 0; // indicates this is not a random access version
	r[4] = (unsigned char)blockSize;
	r=r+5; //1 byte
	sizeToBytes(r, nbConstantBlocks);
	r += sizeof(size_t);
    sizeToBytes(r, (size_t) num_sig);
    r += sizeof(size_t); 
	r += convert_state_to_out(meta, nbBlocks, r);
    r += convert_block2_to_out(r, nbBlocks,num_sig, blk_idx, blk_vals, blk_subidx, blk_sig);
    memcpy(r, oriData+nbBlocks*blockSize, (nbEle%blockSize)*sizeof(float));
    r += (nbEle%blockSize)*sizeof(float);
    unsigned char* c = r;
    unsigned char* o = c+nbConstantBlocks*sizeof(float);
    unsigned char* nc = o+(nbBlocks-nbConstantBlocks)*sizeof(short);
    for (int i=0; i<nbBlocks; i++){
        
        if (meta[i]==0 || meta[i] == 1){
	    memcpy(c, meta+(nbBlocks+i*mSize), sizeof(float));
            c += sizeof(float);
        }else if(meta[i] == 3){
            shortToBytes(o, offsets[i]);
	   
            o += sizeof(short);
            memcpy(nc, meta+(nbBlocks+i*mSize), mSize);
            
	    nc += mSize; 
            memcpy(nc, midBytes+(i*blockSize*sizeof(float)), offsets[i]);
            
	    nc += offsets[i];
	   
        } 
    }

    // return out_size;
    return (uint32_t) (nc-r_old);
}

unsigned char* cuSZx_fast_compress_args_unpredictable_blocked_float(float *oriData, size_t *outSize, float absErrBound, size_t nbEle, int blockSize, float threshold)
{
//    printf("tr thresh abs %f %f\n", threshold, absErrBound);
  //  printf("first: %f %f %f\n", oriData[0], oriData[1], oriData[2]);
    float sparsity_level = SPARSITY_LEVEL;
	float* d_oriData;
    cudaMalloc((void**)&d_oriData, sizeof(float)*nbEle); 
    cudaMemcpy(d_oriData, oriData, sizeof(float)*nbEle, cudaMemcpyHostToDevice); 

	size_t nbBlocks = nbEle/blockSize;
	size_t remainCount = nbEle%blockSize;
	size_t actualNBBlocks = remainCount==0 ? nbBlocks : nbBlocks+1;

    size_t ncBytes = blockSize/4;
    //ncBytes = (blockSize+1)%4==0 ? ncBytes : ncBytes+1; //Bytes to store one non-constant block data.
    size_t mSize = sizeof(float)+1+ncBytes; //Number of bytes for each data block's metadata.
    size_t msz = (1+mSize) * nbBlocks * sizeof(unsigned char);
    size_t mbsz = sizeof(float) * nbEle * sizeof(unsigned char);

    unsigned char *meta = (unsigned char*)malloc(msz);
    short *offsets = (short*)malloc(nbBlocks*sizeof(short));
    unsigned char *midBytes = (unsigned char*)malloc(mbsz);

	unsigned char* d_meta;
	unsigned char* d_midBytes;
	short* d_offsets;

    uint32_t *blk_idx, *d_blk_idx;
    uint8_t *blk_sig, *d_blk_sig;
    uint8_t *blk_subidx, *d_blk_subidx;
    float *blk_vals, *d_blk_vals;
    uint64_t *num_sig, *d_num_sig;

    checkCudaErrors(cudaMalloc((void **)&d_num_sig, sizeof(uint64_t)));
    num_sig = (uint64_t *)malloc(sizeof(uint64_t));
    checkCudaErrors(cudaMalloc((void **)&d_blk_idx, nbBlocks*sizeof(uint32_t)));
    // blk_idx = malloc()
    checkCudaErrors(cudaMalloc((void **)&d_blk_subidx, nbEle*sizeof(uint8_t)));

    checkCudaErrors(cudaMalloc((void **)&d_blk_vals, nbEle*sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&d_blk_sig, nbBlocks*sizeof(uint8_t)));

    checkCudaErrors(cudaMalloc((void**)&d_meta, msz)); 
    //checkCudaErrors(cudaMemcpy(d_meta, meta, msz, cudaMemcpyHostToDevice)); 
    checkCudaErrors(cudaMemset(d_meta, 0, msz));
    checkCudaErrors(cudaMalloc((void**)&d_offsets, nbBlocks*sizeof(short))); 
    checkCudaErrors(cudaMemset(d_offsets, 0, nbBlocks*sizeof(short)));
    checkCudaErrors(cudaMalloc((void**)&d_midBytes, mbsz)); 
    checkCudaErrors(cudaMemset(d_midBytes, 0, mbsz));

    timer_GPU.StartCounter();
    // apply_threshold<<<80,256>>>(d_oriData, threshold, nbEle);
    // cudaDeviceSynchronize();
    dim3 dimBlock(32, blockSize/32);
    dim3 dimGrid(65536, 1);
    const int sMemsize = blockSize * sizeof(float) + dimBlock.y * sizeof(int);
    compress_float<<<dimGrid, dimBlock, sMemsize>>>(d_oriData, d_meta, d_offsets, d_midBytes, absErrBound, blockSize, nbBlocks, mSize, sparsity_level, d_blk_idx, d_blk_subidx,d_blk_vals, threshold, d_blk_sig);
    cudaError_t err = cudaGetLastError();        // Get error code
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    printf("GPU compression timing: %f ms\n", timer_GPU.GetCounter());
    cudaDeviceSynchronize();
    get_numsig<<<1,1>>>(d_num_sig);
    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(num_sig, d_num_sig, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    blk_idx = (uint32_t *)malloc(nbBlocks*sizeof(uint32_t));
    blk_vals= (float *)malloc((*num_sig)*sizeof(float));
    blk_subidx = (uint8_t *)malloc((*num_sig)*sizeof(uint8_t));
    blk_sig = (uint8_t *)malloc(nbBlocks*sizeof(uint8_t));

    checkCudaErrors(cudaMemcpy(meta, d_meta, msz, cudaMemcpyDeviceToHost)); 
    checkCudaErrors(cudaMemcpy(offsets, d_offsets, nbBlocks*sizeof(short), cudaMemcpyDeviceToHost)); 
    checkCudaErrors(cudaMemcpy(midBytes, d_midBytes, mbsz, cudaMemcpyDeviceToHost)); 
    
    
    checkCudaErrors(cudaMemcpy(blk_idx, d_blk_idx, nbBlocks*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(blk_vals,d_blk_vals, (*num_sig)*sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(blk_subidx,d_blk_subidx, (*num_sig)*sizeof(uint8_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(blk_sig,d_blk_sig, (nbBlocks)*sizeof(uint8_t), cudaMemcpyDeviceToHost));

    size_t maxPreservedBufferSize = sizeof(float)*nbEle;
    unsigned char* outBytes = (unsigned char*)malloc(maxPreservedBufferSize);
    memset(outBytes, 0, maxPreservedBufferSize);

    outSize = (size_t *)malloc(sizeof(size_t));
    //outSize[0] = _post_proc(oriData, meta, offsets, midBytes, outBytes, nbEle, blockSize, *num_sig, blk_idx, blk_vals, blk_subidx, blk_sig);

    *outSize = _post_proc(oriData, meta, offsets, midBytes, outBytes, nbEle, blockSize, *num_sig, blk_idx, blk_vals, blk_subidx, blk_sig);
//    printf("Beginning free\n");
    // printf("outsize %p \n", outBytes);
    free(blk_idx);
    free(blk_subidx);
    free(blk_vals);
    free(meta);
    free(offsets);
    free(midBytes);
    checkCudaErrors(cudaFree(d_meta));
    checkCudaErrors(cudaFree(d_offsets));
    checkCudaErrors(cudaFree(d_midBytes));
    return outBytes;
}

void cuSZx_fast_decompress_args_unpredictable_blocked_float(float** newData, size_t nbEle, unsigned char* cmpBytes)
{
    uint32_t *blk_idx, *d_blk_idx;
    uint8_t *blk_subidx, *d_blk_subidx;
    uint8_t *blk_sig, *d_blk_sig;
    float *blk_vals, *d_blk_vals;
    size_t num_sig, *d_num_sig;

	*newData = (float*)malloc(sizeof(float)*nbEle);
    memset(*newData, 0, sizeof(float)*nbEle);
	
	unsigned char* r = cmpBytes;
	r += 4;
	int blockSize = r[0];  //get block size
	if(blockSize == 0)blockSize = 256;
	r++;
	size_t nbConstantBlocks = bytesToLong_bigEndian(r); //get number of constant blocks
	r += sizeof(size_t);
	num_sig = bytesToSize(r);
    r += sizeof(size_t);
	size_t nbBlocks = nbEle/blockSize;
    size_t ncBlocks = 0;
    size_t num_state2_blks = 0;
	// size_t ncBlocks = nbBlocks - nbConstantBlocks; //get number of constant blocks
	size_t stateNBBytes = nbBlocks%4==0 ? nbBlocks/4 : nbBlocks/4+1;
    size_t ncLeading = blockSize/4;
    size_t mSize = sizeof(float)+1+ncLeading; //Number of bytes for each data block's metadata.
	unsigned char* stateArray = (unsigned char*)malloc(nbBlocks);
    unsigned char* d_stateArray;
    cudaMalloc(&d_stateArray, nbBlocks);
	float* constantMedianArray = (float*)malloc(nbConstantBlocks*sizeof(float));			
	
    

    blk_idx = (uint32_t *)malloc(nbBlocks*sizeof(uint32_t));
    blk_vals= (float *)malloc((num_sig)*sizeof(float));
    blk_subidx = (uint8_t *)malloc((num_sig)*sizeof(uint8_t));
    blk_sig = (uint8_t *)malloc(nbBlocks*sizeof(uint8_t));

	// printf("Converting state array\n");
    convert_out_to_state(nbBlocks, r, stateArray);
	// convertByteArray2IntArray_fast_1b_args(nbBlocks, r, stateNBBytes, stateArray); //get the stateArray
	for (size_t i = 0; i < nbBlocks; i++)
    {
        if (stateArray[i] == 2)
        {
            num_state2_blks++;
        }else if(stateArray[i] == 3){
            ncBlocks++;
        }
    }
    
	r += stateNBBytes;
    unsigned char* data = (unsigned char*)malloc(ncBlocks*blockSize*sizeof(float));
    memset(data, 0, ncBlocks*blockSize*sizeof(float));
    // printf("converting block vals\n");
    size_t to_add = convert_out_to_block2(r, nbBlocks, (uint64_t)num_sig, blk_idx, blk_vals, blk_subidx, blk_sig);
    r+= to_add;
    // checkCudaErrors(cudaMalloc((void **)&d_num_sig, sizeof(uint64_t)));
    // num_sig = (uint64_t *)malloc(sizeof(uint64_t));
    checkCudaErrors(cudaMalloc((void **)&d_blk_idx, nbBlocks*sizeof(uint32_t)));
    // blk_idx = malloc()
    checkCudaErrors(cudaMalloc((void **)&d_blk_subidx, num_sig*sizeof(uint8_t)));

    checkCudaErrors(cudaMalloc((void **)&d_blk_vals, num_sig*sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&d_blk_sig, nbBlocks*sizeof(uint8_t)));

    checkCudaErrors(cudaMemcpy(d_blk_idx, blk_idx, nbBlocks*sizeof(uint32_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_blk_vals, blk_vals, (num_sig)*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_blk_subidx, blk_subidx, (num_sig)*sizeof(uint8_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_stateArray, stateArray, nbBlocks, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_blk_sig, blk_sig, nbBlocks*sizeof(uint8_t), cudaMemcpyHostToDevice));


	size_t i = 0, j = 0, k = 0; //k is used to keep track of constant block index
    memcpy((*newData)+nbBlocks*blockSize, r, (nbEle%blockSize)*sizeof(float));
    r += (nbEle%blockSize)*sizeof(float);
	float* fr = (float*)r; //fr is the starting address of constant median values.
	for(i = 0;i < nbConstantBlocks;i++, j+=4) //get the median values for constant-value blocks
		constantMedianArray[i] = fr[i];
    r += nbConstantBlocks*sizeof(float);
    unsigned char* p = r + ncBlocks * sizeof(short);
    for(i = 0;i < ncBlocks;i++){
        int leng = (int)bytesToShort(r)+mSize;
        r += sizeof(short);
        if (leng > blockSize*sizeof(float))
        {
            printf("Warning: compressed block is larger than the original block!\n");
            exit(0);
        }
        memcpy(data+i*blockSize*sizeof(float), p, leng);
        p += leng;
    } 

    unsigned char* d_data;
    float *d_newdata;
    checkCudaErrors(cudaMalloc((void**)&d_data, ncBlocks*blockSize*sizeof(float))); 
    checkCudaErrors(cudaMemcpy(d_data, data, ncBlocks*blockSize*sizeof(float), cudaMemcpyHostToDevice)); 
    checkCudaErrors(cudaMalloc(&d_newdata, nbBlocks*blockSize*sizeof(float)));

    timer_GPU.StartCounter();
    dim3 dimBlock(32, blockSize/32);
    dim3 dimGrid(65536, 1);
    const int sMemsize = blockSize * sizeof(float) + dimBlock.y * sizeof(int);
    decompress_state2<<<nbBlocks, 64>>>(d_newdata, d_stateArray,d_blk_idx, d_blk_vals, d_blk_subidx,blockSize, d_blk_sig);
    decompress_float<<<dimGrid, dimBlock, sMemsize>>>(d_data, blockSize, ncBlocks, mSize);
    cudaError_t err = cudaGetLastError();        // Get error code
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    printf("GPU decompression timing: %f ms\n", timer_GPU.GetCounter());
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(data, d_data, ncBlocks*blockSize*sizeof(float), cudaMemcpyDeviceToHost)); 
    checkCudaErrors(cudaMemcpy(*newData, d_newdata, nbBlocks*blockSize*sizeof(float), cudaMemcpyDeviceToHost));
    float* fdata = (float*)data;

    int nb=0, nc=0;
    for (i=0;i<nbBlocks;i++){
        if (stateArray[i]==0 || stateArray[i]==1){
            float Median = constantMedianArray[nb];
            if (Median>1) printf("data%i:%f\n",i, Median);
            for (j=0;j<blockSize;j++)
                *((*newData)+i*blockSize+j) = Median;
            nb++;
        }else if(stateArray[i]==3){
            for (j=0;j<blockSize;j++)
                *((*newData)+i*blockSize+j) = fdata[nc*blockSize+j];
            nc++;
        }
    }

	free(stateArray);
	free(constantMedianArray);
	free(data);
    cudaFree(d_newdata);
    cudaFree(d_stateArray);
    checkCudaErrors(cudaFree(d_data));

}

__device__ inline void longToBytes_bigEndian_d(unsigned char *b, unsigned long num) 
{
	b[0] = (unsigned char)(num>>56);
	b[1] = (unsigned char)(num>>48);
	b[2] = (unsigned char)(num>>40);
	b[3] = (unsigned char)(num>>32);
	b[4] = (unsigned char)(num>>24);
	b[5] = (unsigned char)(num>>16);
	b[6] = (unsigned char)(num>>8);
	b[7] = (unsigned char)(num);
//	if(dataEndianType==LITTLE_ENDIAN_DATA)
//		symTransform_8bytes(*b);
}

inline void longToBytes_bigEndian_memset(unsigned char *b, unsigned long num) 
{
    checkCudaErrors(cudaMemset(&b[0], (unsigned char)(num>>56), sizeof(char)));
    checkCudaErrors(cudaMemset(&b[1], (unsigned char)(num>>48), sizeof(char)));
    checkCudaErrors(cudaMemset(&b[2], (unsigned char)(num>>40), sizeof(char)));
    checkCudaErrors(cudaMemset(&b[3], (unsigned char)(num>>32), sizeof(char)));
    checkCudaErrors(cudaMemset(&b[4], (unsigned char)(num>>24), sizeof(char)));
    checkCudaErrors(cudaMemset(&b[5], (unsigned char)(num>>16), sizeof(char)));
    checkCudaErrors(cudaMemset(&b[6], (unsigned char)(num>>8), sizeof(char)));
    checkCudaErrors(cudaMemset(&b[7], (unsigned char)(num), sizeof(char)));
//	if(dataEndianType==LITTLE_ENDIAN_DATA)
//		symTransform_8bytes(*b);
}

__device__ inline void shortToBytes_d(unsigned char* b, short value)
{
	lint16 buf;
	buf.svalue = value;
	memcpy(b, buf.byte, 2);
}



__global__ void getNumNonConstantBlocks(size_t nbBlocks, short *offsets, unsigned char *meta, int blockSize, int *nonconstant, int *out_size){
    for (int tid = blockDim.x*blockIdx.x + threadIdx.x; tid < nbBlocks; tid += blockDim.x*gridDim.x){
        if (meta[tid] == 3){ 
            atomicAdd(nonconstant, 1);
            atomicAdd(out_size,1+(blockSize/4)+offsets[tid]);
        }
    }
}

__global__ void generateFlags(unsigned char *states, uint64_t *cBlk_flags, uint64_t *ncBlk_flags,uint64_t* offset_indices,short* offsets, size_t nbBlocks){
    for (int tid = blockDim.x*blockIdx.x + threadIdx.x; tid < nbBlocks; tid += blockDim.x*gridDim.x){
        if (states[tid] == 0 || states[tid] == 1)
        {
            cBlk_flags[tid] = 1;
            ncBlk_flags[tid] = 0;
            offset_indices[tid] = 0;
        }else if(states[tid]==3){
            ncBlk_flags[tid] = 1;
            cBlk_flags[tid] = 0;
            offset_indices[tid] = (uint64_t) offsets[tid];
        }else{
            cBlk_flags[tid] = 0;
            ncBlk_flags[tid] = 0;
            offset_indices[tid] = 0;
        }
        
    }
}

__global__ void nccopy_kernel2(unsigned char * c, unsigned char* o, unsigned char *nc, unsigned char* midBytes, unsigned char* meta,
                        size_t nbBlocks, int blockSize, short *offsets, size_t mSize, uint64_t *cBlk_indices, uint64_t *ncBlk_indices, uint64_t* offset_indices){
   // printf("blockdim %d blockidx %d threadidx %d griddim %d\n", blockDim.x, blockIdx.x, threadIdx.x, gridDim.x);
    int i;
    int num_threads = (blockDim.x*gridDim.x);
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    int blocks_per_thread = nbBlocks/num_threads;
    int start_idx = tid*blocks_per_thread;
    int end_idx = start_idx+blocks_per_thread;

    if (tid == num_threads-1)
    {
        end_idx = nbBlocks;
    }
    
    unsigned char* tmp_o = o+(sizeof(short)*ncBlk_indices[start_idx]);
    unsigned char* tmp_nc= nc+(mSize*ncBlk_indices[i] + offset_indices[i]*ncBlk_indices[i]);
    for (i=start_idx; i<end_idx; i++){
        if(meta[i] == 3){
	
            
            shortToBytes_d(o, offsets[i]);
            tmp_o += sizeof(short);
            memcpy(tmp_nc, meta+(nbBlocks+i*mSize), mSize);
            tmp_nc += mSize; 
            memcpy(tmp_nc, midBytes+(i*blockSize*sizeof(float)), offsets[i]);
            tmp_nc += offsets[i];

            // shortToBytes_d(o+(sizeof(short)*ncBlk_indices[i]), offsets[i]);
            
            // memcpy(nc+(mSize*ncBlk_indices[i] + offset_indices[i]*ncBlk_indices[i]), meta+(nbBlocks+i*mSize), mSize);


            // memcpy(nc+(mSize*(ncBlk_indices[i]+1) + offset_indices[i]*ncBlk_indices[i]), midBytes+(i*blockSize*sizeof(float)), offsets[i]);
        } 
    }
    
}


__global__ void nccopy_kernel(unsigned char * c, unsigned char* o, unsigned char *nc, unsigned char* midBytes, unsigned char* meta,
                        size_t nbBlocks, int blockSize, short *offsets, size_t mSize, uint64_t *cBlk_indices, uint64_t *ncBlk_indices, uint64_t* offset_indices, size_t *final_nc){
   // printf("blockdim %d blockidx %d threadidx %d griddim %d\n", blockDim.x, blockIdx.x, threadIdx.x, gridDim.x);
    int i;
    // if(threadIdx.x==0){
	// printf("c: %ld nc: %ld\n", cBlk_indices[nbBlocks-1], ncBlk_indices[nbBlocks-1]);
    // }
    for (i=blockDim.x*blockIdx.x + threadIdx.x; i<nbBlocks; i+=blockDim.x*gridDim.x){
        //printf("meta %d i: %d\n",meta[i], i); 
        if (meta[i]==0 || meta[i] == 1){
            // printf("cblk\n");
	        memcpy(c+(sizeof(float)*cBlk_indices[i]), meta+(nbBlocks+i*mSize), sizeof(float));
	   
            // printf("cblk done\n");
	    // c += sizeof(float);
	    // float g;
	    // memcpy(&g, (meta+(nbBlocks+i*mSize)),sizeof(float));
	    // printf("%d %f\n",i,g);
        }
        else if(meta[i] == 3){
	
        //     printf("ncblk 1\n");
            shortToBytes_d(o+(sizeof(short)*ncBlk_indices[i]), offsets[i]);
             // o += sizeof(short);

        //     printf("ncblk 2 nbBlocks %d %d \n", nbBlocks, i);
            // printf("nbBlkindices %ld offset_indices %ld\n", ncBlk_indices[i], offset_indices[i]);
        //     printf(" test 1%c\n",meta+(nbBlocks+i*mSize));
        //     printf("test 2%c\n", nc+(mSize*ncBlk_indices[i] + offset_indices[i]*ncBlk_indices[i]));
            memcpy(nc+((mSize*ncBlk_indices[i] + offset_indices[i])), meta+(nbBlocks+i*mSize), mSize);
        //         // nc += mSize; 
                
        //     printf("ncblk 3\n");
            memcpy(nc+(((mSize*ncBlk_indices[i])+mSize + offset_indices[i])), midBytes+(i*blockSize*sizeof(float)), offsets[i]);
        //         // nc += offsets[i];
            
        //     printf("ncblk 4\n");
        }
        if (i==nbBlocks-1)
        {
            *final_nc = (size_t) (((mSize*ncBlk_indices[i])+mSize + offset_indices[i]))+offsets[i];
	}
        
    }
    
}

//__global__ void nccopy_kernel(unsigned char * c, unsigned char* o, unsigned char *nc, unsigned char* midBytes, unsigned char* meta,
//                        size_t nbBlocks, int blockSize, short *offsets, size_t mSize, int *cBlk_indices, int *ncBlk_indices, int* offset_indices){
//    printf("blockdim %d blockidx %d threadidx %d griddim %d\n", blockDim.x, blockIdx.x, threadIdx.x, gridDim.x);
//    int i;
//    for (i=blockDim.x*blockIdx.x + threadIdx.x; i<nbBlocks; i+=blockDim.x*gridDim.x){
        //printf("meta %d i: %d\n",meta[i], i); 
//        if (meta[i]==0 || meta[i] == 1){
            // printf("cblk\n");
//	    memcpy(c+(sizeof(float)*cBlk_indices[i]), meta+(nbBlocks+i*mSize), sizeof(float));

            // printf("cblk done\n");
	    // c += sizeof(float);
	    // float g;
	    // memcpy(&g, (meta+(nbBlocks+i*mSize)),sizeof(float));
	    // printf("%d %f\n",i,g);
//        }else if(meta[i] == 3){
	
//           printf("ncblk 1\n");
//           shortToBytes_d(o+(sizeof(short)*ncBlk_indices[i]), offsets[i]);
            // o += sizeof(short);

//           printf("ncblk 2 nbBlocks %d %d \n", nbBlocks, i);
//	   printf("nbBlkindices %d offset_indices %d\n", ncBlk_indices[i], offset_indices[i]);
//	   memcpy(nc+(mSize*ncBlk_indices[i] + offset_indices[i]*ncBlk_indices[i]), meta+(nbBlocks+i*mSize), mSize);
            // nc += mSize; 
            
//           printf("ncblk 3\n");
//	   memcpy(nc+(mSize*(ncBlk_indices[i]+1) + offset_indices[i]*ncBlk_indices[i]), midBytes+(i*blockSize*sizeof(float)), offsets[i]);
            // nc += offsets[i];
        
//           printf("ncblk 4\n");
//	} 
//    }
    
//}

__global__ void set_nc(unsigned char *nc, short *offsets, uint64_t *offset_indices, uint64_t *ncBlk_indices, size_t mSize, size_t nbBlocks){
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        nc = nc + (mSize*(ncBlk_indices[nbBlocks -1]+1) + offset_indices[nbBlocks - 1]*ncBlk_indices[nbBlocks - 1]) + offsets[nbBlocks-1];
    }
    
}

void ncblkCopy_fast(unsigned char * c, unsigned char* o, unsigned char *nc, unsigned char* midBytes, unsigned char* meta,
                        size_t nbBlocks, int blockSize, short *offsets, size_t mSize, size_t *final_nc){
    uint64_t *cBlk_indices, *ncBlk_indices;
    uint64_t *offset_indices;
    TimingGPU timer2;

    // timer2.StartCounter();
    
    checkCudaErrors(cudaMalloc(&cBlk_indices, sizeof(uint64_t)*nbBlocks));
    checkCudaErrors(cudaMalloc(&ncBlk_indices, sizeof(uint64_t)*nbBlocks));
    checkCudaErrors(cudaMalloc(&offset_indices, sizeof(uint64_t)*nbBlocks));

    generateFlags<<<BLOCKS,THREADS_PER_BLOCK>>>(meta, cBlk_indices, ncBlk_indices, offset_indices, offsets, nbBlocks);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, cBlk_indices, cBlk_indices + nbBlocks, cBlk_indices, 0);
    thrust::exclusive_scan(thrust::device, ncBlk_indices, ncBlk_indices + nbBlocks, ncBlk_indices, 0);
    thrust::exclusive_scan(thrust::device, offset_indices, offset_indices + nbBlocks, offset_indices, 0);

    nccopy_kernel<<<BLOCKS,THREADS_PER_BLOCK>>>(c, o, nc, midBytes, meta, nbBlocks, blockSize, offsets, mSize, cBlk_indices,ncBlk_indices,offset_indices,final_nc);
    // nccopy_kernel2<<<1,1>>>(c, o, nc, midBytes, meta, nbBlocks, blockSize, offsets, mSize, cBlk_indices,ncBlk_indices,offset_indices);

    cudaDeviceSynchronize();

    //printf("nc: %p\n", nc);
    // printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    // set_nc<<<1,1>>>(nc, offsets, offset_indices, ncBlk_indices, mSize, nbBlocks);
    // cudaDeviceSynchronize();
    // printf("ncblockcpy: %f ms\n", timer2.GetCounter());
    checkCudaErrors(cudaFree(cBlk_indices));
    checkCudaErrors(cudaFree(ncBlk_indices));
    checkCudaErrors(cudaFree(offset_indices));
}

void ncblkCopy_h(unsigned char * c, unsigned char* o, unsigned char *nc, unsigned char* midBytes, unsigned char* meta,
                        size_t nbBlocks, int blockSize, short *offsets, size_t mSize){
    unsigned char *tmp_states;
    unsigned char *ncold = nc;
    uint64_t col_off = 0;
    short *tmp_offsets;
    tmp_offsets = (short*)malloc(sizeof(short)*nbBlocks);
    tmp_states = (unsigned char *)malloc(sizeof(char)*nbBlocks);
    checkCudaErrors(cudaMemcpy(tmp_states, meta, sizeof(char)*nbBlocks, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(tmp_offsets,offsets,sizeof(short)*nbBlocks,cudaMemcpyDeviceToHost));
    cudaStream_t stream[3];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);
    cudaStreamCreate(&stream[2]);

    //printf("here\n");
    //checkCudaErrors(cudaMemcpy((void**)&d_offsets, nbBlocks*sizeof(short))); 
    for (int i = 0; i < nbBlocks; i++)
    {
        if(tmp_states[i]==3){
            // shortToBytes_d(o, offsets[i]);
            // buf = (unsigned char*)
            
//	    printf("here2\n");
            cudaMemcpyAsync(o, offsets+i, 2, cudaMemcpyDeviceToDevice, stream[0]);
            o += sizeof(short);
        
    //	    printf("here2.1\n");
            // printf("offsets %ld\n", col_off);
            cudaMemcpyAsync(nc, meta+(nbBlocks+i*mSize), mSize, cudaMemcpyDeviceToDevice, stream[1]);
                // memcpy(nc, meta+(nbBlocks+i*mSize), mSize);
                
            nc += mSize; 
                
    //	    printf("here2.2\n");
            //checkCudaErrors(cudaMemcpy(buf, offsets+i, sizeof(short), cudaMemcpyDeviceToHost));
                
    //	    //printf("here2.3 %d\n", buf);
            cudaMemcpyAsync(nc, midBytes+(i*blockSize*sizeof(float)), (int)tmp_offsets[i], cudaMemcpyDeviceToDevice, stream[2]);
            // memcpy(nc, midBytes+(i*blockSize*sizeof(float)), offsets[i]);
            nc += tmp_offsets[i];
            col_off+=tmp_offsets[i];
       
///	    printf("here2.4\n");
       	}
    }
    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaStreamDestroy(stream[2]);

    free(tmp_states);
    free(tmp_offsets); 
}

__global__ void ncblkCopy(unsigned char * c, unsigned char* o, unsigned char *nc, unsigned char* midBytes, unsigned char* meta,
                        size_t nbBlocks, int blockSize, short *offsets, size_t mSize)
{
    for (int i=blockDim.x*blockIdx.x + threadIdx.x; i<nbBlocks; i+=blockDim.x*gridDim.x){
        
        if (meta[i]==0 || meta[i] == 1){
            memcpy(c, meta+(nbBlocks+i*mSize), sizeof(float));
            c += sizeof(float);
	    // float g;
	    // memcpy(&g, (meta+(nbBlocks+i*mSize)),sizeof(float));
	    // printf("%d %f\n",i,g);
        }else if(meta[i] == 3){
           shortToBytes_d(o, offsets[i]);
            o += sizeof(short);
            memcpy(nc, meta+(nbBlocks+i*mSize), mSize);
            nc += mSize; 
            memcpy(nc, midBytes+(i*blockSize*sizeof(float)), offsets[i]);
            nc += offsets[i];
        } 
    }
}

size_t better_post_proc(size_t *outSize, float *oriData, unsigned char *meta, 
                                short *offsets, unsigned char *midBytes, unsigned char *outBytes, 
                                size_t nbEle, int blockSize, uint64_t num_sig, uint32_t *blk_idx, 
                                float *blk_vals, uint8_t *blk_subidx, uint8_t *blk_sig){
    /**
     * outSize: host pointer
     * float *oriData: device pointer
     * unsigned char* meta: device pointer
     * short *offsets: device pointer
     * 
     * 
     */
    int out_size_h = 0;
    int *out_size_d;
    int tmp_outsize = 0;
    size_t *nc_diff;
    size_t nbConstantBlocks = 0;
    size_t nbBlocks = nbEle/blockSize;
    size_t ncBytes = blockSize/4;
    size_t mSize = sizeof(float)+1+ncBytes; //Number of bytes for each data block's metadata.
    out_size_h += 5+sizeof(size_t)+sizeof(float)*nbBlocks;
    if (nbBlocks%8==0)
        out_size_h += nbBlocks/8;
    else
        out_size_h += nbBlocks/8+1;
    cudaMalloc(&nc_diff, sizeof(size_t));
    int *nonconstant_d, nonconstant_h;
    checkCudaErrors(cudaMalloc((void **)&nonconstant_d, sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&out_size_d, sizeof(int)));

    checkCudaErrors(cudaMemset(nonconstant_d, 0, sizeof(int)));
    checkCudaErrors(cudaMemset(out_size_d, 0, sizeof(int)));


    getNumNonConstantBlocks<<<BLOCKS,THREADS_PER_BLOCK>>>(nbBlocks, offsets, meta, blockSize, nonconstant_d, out_size_d);
    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(&nonconstant_h, nonconstant_d, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&tmp_outsize, out_size_d, sizeof(int), cudaMemcpyDeviceToHost));

    nbConstantBlocks = nbBlocks - nonconstant_h;
    out_size_h+=tmp_outsize;

    out_size_h += (nbBlocks-nbConstantBlocks)*sizeof(short)+(nbEle%blockSize)*sizeof(float);

    //outBytes = (unsigned char*)malloc(out_size);
	unsigned char* r = outBytes;
    unsigned char* r_old = outBytes;
    checkCudaErrors(cudaMemset(r, SZx_VER_MAJOR, sizeof(char)));
    checkCudaErrors(cudaMemset(r+1, SZx_VER_MINOR, sizeof(char)));
    checkCudaErrors(cudaMemset(r+2, 1, sizeof(char)));
    checkCudaErrors(cudaMemset(r+3, 0, sizeof(char)));
    checkCudaErrors(cudaMemset(r+4, blockSize, sizeof(char)));

	r=r+5; //1 byte
	//sizeToBytes(r, nbConstantBlocks);
    longToBytes_bigEndian_memset(r, nbConstantBlocks);
	r += sizeof(size_t);
    //sizeToBytes(r, (size_t) num_sig);
    longToBytes_bigEndian_memset(r, (unsigned long)num_sig);
    r += sizeof(size_t); 
    size_t out_length;

    if(nbBlocks%4==0)
		out_length = nbBlocks/4;
	else
		out_length = nbBlocks/4+1;

    convert_state_to_out_kernel<<<BLOCKS,THREADS_PER_BLOCK>>>(meta, nbBlocks, r, out_length);
    r+=out_length;
    convert_block2_to_out_kernel<<<BLOCKS,THREADS_PER_BLOCK>>>(r, nbBlocks,num_sig, blk_idx, blk_vals, blk_subidx, blk_sig);
    r += nbBlocks*4 + num_sig*sizeof(float) + num_sig*sizeof(uint8_t) + nbBlocks*sizeof(uint8_t);

    checkCudaErrors(cudaMemcpy(r, oriData+nbBlocks*blockSize, (nbEle%blockSize)*sizeof(float), cudaMemcpyDeviceToDevice));
    // memcpy(r, oriData+nbBlocks*blockSize, (nbEle%blockSize)*sizeof(float));
    r += (nbEle%blockSize)*sizeof(float);
    unsigned char* c = r;
    unsigned char* o = c+nbConstantBlocks*sizeof(float);
    unsigned char* nc = o+(nbBlocks-nbConstantBlocks)*sizeof(short);
    // ncblkCopy<<<1,1>>>(c, o, nc, midBytes, meta,nbBlocks, blockSize, offsets, mSize);
    
    // ncblkCopy_h(c, o, nc, midBytes, meta,nbBlocks, blockSize, offsets, mSize);
    ncblkCopy_fast(c, o, nc, midBytes, meta,nbBlocks, blockSize, offsets, mSize, nc_diff);
    // cudaDeviceSynchronize();
    size_t h_nc_diff;
    cudaMemcpy(&h_nc_diff,nc_diff, sizeof(size_t),cudaMemcpyDeviceToHost);
    return (size_t) (nc+h_nc_diff-r_old);
    // checkCudaErrors(cudaMemcpy(outSize, (size_t)(nc-r_old), sizeof(size_t)));
    // *outSize = (size_t) (nc-r_old);
    // return outBytes;
}

__global__ void device_post_proc(size_t *outSize, float *oriData, unsigned char *meta, 
                                short *offsets, unsigned char *midBytes, unsigned char *outBytes, 
                                size_t nbEle, int blockSize, uint64_t num_sig, uint32_t *blk_idx, 
                                float *blk_vals, uint8_t *blk_subidx, uint8_t *blk_sig)
{
    int out_size = 0;

    size_t nbConstantBlocks = 0;
    size_t nbBlocks = nbEle/blockSize;
    size_t ncBytes = blockSize/4;
    size_t mSize = sizeof(float)+1+ncBytes; //Number of bytes for each data block's metadata.
    out_size += 5+sizeof(size_t)+sizeof(float)*nbBlocks;
    if (nbBlocks%8==0)
        out_size += nbBlocks/8;
    else
        out_size += nbBlocks/8+1;
    int s0 = 0;
    int s1 = 0;
    int s2 = 0;
    int s3 = 0;
    for (int i=0; i<nbBlocks; i++){
        if (meta[i]==0 || meta[i]==1 || meta[i] == 2) nbConstantBlocks++;
        else out_size += 1+(blockSize/4)+offsets[i];
    
    	if(meta[i]==0) s0++;
    	if(meta[i]==1) s1++;
    	if(meta[i]==2) s2++;
    	if(meta[i]==3) s3++;
    }
  //  printf("%d %d %d %d\n", s0, s1, s2, s3);
    out_size += (nbBlocks-nbConstantBlocks)*sizeof(short)+(nbEle%blockSize)*sizeof(float);

    //outBytes = (unsigned char*)malloc(out_size);
	unsigned char* r = outBytes;
   // printf("outbytes %p\n",r);
    unsigned char* r_old = outBytes; 
	r[0] = SZx_VER_MAJOR;
	r[1] = SZx_VER_MINOR;
	r[2] = 1;
	r[3] = 0; // indicates this is not a random access version
	r[4] = (unsigned char)blockSize;
	r=r+5; //1 byte
	//sizeToBytes(r, nbConstantBlocks);
    longToBytes_bigEndian_d(r, nbConstantBlocks);
	r += sizeof(size_t);
    //sizeToBytes(r, (size_t) num_sig);

   // printf("outbytes %p\n",r);
    longToBytes_bigEndian_d(r, (unsigned long)num_sig);
    r += sizeof(size_t); 
	r += convert_state_to_out(meta, nbBlocks, r);
   // printf("num sig %d\n", num_sig); 
   // printf("outbytes %p\n",r);
    r += convert_block2_to_out(r, nbBlocks,num_sig, blk_idx, blk_vals, blk_subidx, blk_sig);
    
   // printf("outbytes %p\n",r);
    memcpy(r, oriData+nbBlocks*blockSize, (nbEle%blockSize)*sizeof(float));
    r += (nbEle%blockSize)*sizeof(float);

   // printf("outbytes %p\n",r);
    unsigned char* c = r;
    unsigned char* o = c+nbConstantBlocks*sizeof(float);
    unsigned char* nc = o+(nbBlocks-nbConstantBlocks)*sizeof(short);
    for (int i=0; i<nbBlocks; i++){
        
        if (meta[i]==0 || meta[i] == 1){
            memcpy(c, meta+(nbBlocks+i*mSize), sizeof(float));
            c += sizeof(float);
       
	    // float g;
	    // memcpy(&g, (c-sizeof(float)),sizeof(float));
	    // printf("%d %f\n",i,g);
       	}else if(meta[i] == 3){
           shortToBytes_d(o, offsets[i]);
            o += sizeof(short);
            memcpy(nc, meta+(nbBlocks+i*mSize), mSize);
            nc += mSize; 
            memcpy(nc, midBytes+(i*blockSize*sizeof(float)), offsets[i]);
            nc += offsets[i];
        } 
    }

    // return out_size;
    *outSize = (size_t) (nc-r_old);
   // printf("outBytes 0 %d\n", (int) outBytes[0]);
    // return (uint32_t) (nc-r_old);
}

__global__ void fin_copy(unsigned char* in, unsigned char *out, size_t n){

	for(size_t i = threadIdx.x+blockIdx.x*gridDim.x; i < n; i+=blockDim.x*gridDim.x){
		out[i]=in[i];
	}

}

unsigned char* device_ptr_cuSZx_compress_float(float *oriData, size_t *outSize, float absErrBound, size_t nbEle, int blockSize, float threshold)
{
    /**
     * Assuming the following are device pointers:
     *  float *oriData
     *  size_t *outSize
     *  unsigned char* outBytes
     * 
     */
    // float *dmin,*dmax, *hmin, *hmax;
    // void *d_temp_storage = NULL;
    // size_t temp_storage_bytes = 0;
    timer_GPU.StartCounter();
//     cudaMalloc(&dmin, sizeof(float));
//     cudaMalloc(&dmax, sizeof(float));

//    // dmax = thrust::reduce(oriData, oriData+nbEle, -1, thrust::maximum<float>());
//    // dmin = thrust::reduce(oriData, oriData+nbEle, 1, thrust::minimum<float>());
//     cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, oriData, dmax, nbEle);
//     cudaMalloc(&d_temp_storage, temp_storage_bytes);
//     cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, oriData, dmax, nbEle);

//     cudaFree(d_temp_storage);
//     cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, oriData, dmin, nbEle);
//     cudaMalloc(&d_temp_storage, temp_storage_bytes);
//     cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, oriData, dmin, nbEle);

//     cudaFree(d_temp_storage);
//     // thrust::pair<float *, float *> result = thrust::minmax_element(thrust::device, oriData,oriData+nbEle);
//     //printf("here\n");
//     cudaMemcpy(hmin, dmin, sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemcpy(hmax, dmax,sizeof(float), cudaMemcpyDeviceToHost);
//     absErrBound = absErrBound*(hmax-hmin);
//     threshold = threshold*(hmax-hmin);
    // // printf("%f\n",absErrBound);
    // cudaFree(dmin);
    // cudaFree(dmax);
    float sparsity_level = SPARSITY_LEVEL;

    // Set the input data as the function parameter, this should be a device pointer

	float* d_oriData = oriData;
    // cudaMalloc((void**)&d_oriData, sizeof(float)*nbEle); 
    // cudaMemcpy(d_oriData, oriData, sizeof(float)*nbEle, cudaMemcpyHostToDevice); 

	size_t nbBlocks = nbEle/blockSize;
	size_t remainCount = nbEle%blockSize;
	size_t actualNBBlocks = remainCount==0 ? nbBlocks : nbBlocks+1;

    size_t ncBytes = blockSize/4;
    //ncBytes = (blockSize+1)%4==0 ? ncBytes : ncBytes+1; //Bytes to store one non-constant block data.
    size_t mSize = sizeof(float)+1+ncBytes; //Number of bytes for each data block's metadata.
    size_t msz = (1+mSize) * nbBlocks * sizeof(unsigned char);
    size_t mbsz = sizeof(float) * nbEle * sizeof(unsigned char);

    // These are host pointers and do not need to be allocated

    // unsigned char *meta = (unsigned char*)malloc(msz);
    // short *offsets = (short*)malloc(nbBlocks*sizeof(short));
    // unsigned char *midBytes = (unsigned char*)malloc(mbsz);

	unsigned char* d_meta;
	unsigned char* d_midBytes;
	short* d_offsets;

    uint32_t *blk_idx, *d_blk_idx;
    uint8_t *blk_sig, *d_blk_sig;
    uint8_t *blk_subidx, *d_blk_subidx;
    float *blk_vals, *d_blk_vals;
    uint64_t *num_sig, *d_num_sig;

    checkCudaErrors(cudaMalloc((void **)&d_num_sig, sizeof(uint64_t)));
    num_sig = (uint64_t *)malloc(sizeof(uint64_t));
    checkCudaErrors(cudaMalloc((void **)&d_blk_idx, nbBlocks*sizeof(uint32_t)));
    // blk_idx = malloc()
    checkCudaErrors(cudaMalloc((void **)&d_blk_subidx, nbEle*sizeof(uint8_t)));

    checkCudaErrors(cudaMalloc((void **)&d_blk_vals, nbEle*sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&d_blk_sig, nbBlocks*sizeof(uint8_t)));

    checkCudaErrors(cudaMalloc((void**)&d_meta, msz)); 
    //checkCudaErrors(cudaMemcpy(d_meta, meta, msz, cudaMemcpyHostToDevice)); 
    checkCudaErrors(cudaMemset(d_meta, 0, msz));
    checkCudaErrors(cudaMalloc((void**)&d_offsets, nbBlocks*sizeof(short))); 
    checkCudaErrors(cudaMemset(d_offsets, 0, nbBlocks*sizeof(short)));
    checkCudaErrors(cudaMalloc((void**)&d_midBytes, mbsz)); 
    checkCudaErrors(cudaMemset(d_midBytes, 0, mbsz));

    
    // apply_threshold<<<80,256>>>(d_oriData, threshold, nbEle);
    // cudaDeviceSynchronize();
    dim3 dimBlock(32, blockSize/32);
    dim3 dimGrid(65536, 1);
    const int sMemsize = blockSize * sizeof(float) + dimBlock.y * sizeof(int);
    //printf("Malloc end timestamp: %f ms\n", timer_GPU.GetCounter());
    compress_float<<<dimGrid, dimBlock, sMemsize>>>(d_oriData, d_meta, d_offsets, d_midBytes, absErrBound, blockSize, nbBlocks, mSize, sparsity_level, d_blk_idx, d_blk_subidx,d_blk_vals, threshold, d_blk_sig);
    cudaError_t err = cudaGetLastError();        // Get error code
   // printf("CUDA Error: %s\n", cudaGetErrorString(err));
    //printf("GPU compression timestamp: %f ms\n", timer_GPU.GetCounter());
    cudaDeviceSynchronize();
    get_numsig<<<1,1>>>(d_num_sig);
    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(num_sig, d_num_sig, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // These are allocations and memcpys to host pointers, do not need them

    // blk_idx = (uint32_t *)malloc(nbBlocks*sizeof(uint32_t));
    // blk_vals= (float *)malloc((*num_sig)*sizeof(float));
    // blk_subidx = (uint8_t *)malloc((*num_sig)*sizeof(uint8_t));
    // blk_sig = (uint8_t *)malloc(nbBlocks*sizeof(uint8_t));

    // checkCudaErrors(cudaMemcpy(meta, d_meta, msz, cudaMemcpyDeviceToHost)); 
    // checkCudaErrors(cudaMemcpy(offsets, d_offsets, nbBlocks*sizeof(short), cudaMemcpyDeviceToHost)); 
    // checkCudaErrors(cudaMemcpy(midBytes, d_midBytes, mbsz, cudaMemcpyDeviceToHost)); 
    
    
    // checkCudaErrors(cudaMemcpy(blk_idx, d_blk_idx, nbBlocks*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(blk_vals,d_blk_vals, (*num_sig)*sizeof(float), cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(blk_subidx,d_blk_subidx, (*num_sig)*sizeof(uint8_t), cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(blk_sig,d_blk_sig, (nbBlocks)*sizeof(uint8_t), cudaMemcpyDeviceToHost));


    size_t maxPreservedBufferSize = sizeof(float)*nbEle;
    unsigned char *d_outBytes;
    // unsigned char* outBytes = (unsigned char*)malloc(maxPreservedBufferSize);
    // memset(outBytes, 0, maxPreservedBufferSize);
    checkCudaErrors(cudaMalloc(&d_outBytes, maxPreservedBufferSize));

    size_t *d_outSize;

    checkCudaErrors(cudaMalloc(&d_outSize, sizeof(size_t)));

  //  device_post_proc<<<1,1>>>(d_outSize, d_oriData, d_meta, d_offsets, d_midBytes, d_outBytes, nbEle, blockSize, *num_sig, d_blk_idx, d_blk_vals, d_blk_subidx, d_blk_sig);
    *outSize = better_post_proc(d_outSize, d_oriData, d_meta, d_offsets, d_midBytes, d_outBytes, nbEle, blockSize, *num_sig, d_blk_idx, d_blk_vals, d_blk_subidx, d_blk_sig);
    //cudaDeviceSynchronize();
    
    //checkCudaErrors(cudaMemcpy(outSize, d_outSize, sizeof(size_t), cudaMemcpyDeviceToHost));

    // printf("completed compression\n");
    //free(blk_idx);
    //free(blk_subidx);
    //free(blk_vals);
    // free(meta);
    // free(offsets);
    // free(midBytes);
    checkCudaErrors(cudaFree(d_num_sig));
    checkCudaErrors(cudaFree(d_blk_idx));
    checkCudaErrors(cudaFree(d_blk_subidx));
    checkCudaErrors(cudaFree(d_blk_vals));
    checkCudaErrors(cudaFree(d_blk_sig));

    checkCudaErrors(cudaFree(d_meta));
    checkCudaErrors(cudaFree(d_offsets));
    checkCudaErrors(cudaFree(d_midBytes));

    unsigned char *d_newout;
    
    *outSize = *outSize;
    size_t os = *outSize;
    
    checkCudaErrors(cudaMalloc(&d_newout, os));
    //fin_copy<<<40,256>>>(d_outBytes, d_newout,os);
    checkCudaErrors(cudaMemcpy(d_newout, d_outBytes, os, cudaMemcpyDeviceToDevice));
    cudaDeviceSynchronize(); 

    checkCudaErrors(cudaFree(d_outBytes));
    printf("Compression end timestamp: %f ms\n", timer_GPU.GetCounter());
     
    err = cudaGetLastError();        // Get error code
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    return d_newout;
    //return d_outBytes;
}

__device__ inline long bytesToLong_bigEndian(unsigned char* b) {
	long temp = 0;
	long res = 0;

	res <<= 8;
	temp = b[0] & 0xff;
	res |= temp;

	res <<= 8;
	temp = b[1] & 0xff;
	res |= temp;
	
	res <<= 8;
	temp = b[2] & 0xff;
	res |= temp;
	
	res <<= 8;
	temp = b[3] & 0xff;
	res |= temp;
	
	res <<= 8;
	temp = b[4] & 0xff;
	res |= temp;
	
	res <<= 8;
	temp = b[5] & 0xff;
	res |= temp;
	
	res <<= 8;
	temp = b[6] & 0xff;
	res |= temp;
	
	res <<= 8;
	temp = b[7] & 0xff;
	res |= temp;						
	
	return res;
}

__device__ inline size_t bytesToSize(unsigned char* bytes)
{
	size_t result = bytesToLong_bigEndian(bytes);//8	
	return result;
}

__device__ inline short bytesToShort(unsigned char* bytes)
{
	lint16 buf;
	memcpy(buf.byte, bytes, 2);
	
	return buf.svalue;
}

__global__ void decompress_get_stats(float *newData, size_t nbEle, unsigned char* cmpBytes, 
    size_t *numSigValues, int *bs,
    size_t *numConstantBlks, size_t *numBlks,
    size_t *mSizeptr, unsigned char *newCmpBytes
){
	unsigned char* r = cmpBytes;

    size_t num_sig;
	r += 4;
	int blockSize = (int) r[0];  //get block size
	
	if(blockSize == 0)blockSize = 256;
	r++;
	size_t nbConstantBlocks = bytesToLong_bigEndian(r); //get number of constant blocks
	r += sizeof(size_t);
	num_sig = bytesToSize(r);
    
    r += sizeof(size_t);
	size_t nbBlocks = nbEle/blockSize;
    size_t ncBlocks = 0;
    size_t num_state2_blks = 0;
	// size_t ncBlocks = nbBlocks - nbConstantBlocks; //get number of constant blocks
	size_t stateNBBytes = nbBlocks%4==0 ? nbBlocks/4 : nbBlocks/4+1;
    size_t ncLeading = blockSize/4;
    size_t mSize = sizeof(float)+1+ncLeading; //Number of bytes for each data block's metadata.

    *mSizeptr = mSize;

    *numConstantBlks = nbConstantBlocks;
    *numBlks = nbBlocks;
    *numSigValues = num_sig;
    *bs = blockSize;
    newCmpBytes = r;

}

 void setup_data_stateArray_better(float *newData, size_t nbEle, unsigned char* r, 
    size_t num_sig, int blockSize,
    size_t nbConstantBlocks, size_t nbBlocks, size_t *ncBlks,
    unsigned char *stateArray, unsigned char *newR
){

    //printf("ma\n");
    // blockSize = 256;
    r += 4;
    r++;
    r += sizeof(size_t);
    r += sizeof(size_t);
    int ncBlocks, *ncBlocks_d;
	size_t stateNBBytes = nbBlocks%4==0 ? nbBlocks/4 : nbBlocks/4+1;
    int num_state2_blks, *num_state2_d;
    checkCudaErrors(cudaMalloc((void **)&num_state2_d, sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&ncBlocks_d, sizeof(int)));
    checkCudaErrors(cudaMemset(num_state2_d, 0, sizeof(int)));
    checkCudaErrors(cudaMemset(ncBlocks_d, 0, sizeof(int)));

    //printf("ma2\n");
//	printf("Converting state array\n");
    // printf("cmp %d\n", (int)r[0]);
    // printf("state %d\n", (int)stateArray[0]);
    // convert_out_to_state(nbBlocks, r, stateArray);
    convert_out_to_state_kernel<<<BLOCKS,THREADS_PER_BLOCK>>>(nbBlocks,r,stateArray,stateNBBytes,
                            num_state2_d, ncBlocks_d);
    // printf("state %d\n", (int)stateArray[0]);
    // convertByteArray2IntArray_fast_1b_args(nbBlocks, r, stateNBBytes, stateArray); //get the stateArray
	cudaDeviceSynchronize();
    
    //printf("ma3\n");
	r += stateNBBytes;
    newR = r;
    cudaMemcpy(&ncBlocks, ncBlocks_d, sizeof(int), cudaMemcpyDeviceToHost);
    
    //printf("ma4\n");
    *ncBlks = ncBlocks;

    //printf("ma4\n");
 }

__global__ void setup_data_stateArray(float *newData, size_t nbEle, unsigned char* r, 
    size_t num_sig, int blockSize,
    size_t nbConstantBlocks, size_t nbBlocks, size_t *ncBlks,
    unsigned char *stateArray, unsigned char *newR
){
    // blockSize = 256;
    r += 4;
    r++;
    r += sizeof(size_t);
    r += sizeof(size_t);
    size_t ncBlocks = 0;
	size_t stateNBBytes = nbBlocks%4==0 ? nbBlocks/4 : nbBlocks/4+1;
    size_t num_state2_blks = 0;
//	printf("Converting state array\n");
    // printf("cmp %d\n", (int)r[0]);
    // printf("state %d\n", (int)stateArray[0]);
    convert_out_to_state(nbBlocks, r, stateArray);
    // convert_out_to_state_kernel<<<40,256>>>(nbBlocks,r,stateArray,stateNBBytes);
    // printf("state %d\n", (int)stateArray[0]);
    // convertByteArray2IntArray_fast_1b_args(nbBlocks, r, stateNBBytes, stateArray); //get the stateArray
	for (size_t i = 0; i < nbBlocks; i++)
    {
        if (stateArray[i] == 2)
        {
            num_state2_blks++;
        }else if(stateArray[i] == 3){
            ncBlocks++;
        }
    }
    
	r += stateNBBytes;
    newR = r;
    *ncBlks = ncBlocks;
}

__global__ void decomp_startup_kernel(unsigned char* r, size_t nbConstantBlocks, 
unsigned char *data, int blockSize, size_t mSize, size_t ncBlocks, float *constantMedianArray, uint64_t* g_leng){
    unsigned char * fr = r; //fr is the starting address of constant median values.
    int i = 0, j = 0, k = 0;
  //  printf("%p\n", r);
    unsigned char tmp_r[4];
    tmp_r[0]=fr[0];
    tmp_r[1]=fr[1];
    tmp_r[2]=fr[2];
    tmp_r[3]=fr[3];


//    printf("nbconstant: %f\n", ((float*)tmp_r)[0]);
// nbConstantBlocks
    for(i = blockDim.x*blockIdx.x + threadIdx.x; i < nbConstantBlocks; i += blockDim.x*gridDim.x){ //get the median values for constant-value blocks
	    
    	    tmp_r[0]=fr[4*i];
    	    tmp_r[1]=fr[4*i+1];
    	    tmp_r[2]=fr[4*i+2];
    	    tmp_r[3]=fr[4*i+3];
	    float tmp = ((float*)tmp_r)[0];
	    constantMedianArray[i] = tmp;
	    //printf("%d %f\n", i, tmp);
    }
   

/** PROBLEM AREA, CAN FIX WITH PARALLELIZATION BUT WATCH *FR and *P **/

    // if(threadIdx.x==0 && blockIdx.x==0){
    fr += nbConstantBlocks*sizeof(float);
    unsigned char* p = fr + ncBlocks * sizeof(short);
    unsigned char* basefr = fr;
    unsigned char* basep = p;
    for(i = blockDim.x*blockIdx.x + threadIdx.x;i < ncBlocks;i+=blockDim.x*gridDim.x){
        fr = basefr+(sizeof(short)*i);
        int leng = (int)bytesToShort(fr)+mSize;
        g_leng[i] = (uint64_t)leng;
        // fr += sizeof(short);
        if (leng > blockSize*sizeof(float))
        {
            printf("Warning: compressed block is larger than the original block!\n");
            return;
            // exit(0);
        }
        // memcpy(data+i*blockSize*sizeof(float), p, leng);

        // p += leng;
    }
    
    // }
}

__global__ void decompress_ncblk_kernel(unsigned char* r, size_t nbConstantBlocks, 
unsigned char *data, int blockSize, size_t mSize, size_t ncBlocks, float *constantMedianArray, uint64_t* g_leng){
    unsigned char * fr = r;
    fr += nbConstantBlocks*sizeof(float);
    unsigned char* p = fr + ncBlocks * sizeof(short);
    unsigned char* basefr = fr;
    unsigned char* basep = p;

    for(int i = blockDim.x*blockIdx.x + threadIdx.x;i < ncBlocks;i+=blockDim.x*gridDim.x){
        fr = basefr+(sizeof(short)*i);
        int leng = (int)bytesToShort(fr)+mSize;
        
	
	// g_leng[i] = leng;
        // // fr += sizeof(short);
        // if (leng > blockSize*sizeof(float))
        // {
        //     printf("Warning: compressed block is larger than the original block!\n");
        //     return;
        //     // exit(0);
        // }
        p = basep + g_leng[i];

        memcpy(data+i*blockSize*sizeof(float), p, leng);
	
        // p += leng;
    }
}

void decompress_startup_better(float *newData, size_t nbEle, unsigned char* r, 
    uint32_t *blk_idx, uint8_t *blk_subidx, uint8_t *blk_sig,
    float *blk_vals, size_t num_sig, int blockSize,
    size_t nbConstantBlocks, size_t nbBlocks, size_t ncBlocks,
    unsigned char *stateArray, float* constantMedianArray, unsigned char *data,
    size_t mSize, unsigned char *newCmpBytes
){
    // blockSize = 256;
    size_t nb_tmp = (int) nbEle/blockSize;
    uint64_t* g_leng;
    /**
     * Structures to return:
     * blk_idx, blk_subidx, blk_sig, blk_vals, numSigValues (pointer)
     * bs (pointer to blockSize), numConstantBlks (pointer), numBlks (pointer)
     * ncBlks (pointer), stateArray, constantMedianArray
     */


    size_t stateNBBytes = nb_tmp%4==0 ? nb_tmp/4 : nb_tmp/4+1;
    
    r += 4;
    r++;
    r += sizeof(size_t);
    r += sizeof(size_t);

    r += stateNBBytes;

    convert_out_to_block2_kernel<<<BLOCKS,THREADS_PER_BLOCK>>>(r, nbBlocks, (uint64_t)num_sig, blk_idx, blk_vals, blk_subidx, blk_sig);
    size_t to_add = nbBlocks*4 + num_sig*sizeof(float) + num_sig*sizeof(uint8_t) + nbBlocks*sizeof(uint8_t);
    r+= to_add;

    size_t i = 0, j = 0, k = 0; //k is used to keep track of constant block index
    
    // printf("before mallocs in kernel\n");
    checkCudaErrors(cudaMemcpy((newData)+nbBlocks*blockSize, r, (nbEle%blockSize)*sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMalloc(&g_leng, sizeof(uint64_t)*ncBlocks));
    // memcpy((newData)+nbBlocks*blockSize, r, (nbEle%blockSize)*sizeof(float));

    //printf("before mallocs in kernel %p\n", r);
    r += (nbEle%blockSize)*sizeof(float);
    //printf("r: %p\n", r);
    //printf("%d, %d, %d\n",nbEle, 256, nbEle%256);
    decomp_startup_kernel<<<BLOCKS,THREADS_PER_BLOCK>>>(r, nbConstantBlocks,data, blockSize, mSize, ncBlocks, constantMedianArray, g_leng);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, g_leng, g_leng + ncBlocks, g_leng, 0);

    decompress_ncblk_kernel<<<BLOCKS,THREADS_PER_BLOCK>>>(r, nbConstantBlocks, data, blockSize, mSize, ncBlocks, constantMedianArray, g_leng);
    cudaDeviceSynchronize();
    
    // cudaError_t err = cudaGetLastError();        // Get error code
    
    // printf("CUDA Error: %s\n", cudaGetErrorString(err));
    cudaFree(g_leng);
        
    // err = cudaGetLastError();        // Get error code
    // printf("CUDA Error: %s\n", cudaGetErrorString(err));
    r += nbConstantBlocks*sizeof(float);

    newCmpBytes = r;

}

__global__ void decompress_startup(float *newData, size_t nbEle, unsigned char* r, 
    uint32_t *blk_idx, uint8_t *blk_subidx, uint8_t *blk_sig,
    float *blk_vals, size_t num_sig, int blockSize,
    size_t nbConstantBlocks, size_t nbBlocks, size_t ncBlocks,
    unsigned char *stateArray, float* constantMedianArray, unsigned char *data,
    size_t mSize, unsigned char *newCmpBytes
){
    // blockSize = 256;
    size_t nb_tmp = (int) nbEle/blockSize;
    /**
     * Structures to return:
     * blk_idx, blk_subidx, blk_sig, blk_vals, numSigValues (pointer)
     * bs (pointer to blockSize), numConstantBlks (pointer), numBlks (pointer)
     * ncBlks (pointer), stateArray, constantMedianArray
     */
	
    // size_t ncBlocks = 0;
	// size_t stateNBBytes = nbBlocks%4==0 ? nbBlocks/4 : nbBlocks/4+1;
    // size_t num_state2_blks = 0;
	// printf("Converting state array\n");
    // convert_out_to_state(nbBlocks, r, stateArray);
    // printf("state %d\n", (int)stateArray[0]);
    // // convertByteArray2IntArray_fast_1b_args(nbBlocks, r, stateNBBytes, stateArray); //get the stateArray
	// for (size_t i = 0; i < nbBlocks; i++)
    // {
    //     if (stateArray[i] == 2)
    //     {
    //         num_state2_blks++;
    //     }else if(stateArray[i] == 3){
    //         ncBlocks++;
    //     }
    // }
   // size_t stateNBBytes = nbBlocks%4==0 ? nbBlocks/4 : nbBlocks/4+1;

    size_t stateNBBytes = nb_tmp%4==0 ? nb_tmp/4 : nb_tmp/4+1;
    //printf("%p\n", r);
    r += 4;
    r++;
    r += sizeof(size_t);
    r += sizeof(size_t);
    //printf("statenb %d %d\n", stateNBBytes, nb_tmp);
    r += stateNBBytes;
    // data = (unsigned char*)malloc(ncBlocks*blockSize*sizeof(float));
    // memset(data, 0, ncBlocks*blockSize*sizeof(float));
   // printf("converting block vals %d\n", data[0]);
    size_t to_add = convert_out_to_block2(r, nbBlocks, (uint64_t)num_sig, blk_idx, blk_vals, blk_subidx, blk_sig);
    r+= to_add;

    size_t i = 0, j = 0, k = 0; //k is used to keep track of constant block index
    
    // printf("before mallocs in kernel\n");
    
    memcpy((newData)+nbBlocks*blockSize, r, (nbEle%blockSize)*sizeof(float));

    //printf("before mallocs in kernel %p\n", r);
    r += (nbEle%blockSize)*sizeof(float);
    //printf("r: %p\n", r);
    //printf("%d, %d, %d\n",nbEle, 256, nbEle%256);
    unsigned char * fr = r; //fr is the starting address of constant median values.

  //  printf("%p\n", r);
    unsigned char tmp_r[4];
    tmp_r[0]=r[0];
    tmp_r[1]=r[1];
    tmp_r[2]=r[2];
    tmp_r[3]=r[3];


//    printf("nbconstant: %f\n", ((float*)tmp_r)[0]);
    for(i = 0;i < nbConstantBlocks;i++, j+=4){ //get the median values for constant-value blocks
	    
    	    tmp_r[0]=r[j];
    	    tmp_r[1]=r[j+1];
    	    tmp_r[2]=r[j+2];
    	    tmp_r[3]=r[j+3];
	    float tmp = ((float*)tmp_r)[0];
//	    printf("median: %f\n", tmp);	
	    constantMedianArray[i] = tmp;

	    // printf("%d %f\n", i, tmp);
    }
    //printf("after constantmedian\n");
    r += nbConstantBlocks*sizeof(float);
    unsigned char* p = r + ncBlocks * sizeof(short);
    for(i = 0;i < ncBlocks;i++){
        int leng = (int)bytesToShort(r)+mSize;
        r += sizeof(short);
        if (leng > blockSize*sizeof(float))
        {
            printf("Warning: compressed block is larger than the original block!\n");
            return;
            // exit(0);
        }
//	printf("before memcpy\n");
        memcpy(data+i*blockSize*sizeof(float), p, leng);
  //      printf("after memcpy\n");
	p += leng;
    } 

    newCmpBytes = r;
//    printf("before mallocs in kernel\n");

    // printf("nb blocks: %d\n", nbBlocks);
}

__global__ void cBlkCopy_decompress(int nb, float* constantMedianArray, float *newData, int blockSize, int i){
    int j;
    float Median = constantMedianArray[nb];
    // j = threadIdx.x; j < blockSize; j += blockDim.x
    for (j = threadIdx.x; j < blockSize; j += blockDim.x)
        *((newData)+i*blockSize+j) = Median;
}

__global__ void ncBlkCopy_decompress(int blockSize, float *newData, int nc, float *fdata, int i){
    int j;
    for (j = threadIdx.x; j < blockSize; j += blockDim.x)
        *((newData)+i*blockSize+j) = fdata[nc*blockSize+j];
}

void decompress_post_proc_better(unsigned char *data, float *newData, int blockSize, 
    size_t nbBlocks, size_t ncBlocks, unsigned char *stateArray,
    float *constantMedianArray
){
    // checkCudaErrors(cudaMemcpy(data, d_data, ncBlocks*blockSize*sizeof(float), cudaMemcpyDeviceToHost)); 
    // checkCudaErrors(cudaMemcpy(*newData, d_newdata, nbBlocks*blockSize*sizeof(float), cudaMemcpyDeviceToHost));
    float* fdata = (float*)data;
    int i,j;
    int nb=0, nc=0;
    //printf("h1\n");
    for (i=0;i<nbBlocks;i++){
        unsigned char state;
        cudaMemcpy(&state, &stateArray[i], sizeof(char), cudaMemcpyDeviceToHost);

        if (state==0 || state==1){
            cBlkCopy_decompress<<<1,256>>>(nb, constantMedianArray, newData, blockSize, i);
            nb++;
        }else if(state==3){
            ncBlkCopy_decompress<<<1,256>>>(blockSize, newData, nc, fdata, i);
            nc++;
        }
    }
    cudaDeviceSynchronize();
    //for(int k = 0; k < nbBlocks*blockSize;k++){
//	printf("%f\n", newData[k]);
  //  }
}

__global__ void print_newdata(float *newData, size_t nbBlocks, int blockSize){
    for (size_t i = 0; i < nbBlocks*blockSize; i++)
    {
        printf("%f\n", newData[i]);
    }
    
}

__global__ void generateNbNc(size_t nbBlocks, size_t ncBlocks, unsigned char *stateArray, uint64_t* nbs,  uint64_t* ncs){
    for(int i = blockDim.x*blockIdx.x + threadIdx.x;i < nbBlocks;i+=blockDim.x*gridDim.x){
        unsigned char state = stateArray[i];
        if(state==0||state==1){
            nbs[i] = 1;
            ncs[i] = 0;
        }else if(state==3){
            nbs[i] = 0;
            ncs[i] = 1;
        }else{
            nbs[i] = 0;
            ncs[i] = 0;
        }
    }
}

__global__ void decompress_final_set(unsigned char *data, float *newData, int blockSize, 
    size_t nbBlocks, size_t ncBlocks, unsigned char *stateArray,
    float *constantMedianArray, uint64_t* nb, uint64_t* nc){
    float* fdata = (float*)data;
    for (int i = blockIdx.x;i < nbBlocks;i+=gridDim.x){
        if (stateArray[i]==0 || stateArray[i]==1){
            float Median = constantMedianArray[nb[i]];
            // if (Median>1) printf("data%i:%f\n",i, Median);
            for (int j = threadIdx.x; j < blockSize; j += blockDim.x)
                *((newData)+i*blockSize+j) = Median;
            // nb++;
        }else if(stateArray[i]==3){
            for (int j = threadIdx.x; j < blockSize; j += blockDim.x)
                *((newData)+i*blockSize+j) = fdata[nc[i]*blockSize+j];
            // nc++;
        }
        __syncthreads();
    }
}

void decompress_post_proc_fast(unsigned char *data, float *newData, int blockSize, 
    size_t nbBlocks, size_t ncBlocks, unsigned char *stateArray,
    float *constantMedianArray
){
    // checkCudaErrors(cudaMemcpy(data, d_data, ncBlocks*blockSize*sizeof(float), cudaMemcpyDeviceToHost)); 
    // checkCudaErrors(cudaMemcpy(*newData, d_newdata, nbBlocks*blockSize*sizeof(float), cudaMemcpyDeviceToHost));
    
    int i,j;
    uint64_t *nb, *nc;
    checkCudaErrors(cudaMalloc(&nb, sizeof(uint64_t)*nbBlocks));
    checkCudaErrors(cudaMalloc(&nc, sizeof(uint64_t)*nbBlocks));

    generateNbNc<<<BLOCKS,THREADS_PER_BLOCK>>>(nbBlocks, ncBlocks, stateArray, nb,nc);
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, nb, nb + nbBlocks, nb, 0);
    thrust::exclusive_scan(thrust::device, nc, nc + nbBlocks, nc, 0);

    decompress_final_set<<<nbBlocks,blockSize>>>(data, newData, blockSize,nbBlocks, ncBlocks, stateArray,constantMedianArray, nb, nc);
    cudaDeviceSynchronize();
    cudaFree(nb);
    cudaFree(nc);
}

__global__ void decompress_post_proc(unsigned char *data, float *newData, int blockSize, 
    size_t nbBlocks, size_t ncBlocks, unsigned char *stateArray,
    float *constantMedianArray
){
    // checkCudaErrors(cudaMemcpy(data, d_data, ncBlocks*blockSize*sizeof(float), cudaMemcpyDeviceToHost)); 
    // checkCudaErrors(cudaMemcpy(*newData, d_newdata, nbBlocks*blockSize*sizeof(float), cudaMemcpyDeviceToHost));
    float* fdata = (float*)data;
    int i,j;
    int nb=0, nc=0;
    // if (blockIdx.x == 0)
    // {
    //     for (i=0;i<nbBlocks;i++){
    //         if (stateArray[i]==0 || stateArray[i]==1){
    //             float Median = constantMedianArray[nb];
    //             // if (Median>1) printf("data%i:%f\n",i, Median);
    //             for (j = threadIdx.x; j < blockSize; j += blockDim.x)
    //                 *((newData)+i*blockSize+j) = Median;
    //             nb++;
    //         }
    //     }
    // }else{
    //     for (i=0;i<nbBlocks;i++){
    //         if(stateArray[i]==3){
    //             for (j = threadIdx.x; j < blockSize; j += blockDim.x)
    //                 *((newData)+i*blockSize+j) = fdata[nc*blockSize+j];
    //             nc++;
    //         }
    //     }
    // }
    
    for (i=0;i<nbBlocks;i++){
        if (stateArray[i]==0 || stateArray[i]==1){
            float Median = constantMedianArray[nb];
            // if (Median>1) printf("data%i:%f\n",i, Median);
            for (j = threadIdx.x; j < blockSize; j += blockDim.x)
                *((newData)+i*blockSize+j) = Median;
            nb++;
        }else if(stateArray[i]==3){
            for (j = threadIdx.x; j < blockSize; j += blockDim.x)
                *((newData)+i*blockSize+j) = fdata[nc*blockSize+j];
            nc++;
        }
    }

    //for(int k = 0; k < nbBlocks*blockSize;k++){
//	printf("%f\n", newData[k]);
  //  }
}

float* device_ptr_cuSZx_decompress_float(size_t nbEle, unsigned char* cmpBytes)
{
    /**
     * Assume the following are device pointers
     * 
     * unsigned char* cmpBytes
     * float** newData
     * 
     */
    
    uint32_t *blk_idx;
    uint8_t *blk_subidx;
    uint8_t *blk_sig;
    float *blk_vals, *constantMedianArray;
    size_t *num_sig, *mSize, mSize_h, num_sig_h;
    int *blockSize, bs;
    size_t *nbConstantBlocks, *nbBlocks, *ncBlocks, nbBlocks_h, ncBlocks_h, nbConstantBlocks_h;
    unsigned char *stateArray, *data;
    float *newData;
    timer_GPU.StartCounter();
    unsigned char *oldCmpBytes = cmpBytes;
	//*newData = (float*)malloc(sizeof(float)*nbEle);
//    printf("cmpbytes check %d\n", (int)cmpBytes[0]);
//    printf("new check %f\n", *newData[0]);
    // printf("malloc\n");
    checkCudaErrors(cudaMalloc((void**)&num_sig, sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&blockSize, sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&nbConstantBlocks, sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&nbBlocks, sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&ncBlocks, sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&mSize, sizeof(size_t)));    
    checkCudaErrors(cudaMalloc((void**)&newData, sizeof(float)*nbEle));

    decompress_get_stats<<<1,1>>>(newData, nbEle, cmpBytes, 
        num_sig, blockSize,
        nbConstantBlocks, nbBlocks,
        mSize, cmpBytes
    );
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();        // Get error code
    //printf("CUDA Error: %s\n", cudaGetErrorString(err));
    checkCudaErrors(cudaMemcpy(&nbBlocks_h, nbBlocks, sizeof(size_t), cudaMemcpyDeviceToHost)); 
    checkCudaErrors(cudaMemcpy(&nbConstantBlocks_h, nbConstantBlocks, sizeof(size_t), cudaMemcpyDeviceToHost)); 
    checkCudaErrors(cudaMemcpy(&bs, blockSize, sizeof(int), cudaMemcpyDeviceToHost)); 
    checkCudaErrors(cudaMemcpy(&mSize_h, mSize, sizeof(size_t), cudaMemcpyDeviceToHost)); 
    checkCudaErrors(cudaMemcpy(&num_sig_h, num_sig, sizeof(size_t), cudaMemcpyDeviceToHost)); 


    checkCudaErrors(cudaMalloc((void**)&stateArray, nbBlocks_h));
    checkCudaErrors(cudaMalloc((void**)&constantMedianArray, nbConstantBlocks_h*sizeof(float)));

    checkCudaErrors(cudaMalloc((void**)&blk_idx, nbBlocks_h*sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc((void**)&blk_vals, num_sig_h*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&blk_subidx, num_sig_h*sizeof(uint8_t)));
    checkCudaErrors(cudaMalloc((void**)&blk_sig, nbBlocks_h*sizeof(uint8_t)));

    unsigned char* tmp_r = cmpBytes;
    unsigned char* newR;
    setup_data_stateArray_better(newData, nbEle, tmp_r, 
    num_sig_h, bs,
    nbConstantBlocks_h, nbBlocks_h, &ncBlocks_h,
    stateArray, newR);
    
    
    
   // setup_data_stateArray<<<1,1>>>(newData, nbEle, cmpBytes, 
   //      num_sig_h, bs,
   //      nbConstantBlocks_h, nbBlocks_h, ncBlocks,
   //      stateArray, cmpBytes
   //  );
   // cudaDeviceSynchronize();

   // printf("%s\n", cudaGetErrorString(cudaGetLastError())); 
   // checkCudaErrors(cudaMemcpy(&ncBlocks_h, ncBlocks, sizeof(size_t), cudaMemcpyDeviceToHost)); 

    checkCudaErrors(cudaMalloc((void**)&data, ncBlocks_h*bs*sizeof(float)));

    // err = cudaGetLastError();        // Get error code
    // printf("CUDA start Error: %s\n", cudaGetErrorString(err));
    // cmpBytes = newCmpBytes;
    // data = (unsigned char*)malloc(ncBlocks*blockSize*sizeof(float));
    // memset(data, 0, ncBlocks*blockSize*sizeof(float));
    // stateArray = (unsigned char*)malloc(nbBlocks);
    
    // // unsigned char* d_stateArray;
    // // cudaMalloc(&d_stateArray, nbBlocks);
	// constantMedianArray = (float*)malloc(nbConstantBlocks*sizeof(float));			

    // blk_idx = (uint32_t *)malloc(nbBlocks*sizeof(uint32_t));
    // blk_vals= (float *)malloc((num_sig)*sizeof(float));
    // blk_subidx = (uint8_t *)malloc((num_sig)*sizeof(uint8_t));
    // blk_sig = (uint8_t *)malloc(nbBlocks*sizeof(uint8_t));

    //printf("%s\n", cudaGetErrorString(cudaGetLastError())); 
    //test_nbBlks = (size_t *)malloc(sizeof(size_t));
    // printf("malloc\n");
    
    
    tmp_r = cmpBytes;
    decompress_startup_better(newData, nbEle, tmp_r, 
    blk_idx, blk_subidx, blk_sig,
    blk_vals, num_sig_h, bs,
     nbConstantBlocks_h, nbBlocks_h, ncBlocks_h,
    stateArray, constantMedianArray, data,
    mSize_h, newR);


    // err = cudaGetLastError();        // Get error code
    // printf("CUDA start Error: %s\n", cudaGetErrorString(err));
    //decompress_startup<<<1,1>>>(newData, nbEle, cmpBytes, 
    // blk_idx, blk_subidx, blk_sig,
    // blk_vals, num_sig_h, bs,
    // nbConstantBlocks_h, nbBlocks_h, ncBlocks_h,
    // stateArray, constantMedianArray, data, mSize_h, cmpBytes);
    //cudaDeviceSynchronize();
    // cmpBytes = newCmpBytes;

    //printf("%s\n", cudaGetErrorString(cudaGetLastError())); 

    // unsigned char* d_data;
    float *d_newdata;
    // checkCudaErrors(cudaMalloc((void**)&d_data, ncBlocks*blockSize*sizeof(float))); 
    // checkCudaErrors(cudaMemcpy(d_data, data, ncBlocks*blockSize*sizeof(float), cudaMemcpyHostToDevice)); 
    // printf("nblocks: %d bs: %d ncblock %d\n", nbBlocks_h, bs, ncBlocks_h);
    checkCudaErrors(cudaMalloc(&d_newdata, nbBlocks_h*bs*sizeof(float)));

    // err = cudaGetLastError();        // Get error code
    // printf("CUDA dec main Error: %s\n", cudaGetErrorString(err));
    
    dim3 dimBlock(32, bs/32);
    dim3 dimGrid(65536, 1);
    const int sMemsize = bs * sizeof(float) + dimBlock.y * sizeof(int);
    decompress_state2<<<nbBlocks_h, 64>>>(d_newdata, stateArray,blk_idx, blk_vals, blk_subidx, bs, blk_sig);
    cudaDeviceSynchronize();

    // err = cudaGetLastError();        // Get error code
    // printf("CUDA dec main Error: %s\n", cudaGetErrorString(err));
    decompress_float<<<dimGrid, dimBlock, sMemsize>>>(data, bs, ncBlocks_h, mSize_h);
    //printf("GPU decompression timing: %f ms\n", timer_GPU.GetCounter());
    cudaDeviceSynchronize();

    // err = cudaGetLastError();        // Get error code
    // printf("CUDA dec main Error: %s\n", cudaGetErrorString(err));
    
    // checkCudaErrors(cudaMemcpy(data, d_data, ncBlocks*blockSize*sizeof(float), cudaMemcpyDeviceToHost)); 
    checkCudaErrors(cudaMemcpy(newData, d_newdata, nbBlocks_h*bs*sizeof(float), cudaMemcpyDeviceToDevice));
    cudaFree(d_newdata);

    // decompress_post_proc<<<1,1>>>(data, newData, bs, 
    // nbBlocks_h, ncBlocks_h, stateArray,
    // constantMedianArray);
    // cudaDeviceSynchronize();
    decompress_post_proc_fast(data, newData, bs, 
    nbBlocks_h, ncBlocks_h, stateArray,
    constantMedianArray);
    err = cudaGetLastError();        // Get error code
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    printf("GPU decompression timing: %f ms\n", timer_GPU.GetCounter());
   // print_newdata<<<1,1>>>(newData, nbBlocks_h, bs);
	cudaFree(stateArray);
	cudaFree(constantMedianArray);
	cudaFree(data);
    cudaFree(blk_idx);
    cudaFree(blk_subidx);
    cudaFree(blk_vals);
    cudaFree(blk_sig);
    return newData;

}

