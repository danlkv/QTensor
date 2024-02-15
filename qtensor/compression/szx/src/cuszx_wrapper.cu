#include "cuszx_entry.h"
#include "szx_defines.h"
#include "szx_BytesToolkit.h"
#include "szx_TypeManager.h"
#include "timingGPU.h"

extern "C"{
    unsigned char* cuSZx_integrated_compress(float *data, float r2r_threshold, float r2r_err, size_t nbEle, int blockSize, size_t *outSize){
        float max,min;
        unsigned char* bytes;
        max = data[0];
        min = data[0];
        for (size_t i = 0; i < nbEle; i++)
        {
            if(data[i] > max) max = data[i];
            if(data[i] < min) min = data[i];
        }
        
        float threshold = r2r_threshold*(max-min);
        float errBound = r2r_err*(max-min);
        bytes = cuSZx_fast_compress_args_unpredictable_blocked_float(data, outSize, errBound, nbEle, blockSize, threshold);
   	    // printf("outSize %p\n", bytes);
        return bytes;
    }

    float* cuSZx_integrated_decompress(unsigned char *bytes, size_t nbEle){
        // printf("test\n");
        float**data;
	    cuSZx_fast_decompress_args_unpredictable_blocked_float(data, nbEle, bytes);
        return *data;
    }

    unsigned char* cuSZx_device_compress(float *oriData, size_t *outSize, float absErrBound, size_t nbEle, int blockSize, float threshold){
        return device_ptr_cuSZx_compress_float(oriData, outSize, absErrBound, nbEle, blockSize, threshold);
    }

    float* cuSZx_device_decompress(size_t nbEle, unsigned char* cmpBytes){
        return device_ptr_cuSZx_decompress_float(nbEle, cmpBytes);
    }
    
}
