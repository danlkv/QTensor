import cuszp
import torch
from statcollector import StatCollector
# Create a class that performs compression and decompression on a tensor


class Compressor(torch.nn.Module):
    def __init__(self, err_mode, err_bound, device, num_nodes,statcollector:StatCollector):
        super(Compressor, self).__init__()
        self.err_mode = err_mode
        self.err_bound = err_bound
        self.device = device
        self.compressor = cuszp
        self.num_nodes = num_nodes
        self.sc = statcollector

    def compress(self, x):
        # Ensure float32 type
        if not x.dtype == torch.float32:
            raise TypeError("x must be of type torch.float32")
        x = x.contiguous()
        if self.err_mode == "rel" or self.err_mode == "relative":
            # Value-range error bound
            x_max = torch.max(x)
            x_min = torch.min(x)
            # Compute the err_bound
            err_bound = (x_max - x_min) * self.err_bound
            # print("min =", x_min, "max =", x_max, "err_bound =", err_bound)
            self.sc.add_tensor_stat("Min Value", x_min.item())
            self.sc.add_tensor_stat("Max Value", x_max.item())

        elif self.err_mode == "abs" or self.err_mode == "absolute":
            err_bound = self.err_bound
        else:
            raise ValueError("err_mode must be 'rel / relative' or 'abs / absolute'")
        self.sc.add_tensor_stat("Absolute Error Bound", err_bound.item())

        return CompressedElement(x, self.compressor.compress(x, err_bound, self.err_mode), err_bound, self.device)

    def decompress(self, comp_element):
        if not isinstance(comp_element, CompressedElement):
            raise TypeError("comp_element must be an instance of CompressedElement")
        compressed_size = (
            comp_element.compressed_data.numel()
            * comp_element.compressed_data.element_size()
        )
        decompressed = self.compressor.decompress(
            comp_element.compressed_data,
            comp_element.uncompressed_elements,
            compressed_size,
            comp_element.err_bound,
            self.err_mode,
        )
        # Reshape decompressed to match original shape
        decompressed = decompressed.reshape(comp_element.original_shape)
        return decompressed

    def pack_hook(self, x):
        if (
            x.dtype == torch.float32
            and x.requires_grad
            and not x.is_sparse
            and isinstance(x, torch.Tensor)
            and x.shape[0] == self.num_nodes
        ):
            # print("Packing", x.shape)
            t0 = self.sc.new_clock()
            self.sc.sync_start_time(t0)

            compressed = self.compress(x)

            self.sc.sync_end_time(t0)
            self.sc.increment_epoch_stat("Total Compression Time (s)",self.sc.get_elapsed_time(t0))

            # print("Uncompressed size =", (x.numel() * x.element_size()) / 1024 / 1024)
            # print(
            #     "Compressed size =",
            #     (
            #         compressed.compressed_data.numel()
            #         * compressed.compressed_data.element_size()
            #     )
            #     / 1024
            #     / 1024,
            # )
            # print(
            #     "Compression Ratio = ",
            #     (x.numel() * x.element_size())
            #     / (
            #         compressed.compressed_data.numel()
            #         * compressed.compressed_data.element_size()
            #     ),
            # )
            csize = compressed.compressed_data.numel()*compressed.compressed_data.element_size()
            osize = x.numel() * x.element_size()
            self.sc.add_tensor_stat("Uncompressed Size (bytes)", osize)
            self.sc.add_tensor_stat("Compressed Size (bytes)", csize)
            self.sc.increment_epoch_stat("Average CR", osize/csize)
            self.sc.increment_epoch_stat("Aggregate Uncompressed Tensor Size (bytes)", osize)
            self.sc.increment_epoch_stat("Aggregate Compressed Tensor Size (bytes)", csize)
            # print( "Data Saved", ((x.numel() * x.element_size()) - (compressed.compressed_data.numel() * compressed.compressed_data.element_size()))/1024/1024)
            # print("Testing decompress,", decompressed)
            # print("Compressed data", compressed.compressed_data)
            # print("Decompressed shape =", decompressed.shape)
            # print("X shape = ", x.shape)
            # abs_error = torch.abs(x - decompressed)
            # max_error = torch.max(abs_error)
            # if max_error > self.err_bound * 1.1:
            #     # Print the location of the max error and the values
            #     print("Max error location =", torch.argmax(torch.abs(x - decompressed)))
            #     print("Max error value =", max_error)
            #     location = torch.argmax(torch.abs(x - decompressed))
            #     # Print row and column of max error
            #     print("Row =", int(location / x.shape[1]))
            #     print("Column =", location % x.shape[1])
            #     # Count the number of elements that are > self.err_bound * 1.1
            #     bound_err_cnt = torch.sum(abs_error > self.err_bound * 1.1)
            #     print("Number of elements > err_bound * 1.1 =", bound_err_cnt)
            #     print("X value =", x[int(location / x.shape[1])][location % x.shape[1]])
            #     print(
            #         "Decompressed value =",
            #         decompressed[int(location / x.shape[1])][location % x.shape[1]],
            #     )
            #     raise ValueError(
            #         "Error bound exceeded! Max error = ", max_error
            #     )
            # # Ensure max_error <= err_bound

            # print("Max error =", max_error)
            # Ensure x is freed
            # delete x
            self.sc.increment_epoch_stat("Compressed Tensor Count",1)
            self.sc.register_tensor_row_and_update()


            del x
            # empty cache
            torch.cuda.empty_cache()
            return compressed
        else:
            return x

    def unpack_hook(self, x):
        if isinstance(x, CompressedElement):
            # print("Unpacking", x.name)
            # print("Unpacking")
            t0 = self.sc.new_clock()
            self.sc.sync_start_time(t0)

            decompressed = self.decompress(x)

            self.sc.sync_end_time(t0)
            self.sc.increment_epoch_stat("Total Decompression Time (s)",self.sc.get_elapsed_time(t0))

            # print("Unpacked")
            # print("Unpacked to", decompressed)
            return decompressed
        else:
            return x


# Create class for a compressed element that is used by the Compressor class


class CompressedElement(torch.nn.Module):
    def __init__(self, x, compressed, err_bound, device):
        super(CompressedElement, self).__init__()
        self.device = device
        # self.compressor = cuszp
        self.compressed_data = compressed
        self.uncompressed_elements = x.numel()
        self.original_shape = x.shape
        self.err_bound = err_bound

import numpy as np
import ctypes
from ctypes import *
import random
from qtensor.tools.lazy_import import cupy as cp
import time
import torch

from pathlib import Path



def quant_device_compress(oriData, nbEle, blockSize,threshold):
    #print(nbEle)
    ori_nbEle = nbEle
    variable = ctypes.c_size_t(0)
    outSize = ctypes.pointer(variable)

    oriData = oriData.flatten()
    ori_real = oriData.real
    ori_imag = oriData.imag
    oriData = cp.concatenate((ori_real, ori_imag))
    sample = oriData[::2]
    max_val = cp.amax(oriData).get()
    min_val = cp.amin(oriData).get()
    d = max_val - min_val
    if d.dtype == np.complex64:
        d = d.real
    threshold = threshold*(d)
    s_1 = time.time() 
    truth_values = abs(oriData)<=threshold
    oriData[truth_values] = 0.0
    truth_values = cp.invert(truth_values)
    ori_len = oriData.shape[0]
    nonzero_percent = cp.count_nonzero(oriData)/oriData.shape[0]
    print("Percent nonzero: "+str(nonzero_percent))

    isGrouped = False
    if nonzero_percent<=0.5:
        isGrouped=True
        oriData = oriData[truth_values]
    
    nbEle = oriData.shape[0]
    
    # oriData = cp.reshape(oriData, (-1, blockSize))  # Reshape to blocksize
    tensor = torch.as_tensor(oriData, device='cuda')
    # print("Min val: "+str(cp.amin(oriData).get())+" range: "+str(d))
#    scale = d/255.0
#    zero_point = -1*round(min_val*scale) - 128

    scale = d/((2**8) - 1)
    #zero_point = -1*round(min_val*scale)
    zero_point = -1*round(min_val*scale)+32
#    q_tensor = torch.quantize_per_tensor(tensor, scale, zero_point, dtype=torch.qint8)
    
    q_tensor = torch.quantize_per_tensor(tensor, scale, zero_point, dtype=torch.qint8)
    del tensor
    torch.cuda.empty_cache()
    if isGrouped:
        bitmap = cp.packbits(truth_values)
    else:
        bitmap = None
    del truth_values
    #q_ten2 = torch.dequantize(q_tensor)
    #print(tensor)
    #print(q_ten2)
    #print("Max PW error")
    #print(torch.max(torch.div(torch.abs(torch.sub(tensor[tensor!=0.0],q_ten2[tensor!=0.0])),tensor[tensor!=0.0])))
    return (q_tensor, bitmap, isGrouped), (nbEle/4)+(ori_len/8)


def quant_device_decompress(nbEle, cmpBytes, owner, dtype):
    (q_tensor, bitmap, isGrouped) = cmpBytes
    if isGrouped:
        bitmap = cp.unpackbits(bitmap)
    restored = torch.dequantize(q_tensor)
    arr = cp.asarray(restored)
    # uint8_t* cmpbytes, size_t len, size_t compressed_len, float r2r_error

    # decompressed_ptr = self.cuszx_decompress(isCuPy, cmp_bytes, num_elements_eff)
    # -- Workaround to convert GPU pointer to int
    # p_decompressed_ptr = ctypes.addressof(newData)
    # cast to int64 pointer
    # (effectively converting pointer to pointer to addr to pointer to int64)
    # p_decompressed_int= ctypes.cast(p_decompressed_ptr, ctypes.POINTER(ctypes.c_uint64))
    # decompressed_int = p_decompressed_int.contents
    # # --
    # pointer_for_free = decompressed_int.value
    # # self.decompressed_own.append(decompressed_int.value)
    # mem = cp.cuda.UnownedMemory(decompressed_int.value, nbEle*4, owner, device_id=0)
    # mem_ptr = cp.cuda.memory.MemoryPointer(mem, 0)
    #print("mem ptr")
    #print(mem_ptr)
    # arr = cp.ndarray(shape=(nbEle,), dtype=np.float32, memptr=mem_ptr)
    #print(nbEle)
    if isGrouped:
        res = cp.zeros((nbEle,))
    # ## need to convert newData to cupy
        cp.place(res,bitmap,arr)

        c_res = cp.zeros(int(nbEle/2), np.complex64)
    #c_res.real = arr[0:int(nbEle/2)]
    #c_res.imag = arr[int(nbEle/2):]

        c_res.real = res[0:int(nbEle/2)]
        c_res.imag = res[int(nbEle/2):]
    else:
        c_res = cp.zeros(int(nbEle/2), np.complex64)
        c_res.real = arr[0:int(nbEle/2)]
        c_res.imag = arr[int(nbEle/2):]
    return (c_res, None)

### Example of device compress/decompress wrapper usage
class Comp():
    def __init__(self):
        self.name = "dummy"

def free_compressed(ptr):
    p_ptr = ctypes.addressof(ptr)
    p_int = ctypes.cast(p_ptr, ctypes.POINTER(ctypes.c_uint64))
    decomp_int = p_int.contents
    cp.cuda.runtime.free(decomp_int.value)


if __name__ == "__main__":
    
    DATA_SIZE = int(1024)
    MAX_D = 10.0
    MIN_D = -10.0
    RANGE = MAX_D - MIN_D
    r2r_threshold = 0.002
    r2r_error = 0.0001

    in_vector = np.fromfile("all_sample.bin", dtype=np.complex64)
    #print(np.max(in_vector))
    DATA_SIZE = len(in_vector)
    #range_vr = np.max(in_vector)-np.min(in_vector)
    #r2r_threshold = r2r_threshold*range_vr
    #r2r_error = r2r_error*range_vr
    #in_vector = np.zeros((DATA_SIZE,))
    #for i in range(0,int(DATA_SIZE/4)):
    #    in_vector[i] = 0.0
    #for i in range(int(DATA_SIZE/4), int(2*DATA_SIZE/4)):
    #    in_vector[i] = 5.0
    #for i in range(int(2*DATA_SIZE/4), int(3*DATA_SIZE/4)):
    #    in_vector[i] = random.uniform(MIN_D, MAX_D)
    #for i in range(int(3*DATA_SIZE/4), int(3*DATA_SIZE/4)+6):
    #    in_vector[i] = -7.0
    #for i in range(int(3*DATA_SIZE/4)+6, DATA_SIZE):
    #    in_vector[i] = 0.001

    print(DATA_SIZE)
    #in_vector = in_vector.astype('float32')
    in_vector_gpu = cp.asarray(in_vector)
    
    # variable = ctypes.c_size_t(0)
    # outSize = ctypes.pointer(variable)
    for i in range(200):
        s_time = time.time()
        o_bytes, outSize = quant_device_compress(in_vector_gpu, DATA_SIZE, 256, r2r_threshold)
        print("Time python: "+str(time.time()-s_time))
        # print(outSize[0])
        print("Compress Success...starting decompress ")
        comp = Comp()

        s_time = time.time()
        (d_bytes,ptr )= quant_device_decompress(DATA_SIZE*2, o_bytes, comp, in_vector_gpu.dtype)
        
        # free_compressed(o_bytes[0])
        # cp.cuda.runtime.free(ptr)
        print("Time python: "+str(time.time()-s_time))
    #for i in d_bytes:
    #    print(i)
        print("Decompress Success")