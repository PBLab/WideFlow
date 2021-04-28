import numpy as np

import cupy as cp
from cupy.cuda.memory import MemoryPointer, UnownedMemory
import cupyx.scipy.ndimage as csn

from numba import cuda

update_buffer = cp.RawKernel(r'''
    extern "C" __global__
    void copyKernel(double* x3d, double* x2d, unsigned char* frameIdx, unsigned char* capacity) {
        
        const int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
        const int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
        const int zIndex = blockDim.z * blockIdx.z + threadIdx.z;
        
        if (frameIdx == capacity-1) {
            *frameIdx = 0}
        else {
            *frameIdx++1
        
        if (zIndex < frameIdx & zIndex > frameIdx+1){
            x3d[zIndex][yIndex][xIndex] = x2d[yIndex][xIndex];
        }

    }
    ''', 'copyKernel')


dff = cp.ElementwiseKernel(
    'float32 x, float32 y',
    'float32 z',
    'z = (x - y) / (y + 0.0000000000000001)',
    'dff')


def resize(cp_2d_arr, cp_2d_arr_rs, n_rows_rs, n_cols_rs):
    [n_rows, n_cols] = cp_2d_arr.shape
    trans_mat = cp.eye(3)
    trans_mat[0][0] = n_rows_rs / n_rows
    trans_mat[1][1] = n_cols_rs / n_cols
    csn.affine_transform(cp_2d_arr, trans_mat, output_shape=(n_rows_rs, n_cols_rs), output=cp_2d_arr_rs, mode='opencv')



# def run_preprocesses(cp_2d_arr, processes_list):
#     for process in processes_list:
#         cp_2d_arr = eval(process[0] + "(" + ",".join(process[1:]) + ")")
#
#     return cp_2d_arr


# mem = cp.cuda.BaseMemory(4294967295, (10,10), 0)
# ptr=cp.cuda.MemoryPointer(mem,0)
# a = cp.ndarray((10, 10), memptr=ptr)

from cupy.cuda.memory import MemoryPointer, UnownedMemory
