import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as csn
import cupyx.scipy.signal as css


# update_buffer = cp.RawKernel(r'''
#     extern "C" __global__
#     void copyKernel(double* x3d, double* x2d, int frameIdx, int capacity) {
#
#         const int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
#         const int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
#         const int zIndex = blockDim.z * blockIdx.z + threadIdx.z;
#
#         if (frameIdx == capacity-1){
#             frameIdx = 0;
#             }
#         else {
#             frameIdx = frameIdx + 1;
#             }
#
#         if (zIndex >= frameIdx & zIndex < frameIdx+1){
#             x3d[zIndex][yIndex][xIndex] = x2d[yIndex][xIndex];
#         }
#
#     }
#     ''', 'copyKernel')


# mean_kernel = cp.RawKernel(r'''
#     #include </usr/local/cuda-8.0/include/npps_statistics_functions.h>
# ... extern "C" __global__
# ... void mean(const float* x, float* y) {
# ...     int tid = blockDim.x * blockIdx.x + threadIdx.x;
# ...     nppsMean_StdDev_32f_C1MR_Ctx
# ... }
# ... ''', 'mean')
#
#
# std_kernel = cp.RawKernel(r'''
#     #include </usr/local/cuda-8.0/include/npps_statistics_functions.h>
# ... extern "C" __global__
# ... void std(const float* x, float* y) {
# ...     int tid = blockDim.x * blockIdx.x + threadIdx.x;
# ...     nppsStdDev_32f(x, , y,
# ... }
# ... ''', 'std')



def baseline_calc_carbox(cp_3d_arr):
    dims = (5, 3, 3)
    weights = np.ones(dims, dtype=np.float32) / (dims[0]*dims[1]*dims[2])
    weights = cp.asanyarray(weights, dtype=cp.float32)
    return cp.min(csn.convolve(cp_3d_arr, weights), 0)


def resize(cp_2d_arr, cp_2d_arr_rs):
    [n_rows, n_cols] = cp_2d_arr.shape
    [n_rows_rs, n_cols_rs] = cp_2d_arr_rs.shape
    trans_mat = cp.eye(3)
    trans_mat[0][0] = n_rows_rs / n_rows
    trans_mat[1][1] = n_cols_rs / n_cols
    csn.affine_transform(cp_2d_arr, trans_mat, output_shape=(n_rows_rs, n_cols_rs), output=cp_2d_arr_rs)


def zoom(cp_2d_arr, cp_2d_arr_zm):
    zm_factor = (cp_2d_arr_zm.shape[0] / cp_2d_arr.shape[0], cp_2d_arr_zm.shape[1] / cp_2d_arr.shape[1])
    csn.zoom(cp_2d_arr, zm_factor, cp_2d_arr_zm)


def nd_std(cp_nd_arr, ax):
    n = cp_nd_arr.shape[0]
    return cp.sum(cp.square(cp_nd_arr - cp.sum(cp_nd_arr, axis=ax) / n), axis=ax) / n


def std_threshold(cp_3d_arr, std_map, steps):
    cp_3d_arr_mean = cp.mean(cp_3d_arr, 0)
    cp_3d_arr[cp_3d_arr < cp_3d_arr_mean - steps*std_map] = 0
    return cp_3d_arr


@cp.fuse()
def dff(x2d, bs):
    return cp.divide(x2d - bs, bs + np.finfo(np.float32).eps)


@cp.fuse()
def temporal_cross_corr(x3d, y3d, ptr):
    pass

# def run_preprocesses(cp_2d_arr, processes_list):
#     for process in processes_list:
#         cp_2d_arr = eval(process[0] + "(" + ",".join(process[1:]) + ")")
#
#     return cp_2d_arr


# mem = cp.cuda.BaseMemory(4294967295, (10,10), 0)
# ptr=cp.cuda.MemoryPointer(mem,0)
# a = cp.ndarray((10, 10), memptr=ptr)

from cupy.cuda.memory import MemoryPointer, UnownedMemory
