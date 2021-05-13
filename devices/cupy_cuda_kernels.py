import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as csn



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
    dims = (5, 1, 1)
    weights = np.ones(dims, dtype=np.float32) / (dims[0]*dims[1]*dims[2])
    weights = cp.asanyarray(weights, dtype=cp.float64)
    return cp.min(csn.convolve(cp_3d_arr, weights), 0)


def resize(cp_2d_arr, cp_2d_arr_rs):
    [n_rows, n_cols] = cp_2d_arr.shape
    [n_rows_rs, n_cols_rs] = cp_2d_arr_rs.shape
    trans_mat = cp.eye(3)
    trans_mat[0][0] = n_rows_rs / n_rows
    trans_mat[1][1] = n_cols_rs / n_cols
    csn.affine_transform(cp_2d_arr, trans_mat, output_shape=(n_rows_rs, n_cols_rs), output=cp_2d_arr_rs)
    return cp_2d_arr_rs


def zoom(cp_2d_arr_zm, cp_2d_arr):
    zm_factor = (cp_2d_arr_zm.shape[0] / cp_2d_arr.shape[0], cp_2d_arr_zm.shape[1] / cp_2d_arr.shape[1])
    csn.zoom(cp_2d_arr, zm_factor, cp_2d_arr_zm)
    return cp_2d_arr_zm


def std_threshold(cp_3d_arr, std_map, steps):
    cp_3d_arr_mean = cp.mean(cp_3d_arr, 0)
    cp_3d_arr_rs = cp_3d_arr
    cp_3d_arr_rs[cp_3d_arr_rs < cp_3d_arr_mean - steps*std_map] = 0
    return cp_3d_arr_rs


def cross_corr(x3d, y3d):
    meux = cp.mean(x3d)
    sigx = cp.std(x3d)
    meuy = cp.mean(y3d)
    sigy = cp.std(y3d)
    return cp.mean(cp.divide(cp.mean(cp.multiply(x3d - meux, y3d - meuy)), cp.multiply(sigx, sigy)))
    # return cp.mean(cp.multiply(x3d, y3d))


def extract_rois_timeseries(x3d, rois_dict, shape):
    rois_time_series = [None] * len(rois_dict)
    for i, roi_dict in enumerate(rois_dict):
        pixels_id_list = roi_dict["PixelIdxList"]
        unravel_idx = np.unravel_index(pixels_id_list, shape=shape)
        rois_time_series[i] = cp.mean(x3d[:, unravel_idx[0], unravel_idx[1]], 1)

    return rois_time_series


def cross_corr_cp(x3d, y3d, corr):
    csn.correlate(x3d, y3d, corr)
    return cp.mean(corr)


@cp.fuse()
def dff(x2d, bs):
    return cp.divide(x2d - bs, bs + np.finfo(np.float32).eps)


# def nd_std(cp_nd_arr, ax):
#     n = cp_nd_arr.shape[0]
#     return cp.sum(cp.square(cp_nd_arr - cp.sum(cp_nd_arr, axis=ax) / n), axis=ax) / n


# def run_preprocesses(cp_2d_arr, processes_list):
#     for process in processes_list:
#         cp_2d_arr = eval(process[0] + "(" + ",".join(process[1:]) + ")")
#
#     return cp_2d_arr


# mem = cp.cuda.BaseMemory(4294967295, (10,10), 0)
# ptr=cp.cuda.MemoryPointer(mem,0)
# a = cp.ndarray((10, 10), memptr=ptr)

from cupy.cuda.memory import MemoryPointer, UnownedMemory
