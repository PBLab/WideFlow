import numpy as np
from wideflow.utils.gen_utils import overlapped_blockshaped


def source_sink_saddle(uv):
    '''

    :param uv: vector field - mxnx2 numpy array
    :return:
    '''
    m, n, _ = uv.shape
    poincare_idx = poincare_index(uv)

    j = jacobian(uv)
    delta = np.multiply(j[:, :, 0, 0], j[:, :, 1, 1]) - np.multiply(j[:, :, 0, 1], j[:, :, 1, 0])
    tau = j[:, :, 0, 0] + j[:, :, 1, 1]
    tau_sqr = tau**2

    sss = np.zeros((m, n))
    spiral = np.zeros((m, n))
    for ind in np.argwhere(poincare_idx == 1):
        sss[ind[0], ind[1]], spiral[ind[0], ind[1]] = characterize_critical_point(tau[ind[0], ind[1]], tau_sqr[ind[0], ind[1]], delta[ind[0], ind[1]])

    critics = {'source': np.where(sss == 1), 'sink': np.where(sss == -1), 'spiral': np.where(sss == 2)}
    spirals = np.where(spiral == 1)
    return critics, spirals


def source_sink_saddle2(uv):
    '''

    :param uv: vector field - mxnx2 numpy array
    :return:
    '''
    pass


def jacobian(uv):
    u = uv[:, :, 0]
    v = uv[:, :, 1]

    ux, uy = np.gradient(u, axis=(0, 1))
    vx, vy = np.gradient(v, axis=(0, 1))
    j = np.array([[ux, uy], [vx, vy]])
    return np.moveaxis(j, [0, 1], [2, 3])


def poincare_index(uv):
    m, n, _ = uv.shape

    direction = np.arctan2(uv[:, :, 1], uv[:, :, 0])
    direction_blocks = overlapped_blockshaped(direction, 2, 2)
    tap = np.zeros((m*n, 4))
    tap[:, 0] = direction_blocks[:, 1, 1] - direction_blocks[:, 1, 0]
    tap[:, 1] = direction_blocks[:, 0, 1] - direction_blocks[:, 1, 1]
    tap[:, 2] = direction_blocks[:, 0, 0] - direction_blocks[:, 0, 1]
    tap[:, 3] = direction_blocks[:, 1, 0] - direction_blocks[:, 0, 0]
    tap[tap <= -np.pi / 2] = tap[tap <= -np.pi / 2] + np.pi
    tap[tap > np.pi / 2] = tap[tap > np.pi / 2] - np.pi

    poincare_idx = np.sum(tap, 1) / np.pi
    poincare_idx = np.reshape(poincare_idx, (m, n))
    source_sink_saddle = np.zeros((m, n))
    source_sink_saddle[poincare_idx > 0.9] = 1
    source_sink_saddle[poincare_idx < -0.9] = 1

    return source_sink_saddle


def characterize_critical_point(tau, tau_sqr, delta):
    if delta < 0:
        sss = 0
        spiral = 0
    if delta > 0:
        if tau > 0:
            sss = 1
            if tau_sqr < 4 * delta:
                spiral = 1
            else:
                spiral = 0
        elif tau < 0:
            sss = -1
            if tau_sqr < 4 * delta:
                spiral = 1
            else:
                spiral = 0
        else:
            sss = 2
            spiral = 1
    else:
        sss = None
        spiral = None

    return sss, spiral


def bilinear_intersection(uv):
    pass



import pathlib
from utils.load_matlab_vector_field import load_matlab_OF
import matplotlib.pyplot as plt
from utils.load_tiff import load_tiff
from utils.plot_optical_flow import plot_optical_flow

# # vid_path = str(pathlib.Path('C:/') / 'Users' / 'motar' / 'PycharmProjects' / 'WideFlow' / 'data' / 'OFAMM' / 'ImgSeq.tif')
# vid_path = str(pathlib.Path('C:/') / 'Users' / 'motar' / 'PycharmProjects' / 'WideFlow' / 'data' / 'widefield_fast_acq' / 'wf_spont_crystal_skull1000_1200.tif')
# vid = load_tiff(vid_path)
# # gt_path = str(pathlib.Path('C:/') / 'Users' / 'motar' / 'PycharmProjects' / 'WideFlow' / 'data' / 'OFAMM' / 'ofamm_results.mat')
# gt_path = str(pathlib.Path('C:/') / 'Users' / 'motar' / 'PycharmProjects' / 'WideFlow' / 'data' / 'widefield_fast_acq' / 'uvResults.mat')
# gt_flow = load_matlab_OF(gt_path)
#
# sss = []
# spiral = []
# for uv in gt_flow:
#     # uv30 = gt_flow[30, :, :, :]
#     s, sp = source_sink_saddle(uv)
#     sss.append(s)
#     spiral.append(sp)
#
# frame = 32
# plot_optical_flow(vid, gt_flow, frame, 100, 3)
# z=3
