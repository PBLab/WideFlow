import numpy as np
from scipy import interpolate


def calc_mean_flow(flow):
    '''
    calculate the mean flow, magnitude and direction, of a flow field
    :param flow: 3d or 4d numpy array
                 3d: (time, elements, 2)
                 4d: (time, m, n, 2)
    :return: a list of 2 element tuple of the mean velocity magnitude and direction
    '''
    flow = np.array(flow)
    if np.ndim(flow) > 4 or np.ndim(flow) < 2:
        raise Exception('invalid number of dimention of the "flow" param, must be 2, 3 or 4')
    elif np.ndim(flow) == 2:
        nt, _ = flow.shape
        flow = flow.reshape(nt, 1, 2)
    elif np.ndim(flow) == 4:
        [nt, ny, nx, _] = flow.shape
        flow = flow.reshape(nt, ny * nx, 2)

    mean_flow = np.mean(flow, 1)
    u, v = mean_flow[:, 0], mean_flow[:, 1]
    r = np.sqrt(u ** 2 + v ** 2)
    theta = np.arctan(v / u)
    return r, theta


def sum_vector_field(flow):
    return np.sum(flow, (1, 2))


def average_normal_velocity(flow):
    return np.abs(sum_vector_field(flow)) / sum_vector_field(np.abs(flow))


def get_streamline(uv, pos):
    '''

    :param uv: vector field - numpy array of size (m, n, 2)
    :param pos: initial position [y0, x0]
    :return:
    '''
    m, n, _ = uv.shape
    u, v = uv[:, :, 0], uv[:, :, 1]
    x = np.linspace(0, n, n)
    y = np.linspace(0, m, m)

    u_interp = interpolate.RegularGridInterpolator((x, y), u)
    v_interp = interpolate.RegularGridInterpolator((x, y), v)

    sline = [pos]
    while True:  # might be infinite loop for spiral flow
        pos = [pos[0] + u_interp(pos)[0], pos[1] + v_interp(pos)[0]]
        sline.append(pos)
        if pos[0] <= 0 or pos[0] >= m or pos[1] <= 0 or pos[1] >= n or (pos[0] == sline[-2][0] and pos[1] == sline[-2][1]):
            sline = sline[:-1]
            break

    return np.stack(sline, axis=0)


def get_temporal_streamline(uvt, pos):
    """

    :param uvt: ndarray (nt, m, n, 2)
    :param pos: spatial initial position [y0, x0]
    :return:
    """
    nt, m, n, _ = uvt.shape
    ut, vt = uvt[:, :, :, 0], uvt[:, :, :, 1]
    x = np.linspace(0, n, n, n)
    y = np.linspace(0, m, m, m)
    t = np.linspace(0, nt, nt, nt)

    u_interp = interpolate.RegularGridInterpolator((t, x, y), ut)
    v_interp = interpolate.RegularGridInterpolator((t, x, y), vt)

    pos = pos + [0]
    sline = [pos]
    for i in range(1, nt):
        pos = [pos[0] + u_interp(pos)[0], pos[1] + v_interp(pos)[0], i]
        sline.append(pos)
        if pos[0] <= 0 or pos[0] >= m or pos[1] <= 0 or pos[1] >= n or (pos[0] == sline[-2][0] and pos[1] == sline[-2][1]):
            sline = sline[:-1]
            break

    return np.stack(sline, axis=0)


def get_roi_stream_connectivity(uv, outline, roi_list):
    """
    :param uv: vector field: numpy array of size (m, n, 2)
    :param outline: tuple of two np arrays: (row indices, columns indices) of the roi boundry
    :return:
    """
    streamLines = []
    for pos in outline:
        sline = get_streamline(uv, pos)
        travel_list = find_stramline_travel_list(sline, roi_list)
        streamLines.append(sline)


def find_stramline_travel_list(pos, rois_dict, shape):
    travel_list = []
    pos_flat = np.ravel_multi_index(pos, shape)
    for i, posi in enumerate(pos):
        rois_list = []
        for roi_name, roi_dict in rois_dict.items():  # filtrate rois which the pixel is out of their bounds
            tl_br = roi_dict["top_left_bottom_rigth"]
            if tl_br[0][0] < posi[0] < tl_br[1][0] and tl_br[0][1] < posi[1] < tl_br[1][1]:
                rois_list.append(roi_name)

        for r in rois_list:
            if pos_flat[i] in roi_dict[r]["PixelIdxList"]:
                travel_list.append(roi_dict[r]["Index"])
                break

    return travel_list


def streamline_path_integral(uv, streamline):
    pass


# import pathlib
# from utils.load_matlab_vector_field import load_matlab_OF
# gt_path = str(pathlib.Path('C:/') / 'Users' / 'motar' / 'PycharmProjects' / 'WideFlow' / 'data' / 'ofamm_results.mat')
# gt_flow = load_matlab_OF(gt_path)
# nt, ny, nx, _ = gt_flow.shape
# gt_flowc = gt_flow[:,10:,10:,:]
# tsline = get_temporal_streamline(gt_flowc, [20, 80])