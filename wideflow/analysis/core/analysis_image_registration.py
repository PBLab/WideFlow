from scipy.ndimage import map_coordinates
import numpy as np
from wideflow.utils.create_matching_points import MatchingPointSelector
from skimage.transform import PiecewiseAffineTransform, warp_coords


def registration(video, match_p_src, match_p_dst, cortex_map, accept=False):
    new_shape = cortex_map.shape
    vid_shape = video.shape
    reference_frame = video[0]

    if accept:
        dst_n_rows, dst_n_cols = cortex_map.shape
        tform = PiecewiseAffineTransform()
        tform.estimate(match_p_src, match_p_dst)
        warp_coor = warp_coords(tform.inverse, (dst_n_rows, dst_n_cols))
        src_cols = np.reshape(warp_coor[0], (dst_n_rows * dst_n_cols, 1))
        src_rows = np.reshape(warp_coor[1], (dst_n_rows * dst_n_cols, 1))

    else:
        mps = MatchingPointSelector(reference_frame, cortex_map * np.random.random(cortex_map.shape),
                                    match_p_src,
                                    match_p_dst,
                                    25)
        src_cols = mps.src_cols
        src_rows = mps.src_rows

    coordinates = np.array([src_cols, src_rows])

    # start registration
    video_reg = np.ndarray((vid_shape[0], new_shape[0], new_shape[1]), dtype=video.dtype)
    for i, frame in enumerate(video):
        video_reg[i] = np.reshape(map_coordinates(frame, coordinates, order=1), new_shape)

    return video_reg
