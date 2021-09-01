from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
import numpy as np
import cv2

from Imaging.utils.create_matching_points import MatchingPointSelector, select_matching_points


def registration(video, reference_image, ref_bbox, match_p_src, match_p_dst, cortex_map):
    new_shape = cortex_map.shape
    ref_shape = reference_image.shape
    vid_shape = video.shape
    reference_frame = video[0]
    if ref_shape[0] != vid_shape[1] or ref_shape[1] != vid_shape[2]:
        resize = True
        reference_frame = cv2.resize(reference_frame, ref_shape)
    else:
        resize = False

    # find video roi and registration coordinates
    ref_image_roi = reference_image[ref_bbox[2]: ref_bbox[3], ref_bbox[0]: ref_bbox[1]]
    corr = fftconvolve(reference_frame, np.fliplr(np.flipud(ref_image_roi)))
    (yi, xi) = np.unravel_index(np.argmax(corr), corr.shape)
    yi = yi - (corr.shape[0] - reference_frame.shape[0])
    xi = xi - (corr.shape[1] - reference_frame.shape[1])
    bbox = (xi, xi + (ref_bbox[1] - ref_bbox[0]), yi, yi + (ref_bbox[3] - ref_bbox[2]))
    reference_frame_roi = reference_frame[bbox[2]: bbox[3], bbox[0]: bbox[1]]

    mps = MatchingPointSelector(reference_frame_roi, cortex_map * np.random.random(cortex_map.shape),
                                match_p_src,
                                match_p_dst,
                                25)

    src_cols = mps.src_cols
    src_rows = mps.src_rows
    coordinates = np.array([src_cols, src_rows])

    # start registration
    video_reg = np.ndarray((vid_shape[0], new_shape[0], new_shape[1]), dtype=video.dtype)
    for i, frame in enumerate(video):
        if resize:
            frame = cv2.resize(frame, ref_shape)
        frame_roi = frame[bbox[2]: bbox[3], bbox[0]: bbox[1]]
        video_reg[i] = np.reshape(map_coordinates(frame_roi, coordinates, order=1), new_shape)

    return video_reg