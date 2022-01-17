import numpy as np
from scipy.signal import fftconvolve
import cv2


def crop(video, ref_bbox=None, reference_image=None):

    vid_shape = video.shape
    reference_frame = video[0]

    if ref_bbox != None and reference_image is not None:
        ref_shape = reference_image.shape
        if ref_shape[0] != vid_shape[1] or ref_shape[1] != vid_shape[2]:
            resize = True
            reference_frame = cv2.resize(reference_frame, ref_shape)
        else:
            resize = False

        ref_image_roi = reference_image[ref_bbox[2]: ref_bbox[3], ref_bbox[0]: ref_bbox[1]]
        corr = fftconvolve(reference_frame, np.fliplr(np.flipud(ref_image_roi)))
        (yi, xi) = np.unravel_index(np.argmax(corr), corr.shape)
        yi = yi - (corr.shape[0] - reference_frame.shape[0])
        xi = xi - (corr.shape[1] - reference_frame.shape[1])
        bbox = (xi, xi + (ref_bbox[1] - ref_bbox[0]), yi, yi + (ref_bbox[3] - ref_bbox[2]))

    else:
        pass  # choose ROI bbox manualy

    # crop
    video_crop = np.zeros((vid_shape[0], bbox[3] - bbox[2], bbox[1] - bbox[0]), dtype=video.dtype)
    for i, frame in enumerate(video):
        if resize:
            frame = cv2.resize(frame, ref_shape)
        video_crop[i] = frame[bbox[2]: bbox[3], bbox[0]: bbox[1]]

    return video_crop
