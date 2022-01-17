import numpy as np
from tifffile import TiffWriter
from skimage.transform import resize


def traces_to_video(traces, rois_dict, shape, path):
    n = len(traces['roi_1'])
    video = np.zeros((n, shape[0], shape[1]), dtype=np.float32)
    for roi_key, roi_trace in traces.items():
        roi_pixels_list = rois_dict[roi_key]["PixelIdxList"]
        pixels_inds = np.unravel_index(roi_pixels_list, (shape[1], shape[0]))
        for i in range(n):
            video[i, pixels_inds[1], pixels_inds[0]] = traces[roi_key][i]

    video_rs = np.zeros((n, int(shape[0]/4), int(shape[1]/4)), dtype=np.float32)
    for i in range(n):
        video_rs[i] = resize(video[i], output_shape=video_rs.shape[-2:])
    del video

    with TiffWriter(path) as tif:
        tif.write(video_rs, contiguous=True)