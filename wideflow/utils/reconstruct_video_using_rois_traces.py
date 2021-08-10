import numpy as np


def reconstruct_video_using_rois_traces(rois_traces, rois_dict, shape):
    n_frames = len(rois_traces[[*rois_traces][0]])
    recon_vid = np.zeros((n_frames, shape[0], shape[1]), dtype=rois_traces[[*rois_traces][0]].dtype)
    for i in range(n_frames):
        for roi_name, roi_dict in rois_dict.items():
            roi_pixels_list = roi_dict["PixelIdxList"]
            pixels_inds = np.unravel_index(roi_pixels_list, (shape[1], shape[0]))
            recon_vid[i, pixels_inds[1], pixels_inds[0]] = rois_traces[roi_name][i]

    return recon_vid

