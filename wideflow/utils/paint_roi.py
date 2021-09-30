import numpy as np
import matplotlib.pyplot as plt


def paint_roi(rois_dict, cortex_map, rois_names, rois_vals=None, ax=None):
    if rois_vals is None:
        rois_vals = [1] * len(rois_names)
    elif len(rois_names) != len(rois_vals):
        raise Exception('The number of elements in rois_names and rois_vals should be the same')

    paint_map = np.zeros((cortex_map.shape))
    paint_map[:] = cortex_map[:]
    for i, roi_name in enumerate(rois_names):
        roi_pixels_list = rois_dict[roi_name]["PixelIdxList"]
        pixels_inds = np.unravel_index(roi_pixels_list, (cortex_map.shape[1], cortex_map.shape[0]))
        paint_map[pixels_inds[1], pixels_inds[0]] = rois_vals[i]
    if ax is None:
        plt.figure()
        plt.imshow(paint_map)
    else:
        ax.imshow(paint_map)
    # plt.title(' '.join(rois_names))

