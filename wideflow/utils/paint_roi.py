import numpy as np
import matplotlib.pyplot as plt


def paint_roi(rois_dict, cortex_map, rois_names, rois_vals=None, ax=None):
    if rois_vals is None:
        rois_vals = {}
        for roi_name in rois_names:
            rois_vals[roi_name] = 1

    elif len(rois_names) != len(rois_vals):
        raise Exception('The number of elements in rois_names and rois_vals should be the same')

    paint_map = np.zeros((cortex_map.shape))
    paint_map[:] = cortex_map[:]
    for roi_name in rois_names:
        roi_pixels_list = rois_dict[roi_name]["PixelIdxList"]
        pixels_inds = np.unravel_index(roi_pixels_list, (cortex_map.shape[1], cortex_map.shape[0]))
        paint_map[pixels_inds[1], pixels_inds[0]] = rois_vals[roi_name]
    if ax is None:
        plt.figure()
        plt.imshow(paint_map)
    else:
        return ax.imshow(paint_map)

    # plt.title(' '.join(rois_names))

