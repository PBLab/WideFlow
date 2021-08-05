import numpy as np
import matplotlib.pyplot as plt


def paint_roi(rois_dict, cortex_map, rois_names):
    paint_map = np.zeros((cortex_map.shape))
    paint_map[:] = cortex_map[:]
    for roi_name in rois_names:
        roi_pixels_list = rois_dict[roi_name]["PixelIdxList"]
        pixels_inds = np.unravel_index(roi_pixels_list, (cortex_map.shape[1], cortex_map.shape[0]))
        paint_map[pixels_inds[1], pixels_inds[0]] = 1
    plt.figure()
    plt.imshow(paint_map)
    plt.title(' '.join(rois_names))

