import numpy as np


def calc_rois_proximity(rois_dict, metric_roi):
    metric_roi_center = rois_dict[metric_roi]["Centroid"]
    rois_proximity = {}
    for roi_key, roi_dict in rois_dict.items():
        roi_center = roi_dict["Centroid"]
        dist = np.sqrt((metric_roi_center[0] - roi_center[0]) ** 2 + (metric_roi_center[1] - roi_center[1]) ** 2)
        rois_proximity[roi_key] = dist
        #rois_proximity[roi_key] = 0.029*dist #added by LK to convert to mm
    return rois_proximity
