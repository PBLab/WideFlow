import numpy as np


rois_dict_left = {}
for i, (key, val) in enumerate(rois_dict.items()):
    if i < len(rois_dict):
        rois_dict_left[key] = val
rois_dict_left_hemi2680 = {}
for roi_key, roi_dict in rois_dict_left.items():
    inds = np.unravel_index(roi_dict['PixelIdxList'], (337, 297))
    inds = np.squeeze(np.array((inds[1], inds[0])))
    pix_inds = np.ravel_multi_index(inds, (297, 168), order='F')

    inds = np.unravel_index(roi_dict['outline'], (337, 297))
    inds = np.squeeze(np.array((inds[1], inds[0])))
    out_inds = np.ravel_multi_index(inds, (297, 168), order='F')

    rois_dict_left_hemi2680[roi_key] = {}
    rois_dict_left_hemi2680[roi_key]['Index'] = roi_dict['Index']
    rois_dict_left_hemi2680[roi_key]['Area'] = roi_dict['Area']
    rois_dict_left_hemi2680[roi_key]['Centroid'] = roi_dict['Centroid']
    rois_dict_left_hemi2680[roi_key]['PixelIdxList'] = pix_inds
    rois_dict_left_hemi2680[roi_key]['outline'] = out_inds
    rois_dict_left_hemi2680[roi_key]['top_left_bottom_rigth'] = roi_dict['top_left_bottom_rigth']
    rois_dict_left_hemi2680[roi_key]['name'] = roi_dict['name']