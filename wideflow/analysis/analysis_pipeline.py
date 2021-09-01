from  analysis.core import *
from analysis.utils import *
from utils.load_tiff import load_tiff
from utils.load_bbox import load_bbox
from utils.load_matching_points import load_matching_points
from utils.load_matlab_vector_field import load_extended_rois_list

import os
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt

# load cortex data
cortex_file_path = "C:\\Users\\motar\\PycharmProjects\\WideFlow\\data\\cortex_map\\allen_2d_cortex.h5"
with h5py.File(cortex_file_path, 'r') as f:
    cortex_mask = np.transpose(f["mask"][()])
    cortex_map = np.transpose(f["map"][()])

rois_dict_path = "C:\\Users\\motar\\PycharmProjects\\WideFlow\\data\\cortex_map\\allen_2d_cortex_rois_extended.h5"
rois_dict = load_extended_rois_list(rois_dict_path)

# load session data
register = False
dir_path = 'Z:/Rotem/WideFlow prj/3422/01082021/'
metadata, config = load_session_metadata(dir_path)

n_channels = config["camera_config"]["attr"]["channels"]
dff_bs_n_frames = config["acquisition_config"]["capacity"]

# load reference image, bbox, and matching points
reference_dir_path = 'Z:/Rotem/WideFlow prj/3422/'
reference_image = load_tiff(reference_dir_path + 'reference_image.tif')
ref_bbox = load_bbox(reference_dir_path + 'bbox.txt')
match_p_src, match_p_dst = load_matching_points(reference_dir_path + 'matching_points.txt')
match_p_src = np.array(match_p_src)
match_p_dst = np.array(match_p_dst)

# run analysis by parts to avoid memory overflow
concat_rois_traces = {}
for i in range(n_channels):
    concat_rois_traces[f'channel_{i}'] = {}
    for key in rois_dict:
        concat_rois_traces[f'channel_{i}'][key] = np.empty((0, ), dtype=np.float16)

wf_video_paths = []
for file in os.listdir(dir_path):
    if file.endswith(".tif"):
        wf_video_paths.append(os.path.join(dir_path, file))

for p, tif_path in enumerate(wf_video_paths):
    print(f"starting analysis for tiff part: {p}")
    wf_data = load_tiff(tif_path)
    if p != 0:
        wf_data = np.concatenate((wf_data_remainder, wf_data), axis=0)
    wf_data_remainder = wf_data[-dff_bs_n_frames:, :, :]

    # register
    if register:
        wf_data = registration(wf_data, reference_image, ref_bbox, match_p_src, match_p_dst, cortex_map)
    else:
        temp = np.ndarray((wf_data.shape[0], cortex_mask.shape[0], cortex_mask.shape[1]), dtype=wf_data.dtype)
        for i, frame in enumerate(wf_data):
            temp[i] = cv2.resize(frame, (cortex_mask.shape[1], cortex_mask.shape[0]))
        wf_data = temp
        del temp

    # masking
    wf_data = mask(wf_data, cortex_mask)

    # deinterleave
    wf_data = deinterleave(wf_data, n_channels)

    wf_data['channel_0'] = wf_data['channel_0'][
        np.mean(wf_data['channel_0'], axis=(1, 2)) > 12000]  # TODO: only for strobing failure
    # dff
    wf_data = calc_dff(wf_data, "t_prj", dff_bs_n_frames)
    # remove remainder
    if p != 0:
        for ch in wf_data:
            wf_data[ch] = wf_data[ch][dff_bs_n_frames:, :, :]

    # hemodynamics attenuation
    hemodynamics_attenuation()

    # ROIs traces extraction
    rois_traces = extract_roi_traces(wf_data, rois_dict, cortex_mask.shape)

    # concatenate to previous data
    for ch, ch_rois_traces in rois_traces.items():
        for roi, trace in ch_rois_traces.items():
            concat_rois_traces[ch][roi] = np.concatenate((concat_rois_traces[ch][roi], trace))

# save rois traces
print("saving results")
with h5py.File(dir_path + 'session_ROIs_traces_Tprj.h5', 'w') as f:
    for ch, ch_traces_dict in concat_rois_traces.items():
        f.create_group(ch)
        for roi, trace in ch_traces_dict.items():
            f.create_dataset(f'{ch}/{roi}', data=trace)
