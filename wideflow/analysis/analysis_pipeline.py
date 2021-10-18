from wideflow.analysis.core import *
from wideflow.analysis.utils import *
from wideflow.utils.load_tiff import load_tiff
from wideflow.utils.load_bbox import load_bbox
from wideflow.utils.load_matching_points import load_matching_points
from wideflow.utils.load_matlab_vector_field import load_extended_rois_list
from wideflow.utils.decompose_dict_and_h5_groups import decompose_dict_to_h5_groups

import os
import h5py
import numpy as np
import cv2

# project path
project_path = '/data/Rotem/WideFlow prj/'
mouse_id = '3422'
session_name = '20211014_nf'


# analysis global parameters

crop = False
register = True

dff_bs_method = "moving_avg"
accept_transform_matching_points = False
hemo_correct_ch = ["channel_0", "channel_1"]
global_params = {
    "crop": crop,
    "register": register,
    "dff_bs_method": dff_bs_method,
    "accept_transform_matching_points": accept_transform_matching_points,
    "hemo_correct_ch": hemo_correct_ch
}

# load cortex data
cortex_file_path = os.path.abspath(os.path.join(os.path.pardir, '../', 'data', 'cortex_map', 'allen_2d_cortex.h5'))
with h5py.File(cortex_file_path, 'r') as f:
    cortex_mask = np.transpose(f["mask"][()])
    cortex_map = np.transpose(f["map"][()])

rois_dict_path = os.path.abspath(os.path.join(os.path.pardir, '../', 'data', 'cortex_map', 'allen_2d_cortex_rois_extended.h5'))
rois_dict = load_extended_rois_list(rois_dict_path)

# load session metadata and configurations
session_path = project_path + mouse_id + '/' + session_name + '/'
metadata, config = load_session_metadata(session_path)
if os.path.exists(session_path + 'regression_coeff_map.npy'):
    regression_coeff_map = np.load(session_path + "regression_coeff_map.npy")
elif os.path.exists(project_path + mouse_id + '/hemodynamics_regression_map.npy'):
    regression_coeff_map = np.load(project_path + mouse_id + "/hemodynamics_regression_map.npy")
else:
    regression_coeff_map = None

n_channels = config["camera_config"]["attr"]["channels"]
dff_bs_n_frames = config["acquisition_config"]["capacity"]

# load reference image, bbox, and matching points
reference_dir_path = project_path + mouse_id + '/'
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
for file in os.listdir(session_path):
    if file.endswith(".tif"):
        wf_video_paths.append(os.path.join(session_path, file))

wf_video_paths = sort_video_path_list(wf_video_paths)
for p, tif_path in enumerate(wf_video_paths):
    print(f"starting analysis for tiff part: {p}")
    wf_data = load_tiff(tif_path)
    if p != 0:
        wf_data = np.concatenate((wf_data_remainder, wf_data), axis=0)
    wf_data_remainder = wf_data[-dff_bs_n_frames*n_channels:, :, :]

    # crop
    if crop:
        wf_data = crop(wf_data, ref_bbox, reference_image)

    # register
    if register:
        wf_data = registration(wf_data, match_p_src, match_p_dst, cortex_map, accept_transform_matching_points)
        accept_transform_matching_points = True
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

    # dff
    wf_data = calc_dff(wf_data, dff_bs_method, dff_bs_n_frames)
    # remove remainder
    if p != 0:
        for ch in wf_data:
            wf_data[ch] = wf_data[ch][dff_bs_n_frames:, :, :]

    # hemodynamics attenuation
    if hemo_correct_ch != None:
        regression_coeff_map = hemodynamics_attenuation(wf_data, regression_coeff_map, hemo_correct_ch, dff_bs_n_frames)

    # ROIs traces extraction
    rois_traces = extract_roi_traces(wf_data, rois_dict, cortex_mask.shape)

    # concatenate to previous data
    for ch, ch_rois_traces in rois_traces.items():
        for roi, trace in ch_rois_traces.items():
            concat_rois_traces[ch][roi] = np.concatenate((concat_rois_traces[ch][roi], trace))

# results statistics
neuronal_response_stats, behavioral_response_stats, statistics_global_params = \
    analysis_statistics(concat_rois_traces, metadata, config)

# save rois traces
with h5py.File(project_path + 'results/' + 'sessions_dataset.h5', 'a') as f:
    main_group = f[mouse_id]
    session_group = main_group.create_group(session_name)  # create group named by session name

    data_group = session_group.create_group('rois_traces')
    for ch, ch_traces_dict in concat_rois_traces.items():
        ch_group = data_group.create_group(ch)
        for roi, trace in ch_traces_dict.items():
            ch_group.create_dataset(roi, data=trace)

    stats_group = session_group.create_group('statistics')
    behavioral_stats_group = stats_group.create_group('behavioral_response')
    decompose_dict_to_h5_groups(f, behavioral_response_stats, behavioral_stats_group.name + '/')
    neuronal_stats_group = stats_group.create_group('neuronal_response')
    decompose_dict_to_h5_groups(f, neuronal_response_stats, neuronal_stats_group.name + '/')
    glob_param_stats_group = stats_group.create_group('global_parameters')
    decompose_dict_to_h5_groups(f, statistics_global_params, glob_param_stats_group.name + '/')

    session_group.create_dataset('regression_coeff_map', data=regression_coeff_map)


if not os.path.isdir(session_path + 'analysis_results'):
    os.mkdir(session_path + 'analysis_results')
plot_figures(session_path + 'analysis_results/', metadata, config, concat_rois_traces,
             neuronal_response_stats, behavioral_response_stats, statistics_global_params, rois_dict)


