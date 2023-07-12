import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import numpy as np
from scipy.stats import linregress, ttest_rel, wilcoxon
import h5py

from utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict
from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
from analysis.utils.rois_proximity import calc_rois_proximity
from utils.load_config import load_config
from utils.load_rois_data import load_rois_data

from analysis.plots import *
from analysis.utils.peristimulus_time_response import calc_pstr
from scipy.ndimage.filters import maximum_filter1d

from scipy.signal import find_peaks, peak_prominences, peak_widths, savgol_filter
from Imaging.utils.numba_histogram import numba_histogram

def percentile_update_procedure(threshold, samples, percentile, nbins):
    hist, bins = numba_histogram(samples, nbins, density=True)
    bins_width = np.diff(bins)
    p_density = hist * bins_width
    prob = np.cumsum(p_density)
    percentile_inds = np.where(prob > np.percentile(prob, percentile))[0]
    if len(percentile_inds):
        return bins[percentile_inds[0]], p_density, bins, prob
    else:
        return threshold, p_density, bins, prob



base_path = '/data/Lena/WideFlow_prj'
dataset_path = f'{base_path}/Results/results_exp2.h5'
#sessions_xxx
#base_path = '/data/Rotem/WideFlow prj'
#dataset_path = f'{base_path}/results/sessions_20220320.h5'

mice_id = ['21ML', '21ML','21ML']
#mice_id = ['2604', '2604','2604']
sessions_names = [
    # '20221128_MNL_NF1',
    # #'20221129_MR_NF2',
    # '20221130_MNL_NF3',
    # '20221201_MNL_NF4',
    # '20221202_MNL_NF5',
    # '20221218_MNL_NF6',
    # '20221219_MNL_NF7',
    # '20221221_MNL_NF8',
    # '20221222_MNL_NF9',
    # '20221223_MNL_NF10',
    # '20221224_MNL_NF11',
    # '20230102_MNL_NF12',
    # '20230103_MNL_NF13',
    # '20230104_MNL_NF14',
    # '20230109_MNL_NF15',
    # '20230116_MNL_NF16',
    # '20230117_MNL_NF17'
    # '20230118_MNL_NF18_tiffs',
    # '20230119_MNL_NF19_tiffs',
    # '20230122_MNL_NF20_tiffs',
    # '20230123_MR_NF21',
    # '20230129_MR_NF22',
    # '20230201_MR_NF23'
    # '20230122_FL_NF4',
    # '20230123_FL_NF5',
    # '20230124_FL_NF6',
    # '20230129_FL_NF7',
    # '20230131_FL_NF8',
    # '20230201_FL_NF9'
    # '20220320_neurofeedback',
    # '20220321_neurofeedback',
    # '20220322_neurofeedback',
    # '20220323_neurofeedback',
    # '20220324_neurofeedback'
    #'MR_NF18_tiffs',
    # 'MR_NF19_tiffs',
    # 'MNL_NF21_tiffs',
    # 'MNL_NF22_tiffs',
    # 'MNL_NF23_tiffs'
    # '20230120_FL_NF3_tiffs3',
    # 'FL_NF4_tiffs',
    # 'FL_NF5_tiffs',
    # 'FL_NF6_tiffs',
    # 'FL_NF7_tiffs',
    # 'FL_NF8_tiffs',
    # 'FL_NF9_tiffs'
    # '20221202_MNL_NF5',
    # '20221221_MNL_NF8',
    # '20221223_MNL_NF10',
    # '20230102_MNL_NF12',
    # '20230103_MNL_NF13',
    # '20230104_MNL_NF14',
    # '20230116_MNL_NF16',
    # '20230117_MNL_NF17'
    '20221128_MR_NF1_tiffs',
    '20221129_MR_NF2_tiffs',
    '20221130_MR_NF3_tiffs',
    '20221201_MR_NF4_tiffs',
    '20221218_MR_NF6_tiffs',
    '20221219_MR_NF7_tiffs',
    '20221222_MR_NF9_tiffs',
    '20221223_MR_NF10_tiffs',
    '20230102_MR_NF12_tiffs',
    '20230103_MR_NF13_tiffs',
    '20230104_MR_NF14_tiffs',
    '20230116_MR_NF16_tiffs'

]
start_frame = 0

sessions_data = {}
sessions_metadata = {}
sessions_config = {}
sup_data = {}
for mouse_id in mice_id:
    # load sessions data __________________________________________________________
    sessions_data[mouse_id] = {}
    with h5py.File(dataset_path, 'a') as f:
        decompose_h5_groups_to_dict(f, sessions_data[mouse_id], f'/{mouse_id}/')

    # load sessions supplementary data __________________________________________________________
    cortex_map_path = f'{base_path}/{mouse_id}/functional_parcellation_cortex_map.h5'
    rois_dict_path = f'{base_path}/{mouse_id}/functional_parcellation_rois_dict.h5'
    # cortex_map_path = f'{base_path}/{mouse_id}/functional_parcellation_cortex_map.h5'
    # rois_dict_path = f'{base_path}/{mouse_id}/functional_parcellation_rois_dict_left_hemi.h5'
    #20221122_{mouse_id}_CRC3functional_parcellation_cortex_map.h5
    #20221122_{mouse_id}_CRC3functional_parcellation_rois_dict.h5
    #FLfunctional_parcellation_cortex_map_CRC3.h5
    #FLfunctional_parcellation_rois_dict_CRC3.h5
    rois_dict = load_rois_data(rois_dict_path)
    sup_data[mouse_id] = {}
    with h5py.File(cortex_map_path, 'r') as f:
        cortex_mask = f["mask"][()]
        cortex_map = f["map"][()]
    cortex_mask = cortex_mask[:, :168]
    cortex_mask[cortex_mask == 0] = None
    cortex_map = cortex_map[:, :168]
    cortex_map[cortex_map == 0] = None
    sup_data[mouse_id]['cortex_map'] = cortex_map
    sup_data[mouse_id]['cortex_mask'] = cortex_mask
    sup_data[mouse_id]['rois_dict'] = rois_dict

    sessions_metadata[mouse_id] = {}
    sessions_config[mouse_id] = {}
    for sess_name in sessions_names:
        # load sessions config __________________________________________________________
        config = load_config(f'{base_path}/{mouse_id}/{sess_name}/session_config.json')
        sessions_config[mouse_id][sess_name] = config

        # load sessions metadata __________________________________________________________
        [timestamp, cue, metric_result, threshold, serial_readout] = extract_from_metadata_file(
            f'{base_path}/{mouse_id}/{sess_name}/metadata.txt')
        # f_start = 0
        # f_finish = 70000
        timestamp = timestamp[start_frame:]
        cue = cue[start_frame:]
        metric_result = metric_result[start_frame:]
        threshold = threshold[start_frame:]
        serial_readout = serial_readout[start_frame:]
        dt = np.mean(np.diff(timestamp))
        serial_readout = 1 - np.array(serial_readout)
        serial_readout = maximum_filter1d(serial_readout, 2)[::2]
        cue = maximum_filter1d(cue, 2)[::2]

        metric_roi = config['analysis_pipeline_config']['args']['metric_args'][1][0]
        metric_roi_idx = int(metric_roi[-2:]) - 1
        metric_bool_arr = [True if metric_roi == key else False for key in rois_dict.keys()]
        non_metric_bool_arr = np.invert(metric_bool_arr)
        rois_proximity = calc_rois_proximity(rois_dict, metric_roi)

        sessions_metadata[mouse_id][sess_name] = {
            "timestamp": timestamp, "cue": cue, "metric_result": metric_result,
            "threshold": threshold, "serial_readout": serial_readout, "metric_roi": metric_roi,
            "metric_bool_arr": metric_bool_arr, "non_metric_bool_arr": non_metric_bool_arr,
            "rois_proximity": rois_proximity}

    # calculate pstr max __________________________________________________________
    #b = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,25,26] #LK indecies to exclude reward number 14 because it takes over the whole graph
    #b = 26

    pstr_zscore_max_nm, pstr_zscore_max_m = [], []
    for i, sess_name in enumerate(sessions_names):
        pstr_zscore_max_m.append([])
        pstr_zscore_max_nm.append([])
        for j, roi_key in enumerate(rois_dict.keys()):
            if roi_key == sessions_metadata[mouse_id][sess_name]['metric_roi']:
                pstr_zscore_max_m[i].append(np.max(
                    sessions_data[mouse_id][sess_name]['post_session_analysis']['dff_delta5'][
                        'zscore_pstr_sim_reward_fix_th'][
                        j]))
                # pstr_zscore_max_m[i].append(np.max(
                #     sessions_data[mouse_id][sess_name]['post_session_analysis']['dff_delta5'][
                #         'zscore_pstr_sim_reward_fix_th'][
                #         j,b]))
            else:
                pstr_zscore_max_nm[i].append(np.max(
                    sessions_data[mouse_id][sess_name]['post_session_analysis']['dff_delta5'][
                        'zscore_pstr_sim_reward_fix_th'][
                        j]))
                # pstr_zscore_max_nm[i].append(np.max(
                #     sessions_data[mouse_id][sess_name]['post_session_analysis']['dff_delta5'][
                #         'zscore_pstr_sim_reward_fix_th'][
                #         j,b]))

        x = np.expand_dims(np.arange(len(sessions_names)).transpose(), axis=1)
        max_vals = {}
        max_slope = {}
        max_score = {}
        max_pval = {}
        for j, key in enumerate(rois_dict.keys()):
            mx = []
            for sess_name in sessions_names:
                mx.append(np.max(
                    sessions_data[mouse_id][sess_name]['post_session_analysis']['dff_delta5'][
                        'zscore_pstr_sim_reward_fix_th'][
                        j]))
            max_vals[key] = mx

            reg = linregress(x.flatten(), np.array([max_vals[key]]).flatten())
            max_slope[key] = reg.slope
            max_score[key] = reg.rvalue ** 2
            max_pval[key] = reg.pvalue


    sessions_data[mouse_id]["pstr_zscore_max_nm"] = np.transpose(np.array(pstr_zscore_max_nm))
    sessions_data[mouse_id]["pstr_zscore_max_m"] = np.transpose(np.array(pstr_zscore_max_m))
    sessions_data[mouse_id]["max_slope"] = max_slope
    sessions_data[mouse_id]["max_score"] = max_score
    sessions_data[mouse_id]["max_pval"] = max_pval

    # calculate number of rewards __________________________________________________________
    num_of_rewards_nm, num_of_rewards_m = [], []
    for i, sess_name in enumerate(sessions_names):
        num_of_rewards_m.append([])
        num_of_rewards_nm.append([])
        for roi_key, roi_val in sessions_data[mouse_id][sess_name]['post_session_analysis']['dff_delta5'][
            'zscore_sim_reward_timing_fix_th'].items():
            if roi_key == sessions_metadata[mouse_id][sess_name]['metric_roi']:
                num_of_rewards_m[i].append(len(roi_val))
            else:
                num_of_rewards_nm[i].append(len(roi_val))

    rewards_counts = {}
    reward_slope = {}
    reward_score = {}
    reward_pval = {}
    for key in rois_dict.keys():
        rewards_counts[key] = []
        for sess_name in sessions_names:
            rewards_counts[key].append(len(
                sessions_data[mouse_id][sess_name]['post_session_analysis']['dff_delta5'][
                    'zscore_sim_reward_timing_fix_th'][
                    key]))

        reg = linregress(x.flatten(), np.array([rewards_counts[key]]).flatten())
        reward_slope[key] = reg.slope
        reward_score[key] = reg.rvalue ** 2
        reward_pval[key] = reg.pvalue

    sessions_data[mouse_id]["num_of_rewards_nm"] = np.transpose(np.array(num_of_rewards_nm))
    sessions_data[mouse_id]["num_of_rewards_m"] = np.transpose(np.array(num_of_rewards_m))
    sessions_data[mouse_id]["reward_slope"] = reward_slope
    sessions_data[mouse_id]["reward_score"] = reward_score
    sessions_data[mouse_id]["reward_pval"] = reward_pval

    # calculate threshold crossing probability __________________________________________________________
    p_th_m = []
    p_th_nm = []
    traces_p_th_m = []
    traces_p_th_nm = []
    metric_dens_m = []
    metric_dens_nm = []
    metric_th_idx_m = []
    metric_th_idx_nm = []
    metric_bins_m = []
    metric_bins_nm = []
    traces_dens_m = []
    traces_dens_nm = []
    traces_th_idx_m = []
    traces_th_idx_nm = []
    traces_bins_m = []
    traces_bins_nm = []

    # threshold = sessions_data[mouse_id]['20221128_MNL_NF1']['post_session_analysis']['analysis_parameters'][
    #     'fixed_threshold']
    threshold = 2.8
    dff_threshold = 0
    percentile = 95
    nbins = 100
    metric_dens = []
    metric_bins = []
    traces_dens = []
    traces_bins = []
    for sess_name in sessions_names:
        metric_bool_arr = sessions_metadata[mouse_id][sess_name]['metric_bool_arr']
        non_metric_bool_arr = np.invert(metric_bool_arr)

        metric = sessions_data[mouse_id][sess_name]['post_session_analysis']['dff_delta5']['zscore']#[:,f_start//2:f_finish//2]
        metric_filter = savgol_filter(metric, window_length=7, polyorder=2)
        traces = sessions_data[mouse_id][sess_name]['post_session_analysis']['dff']['traces']#[:,f_start//2:f_finish//2]
        traces_filter = savgol_filter(traces, window_length=7, polyorder=2)
        # peaks_prom[sess_name] = []

        p_th = np.ndarray(traces_filter.shape[0])
        traces_p_th = np.ndarray(traces_filter.shape[0])
        metric_dens = []
        metric_bins = []
        metric_th_idx = np.ndarray(traces_filter.shape[0])
        traces_dens = []
        traces_bins = []
        traces_th_idx = np.ndarray(traces_filter.shape[0])
        auc = np.ndarray(traces_filter.shape[0])
        for i, (trace, metric) in enumerate(zip(traces_filter, metric_filter)):
            _, p_density, bins, prob = percentile_update_procedure(threshold, metric, percentile, nbins)
            metric_dens.append(p_density)
            metric_bins.append(bins)
            idx = np.where(bins > threshold)[0]
            if len(idx):
                p_th[i] = 1 - prob[idx[0] - 1]
                metric_th_idx[i] = (idx[0] - 1)
            else:
                p_th[i] = 0
                metric_th_idx[i] = 99


            _, p_density, bins, prob = percentile_update_procedure(0, trace, percentile, nbins)
            traces_dens.append(p_density)
            traces_bins.append(bins)
            idx = np.where(bins > dff_threshold)[0]
            if len(idx):
                traces_p_th[i] = 1 - prob[idx[0] - 1]
                traces_th_idx[i] = (idx[0] - 1)
            else:
                traces_p_th[i] = 0
                traces_th_idx[i] = 99


        p_th_m.append(p_th[metric_bool_arr])
        p_th_nm.append(p_th[non_metric_bool_arr])
        traces_p_th_m.append(traces_p_th[metric_bool_arr])
        traces_p_th_nm.append(traces_p_th[non_metric_bool_arr])

        metric_dens_m.append(np.array(metric_dens)[metric_bool_arr])
        metric_dens_nm.append(np.array(metric_dens)[non_metric_bool_arr])
        metric_th_idx_m.append(metric_th_idx[metric_bool_arr])
        metric_th_idx_nm.append(metric_th_idx[non_metric_bool_arr])
        metric_bins_m.append(np.array(metric_bins)[metric_bool_arr])
        metric_bins_nm.append(np.array(metric_bins)[non_metric_bool_arr])

        traces_dens_m.append(np.array(traces_dens)[metric_bool_arr])
        traces_dens_nm.append(np.array(traces_dens)[non_metric_bool_arr])
        traces_th_idx_m.append(traces_th_idx[metric_bool_arr])
        traces_th_idx_nm.append(traces_th_idx[non_metric_bool_arr])
        traces_bins_m.append(np.array(traces_bins)[metric_bool_arr])
        traces_bins_nm.append(np.array(traces_bins)[non_metric_bool_arr])

    p_th_slope = {}
    p_th_score = {}
    p_th_pval = {}
    y = np.ndarray((len(rois_dict), len(sessions_names)))
    y[metric_bool_arr, :] = np.array(p_th_m).transpose()
    y[non_metric_bool_arr, :] = np.array(p_th_nm).transpose()

    traces_p_th_slope = {}
    traces_p_th_score = {}
    traces_p_th_pval = {}
    traces_y = np.ndarray((len(rois_dict), len(sessions_names)))
    traces_y[metric_bool_arr, :] = np.array(traces_p_th_m).transpose()
    traces_y[non_metric_bool_arr, :] = np.array(traces_p_th_nm).transpose()
    for j, key in enumerate(rois_dict.keys()):
        reg = linregress(x.flatten(), y[j].flatten())
        p_th_slope[key] = reg.slope
        p_th_score[key] = reg.rvalue ** 2
        p_th_pval[key] = reg.pvalue

        reg = linregress(x.flatten(), traces_y[j].flatten())
        traces_p_th_slope[key] = reg.slope
        traces_p_th_score[key] = reg.rvalue ** 2
        traces_p_th_pval[key] = reg.pvalue

    sessions_data[mouse_id]["p_th_m"] = np.array(np.transpose(p_th_m))
    sessions_data[mouse_id]["p_th_nm"] = np.array(np.transpose(p_th_nm))
    sessions_data[mouse_id]["p_th_slope"] = p_th_slope
    sessions_data[mouse_id]["p_th_score"] = p_th_score
    sessions_data[mouse_id]["p_th_pval"] = p_th_pval

    sessions_data[mouse_id]["traces_p_th_m"] = np.array(np.transpose(traces_p_th_m))
    sessions_data[mouse_id]["traces_p_th_nm"] = np.array(np.transpose(traces_p_th_nm))
    sessions_data[mouse_id]["traces_p_th_slope"] = traces_p_th_slope
    sessions_data[mouse_id]["traces_p_th_score"] = traces_p_th_score
    sessions_data[mouse_id]["traces_p_th_pval"] = traces_p_th_pval

    sessions_data[mouse_id]["metric_dens_m"] = metric_dens_m
    sessions_data[mouse_id]["metric_dens_nm"] = metric_dens_nm
    sessions_data[mouse_id]["metric_th_idx_m"] = metric_th_idx_m
    sessions_data[mouse_id]["metric_th_idx_nm"] = metric_th_idx_nm
    sessions_data[mouse_id]["metric_bins_m"] = metric_bins_m
    sessions_data[mouse_id]["metric_bins_nm"] = metric_bins_nm

    sessions_data[mouse_id]["traces_dens_m"] = traces_dens_m
    sessions_data[mouse_id]["traces_dens_nm"] = traces_dens_nm
    sessions_data[mouse_id]["traces_th_idx_m"] = traces_th_idx_m
    sessions_data[mouse_id]["traces_th_idx_nm"] = traces_th_idx_nm
    sessions_data[mouse_id]["traces_bins_m"] = traces_bins_m
    sessions_data[mouse_id]["traces_bins_nm"] = traces_bins_nm

    # correlation calculations _______________________________________________________________________________
    for sess_name in sessions_names:
        metric_pstr_cat, metric_pstr_avg, metric_pstr_ful = {}, {}, {}
        d5_pstr_cat, d5_pstr_avg, d5_pstr_ful = {}, {}, {}
        dff_pstr_cat, dff_pstr_avg, dff_pstr_full = {}, {}, {}
        delta_t = int(
            sessions_data[mouse_id][sess_name]['post_session_analysis']['dff_delta5']['zscore_pstr'].shape[1] / 2)
        for i, key in enumerate(rois_dict.keys()):
            # metric pstr
            pstr_mat = calc_pstr(sessions_metadata[mouse_id][sess_name]['cue'],
                                 sessions_data[mouse_id][sess_name]['post_session_analysis']['dff_delta5']['zscore'][i],
                                 delta_t)
            pstr_mat = pstr_mat[:, :int(pstr_mat.shape[1] / 2) + 1]  # calculate based on pre-stim only
            metric_pstr_avg[key] = np.mean(pstr_mat, axis=0)
            metric_pstr_cat[key] = np.reshape(pstr_mat, newshape=(np.prod(pstr_mat.shape),))

            # delta5 diff pstr
            pstr_mat = calc_pstr(sessions_metadata[mouse_id][sess_name]['cue'],
                                 sessions_data[mouse_id][sess_name]['post_session_analysis']['dff_delta5']['traces'][i],
                                 delta_t)
            pstr_mat = pstr_mat[:, :int(pstr_mat.shape[1] / 2) + 1]  # calculate based on pre-stim only
            d5_pstr_avg[key] = np.mean(pstr_mat, axis=0)
            d5_pstr_cat[key] = np.reshape(pstr_mat, newshape=(np.prod(pstr_mat.shape),))

            # dff pstr
            pstr_mat = calc_pstr(sessions_metadata[mouse_id][sess_name]['cue'],
                                 sessions_data[mouse_id][sess_name]['post_session_analysis']['dff']['traces'][i],
                                 delta_t)
            pstr_mat = pstr_mat[:, :int(pstr_mat.shape[1] / 2) + 1]  # calculate based on pre-stim only
            dff_pstr_avg[key] = np.mean(pstr_mat, axis=0)
            dff_pstr_cat[key] = np.reshape(pstr_mat, newshape=(np.prod(pstr_mat.shape),))

        metric_roi = sessions_metadata[mouse_id][sess_name]['metric_roi']
        metric_index = int(metric_roi[4:]) - 1
        metric_corr, metric_corr_avg, metric_corr_full = {}, {}, {}
        dff_corr, dff_corr_avg, dff_corr_full = {}, {}, {}
        d5_corr, d5_corr_avg, d5_corr_full = {}, {}, {}
        for i, (key, val) in enumerate(sessions_metadata[mouse_id][sess_name]['rois_proximity'].items()):
            metric_corr[key] = np.corrcoef(metric_pstr_cat[key], metric_pstr_cat[metric_roi])[0, 1]
            metric_corr_avg[key] = np.corrcoef(metric_pstr_avg[key], metric_pstr_avg[metric_roi])[0, 1]
            metric_corr_full[key] = np.corrcoef(sessions_data[mouse_id][sess_name]['post_session_analysis']['dff_delta5']['zscore'][i],
                                                sessions_data[mouse_id][sess_name]['post_session_analysis']['dff_delta5']['zscore'][metric_index])[0, 1]

            d5_corr[key] = np.corrcoef(d5_pstr_cat[key], d5_pstr_cat[metric_roi])[0, 1]
            d5_corr_avg[key] = np.corrcoef(d5_pstr_avg[key], d5_pstr_avg[metric_roi])[0, 1]
            d5_corr_full[key] = np.corrcoef(sessions_data[mouse_id][sess_name]['post_session_analysis']['dff_delta5']['traces'][i],
                                                sessions_data[mouse_id][sess_name]['post_session_analysis']['dff_delta5']['traces'][metric_index])[0, 1]

            dff_corr[key] = np.corrcoef(dff_pstr_cat[key], dff_pstr_cat[metric_roi])[0, 1]
            dff_corr_avg[key] = np.corrcoef(dff_pstr_avg[key], dff_pstr_avg[metric_roi])[0, 1]
            dff_corr_full[key] = np.corrcoef(sessions_data[mouse_id][sess_name]['post_session_analysis']['dff']['traces'][i],
                                                sessions_data[mouse_id][sess_name]['post_session_analysis']['dff']['traces'][metric_index])[0, 1]

        sessions_data[mouse_id][sess_name]["metric_corr"] = metric_corr
        sessions_data[mouse_id][sess_name]["metric_corr_cat"] = metric_corr
        sessions_data[mouse_id][sess_name]["metric_corr_avg"] = metric_corr_avg
        sessions_data[mouse_id][sess_name]["metric_corr_full"] = metric_corr_full

        sessions_data[mouse_id][sess_name]["d5_corr"] = d5_corr
        sessions_data[mouse_id][sess_name]["d5_corr_cat"] = d5_corr
        sessions_data[mouse_id][sess_name]["d5_corr_avg"] = d5_corr_avg
        sessions_data[mouse_id][sess_name]["d5_corr_full"] = d5_corr_full

        sessions_data[mouse_id][sess_name]["dff_corr"] = dff_corr
        sessions_data[mouse_id][sess_name]["dff_corr_cat"] = dff_corr
        sessions_data[mouse_id][sess_name]["dff_corr_avg"] = dff_corr_avg
        sessions_data[mouse_id][sess_name]["dff_corr_full"] = dff_corr_full

        uncorr_arg = np.argmin(np.abs(list(metric_corr.values())))
        uncorr_roi = f'roi_{uncorr_arg + 1:02d}'
        metric_uncorr = {}
        for key, val in sessions_metadata[mouse_id][sess_name]['rois_proximity'].items():
            metric_uncorr[key] = np.corrcoef(metric_pstr_cat[key], metric_pstr_cat[uncorr_roi])[0, 1]
        sessions_data[mouse_id][sess_name]["metric_uncorr"] = metric_uncorr



# Plot ________________________________________________________________________________________
mouse_zero = mice_id[1]
session_zero = sessions_names[-1]

f = plt.figure(constrained_layout=True)
gs = f.add_gridspec(2, 2)
x = np.arange(len(sessions_names))
n_sessions = len(sessions_names)
#sessions_ticklabels = [f'day {i}' for i in range(n_sessions)]
sessions_ticklabels = [f'{i}' for i in sessions_names]
corr_num = 7

cmap = copy.deepcopy(plt.cm.get_cmap('plasma'))
c_list = cmap.colors[::int(256 / 2)]
for c in c_list:
    c.append(0.6)

##################################################################################
ax00 = f.add_subplot(gs[0, 0])
plot_box_plot(ax00, sessions_data[mouse_zero]["pstr_zscore_max_nm"], sessions_data[mouse_zero]["pstr_zscore_max_m"],
              set_title="",
              set_xticklabels={'labels': sessions_ticklabels, 'rotation': 90},
              set_ylabel='Z-Score',
              legend=["ROIs", "Target ROI"])
ax00.spines['right'].set_visible(False)
ax00.spines['top'].set_visible(False)
ax00.set_ylabel('Z-Score', fontsize=14)

# ax10 = f.add_subplot(gs[1, 0])
# xs, ys = [], []
# for i, (c, s) in enumerate(zip(sessions_data[mouse_zero][session_zero]['metric_corr'].values(),
#                                sessions_data[mouse_zero]["max_slope"].values())):
#     if s < 0.6 and s > -0.2:
#         xs.append(c)
#         ys.append(s)
# xs = np.array([xs]).transpose()
# ys = np.array([ys]).transpose()
# p = xs.argsort(axis=0)
# xs = xs[p[:, 0]]
# ys = ys[p[:, 0]]
#
# x_corr, y_corr = xs[-corr_num:], ys[-corr_num:]
# reg_corr = linregress(x_corr.flatten(), y_corr.flatten())
# ax10.scatter(x_corr[:-1], y_corr[:-1], c=c_list[0], s=16)
# ax10.scatter(x_corr[-1], y_corr[-1], c=c_list[1], s=16)
# p1 = ax10.plot(np.linspace(np.min(x_corr), np.max(x_corr), x_corr.shape[0]),
#                np.linspace(np.min(x_corr), np.max(x_corr), x_corr.shape[0]) * reg_corr.slope + reg_corr.intercept,
#                c='k')
# pstr_max_corr_score = reg_corr.rvalue ** 2
# pstr_max_corr_pval = reg_corr.pvalue
# pstr_max_corr_slope = reg_corr.slope
#
# x_uncorr, y_uncorr = xs[:-corr_num], ys[:-corr_num]
# reg_uncorr = linregress(x_uncorr.flatten(), y_uncorr.flatten())
# ax10.scatter(x_uncorr, y_uncorr, c=c_list[0], s=16)
# p2 = ax10.plot(np.linspace(np.min(x_uncorr), np.max(x_uncorr), x_uncorr.shape[0]),
#                np.linspace(np.min(x_uncorr), np.max(x_uncorr), x_uncorr.shape[0]) * reg_uncorr.slope +
#                reg_uncorr.intercept,
#                c='k', linestyle='--')
# pstr_max_uncorr_score = reg_uncorr.rvalue ** 2
# pstr_max_uncorr_pval = reg_uncorr.pvalue
# pstr_max_uncorr_slope = reg_uncorr.slope
#
# ax10.set_ylabel("Slope", fontsize=14)
# ax10.set_xlabel("Correlation", fontsize=14)
# ax10.spines['right'].set_visible(False)
# ax10.spines['top'].set_visible(False)
# ax10.legend([p1[0], p2[0]], ["high correlation", "low correlation"])
#
#
ax10 = f.add_subplot(gs[1, 0])
pstr_max_nm_before = []
pstr_max_nm_after = []
pstr_max_nm_before_after_ratio = []
pstr_max_m_before = []
pstr_max_m_after = []
pstr_max_m_before_after_ratio = []
for mouse_id, mouse_data in sessions_data.items():
    pstr_max_nm_before.append(np.median(mouse_data['pstr_zscore_max_nm'][:, 0]))
    pstr_max_nm_after.append(np.median(mouse_data['pstr_zscore_max_nm'][:, -1]))
    pstr_max_nm_before_after_ratio.append(np.array(np.median(np.median(mouse_data['pstr_zscore_max_nm'][:, -1]) / (np.array(mouse_data['pstr_zscore_max_nm'][:, 0])+1e-10))))
    pstr_max_m_before.append(np.median(mouse_data['pstr_zscore_max_m'][:, 0]))
    pstr_max_m_after.append(np.median(mouse_data['pstr_zscore_max_m'][:, -1]))
    pstr_max_m_before_after_ratio.append(np.median(np.median(np.array(mouse_data['pstr_zscore_max_m'][:, -1]) / (np.array(mouse_data['pstr_zscore_max_m'][:, 0])+1e-10))))


bars = ax10.bar((1, 2, 4, 5),
                (np.mean(pstr_max_nm_before), np.mean(pstr_max_nm_after), np.mean(pstr_max_m_before),
                 np.mean(pstr_max_m_after)),
                color=(c_list[0], c_list[0], c_list[1], c_list[1]))

ratio = np.array(pstr_max_m_before_after_ratio) / np.array(pstr_max_nm_before_after_ratio)
ax10t = ax10.twinx()
ax10t.bar((7, ), np.mean(ratio), color='gray')
ax10t.plot((6.5, 7.5), (ratio[0], ratio[0]), color='k', linestyle='-')
#ax20t.plot((6.5, 7.5), (ratio[1], ratio[1]), color='k', linestyle=':')
#ax20t.plot((6.5, 7.5), (ratio[2], ratio[2]), color='k', linestyle='--')

x, y = [], []
for nb, na, mb, ma in zip(pstr_max_nm_before,
                          pstr_max_nm_after,
                          pstr_max_m_before,
                          pstr_max_m_after):
    y.extend([nb, na, mb, ma])
    x.extend([1, 2, 4, 5])
# ax20.scatter(x[:4], y[:4], c='k', zorder=2, marker='o')
# ax20.scatter(x[4:8], y[4:8], c='k', zorder=2, marker='x')
# ax20.scatter(x[8:], y[8:], c='k', zorder=2, marker='*')
ax10.plot([1, 2], [pstr_max_nm_before[0], pstr_max_nm_after[0]], color='k', linestyle='-')
#ax20.plot([1, 2], [pstr_max_nm_before[1], pstr_max_nm_after[1]], color='k', linestyle=':')
#ax20.plot([1, 2], [pstr_max_nm_before[2], pstr_max_nm_after[2]], color='k', linestyle='--')
ax10.plot([4, 5], [pstr_max_m_before[0], pstr_max_m_after[0]], color='k', linestyle='-')
#ax20.plot([4, 5], [pstr_max_m_before[1], pstr_max_m_after[1]], color='k', linestyle=':')
#ax20.plot([4, 5], [pstr_max_m_before[2], pstr_max_m_after[2]], color='k', linestyle='--')

ax10.set_xticks([1, 2, 4, 5, 7])
ax10.set_xticklabels(['before\ntraining', "after\ntraining", 'before\ntraining', "after\ntraining", 'growth\nratio'], fontsize=12)

ax10.set_ylim(2.4, 4)
ax10t.set_ylim(0, 1.1)
ax10.set_ylabel("Z-Score", fontsize=14)
# ax20.legend(bars[:2], ['non_metric', 'metric'])

ax10.spines['right'].set_visible(False)
ax10.spines['top'].set_visible(False)

##################################################################################
ax01 = f.add_subplot(gs[0, 1])
plot_box_plot(ax01, sessions_data[mouse_zero]["p_th_nm"], sessions_data[mouse_zero]["p_th_m"],
              set_title="",
              set_xticklabels={'labels': sessions_ticklabels, 'rotation': 90},
              set_ylabel='Probability',
              legend=["ROIs", "Target ROI"])
ax01.spines['right'].set_visible(False)
ax01.spines['top'].set_visible(False)
ax01.set_ylabel('Probability', fontsize=14)

# ax11 = f.add_subplot(gs[1, 1])
# xs, ys = [], []
# for i, (c, s) in enumerate(zip(sessions_data[mouse_zero][session_zero]['metric_corr'].values(),
#                                sessions_data[mouse_zero]["p_th_slope"].values())):
#     # if s < 0.6 and s > -0.2:
#     xs.append(c)
#     ys.append(s)
# xs = np.array([xs]).transpose()
# ys = np.array([ys]).transpose()
# p = xs.argsort(axis=0)
# xs = xs[p[:, 0]]
# ys = ys[p[:, 0]]
#
# x_corr, y_corr = xs[-corr_num:], ys[-corr_num:]
# reg_corr = linregress(x_corr.flatten(), y_corr.flatten())
# ax11.scatter(x_corr[:-1], y_corr[:-1], c=c_list[0], s=16)
# ax11.scatter(x_corr[-1], y_corr[-1], c=c_list[1], s=16)
# p1 = ax11.plot(np.linspace(np.min(x_corr), np.max(x_corr), x_corr.shape[0]),
#                np.linspace(np.min(x_corr), np.max(x_corr), x_corr.shape[0]) * reg_corr.slope + reg_corr.intercept,
#                c='k')
# p_th_corr_score = reg_corr.rvalue ** 2
# p_th_corr_pval = reg_corr.pvalue
# p_th_corr_slope = reg_corr.slope
#
# x_uncorr, y_uncorr = xs[:-corr_num], ys[:-corr_num]
# reg_uncorr = linregress(x_uncorr.flatten(), y_uncorr.flatten())
#
# ax11.scatter(x_uncorr, y_uncorr, c=c_list[0], s=16)
# p2 = ax11.plot(np.linspace(np.min(x_uncorr), np.max(x_uncorr), x_uncorr.shape[0]),
#                np.linspace(np.min(x_uncorr), np.max(x_uncorr), x_uncorr.shape[0]) * reg_uncorr.slope +
#                reg_uncorr.intercept,
#                c='k', linestyle='--')
# p_th_uncorr_score = reg_uncorr.rvalue ** 2
# p_th_uncorr_pval = reg_uncorr.pvalue
# p_th_uncorr_slope = reg_uncorr.slope
#
# ax11.set_ylabel("Slope", fontsize=14)
# ax11.set_xlabel("Correlation", fontsize=14)
# ax11.legend([p1[0], p2[0]], ["high correlation", "low correlation"])
# ax11.spines['right'].set_visible(False)
# ax11.spines['top'].set_visible(False)
#
ax11 = f.add_subplot(gs[1, 1])
p_th_nm_before = []
p_th_nm_after = []
p_th_nm_before_after_ratio = []
p_th_m_before = []
p_th_m_after = []
p_th_m_before_after_ratio = []
for mouse_id, mouse_data in sessions_data.items():
    p_th_nm_before.append(np.median(mouse_data['p_th_nm'][:, 0]))
    p_th_nm_after.append(np.median(mouse_data['p_th_nm'][:, -1]))
    p_th_nm_before_after_ratio.append(np.array(np.median(np.median(mouse_data['p_th_nm'][:, -1]) / (
                np.array(mouse_data['p_th_nm'][:, 0]) + 1e-10))))
    p_th_m_before.append(np.median(mouse_data['p_th_m'][:, 0]))
    p_th_m_after.append(np.median(mouse_data['p_th_m'][:, -1]))
    p_th_m_before_after_ratio.append(np.array(np.median(np.median(mouse_data['p_th_m'][:, -1]) / (
            np.array(mouse_data['p_th_m'][:, 0]) + 1e-10))))

bars = ax11.bar((1, 2, 4, 5),
                (np.mean(p_th_nm_before), np.mean(p_th_nm_after),
                 np.mean(p_th_m_before), np.mean(p_th_m_after)),
                color=(c_list[0], c_list[0], c_list[1], c_list[1]))

ratio = np.array(p_th_m_before_after_ratio) / np.array(p_th_nm_before_after_ratio)
ax11t = ax11.twinx()
ax11t.bar((7, ), np.mean(ratio), color='gray')
ax11t.plot((6.5, 7.5), (ratio[0], ratio[0]), color='k', linestyle='-')
# ax11t.plot((6.5, 7.5), (ratio[1], ratio[1]), color='k', linestyle=':')
# ax11t.plot((6.5, 7.5), (ratio[2], ratio[2]), color='k', linestyle='--')

x, y = [], []
for nb, na, mb, ma in zip(p_th_nm_before,
                          p_th_nm_after,
                          p_th_m_before,
                          p_th_m_after):
    y.extend([nb, na, mb, ma])
    x.extend([1, 2, 4, 5])
# ax21.scatter(x[:4], y[:4], c='k', zorder=2, marker='o')
# ax21.scatter(x[4:8], y[4:8], c='k', zorder=2, marker='x')
# ax21.scatter(x[8:], y[8:], c='k', zorder=2, marker='*')
ax11.plot([1, 2], [p_th_nm_before[0], p_th_nm_after[0]], color='k', linestyle='-')
# ax11.plot([1, 2], [p_th_nm_before[1], p_th_nm_after[1]], color='k', linestyle=':')
# ax11.plot([1, 2], [p_th_nm_before[2], p_th_nm_after[2]], color='k', linestyle='--')
ax11.plot([4, 5], [p_th_m_before[0], p_th_m_after[0]], color='k', linestyle='-')
# ax11.plot([4, 5], [p_th_m_before[1], p_th_m_after[1]], color='k', linestyle=':')
# ax11.plot([4, 5], [p_th_m_before[2], p_th_m_after[2]], color='k', linestyle='--')

ax11.set_xticks([1, 2, 4, 5, 7])
ax11.set_xticklabels(['before\ntraining', "after\ntraining", 'before\ntraining', "after\ntraining", 'growth\nratio'], fontsize=12)

# ax21.set_ylim(0, 50)
ax11t.set_ylim(0, 5)
ax11.set_ylabel("Probability", fontsize=14)
# ax21.legend(bars[:2], ['non_metric', 'metric'])
ax11.spines['right'].set_visible(False)
ax11.spines['top'].set_visible(False)

plt.show()