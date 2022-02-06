from utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict
from analysis.utils.load_session_metadata import load_session_metadata
from analysis.utils.peristimulus_time_response import calc_pstr

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks, peak_prominences

from wideflow.utils.load_rois_data import load_rois_data
import collections

import h5py

temporal_window = 1000  # in milliseconds, used to calculate divergence metric and pstr
smoo_kernel_size = 11
max_f_to_peak = 5
fixed_threshold = 1.5


def calc_z_score(x):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    return (x - x_mean) / (x_std + np.finfo(np.float32).eps)


def calc_divergence_score(x, samples_window):
    n_vars, n_samples = x.shape
    # shift rows to zero mean and fix for global slop
    for i in range(n_vars):
        reg = LinearRegression().fit(np.arange(n_samples).reshape(-1, 1), x[i].reshape(-1, 1))
        x[i] = x[i] - np.arange(n_samples) * reg.coef_[0] - reg.intercept_[0]

    x = np.pad(x, [[0, 0], [samples_window, 0]], 'edge')
    x_slop = np.zeros((n_vars, n_samples), dtype=np.float32)
    for j in range(n_vars):
        xj = x[j]
        for i in range(n_samples):
            reg = LinearRegression().fit(np.arange(samples_window).reshape(-1, 1),
                                         xj[i: i + samples_window].reshape(-1, 1))
            x_slop[j, i] = reg.coef_[0]

    return x_slop


def calc_cross_corr_euclidean_dist(x, samples_window):
    n_vars, n_samples = x.shape
    x = np.pad(x, [[0, 0], [samples_window, 0]], 'edge')

    # calculate the temporal correlation coefficient matrix
    corr_mat = np.zeros((n_samples, n_vars, n_vars))
    for i in range(samples_window, n_samples + samples_window):
        corr_mat[i] = np.corrcoef(x[:, i-samples_window: i])

    # calculate the euclidean distance of each sample to


def calc_diff(x, delta_t):
    x = np.pad(x, [[0, 0], [delta_t, 0]])
    return x[:, delta_t:] - x[:, :-delta_t]


def calc_time_to_peak(x, smoo_kernel_size=11, delta_t=5, peaks=None):
    if peaks is None:
        x_mean = np.mean(x, axis=0)
        x_mean_conv = np.convolve(x_mean, np.ones((smoo_kernel_size, ))/smoo_kernel_size, mode='same')
        peaks = find_peaks(x_mean_conv)[0]
        prominence = peak_prominences(x_mean_conv, peaks)
        prominence_mean = np.mean(prominence[0])
        prominence_sd = np.std(prominence[0])
        prominence_th = prominence[0] > prominence_mean + prominence_sd
        peaks = peaks[prominence_th]

    n_peaks = len(peaks)
    n_vars, n_samples = x.shape
    peaks_delta = np.zeros((n_vars, n_peaks))
    x_pad = np.pad(x, [[0, 0], [delta_t, delta_t]])
    for i, xi in enumerate(x_pad):
        for j, peak in enumerate(peaks + delta_t):
            xi_prox = xi[peak - delta_t: peak + delta_t + 1]
            peaks_delta[i, j] = np.argmax(xi_prox) - delta_t

    return peaks_delta, peaks


def calc_simulated_rewards(x, th, min_dst, prominence_th=False):
    peaks_inds = find_peaks(x, height=th, distance=min_dst)[0]
    if prominence_th:
        prominence = peak_prominences(x, peaks_inds)
        prominence_mean = np.mean(prominence[0])
        prominence_sd = np.std(prominence[0])
        prominence_th = prominence[0] > prominence_mean + prominence_sd
        peaks_inds = peaks_inds[prominence_th]
    peaks = np.zeros(x.shape, dtype=np.bool)
    peaks[peaks_inds] = 1
    return peaks, peaks_inds


def pstr_process(metric, samples_window, th, min_dst, prominence_th=False):
    pstr_mat = []
    peaks_inds_arr = []
    for i, trace in enumerate(metric):
        peaks, peaks_inds = calc_simulated_rewards(trace, th, min_dst, prominence_th)
        pstr = calc_pstr(peaks, trace, samples_window)
        if pstr.ndim > 1:
            pstr = np.mean(pstr, axis=0)
        pstr_mat.append(pstr)
        peaks_inds_arr.append(peaks_inds)

    pstr_mat = np.array(pstr_mat)
    return pstr_mat, peaks_inds_arr


def calc_adjusted_reward_pstr(traces, n_samp, height, threshold, min_dst, prominence_th):
    pstr_3d_mat = np.zeros((n_rois, n_rois, 2 * n_samp + 1), dtype=np.float32)
    for i, (roi_name, roi_trace) in enumerate(traces.items()):
        peaks, peaks_inds = calc_simulated_rewards(roi_trace, height, threshold, min_dst, prominence_th)
        for j, (roi_name, roi_trace) in enumerate(traces.items()):
            pstr = np.mean(calc_pstr(peaks, roi_trace, n_samp),
                           axis=0)  # pstr return as matrix with each row is response to different reward
            pstr_3d_mat[i, j] = pstr

    pstr_mat_avg = np.zeros((n_rois + 1, 2 * n_samp + 1),  dtype=np.float32)  # first row will be reward-calc_roi and the other - all rois
    for i, pstr_mat in enumerate(pstr_3d_mat):
        if np.isnan(pstr_mat[0, 0]):
            continue
        pstr_mat_avg[0] += pstr_mat[i]
        pstr_mat[i] = np.zeros((2 * n_samp + 1, ), dtype=np.float32)
        pstr_mat_avg[1:] = pstr_mat

    pstr_mat_avg[0] = pstr_mat_avg[0] / n_rois
    pstr_mat_avg[1:] = pstr_mat_avg[1:] / (n_rois - 1)

    return pstr_mat_avg


rois_dict_path = '/data/Rotem/Wide Field/WideFlow/data/cortex_map/allen_2d_cortex_rois.h5'
cortex_map_path = '/data/Rotem/Wide Field/WideFlow/data/cortex_map/allen_2d_cortex.h5'
rois_dict = load_rois_data(rois_dict_path)
rois_dict = collections.OrderedDict(sorted(rois_dict.items()))
n_rois = len(rois_dict)
with h5py.File(cortex_map_path, 'r') as f:
    cortex_mask = np.transpose(f["mask"][()])
    cortex_map = np.transpose(f["map"][()])

base_path = '/data/Rotem/WideFlow prj/'
dataset_path = base_path + 'results/sessions_dataset.h5'
statistics_path = base_path + 'results/sessions_statistics.h5'

mouse_id = '2680'
sessions_list = [
    # '20211125_neurofeedback',
    '20211130_neurofeedback',
    # '20211206_neurofeedback',
    # '20211208_neurofeedback',
    # '20211219_neurofeedback'
]

# frames to exclude for each session. frames indexing of one channel
# for mouse ID 2680
sessions_exclution_list = {
    '20211125_neurofeedback': [],
    '20211130_neurofeedback': [],
    '20211206_neurofeedback': np.arange(400).tolist(),
    '20211208_neurofeedback': np.arange(11700, 11850).tolist() + np.arange(13800, 14720).tolist() + np.arange(21500, 21600).tolist(),
    '20211219_neurofeedback': np.arange(200).tolist() +
                              np.arange(13500, 14000).tolist() +
                              np.arange(24200, 25000).tolist() +
                              np.arange(28200, 30000).tolist()
}

# for mouse ID 2683
# sessions_exclution_list = {
#     '20211125_neurofeedback': np.arange(11950, 12050).tolist(),
#     '20211130_neurofeedback': [],
#     '20211206_neurofeedback':  np.arange(29900, 30000).tolist(),
#     '20211208_neurofeedback': [],
#     '20211219_neurofeedback': []
# }


# load data
n_sessions = len(sessions_list)
sessions_data = {}
sessions_metadata = {}
sessions_config = {}
for session_name in sessions_list:
    sessions_data[session_name] = {}
    sessions_metadata[session_name] = {}
    sessions_config[session_name] = {}
    with h5py.File(dataset_path, 'r') as f:
        decompose_h5_groups_to_dict(f, sessions_data[session_name], f'/{mouse_id}/{session_name}/')
        if len(sessions_data[session_name]) > 1:
            del sessions_data[session_name]['channel_1']
        sessions_metadata[session_name], sessions_config[session_name] = load_session_metadata(f'{base_path}{mouse_id}/{session_name}/')

print(f"running analysis for mouse {mouse_id}")
# calculate different knd of metrics
for session_name in sessions_list:
    print(f"calculating statistics for session {session_name}")
    threshold = np.max(sessions_metadata[session_name]["threshold"])

    n_channels = sessions_config[session_name]['camera_config']['attr']['channels']
    time_stemps = np.array(sessions_metadata[session_name]["timestamp"][::n_channels])
    time_intervals = time_stemps[1:] - time_stemps[:-1]
    avg_interval = np.mean(time_intervals) * 1000
    samples_window = int(temporal_window / avg_interval)

    min_reward_interval = avg_interval * sessions_config[session_name]['feedback_config']['inter_feedback_delay'] / 1000

    reward = np.array(sessions_metadata[session_name]["cue"])
    if n_channels > 1:
        reward = reward[::n_channels]
        for i in range(1, n_channels):
            reward = np.maximum(reward, np.array(sessions_metadata[session_name]["cue"])[i::n_channels])

    traces = np.array(list(sessions_data[session_name]["channel_0"].values()))
    # exclude corrupted frames
    ex_list = sessions_exclution_list[session_name]
    ex_bo = np.array([False if i in ex_list else True for i in range(traces.shape[1])])
    traces = traces[:, ex_bo]
    reward = reward[ex_bo]

    n_vars, n_samples = traces.shape[0], traces.shape[1]

    # dff Intensity___________________________________________________________________________________________________
    print("     dff stats")
    traces = traces - np.expand_dims(np.mean(traces, axis=1), 1)
    # Z-score
    traces_zscore = calc_z_score(traces)
    # pstr
    traces_pstr = np.zeros((n_vars, samples_window * 2 + 1))
    traces_zscore_pstr = np.zeros((n_vars, samples_window * 2 + 1))
    for i in range(n_vars):
        traces_pstr[i] = np.mean(calc_pstr(reward, traces[i], samples_window), axis=0)
        traces_zscore_pstr[i] = np.mean(calc_pstr(reward, traces_zscore[i], samples_window), axis=0)
    # rewards
    traces_zscore_rewards = np.mean(traces_zscore > threshold, axis=1)
    traces_zscore_rewards_fixed_th = np.mean(traces_zscore > fixed_threshold, axis=1)

    # simulated reward pstr
    traces_sim_pstr_mat, peaks_inds_arr = pstr_process(traces, samples_window, None, min_reward_interval, prominence_th=True)
    traces_zscore_sim_pstr_mat, peaks_inds_arr = pstr_process(traces_zscore, samples_window, fixed_threshold, min_reward_interval)

    # regression divergence value_____________________________________________________________________________________
    print("     regression divergence stats")
    divergence = calc_divergence_score(traces, samples_window)
    # Divergence Z-score
    divergence_zscore = calc_z_score(divergence)
    # pstr
    divergence_pstr = np.zeros((n_vars, samples_window * 2 + 1))
    divergence_zscore_pstr = np.zeros((n_vars, samples_window * 2 + 1))
    for i in range(n_vars):
        divergence_pstr[i] = np.mean(calc_pstr(reward, divergence[i], samples_window), axis=0)
        divergence_zscore_pstr[i] = np.mean(calc_pstr(reward, divergence_zscore[i], samples_window), axis=0)
    # rewards
    divergence_zscore_rewards = np.mean(divergence_zscore > threshold, axis=1)
    divergence_zscore_rewards_fixed_th = np.mean(divergence_zscore > fixed_threshold, axis=1)

    # simulated reward pstr
    divergence_sim_pstr_mat, peaks_inds_arr = pstr_process(divergence, samples_window, None, min_reward_interval, prominence_th=True)
    divergence_zscore_sim_pstr_mat, peaks_inds_arr = pstr_process(divergence_zscore, samples_window, fixed_threshold, min_reward_interval)

    # Time to peak____________________________________________________________________________________________________
    print("     Time to peak stats")
    peaks_delta, peaks = calc_time_to_peak(traces, smoo_kernel_size, max_f_to_peak)

    # delta_t difference______________________________________________________________________________________________
    print("     delta_t difference stats")
    diff1 = calc_diff(traces, 1)
    diff2 = calc_diff(traces, 2)
    diff3 = calc_diff(traces, 3)
    diff5 = calc_diff(traces, 5)
    # delta_t difference Z-score
    diff1_zscore = calc_z_score(diff1)
    diff2_zscore = calc_z_score(diff2)
    diff3_zscore = calc_z_score(diff3)
    diff5_zscore = calc_z_score(diff5)
    # pstr
    diff1_pstr = np.zeros((n_vars, samples_window * 2 + 1))
    diff2_pstr = np.zeros((n_vars, samples_window * 2 + 1))
    diff3_pstr = np.zeros((n_vars, samples_window * 2 + 1))
    diff5_pstr = np.zeros((n_vars, samples_window * 2 + 1))
    diff1_zscore_pstr = np.zeros((n_vars, samples_window * 2 + 1))
    diff2_zscore_pstr = np.zeros((n_vars, samples_window * 2 + 1))
    diff3_zscore_pstr = np.zeros((n_vars, samples_window * 2 + 1))
    diff5_zscore_pstr = np.zeros((n_vars, samples_window * 2 + 1))
    for i in range(n_vars):
        diff1_pstr[i] = np.mean(calc_pstr(reward, diff1[i], samples_window), axis=0)
        diff2_pstr[i] = np.mean(calc_pstr(reward, diff2[i], samples_window), axis=0)
        diff3_pstr[i] = np.mean(calc_pstr(reward, diff3[i], samples_window), axis=0)
        diff5_pstr[i] = np.mean(calc_pstr(reward, diff5[i], samples_window), axis=0)
        diff1_zscore_pstr[i] = np.mean(calc_pstr(reward, diff1_zscore[i], samples_window), axis=0)
        diff2_zscore_pstr[i] = np.mean(calc_pstr(reward, diff2_zscore[i], samples_window), axis=0)
        diff3_zscore_pstr[i] = np.mean(calc_pstr(reward, diff3_zscore[i], samples_window), axis=0)
        diff5_zscore_pstr[i] = np.mean(calc_pstr(reward, diff5_zscore[i], samples_window), axis=0)
    # rewards
    diff1_zscore_rewards = np.mean(diff1_zscore_pstr > threshold, axis=1)
    diff1_zscore_rewards_fixed_th = np.mean(diff1_zscore_pstr > fixed_threshold, axis=1)
    diff2_zscore_rewards = np.mean(diff2_zscore_pstr > threshold, axis=1)

    diff2_zscore_rewards_fixed_th = np.mean(diff2_zscore_pstr > fixed_threshold, axis=1)
    diff3_zscore_rewards = np.mean(diff3_zscore_pstr > threshold, axis=1)
    diff3_zscore_rewards_fixed_th = np.mean(diff3_zscore_pstr > fixed_threshold, axis=1)
    diff5_zscore_rewards = np.mean(diff5_zscore_pstr > threshold, axis=1)
    diff5_zscore_rewards_fixed_th = np.mean(diff5_zscore_pstr > fixed_threshold, axis=1)

    # simulated reward pstr
    diff1_sim_pstr_mat, peaks_inds_arr = pstr_process(diff1, samples_window, None, min_reward_interval, prominence_th=True)
    diff1_zscore_sim_pstr_mat, peaks_inds_arr = pstr_process(diff1_zscore, samples_window, fixed_threshold, min_reward_interval)
    diff3_sim_pstr_mat, peaks_inds_arr = pstr_process(diff3, samples_window, None, min_reward_interval, prominence_th=True)
    diff3_zscore_sim_pstr_mat, peaks_inds_arr = pstr_process(diff3_zscore, samples_window, fixed_threshold, min_reward_interval)
    diff5_sim_pstr_mat, peaks_inds_arr = pstr_process(diff5, samples_window, None, min_reward_interval, prominence_th=True)
    diff5_zscore_sim_pstr_mat, peaks_inds_arr = pstr_process(diff5_zscore, samples_window, fixed_threshold, min_reward_interval)

    print("writing to tiff")
    with h5py.File(statistics_path, 'a') as f:
        mouse_grp = f[mouse_id]
        session_grp = mouse_grp.create_group(session_name)

        traces_grp = session_grp.create_group("traces")
        traces_grp.create_dataset('dff', data=traces)
        traces_grp.create_dataset('zscore', data=traces_zscore)
        traces_grp.create_dataset('pstr', data=traces_pstr)
        traces_grp.create_dataset('zscore_pstr', data=traces_zscore_pstr)
        traces_grp.create_dataset(f'zscore_reward_th{threshold}sd', data=traces_zscore_rewards)
        traces_grp.create_dataset(f'zscore_reward_th{fixed_threshold}sd', data=traces_zscore_rewards_fixed_th)
        traces_grp.create_dataset(f'pstr_sim_reward', data=traces_sim_pstr_mat)
        traces_grp.create_dataset(f'pstr_zscore_sim_reward', data=traces_zscore_sim_pstr_mat)

        divergence_grp = session_grp.create_group("regression_divergence")
        divergence_grp.create_dataset('divergence', data=divergence)
        divergence_grp.create_dataset('zscore', data=divergence_zscore)
        divergence_grp.create_dataset('pstr', data=divergence_pstr)
        divergence_grp.create_dataset('zscore_pstr', data=divergence_zscore_pstr)
        divergence_grp.create_dataset(f'zscore_th{threshold}sd', data=divergence_zscore_rewards)
        divergence_grp.create_dataset(f'zscore_reward_th{fixed_threshold}sd', data=divergence_zscore_rewards_fixed_th)
        divergence_grp.create_dataset(f'pstr_sim_reward', data=divergence_sim_pstr_mat)
        divergence_grp.create_dataset(f'pstr_zscore_sim_reward', data=divergence_zscore_sim_pstr_mat)

        time_to_peak_grp = session_grp.create_group("time_to_peak")
        time_to_peak_grp.create_dataset('time_to_peak_delta', data=peaks_delta)
        time_to_peak_grp.create_dataset('time_to_peak_peaks', data=peaks)

        delta_f_diff_grp = session_grp.create_group("delta_frames_diff")
        delta_1_grp = delta_f_diff_grp.create_group("delta_frames_1")
        delta_1_grp.create_dataset('diff', data=diff1)
        delta_1_grp.create_dataset('diff_zscore', data=diff1_zscore)
        delta_1_grp.create_dataset('diff_pstr', data=diff1_pstr)
        delta_1_grp.create_dataset('zscore_pstr', data=diff1_zscore_pstr)
        delta_1_grp.create_dataset(f'zscore_th{threshold}sd', data=diff1_zscore_rewards)
        delta_1_grp.create_dataset(f'zscore_reward_th{fixed_threshold}sd', data=diff1_zscore_rewards_fixed_th)
        delta_1_grp.create_dataset(f'pstr_sim_reward', data=diff1_sim_pstr_mat)
        delta_1_grp.create_dataset(f'pstr_zscore_sim_reward', data=diff1_zscore_sim_pstr_mat)

        delta_3_grp = delta_f_diff_grp.create_group("delta_frames_3")
        delta_3_grp.create_dataset('diff', data=diff3)
        delta_3_grp.create_dataset('diff_zscore', data=diff3_zscore)
        delta_3_grp.create_dataset('diff_pstr', data=diff3_pstr)
        delta_3_grp.create_dataset('zscore_pstr', data=diff3_zscore_pstr)
        delta_3_grp.create_dataset(f'zscore_th{threshold}sd', data=diff3_zscore_rewards)
        delta_3_grp.create_dataset(f'zscore_reward_th{fixed_threshold}sd', data=diff3_zscore_rewards_fixed_th)
        delta_3_grp.create_dataset(f'pstr_sim_reward', data=diff3_sim_pstr_mat)
        delta_3_grp.create_dataset(f'pstr_zscore_sim_reward', data=diff3_zscore_sim_pstr_mat)

        delta_5_grp = delta_f_diff_grp.create_group("delta_frames_5")
        delta_5_grp.create_dataset('diff', data=diff5)
        delta_5_grp.create_dataset('diff_zscore', data=diff5_zscore)
        delta_5_grp.create_dataset('diff_pstr', data=diff5_pstr)
        delta_5_grp.create_dataset('zscore_pstr', data=diff5_zscore_pstr)
        delta_5_grp.create_dataset(f'zscore_th{threshold}sd', data=diff5_zscore_rewards)
        delta_5_grp.create_dataset(f'zscore_reward_th{fixed_threshold}sd', data=diff5_zscore_rewards_fixed_th)
        delta_5_grp.create_dataset(f'pstr_sim_reward', data=diff5_sim_pstr_mat)
        delta_5_grp.create_dataset(f'pstr_zscore_sim_reward', data=diff5_zscore_sim_pstr_mat)

        stats_grp = session_grp.create_group("statistics_parameters")
        stats_grp.create_dataset("temporal_window_ms", data=temporal_window)
        stats_grp.create_dataset("smoo_kernel_size", data=smoo_kernel_size)
        stats_grp.create_dataset("max_f_to_peak", data=max_f_to_peak)
        stats_grp.create_dataset("fixed_threshold", data=fixed_threshold)
        stats_grp.create_dataset("excluded_frames", data=sessions_exclution_list[session_name])

