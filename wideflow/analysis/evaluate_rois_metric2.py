from utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict
from analysis.utils.load_session_metadata import load_session_metadata
from analysis.utils.peristimulus_time_response import calc_pstr

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks, peak_prominences, convolve2d

import h5py

temporal_window = 1000  # in milliseconds, used to calculate divergence metric and pstr
smoo_kernel_size = 11
max_f_to_peak = 5
fixed_threshold = 2.3
auc_window_length = 5000
exclude_post_rewards_frames = 30  # exclude from the analysis n frames following rewards due to stimulation artifacts


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


def pstr_process(metric, samples_window, th, min_dst, prominence_th=False, ex_list=[], ex_n=0):
    pstr_mat = []
    peaks_inds_arr = []
    for trace in metric:
        peaks, peaks_inds = calc_simulated_rewards(trace, th, min_dst, prominence_th)
        if len(ex_list) > 0 and ex_n > 0:
            j = 0
            for index in peaks_inds:
                if 1 in ex_list[np.max((0, index-ex_n)): index-1]:
                    peaks_inds = np.delete(peaks_inds, j)
                    peaks[index] = False
                    j = j - 1
                j = j + 1
        pstr = calc_pstr(peaks, trace, samples_window)
        if pstr.ndim > 1:
            pstr = np.mean(pstr, axis=0)
        if np.isnan(np.sum(pstr)):
            pstr = np.zeros(pstr.shape)
        pstr_mat.append(pstr)
        peaks_inds_arr.append(peaks_inds)

    pstr_mat = np.array(pstr_mat)
    return pstr_mat, peaks_inds_arr


def moving_average_auc(x, window_length):
    n_vars, n_samples = x.shape
    window = np.ones((window_length, )) / window_length
    auc = np.zeros((n_vars, n_samples - window_length + 1))
    for i, trace in enumerate(x):
        auc[i] = np.convolve(trace, window, 'valid')
    return auc


def smoo_rewards(inds_list, shape):
    x = np.zeros([shape[0], shape[1]-10000 + 1])
    k = np.ones((10000,))
    for i, inds in enumerate(inds_list):
        temp = np.zeros((shape[1], ))
        temp[inds] = 1
        x[i] = np.convolve(temp, k, 'valid')
    return x


base_path = '/data/Rotem/WideFlow prj/'
dataset_path = base_path + 'results/sessions_20220220.h5'

mouse_id = '2680'
sessions_list = [
    # '20220220_neurofeedback',
    # '20220221_neurofeedback',
    '20220222_neurofeedback',
    '20220223_neurofeedback',
    '20220224_neurofeedback',
    '20220227_neurofeedback',
    '20220228_neurofeedback',
    '20220302_neurofeedback',
    '20220303_neurofeedback',
    # '20220306_neurofeedback',
    '20220307_neurofeedback',
]

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
        sessions_metadata[session_name], sessions_config[session_name] = load_session_metadata(f'{base_path}{mouse_id}/{session_name}/')

print(f"running analysis for mouse {mouse_id}")
# calculate different knd of metrics
for session_name in sessions_list:
    print(f"calculating statistics for session {session_name}")
    sess_threshold = np.mean(sessions_metadata[session_name]["threshold"])

    n_channels = sessions_config[session_name]['camera_config']['attr']['channels']
    time_stemps = np.array(sessions_metadata[session_name]["timestamp"][::n_channels])
    time_intervals = time_stemps[1:] - time_stemps[:-1]
    avg_interval = np.mean(time_intervals) * 1000  # convert to milli seconds
    samples_window = int(temporal_window / avg_interval)

    min_reward_interval = avg_interval * sessions_config[session_name]['feedback_config']['inter_feedback_delay'] / 1000

    reward = np.array(sessions_metadata[session_name]["cue"])
    if n_channels > 1:  # a fix when examining one channel while two channel acquisition is used
        reward = reward[::n_channels]
        for i in range(1, n_channels):
            reward = np.maximum(reward, np.array(sessions_metadata[session_name]["cue"])[i::n_channels])


    traces = np.array(list(sessions_data[session_name]['rois_traces']["channel_0"].values()))  # dff values after hemodynamics corrections
    n_vars, n_samples = traces.shape[0], traces.shape[1]
    if exclude_post_rewards_frames > 0:
        exclude_frames = np.ones((n_samples, ))
        for i in range(n_samples):
            if 1 in reward[np.max((0, i-exclude_post_rewards_frames)):i]:
                exclude_frames[i] = 0

        traces_nr = traces[np.ix_(np.arange(traces.shape[0]), np.argwhere(exclude_frames)[:, 0])]


    rois_key = list(sessions_data[session_name]['rois_traces']["channel_0"].keys())
    # traces = traces - np.expand_dims(np.mean(traces, axis=1), 1)


    # dff Intensity___________________________________________________________________________________________________
    print("     dff stats")
    # Z-score
    traces_zscore = calc_z_score(traces)

    # pstr for real time rewards timing
    traces_pstr = np.zeros((n_vars, samples_window * 2 + 1))
    traces_zscore_pstr = np.zeros((n_vars, samples_window * 2 + 1))
    for i in range(n_vars):
        traces_pstr[i] = np.mean(calc_pstr(reward, traces[i], samples_window), axis=0)
        traces_zscore_pstr[i] = np.mean(calc_pstr(reward, traces_zscore[i], samples_window), axis=0)

    # pstr for simulated rewards timing
    traces_sim_pstr_mat, dff_peaks_inds_arr = pstr_process(traces, samples_window, None, min_reward_interval, prominence_th=True, ex_list=reward, ex_n=exclude_post_rewards_frames)
    traces_zscore_sim_pstr_mat_fix_th, dff_zscore_peaks_inds_arr_fix_th = pstr_process(traces_zscore, samples_window, fixed_threshold, min_reward_interval, ex_list=reward, ex_n=exclude_post_rewards_frames)
    traces_zscore_sim_pstr_mat_session_th, dff_zscore_peaks_inds_arr_session_th = pstr_process(traces_zscore, samples_window, sess_threshold, min_reward_interval, ex_list=reward, ex_n=exclude_post_rewards_frames)
    dff_zscore_peaks_inds_arr_fix_th_conv_10k = smoo_rewards(dff_zscore_peaks_inds_arr_fix_th, (n_vars, n_samples))
    dff_zscore_peaks_inds_arr_session_th_conv_10k = smoo_rewards(dff_zscore_peaks_inds_arr_session_th, (n_vars, n_samples))

    # area under the curve (AUC) and mooving average AUC
    traces_auc = np.mean(traces, axis=1)
    traces_zscore_auc = np.mean(traces_zscore, axis=1)
    traces_moving_avg_auc = moving_average_auc(traces, auc_window_length)
    traces_zscore_moving_avg_auc = moving_average_auc(traces_zscore, auc_window_length)

    # delta_t difference______________________________________________________________________________________________
    print("     delta_t difference stats")
    diff5 = calc_diff(traces, 5)
    # Z-score
    diff5_zscore = calc_z_score(diff5)

    # pstr for real time rewards timing
    diff5_pstr = np.zeros((n_vars, samples_window * 2 + 1))
    diff5_zscore_pstr = np.zeros((n_vars, samples_window * 2 + 1))
    for i in range(n_vars):
        diff5_pstr[i] = np.mean(calc_pstr(reward, diff5[i], samples_window), axis=0)
        diff5_zscore_pstr[i] = np.mean(calc_pstr(reward, diff5_zscore[i], samples_window), axis=0)

    # simulated reward pstr
    diff5_sim_pstr_mat, diff5_peaks_inds_arr = pstr_process(diff5, samples_window, None, min_reward_interval, prominence_th=True, ex_list=reward, ex_n=exclude_post_rewards_frames)
    diff5_zscore_sim_pstr_mat_fix_th, diff5_zscore_peaks_inds_arr_fix_th = pstr_process(diff5_zscore, samples_window, fixed_threshold, min_reward_interval, ex_list=reward, ex_n=exclude_post_rewards_frames)
    diff5_zscore_sim_pstr_mat_session_th, diff5_zscore_peaks_inds_arr_session_th = pstr_process(diff5_zscore, samples_window, sess_threshold, min_reward_interval, ex_list=reward, ex_n=exclude_post_rewards_frames)
    diff5_zscore_peaks_inds_arr_fix_th_conv_10k = smoo_rewards(diff5_zscore_peaks_inds_arr_fix_th, (n_vars, n_samples))
    diff5_zscore_peaks_inds_arr_session_th_conv_10k = smoo_rewards(diff5_zscore_peaks_inds_arr_session_th, (n_vars, n_samples))

    # area under the curve (AUC) and mooving average AUC
    diff5_auc = np.mean(diff5, axis=1)
    diff5_zscore_auc = np.mean(diff5_zscore, axis=1)
    diff5_moving_avg_auc = moving_average_auc(diff5, auc_window_length)
    diff5_zscore_moving_avg_auc = moving_average_auc(diff5_zscore, auc_window_length)


#############################################################################################
#############################################################################################
#############################################################################################
    print("writing to tiff")
    with h5py.File(dataset_path, 'a') as f:
        mouse_grp = f[mouse_id]
        session_grp = mouse_grp[session_name]
        if 'post_session_analysis' in session_grp.keys():
           del session_grp['post_session_analysis']

        eval_grp = session_grp.create_group('post_session_analysis')
        #################################################################################
        if 'dff' not in eval_grp.keys():
            traces_grp = eval_grp.create_group('dff')
        else:
            traces_grp = eval_grp['dff']

        traces_grp.create_dataset('traces', data=traces)
        traces_grp.create_dataset('traces_pstr', data=traces_pstr)
        traces_grp.create_dataset('traces_pstr_sim_reward', data=traces_sim_pstr_mat)
        reward_timing_grp = traces_grp.create_group('traces_sim_reward_timing')
        for i in range(n_vars):
            reward_timing_grp.create_dataset(rois_key[i], data=dff_peaks_inds_arr[i])

        traces_grp.create_dataset('traces_auc', data=traces_auc)
        traces_grp.create_dataset('traces_moving_avg_auc', data=traces_moving_avg_auc)

        traces_grp.create_dataset('zscore', data=traces_zscore)
        traces_grp.create_dataset('zscore_pstr', data=traces_zscore_pstr)
        traces_grp.create_dataset(f'zscore_pstr_sim_reward_fix_th', data=traces_zscore_sim_pstr_mat_fix_th)
        reward_timing_grp = traces_grp.create_group('zscore_sim_reward_timing_fix_th')
        for i in range(n_vars):
            reward_timing_grp.create_dataset(rois_key[i], data=dff_zscore_peaks_inds_arr_fix_th[i])

        traces_grp.create_dataset(f'zscore_pstr_sim_reward_session_th', data=traces_zscore_sim_pstr_mat_session_th)
        reward_timing_grp = traces_grp.create_group('zscore_sim_reward_timing_session_th')
        for i in range(n_vars):
            reward_timing_grp.create_dataset(rois_key[i], data=dff_zscore_peaks_inds_arr_session_th[i])

        traces_grp.create_dataset('zscore_auc', data=traces_zscore_auc)
        traces_grp.create_dataset('zscore_moving_avg_auc', data=traces_zscore_moving_avg_auc)

        traces_grp.create_dataset('zscore_convoluted_peaks_kernel_10k_fix_th', data=dff_zscore_peaks_inds_arr_fix_th_conv_10k)
        traces_grp.create_dataset('zscore_convoluted_peaks_kernel_10k_session_th', data=dff_zscore_peaks_inds_arr_session_th_conv_10k)

        #################################################################################
        delta_5_grp = eval_grp.create_group("dff_delta5")

        delta_5_grp.create_dataset('traces', data=diff5)
        delta_5_grp.create_dataset('traces_pstr', data=diff5_pstr)
        delta_5_grp.create_dataset('traces_pstr_sim_reward', data=diff5_sim_pstr_mat)
        reward_timing_grp = delta_5_grp.create_group('traces_sim_reward_timing')
        for i in range(n_vars):
            reward_timing_grp.create_dataset(rois_key[i], data=diff5_peaks_inds_arr[i])

        delta_5_grp.create_dataset('traces_auc', data=diff5_auc)
        delta_5_grp.create_dataset('traces_moving_avg_auc', data=diff5_moving_avg_auc)

        delta_5_grp.create_dataset('zscore', data=diff5_zscore)
        delta_5_grp.create_dataset('zscore_pstr', data=diff5_zscore_pstr)
        delta_5_grp.create_dataset(f'zscore_pstr_sim_reward_fix_th', data=diff5_zscore_sim_pstr_mat_fix_th)
        reward_timing_grp = delta_5_grp.create_group('zscore_sim_reward_timing_fix_th')
        for i in range(n_vars):
            reward_timing_grp.create_dataset(rois_key[i], data=diff5_zscore_peaks_inds_arr_fix_th[i])

        delta_5_grp.create_dataset(f'zscore_pstr_sim_reward_session_th', data=diff5_zscore_sim_pstr_mat_session_th)
        reward_timing_grp = delta_5_grp.create_group('zscore_sim_reward_timing_session_th')
        for i in range(n_vars):
            reward_timing_grp.create_dataset(rois_key[i], data=diff5_zscore_peaks_inds_arr_session_th[i])

        delta_5_grp.create_dataset('zscore_auc', data=diff5_zscore_auc)
        delta_5_grp.create_dataset('zscore_moving_avg_auc', data=diff5_zscore_moving_avg_auc)

        delta_5_grp.create_dataset('zscore_convoluted_peaks_kernel_10k_fix_th', data=diff5_zscore_peaks_inds_arr_fix_th_conv_10k)
        delta_5_grp.create_dataset('zscore_convoluted_peaks_kernel_10k_session_th', data=diff5_zscore_peaks_inds_arr_session_th_conv_10k)

        #################################################################################

        stats_grp = eval_grp.create_group("analysis_parameters")
        stats_grp.create_dataset("temporal_window_ms", data=temporal_window)
        stats_grp.create_dataset("smoo_kernel_size", data=smoo_kernel_size)
        stats_grp.create_dataset("max_f_to_peak", data=max_f_to_peak)
        stats_grp.create_dataset("fixed_threshold", data=fixed_threshold)
        stats_grp.create_dataset("auc_window_length", data=auc_window_length)
        # stats_grp.create_dataset("excluded_frames", data=sessions_exclution_list[session_name])

