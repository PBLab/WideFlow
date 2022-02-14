import matplotlib.pyplot as plt
import numpy as np
import pickle
from analysis.utils.peristimulus_time_response import calc_pstr
from utils.interleave_matrix import interleave_matrix


def plot_figures(results_path, metadata, config, rois_traces, neuronal_response_stats, behavioral_response_stats, statistics_global_param, rois_dict):
    if "rois_names" in config["analysis_pipeline_config"]["args"]:
        metric_rois_names = config["analysis_pipeline_config"]["args"]["rois_names"]
    else:
        metric_rois_names = []

    cue, serial_readout, timestamp, metric_result, threshold = metadata["cue"], metadata["serial_readout"], metadata["timestamp"], metadata["metric_result"], metadata["threshold"]
    timediff = np.array(timestamp)[1:] - np.array(timestamp)[:-1]
    dt = int(np.mean(timediff) * 1000)  # in milliseconds
    delta_t = statistics_global_param["delta_t"]
    n_std = statistics_global_param['threshold_nstd']
    sigma = statistics_global_param['sdf_sigma']

    n_channels = len(list(rois_traces.keys()))
    cue = np.array(metadata["cue"])
    if n_channels > 1:
        cue_ch = cue[::n_channels]
        for i in range(1, n_channels):
            cue_ch = np.maximum(cue_ch, np.array(metadata["cue"])[i::n_channels])
    else:
        cue_ch = cue

    threshold_ch = threshold[::2]
    metric_result_ch = metric_result[::2]

    neuronal_resp = {}
    for ch_key, ch_val in rois_traces.items():
        neuronal_resp[ch_key] = {}
        for roi_key, roi_val in ch_val.items():
            if roi_key in metric_rois_names:
                neuronal_resp[ch_key][roi_key] = calc_pstr(cue_ch, roi_val, delta_t)

    plot_trials_pstr(results_path, neuronal_resp['channel_0'], dt, delta_t, rois_dict)
    plot_segments_pstrs(results_path, cue_ch, rois_traces['channel_0'], metric_rois_names, dt, delta_t, 5, 2000)
    plot_pstr(results_path, neuronal_response_stats['channel_0'], dt, delta_t, rois_dict, metric_rois_names)
    plot_sdf(results_path, behavioral_response_stats, dt, delta_t)
    plot_cue_response(results_path, cue, serial_readout)

    plot_rois_traces(results_path, rois_traces['channel_0'], rois_dict)
    plot_std_th_traces_scatter_plot(results_path, rois_traces['channel_0'], threshold_ch, cue_ch, rois_dict)
    plot_metric_result(results_path, metric_result_ch, cue_ch, threshold_ch, rois_traces['channel_0'], metric_rois_names)


def plot_pstr(results_path, neuronal_response_stats, dt, delta_t, rois_dict, bold_list):
    frames_time = np.arange(-delta_t[0]*dt, delta_t[1]*dt, dt)
    legend_list = []
    pstr_mat = np.zeros((len(rois_dict), len(frames_time)))
    f, ax = plt.subplots()
    for i, (roi, roi_pstr_stats) in enumerate(neuronal_response_stats.items()):
        if roi in bold_list:
            ax.plot(frames_time, roi_pstr_stats["pstr"], linewidth=2)
        else:
            ax.plot(frames_time, roi_pstr_stats["pstr"], linewidth=0.5)
        pstr_mat[i] = roi_pstr_stats["pstr"]
        legend_list.append(rois_dict[roi]['name'])

    mean = np.mean(pstr_mat, axis=0)
    std = np.std(pstr_mat, axis=0)
    ax.plot(frames_time, mean, '--', 'k')
    ax.fill_between(frames_time, mean - std, mean + std, color='gray', alpha=0.2)
    legend_list.extend(['ROIs pstr mean', 'ROIs pstr std'])

    ax.legend(legend_list, ncol=3)
    ax.set_ylabel("pstr")
    ax.set_xlabel("Time [ms]")
    f.suptitle(f'ROIs Peristimulus Time Response')
    ax.axvline(x=0, color='k')

    save_figure(results_path + 'rois_pstr.fig.pickle', f)


def plot_trials_pstr(results_path, neuronal_response, dt, delta_t, rois_dict):
    frames_time = np.arange(-delta_t[0] * dt, delta_t[1] * dt, dt)
    for roi, roi_pstr in neuronal_response.items():
        f, ax = plt.subplots()
        legend_list = []
        for i, trial in enumerate(roi_pstr):
            ax.plot(frames_time, trial)
            legend_list.append(f'trial {i}')

        ax.legend(legend_list)
        ax.set_ylabel("pstr")
        ax.set_xlabel("Time [ms]")
        name = rois_dict[roi]['name']
        f.suptitle(f'{name} Peristimulus Time Response')
        ax.axvline(x=0, color='k')
        save_figure(results_path + name + 'pstr.fig.pickle', f)


def plot_segments_pstrs(results_path, cues, rois_traces, rois_names, dt, delta_t, n_partitions, frames_overlap=0):
    n_frames = len(rois_traces[list(rois_traces.keys())[0]])
    _n_frames = n_frames + (n_partitions - 1)*frames_overlap
    segment_length = int(_n_frames / n_partitions)
    segments_start_inds = np.arange(0, _n_frames, segment_length) - frames_overlap * np.arange(n_partitions)
    segments_end_inds = np.arange(0, _n_frames, segment_length) - frames_overlap * np.arange(n_partitions) + segment_length
    segments_inds = [[i0, i1] for (i0, i1) in zip(segments_start_inds, segments_end_inds)]

    legend_list = [f'segment {i} pstr - frames: {seg[0]} - {seg[1]}' for i, seg in enumerate(segments_inds)]
    frames_time = np.arange(-delta_t[0] * dt, delta_t[1] * dt, dt)
    for roi_name, roi_trace in rois_traces.items():
        if roi_name in rois_names:
            f, ax = plt.subplots()
            for seg_idx in segments_inds:
                pstr = calc_pstr(cues[seg_idx[0]:seg_idx[1]], roi_trace[seg_idx[0]:seg_idx[1]], delta_t)
                ax.plot(frames_time, np.mean(pstr, axis=0))
            f.suptitle(
                f'{roi_name} Peristimulus Time Response for {n_partitions} Consecutive Segments with Overlap of {frames_overlap} Frames')

            ax.axvline(x=0, color='k')
            ax.legend(legend_list)
            ax.set_ylabel("pstr")
            ax.set_xlabel("Time [ms]")

            save_figure(results_path + roi_name + '_session evolved pstr.fig.pickle', f)


def plot_sdf(results_path, behavioral_response_stats, dt, delta_t):
    f, ax = plt.subplots()

    frames_time = np.arange(-delta_t[0] * dt, delta_t[1] * dt, dt)
    mean_spike_rate = behavioral_response_stats["mean_spike_rate"]

    ax.plot(frames_time, behavioral_response_stats["sdf"])
    ax.plot(frames_time, mean_spike_rate * np.ones((len(frames_time), )))

    ax.set_ylabel("sdf")
    ax.set_xlabel("Time [ms]")
    ax.legend(["sdf", "mean spikes rate"])
    f.suptitle("Stim-Lick SDF")
    ax.axvline(x=0, color='k')

    save_figure(results_path + 'stim_lick_sdf.fig.pickle', f)


def plot_cue_response(results_path, cue, serial_readout):
    cue = np.array(cue)
    serial_readout = np.array(serial_readout)

    f, ax = plt.subplots()
    ax.plot(2*cue)
    ax.plot(1.5*(1 - serial_readout))

    kernel = np.ones((5000,))
    conv_cue = np.convolve(np.array(cue), kernel, 'same')
    conv_cue = conv_cue / conv_cue.max()
    ax.plot(conv_cue)

    conv_resp = np.convolve(1 - np.array(serial_readout), kernel, 'same')
    conv_resp = conv_resp / conv_resp.max()
    ax.plot(conv_resp)


    ax.legend(["cues", "response", "convolved cues - kernel 5k frames", "convolved response - kernel 5k frames"])
    f.suptitle("Session Cues and Licks Timing")
    ax.set_xlabel("frames")

    save_figure(results_path + 'cues_responses_plot.fig.pickle', f)


def plot_rois_traces(results_path, traces, rois_dict):
    legend = []
    f, ax = plt.subplots()
    for roi, trace in traces.items():
        legend.append(rois_dict[roi]['name'])
        ax.plot(trace)

    ax.legend(legend, ncol=3)
    f.suptitle(f"ROIs traces")
    ax.set_xlabel("frames")
    ax.set_ylabel("dFF")

    save_figure(results_path + 'rois_traces.fig.pickle', f)


def plot_std_th_traces_scatter_plot(results_path, rois_traces, threshold, cues, rois_dict):

    n_rois = len(rois_traces)
    f, ax = plt.subplots()

    traces_mat = np.array([v for _, v in rois_traces.items()])

    n_frames = traces_mat.shape[1]

    traces_mean = np.mean(traces_mat, axis=0)
    traces_std = np.std(traces_mat, axis=0)

    threshold = np.array(threshold)

    scatter_mat = ((traces_mat - traces_mean) / traces_std) >= threshold
    plt.imshow(scatter_mat, aspect='auto')

    # set x ticks labels and colors
    xticks = list(np.arange(0, n_frames, 5000))
    xticks_colors = ['k'] * len(xticks)
    xticks_labels = [str(x) for x in xticks]

    cues_ticks = np.where(cues)[0]
    cues_ticks_labels = ['c'] * len(cues_ticks)
    cues_ticks_colors = ['r'] * len(cues_ticks)

    xticks.extend(cues_ticks)
    xticks_labels.extend(cues_ticks_labels)
    xticks_colors.extend(cues_ticks_colors)

    xticks_nest = [[tic, lbl, col] for tic, lbl, col in sorted(zip(xticks, xticks_labels, xticks_colors))]
    xticks, xticks_labels, xticks_colors = [], [], []
    for t, l, c in xticks_nest:
        xticks.append(t)
        xticks_labels.append(l)
        xticks_colors.append(c)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks_labels)

    ylabels = [f'{k}: {rois_dict[k]["name"]}' for k, _ in rois_traces.items()]
    ax.set_yticks(np.arange(n_rois))
    ax.set_yticklabels(ylabels)
    f.suptitle(f'ROIs Scatter Plot, threshold - Metric Threshold')

    ax.grid()
    save_figure(results_path + 'threshold_rois_traces_scatter_plot.fig.pickle', f)


def plot_metric_result(results_path, metric_results, cues, threshold, rois_traces, metric_rois_names):
    metric_results = np.array(metric_results)
    n_frames = len(metric_results)

    traces_mat = np.array([v for k, v in rois_traces.items()])
    traces_n_frames = traces_mat.shape[1]
    if n_frames > traces_n_frames:
        dup = int(n_frames / traces_n_frames)
        traces_mat = interleave_matrix(traces_mat, dup)

    traces_mean = np.mean(traces_mat, axis=0)
    traces_std = np.std(traces_mat, axis=0)

    metric_traces = []
    for i, roi_name in enumerate(rois_traces.keys()):
        if roi_name in metric_rois_names:
            metric_traces.append(traces_mat[i])

    metric_traces = np.array(metric_traces)
    if metric_traces.ndim > 1:
        metric_traces = np.mean(metric_traces, axis=0)

    traces_std_steps = (metric_traces - traces_mean) / (traces_std + np.finfo(float).eps)
    f, ax = plt.subplots()
    ax.plot(metric_results, color='tab:red')
    ax.plot(np.arange(n_frames), traces_std_steps, '--', color='tab:red')
    ax.plot(threshold, color='g')
    ax.vlines(np.where(cues), ymin=np.min(metric_results), ymax=np.max(metric_results), color='k')
    ax.set_ylabel('metric result', color='tab:red')
    leg = ['metric_results', 'traces_std_steps', 'threshold', 'reward timing']

    ax2 = ax.twinx()
    ax2.plot(metric_traces, color='tab:blue')
    ax2.plot(traces_mean, '--', color='tab:blue')
    ax2.fill_between(np.arange(n_frames), traces_mean - 2*traces_std, traces_mean + 2*traces_std, color='gray', alpha=0.2)
    ax2.fill_between(np.arange(n_frames), traces_mean - traces_std, traces_mean + traces_std, color='gray', alpha=0.2)
    leg.extend(['metric ROIs traces mean', 'all ROIs traces mean', '1 std', '2 std'])

    ax2.set_ylabel('metric - ROIs DFF - trace ', color='tab:blue')
    ax2.set_xlabel('Frames')
    f.legend(leg)
    f.suptitle('Metric vs Threshold, metric ROI trace')
    save_figure(results_path + 'Metric vs Threshold vs ROI trace.fig.pickle', f)


def save_figure(path, f):
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.pause(2)
    f.tight_layout()
    pickle.dump(f, open(path, 'wb'))
    plt.close(f)



