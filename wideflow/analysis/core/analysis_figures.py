import matplotlib.pyplot as plt
import numpy as np
from wideflow.analysis.utils.dynamic_threshold_scatter_activity import dynamic_threshold
from wideflow.analysis.utils.peristimulus_time_response import calc_pstr, calc_sdf


def plot_figures(results_path, metadata, config, rois_traces, neuronal_response_stats, behavioral_response_stats, statistics_global_param, rois_dict):
    if "rois_names" in config["analysis_pipeline_config"]["args"]:
        metric_rois_names = config["analysis_pipeline_config"]["args"]["rois_names"]
    else:
        metric_rois_names = []

    cue, serial_readout, timestamp = metadata["cue"], metadata["serial_readout"], metadata["timestamp"]
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

    neuronal_resp = {}
    for ch_key, ch_val in rois_traces.items():
        neuronal_resp[ch_key] = {}
        for roi_key, roi_val in ch_val.items():
            if roi_key in metric_rois_names:
                neuronal_resp[ch_key][roi_key] = calc_pstr(cue_ch, roi_val, delta_t)

    plot_trials_pstr(results_path, neuronal_resp, dt, delta_t, rois_dict)
    plot_pstr(results_path, neuronal_response_stats, dt, delta_t, rois_dict, metric_rois_names)
    plot_sdf(results_path, behavioral_response_stats, dt, delta_t)
    plot_cue_response(results_path, cue, serial_readout)
    plot_rois_traces(results_path, rois_traces, rois_dict)
    plot_th_traces_scatter_plot(results_path, rois_traces, n_std, rois_dict)
    plot_th_traces_pstr(results_path, cue_ch, neuronal_response_stats, n_std, rois_dict, dt, delta_t, sigma)


def save_figure(path):
    manager = plt.get_current_fig_manager()
    # manager.resize(3000, 1500)
    manager.window.showMaximized()
    plt.tight_layout()
    plt.pause(2)
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def plot_pstr(results_path, neuronal_response_stats, dt, delta_t, rois_dict, bold_list):
    frames_time = np.arange(-delta_t[0]*dt, delta_t[1]*dt, dt)
    legend_list = []
    for ch, ch_val in neuronal_response_stats.items():
        plt.figure()
        for roi, roi_pstr_stats in ch_val.items():
            if roi in bold_list:
                plt.plot(frames_time, roi_pstr_stats["pstr"], linewidth=2)
            else:
                plt.plot(frames_time, roi_pstr_stats["pstr"], linewidth=0.5)
            legend_list.append(rois_dict[roi]['name'])

        plt.legend(legend_list, ncol=3)
        plt.ylabel("pstr")
        plt.xlabel("Time [ms]")
        plt.title(f'ROIs Peristimulus Time Response - channel {ch}')
        plt.axvline(x=0, color='k')

        save_figure(results_path + 'rois_pstr_of_' + ch + '.png')


def plot_trials_pstr(results_path, neuronal_response, dt, delta_t, rois_dict):
    frames_time = np.arange(-delta_t[0] * dt, delta_t[1] * dt, dt)
    for ch, ch_val in neuronal_response.items():
        for roi, roi_pstr in ch_val.items():
            plt.figure()
            legend_list = []
            for i, trial in enumerate(roi_pstr):
                plt.plot(frames_time, trial)
                legend_list.append(f'trial {i}')

            plt.legend(legend_list)
            plt.ylabel("pstr")
            plt.xlabel("Time [ms]")
            name = rois_dict[roi]['name']
            plt.title(f'{name} Peristimulus Time Response - channel {ch}')
            plt.axvline(x=0, color='k')
            save_figure(results_path + name + 'pstr_of_' + ch + '.png')


def plot_sdf(results_path, behavioral_response_stats, dt, delta_t):
    plt.figure(figsize=(30.0, 10.0))

    frames_time = np.arange(-delta_t[0] * dt, delta_t[1] * dt, dt)
    mean_spike_rate = behavioral_response_stats["mean_spike_rate"]

    plt.plot(frames_time, behavioral_response_stats["sdf"])
    plt.plot(frames_time, mean_spike_rate * np.ones((len(frames_time), )))

    plt.ylabel("sdf")
    plt.xlabel("Time [ms]")
    plt.legend(["sdf", "mean spikes rate"])
    plt.title("Stim-Lick SDF")
    plt.axvline(x=0, color='k')

    save_figure(results_path + 'stim_lick_sdf.png')


def plot_cue_response(results_path, cue, serial_readout):
    cue = np.array(cue)
    serial_readout = np.array(serial_readout)

    plt.figure(figsize=(30.0, 10.0))
    plt.plot(2*cue)
    plt.plot(1.5*(1 - serial_readout))

    kernel = np.ones((5000,))
    conv_cue = np.convolve(np.array(cue), kernel, 'same')
    conv_cue = conv_cue / conv_cue.max()
    plt.plot(conv_cue)

    conv_resp = np.convolve(1 - np.array(serial_readout), kernel, 'same')
    conv_resp = conv_resp / conv_resp.max()
    plt.plot(conv_resp)


    plt.legend(["cues", "response", "convolved cues - kernel 5k frames", "convolved response - kernel 5k frames"])
    plt.title("cues and responses")
    plt.xlabel("frames")
    plt.title("Session Cues and Licks Timing")

    save_figure(results_path + 'cues_responses_plot.png')


def plot_rois_traces(results_path, traces, rois_dict):
    legend = []
    for ch, ch_val in traces.items():
        plt.figure()
        for roi, trace in ch_val.items():
            legend.append(rois_dict[roi]['name'])
            plt.plot(trace)

        plt.legend(legend, ncol=3)
        plt.title(f"ROIs traces - channel {ch}")
        plt.xlabel("frames")
        plt.ylabel("dFF")

        save_figure(results_path + 'rois_traces_' + ch + '.png')


def plot_th_traces_scatter_plot(results_path, traces, n_std, rois_dict):
    n_rois = len(traces['channel_0'])
    n_samples = len(traces['channel_0']['roi_1'])
    for ch, ch_val in traces.items():
        fig, ax = plt.subplots(1, 1)
        labels = []
        traces_mat = np.zeros((n_rois, n_samples))
        for i, (roi, trace) in enumerate(ch_val.items()):
            traces_mat[i] = trace
            labels.append(rois_dict[roi]['name'])

        scatter_mat = dynamic_threshold(traces_mat, n_std)
        plt.imshow(scatter_mat, aspect='auto')
        ax.set_yticks(np.arange(n_rois))
        ax.set_yticklabels(labels)
        plt.suptitle(f'ROIs Scatter Plot, threshold of {n_std} std - channel {ch}')

        save_figure(results_path + 'threshold_rois_traces_scatter_plot' + ch + '.png')


def plot_th_traces_pstr(results_path, stimuli, traces, n_std, rois_dict, dt, delta_t, sigma):
    n_rois = len(traces['channel_0'])
    n_samples = len(traces['channel_0']['roi_1'])
    frames_time = np.arange(-delta_t[0] * dt, delta_t[1] * dt, dt)
    for ch, ch_val in traces.items():
        plt.figure()
        legend_list = []
        traces_mat = np.zeros((n_rois, n_samples))
        for i, (roi, trace) in enumerate(ch_val.items()):
            traces_mat[i] = trace
            legend_list.append(rois_dict[roi]['name'])

        scatter_mat = dynamic_threshold(traces_mat, n_std)
        for i, trace in enumerate(scatter_mat):
            pstr = calc_sdf(stimuli, trace, delta_t, sigma)
            plt.plot(frames_time, np.mean(pstr, axis=0))

        plt.legend(legend_list, ncol=3)
        plt.ylabel("pstr")
        plt.xlabel("Time [ms]")
        plt.title(f'ROIs Peristimulus Time Response - channel {ch}')
        plt.axvline(x=0, color='k')
        save_figure(results_path + 'rois_threshold_traces_pstr_of_' + ch + '.png')

