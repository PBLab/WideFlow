import matplotlib.pyplot as plt
import numpy as np
from wideflow.analysis.utils.dynamic_threshold_scatter_activity import dynamic_threshold


def plot_figures(results_path, metadata, rois_traces, neuronal_response_stats, behavioral_response_stats, statistics_global_param, rois_dict):
    cue, serial_readout, timestamp = metadata["cue"], metadata["serial_readout"], metadata["timestamp"]
    timediff = np.array(timestamp)[1:] - np.array(timestamp)[:-1]
    dt = int(np.mean(timediff) * 1000)  # in milliseconds

    delta_t = statistics_global_param["delta_t"]
    plot_pstr(results_path, neuronal_response_stats, dt, delta_t, rois_dict)
    plot_sdf(results_path, behavioral_response_stats, dt, delta_t)
    plot_cue_response(results_path, cue, serial_readout)
    plot_rois_traces(results_path, rois_traces, rois_dict)
    plot_th_traces_scatter_plot(results_path, rois_traces, statistics_global_param['threshold_nstd'], rois_dict)


def save_figure(path):
    manager = plt.get_current_fig_manager()
    manager.resize(3000, 1500)
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def plot_pstr(results_path, neuronal_response_stats, dt, delta_t, rois_dict):
    channels_keys = list(neuronal_response_stats.keys())

    frames_time = np.arange(-delta_t[0]*dt, delta_t[1]*dt, dt)
    legend_list = []
    for ch, ch_val in neuronal_response_stats.items():
        plt.figure()
        for roi, roi_pstr_stats in ch_val.items():
            plt.plot(frames_time, roi_pstr_stats["pstr"])
            legend_list.append(rois_dict[roi]['name'])

        plt.legend(legend_list, ncol=3)
        plt.ylabel("pstr")
        plt.xlabel("Time [ms]")
        plt.title(f'ROIs Peristimulus Time Response - channel {ch}')
        plt.axvline(x=0, color='k')

        save_figure(results_path + 'rois_pstr_of_' + ch + '.png')


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

    conv_resp = np.convolve(np.array(serial_readout), kernel, 'same')
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
    for ch, ch_val in traces.items():
        plt.figure()
        labels = []
        traces_mat = np.zeros(())
        for i, (roi, trace) in enumerate(ch_val.items()):
            traces_mat[i] = trace
            labels.append(rois_dict[roi]['name'])

        scatter_mat = dynamic_threshold(traces_mat, n_std)
        plt.imshow(scatter_mat, aspect='auto')

        save_figure(results_path + 'threshold_rois_traces_scatter_plot' + ch + '.png')