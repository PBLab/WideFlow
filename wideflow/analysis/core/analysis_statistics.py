from analysis.utils.peristimulus_time_response import calc_pstr, calc_sdf
import numpy as np

delta_t = [100, 200]
bin_width = 5
sdf_sigma = 5


def analysis_statistics(rois_traces, metadata, config):
    global_params = {"delta_t": delta_t, "bin_width": bin_width, "sdf_sigma": sdf_sigma, 'threshold_nstd': 2}

    # evaluate neuronal response
    n_channels = len(list(rois_traces.keys()))
    neuronal_response_stats = {}
    cue = np.array(metadata["cue"])
    if n_channels > 1:
        cue = cue[::n_channels]
        for i in range(1, n_channels):
            cue = np.maximum(cue, np.array(metadata["cue"])[i::n_channels])

    for ch_key, ch_val in rois_traces.items():
        neuronal_response_stats[ch_key] = {}
        for roi_key, roi_val in ch_val.items():
            neuronal_resp = calc_pstr(cue, roi_val, delta_t)
            rois_pstr_stats, rois_str_pre_stats, rois_str_post_stats = analyze_pstr(neuronal_resp)
            std = np.std(roi_val)
            mean = np.mean(roi_val)
            neuronal_response_stats[ch_key][roi_key] = {
                "pstr": np.mean(neuronal_resp, axis=0),
                "pstr_stats": rois_pstr_stats,
                "str_pre_stats": rois_str_pre_stats,
                "str_post_stats": rois_str_post_stats,
                "std": std,
                "mean": mean,
                "delta_t": delta_t
            }

    # evaluate behavioral response
    behavioral_resp = calc_sdf(metadata["cue"], 1-np.array(metadata["serial_readout"]), delta_t, sdf_sigma)
    p_lick_trials, p_lick_pre, p_lick_post = analyze_sdf(behavioral_resp)
    mean_spike_rate = len(np.where(1-np.array(metadata["serial_readout"] == 1))[0]) / len(np.array(metadata["serial_readout"]))
    behavioral_response_prob = {
        "delta_t": delta_t,
        "sdf": np.mean(behavioral_resp, axis=0),
        "mean_spike_rate": mean_spike_rate,
        "p_lick_trials": p_lick_trials,
        "p_lick_pre": p_lick_pre,
        "p_lick_post": p_lick_post
    }

    return neuronal_response_stats, behavioral_response_prob, global_params


def analyze_pstr(pstr):
    # inter trials analysis
    pstr_smoo = np.ndarray(pstr.shape, dtype=pstr.dtype)
    pre_str_peak, pre_str_mu, pre_str_max, pre_str_std, pre_str_med = [], [], [], [], []
    post_str_peak, post_str_mu, post_str_max, post_str_std, post_str_med = [], [], [], [], []
    for i, pstri in enumerate(pstr):
        pstr_smoo[i] = np.convolve(pstri, np.ones((5,)), mode='same')

        str_pre = pstr_smoo[i][:delta_t[0]]
        pre_str_peak.append(np.argmax(str_pre))
        pre_str_mu.append(np.mean(str_pre))
        pre_str_max.append(str_pre[pre_str_peak[i]])
        pre_str_std.append(np.std(str_pre))
        pre_str_med.append(np.median(str_pre))

        str_post = pstr_smoo[i][delta_t[0]:]
        post_str_peak.append(np.argmax(str_post))
        post_str_mu.append(np.mean(str_post))
        post_str_max.append(str_post[post_str_peak[i]])
        post_str_std.append(np.std(str_post))
        post_str_med.append(np.median(str_pre))

    pstr_stats = {
        "pre_stim":{
            "peak_delay": pre_str_peak,
            "average": pre_str_mu,
            "max": pre_str_max,
            "std": pre_str_std,
            "median": pre_str_med
        },
        "post_stim": {
            "peak_delay": post_str_peak,
            "average": post_str_mu,
            "max": post_str_max,
            "std": post_str_std,
            "median": post_str_med
        }
    }
    # cross trials analysis
    pstr_avg = np.mean(pstr_smoo, axis=0)
    str_pre = pstr_avg[:delta_t[0]]
    str_pre_stats = {
        "peak_delay": np.argmax(str_pre),
        "average": np.mean(str_pre),
        "max": np.max(str_pre),
        "std": np.std(str_pre),
        "median": np.median(str_pre),
    }
    str_post = pstr_avg[delta_t[0]:]
    str_post_stats = {
        "peak_delay": np.argmax(str_post),
        "average": np.mean(str_post),
        "max": np.max(str_post),
        "std": np.std(str_post),
        "median": np.median(str_post),
    }

    return pstr_stats, str_pre_stats, str_post_stats


def analyze_sdf(sdf):
    p_trials = []
    # inter trials analysis
    for i, sdfi in enumerate(sdf):
        # sdfi_norm = sdfi / np.sum(sdfi)  # normalize to get probability density
        p_pre = np.sum(sdfi[:delta_t[0]])
        p_post = np.sum(sdfi[delta_t[0]:])
        p_trials.append([p_pre, p_post])

    # cross trials analysis
    sdf_avg = np.mean(sdf, axis=0)
    sdf_norm = sdf_avg / np.sum(sdf_avg)  # normalize to get probability density
    p_pre_avg = np.sum(sdf_norm[:delta_t[0]])  # pre-stimulus response probability
    p_post_avg = np.sum(sdf_norm[delta_t[0]:])  # post-stimulus response probability

    return p_trials, p_pre_avg, p_post_avg

