import numpy as np



def calc_dff(wf_data, bs_calc_method, dff_bs_n_frames=None):
    dff = {}
    for ch, video in wf_data.items():
        if bs_calc_method == "moving_avg":
            bs = calc_baseline_moving_avg(np.float32(video), dff_bs_n_frames)
        if bs_calc_method == "t_prj":
            bs = calc_baseline_t_prj(np.float32(video))

        dff[ch] = np.divide(video - bs, bs + np.finfo(np.float32).eps)

    return dff


def calc_baseline_moving_avg(video, dff_bs_n_frames):
    bs = np.ndarray(video.shape, dtype=video.dtype)
    bs[0] = video[0]
    for i in range(1, dff_bs_n_frames):
        bs[i] = np.mean(video[0: i, :, :], axis=0)

    for i in range(dff_bs_n_frames, video.shape[0]):
        bs[i] = np.mean(video[i - dff_bs_n_frames: i, :, :], axis=0)

    return bs


def calc_baseline_t_prj(video):
    bs = np.mean(video, axis=0)

    return bs