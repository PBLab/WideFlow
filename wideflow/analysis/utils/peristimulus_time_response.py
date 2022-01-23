import numpy as np


def calc_pstr(stimuli, responses, delta_t, bin_width=1):
    stimuli = np.pad(np.array(stimuli), delta_t)
    responses = np.pad(np.array(responses), delta_t)

    if not isinstance(delta_t, list) and not isinstance(delta_t, np.ndarray):
        delta_t = [delta_t, delta_t + 1]

    if bin_width > 1:
        bins = np.arange(-delta_t[0], delta_t[1], bin_width)
        hist_inds = np.digitize(np.arange(-delta_t, delta_t), bins) - 1

    stimuli_inds = np.where(stimuli == 1)[0]
    pstr = np.zeros((len(stimuli_inds), int(np.ceil((delta_t[0] + delta_t[1])/ bin_width))))
    for i, idx in enumerate(stimuli_inds):
        pstri = responses[idx-delta_t[0]: idx+delta_t[1]]
        if bin_width > 1:
            np.add.at(pstr[i], hist_inds, pstri)
        else:
            pstr[i] = pstri

    return pstr


def calc_sdf(stimuli, responses, delta_t, sigma):
    stimuli = np.pad(np.array(stimuli), delta_t)
    responses = np.pad(np.array(responses), delta_t)

    stimuli_inds = np.where(stimuli == 1)[0]
    if not isinstance(delta_t, list) and not isinstance(delta_t, np.ndarray):
        delta_t = [delta_t, delta_t]

    trial_len = delta_t[0] + delta_t[1]
    sdf = np.zeros((len(stimuli_inds), trial_len))
    time_vec = np.arange(trial_len)
    for i, stim_idx in enumerate(stimuli_inds):
        pstri = responses[stim_idx-delta_t[0]: stim_idx+delta_t[1]]
        resp_inds = np.where(pstri == 1)[0]
        gauss = np.ndarray((len(resp_inds), trial_len))
        for j, mu in enumerate(resp_inds):
            # calculate the gaussian for each response
            p1 = -0.5 * np.power((time_vec - mu) / sigma, 2)
            p2 = (sigma * np.sqrt(2*np.pi))
            gauss[j] = np.exp(p1) / p2
        sdf[i] = np.sum(gauss, axis=0)

    return sdf

