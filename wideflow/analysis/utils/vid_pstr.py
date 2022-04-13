import numpy as np


def vid_pstr(vid, cues, delta_t):
    if type(delta_t) is int:
        delta_t = [delta_t, delta_t]

    cues_inds = np.where(np.array(cues) == 1)[0]
    nt, ny, nx = vid.shape
    pstr = np.zeros((delta_t[0] + delta_t[1] + 1, ny, nx))
    n = 0
    for idx in cues_inds:
        if (idx - delta_t[0] < 0) or (idx + delta_t[1] + 1 > nt):
            continue
        pstr += vid[idx - delta_t[0]: idx + delta_t[1] + 1, :, :]
        n += 1
    return pstr / n
