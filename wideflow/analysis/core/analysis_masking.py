import numpy as np


def mask(video, mask):
    for i in range(video.shape[0]):
        video[i] = np.multiply(video[i], mask)

    return video