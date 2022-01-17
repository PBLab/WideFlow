import numpy as np
from scipy.signal import fftconvolve


def find_2d_max_correlation_coordinates(image1, image2):
    corr = fftconvolve(image1, np.fliplr(np.flipud(image2)))
    (yi, xi) = np.unravel_index(np.argmax(corr), corr.shape)
    yi = yi - (corr.shape[0] - image1.shape[0])
    xi = xi - (corr.shape[1] - image1.shape[1])

    return xi, yi

