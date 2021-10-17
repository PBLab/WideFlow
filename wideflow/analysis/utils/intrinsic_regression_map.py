import numpy as np
from scipy.signal import convolve2d


def calc_regression_map(data, hemo_data, smooth_fac):
    n_samples, n_rows, n_cols = data.shape

    kernel = np.ones((smooth_fac, smooth_fac)) / np.square(smooth_fac)
    for i, frame in enumerate(data):
        data[i] = convolve2d(frame, kernel, 'same')
    for i, frame in enumerate(hemo_data):
        hemo_data[i] = convolve2d(frame, kernel, 'same')

    regression_coeff = np.zeros((2, n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            [theta, _, _, _] = np.linalg.lstsq(
                np.stack((hemo_data[:, i, j], np.ones((n_samples,))), axis=1),
                data[:, i, j],
                rcond=None)
            regression_coeff[0][i, j] = theta[0]
            regression_coeff[1][i, j] = theta[1]

    return regression_coeff


# def update_regression_model(data, hemo_data, regression_coeff):
#     n_samples, n_rows, n_cols = data.shape
#     for i in range(n_rows):
#         for j in range(n_cols):
#             regression_coeff += y-