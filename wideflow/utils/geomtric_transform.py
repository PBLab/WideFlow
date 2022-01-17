import cv2
import numpy as np


def compute_homography(mp_src, mp_dst, inliers_percent, max_err):
    _, n_match = mp_src.shape

    p = 0.99
    d = 4
    wn = np.power(1 - inliers_percent, d)
    k = int(np.log(1 - p) / np.log(1 - wn))

    max_inliers = np.ceil(n_match * inliers_percent)
    H_best = compute_homography_naive(mp_src, mp_dst)
    best_err = 1e9
    for i in range(k):
        sampled_idx = np.random.choice(range(n_match), np.random.randint(d, n_match), replace=False)
        sampled_mp_src = mp_src[:, sampled_idx]
        sampled_mp_sdt = mp_dst[:, sampled_idx]
        H = compute_homography_naive(sampled_mp_src, sampled_mp_sdt)
        fit_percent, dist_mse = test_homography(H, mp_src, mp_dst, max_err)
        if fit_percent * n_match >= max_inliers:
            if dist_mse < best_err:
                H_best = H

    return H_best


def compute_homography_naive(mp_src, mp_dst):
    '''
    Solving for the homography matrix parameters using the Direct Linear Transformation (DLT) algorithm
    :param mp_src: source matching points coordinates - 2d numpy array of size (2, n)
    :param mp_dst: destination matching points coordinates -2d numpy array of size (2, n)
    :return:
    '''

    _, n_match = mp_src.shape
    # convert the source and destination points to homogenous coordinates
    mp_src = np.vstack((mp_src, np.ones((1, n_match))))
    mp_dst = np.vstack((mp_dst, np.ones((1, n_match))))

    A = np.zeros((2 * n_match, 9))
    for i in range(n_match):
        A[2 * i] = np.concatenate((mp_src[:, i],
                    [0, 0, 0],
                    -mp_dst[0][i] * mp_src[:, i]))
        A[2 * i + 1] = np.concatenate(([0, 0, 0],
                        mp_src[:, i],
                        -mp_dst[1][i]*mp_src[:, i]))

    U, S, V = np.linalg.svd(A)
    H = V[8].reshape((3, 3))
    H = H / H[2, 2]
    return H


def test_homography(H, mp_src, mp_dst, max_err):
    _, n_match = mp_src.shape
    dist_mse = 0
    n_inliers = 0
    for i in range(n_match):
        xsrc, ysrc = mp_src[0, i], mp_src[1, i]
        xdst, ydst = mp_dst[0, i], mp_dst[1, i]
        x_err = xdst - (H[0, 0] * xsrc + H[0, 1] * ysrc + H[0, 2]) / (H[2, 0] * xsrc + H[2, 1] * ysrc + H[2, 2])
        y_err = ydst - (H[1, 0] * xsrc + H[1, 1] * ysrc + H[1, 2]) / (H[2, 0] * xsrc + H[2, 1] * ysrc + H[2, 2])
        if np.sqrt(x_err ** 2 + y_err ** 2) <= max_err:
            dist_mse += x_err ** 2 + y_err ** 2
            n_inliers += 1

    return n_inliers / n_match, dist_mse / (n_inliers+1e-9)