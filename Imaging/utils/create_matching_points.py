import cv2
import numpy as np
import scipy.io


def select_matching_points(src, dst, n_pairs):

    def select_point(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN and len(match_p_src) < n_pairs and param == 1:
            match_p_src.append((x, y))
            cv2.circle(src, (x, y), 4, (0, 0, 255), 2)
            cv2.imshow("source", src)
        elif event == cv2.EVENT_LBUTTONDOWN and len(match_p_dst) < n_pairs and param == 2:
            match_p_dst.append((x, y))
            cv2.circle(dst, (x, y), 4, (0, 255, 0), 2)
            cv2.imshow("Allen Cortex Map", dst)

    cv2.namedWindow("source", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("source", 600, 600)
    cv2.setMouseCallback("source", select_point, param=1)
    cv2.imshow("source", src)
    cv2.moveWindow("source", 1, 1)

    cv2.namedWindow("Allen Cortex Map", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Allen Cortex Map", 600, 600)
    cv2.setMouseCallback("Allen Cortex Map", select_point, param=2)
    cv2.imshow("Allen Cortex Map", dst)
    cv2.moveWindow("Allen Cortex Map", 650, 1)

    # keep looping until n_pairs points have been selected
    match_p_dst = []
    match_p_src = []
    n_points = 2 * n_pairs
    while (len(match_p_src)+len(match_p_dst)) < n_points:
        print(f"{len(match_p_dst)} out of {n_pairs} matching pairs where selected")
        print('Press any key when finished marking the points!! ')
        cv2.waitKey(0)

    match_p_src = np.array(match_p_src, dtype=float)
    match_p_dst = np.array(match_p_dst, dtype=float)
    cv2.destroyAllWindows()

    return match_p_src, match_p_dst


# match_p_src, match_p_dst = select_matching_points(np.random.random((100, 100)), np.random.random((100, 100)), 13)
# print(match_p_dst, match_p_src)