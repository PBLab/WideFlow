import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from scipy.ndimage import map_coordinates
from skimage.transform import PiecewiseAffineTransform, warp_coords


def select_matching_points(src_np, dst_np, n_pairs, src_p_init=[], dst_p_init=[]):

    src_np = np.divide(src_np - np.min(src_np), np.max(src_np) - np.min(src_np) + np.finfo(np.float32).eps)
    src = cv2.UMat(np.stack((src_np, src_np, src_np), axis=2))
    dst_np = np.divide(dst_np - np.min(dst_np), np.max(dst_np) - np.min(dst_np) + np.finfo(np.float32).eps)
    dst = cv2.UMat(np.stack((dst_np, dst_np, dst_np), axis=2))

    r = 4

    def select_point(event, x, y, flags, param):
        del_p_idx = -1
        if event == cv2.EVENT_LBUTTONDOWN and len(match_p_src) < n_pairs and param == 1:
            for i, p in enumerate(match_p_src):
                dist = np.linalg.norm(np.array(p) - np.array((x, y)))
                if dist < r:
                    del_p_idx = i
                    break
            if del_p_idx == -1:
                match_p_src.append((x, y))
            else:
                del match_p_src[del_p_idx]

            src_circ = cv2.UMat(np.stack((src_np, src_np, src_np), axis=2))
            for p in match_p_src:
                cv2.circle(src_circ, p, r, (0, 0, 1), 2)
            cv2.imshow("source", src_circ)

        elif event == cv2.EVENT_LBUTTONDOWN and len(match_p_dst) < n_pairs and param == 2:
            for i, p in enumerate(match_p_dst):
                dist = np.linalg.norm(np.array(p) - np.array((x, y)))
                if dist < r:
                    del_p_idx = i
                    break
            if del_p_idx == -1:
                match_p_dst.append((x, y))
            else:
                del match_p_dst[del_p_idx]

            dst_circ = cv2.UMat(np.stack((dst_np, dst_np, dst_np), axis=2))
            for p in match_p_dst:
                cv2.circle(dst_circ, p, r, (0, 1, 0), 2)
            cv2.imshow("Allen Cortex Map", dst_circ)

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
    for p in src_p_init:
        match_p_src.append((p[1], p[0]))
        cv2.circle(src, (p[1], p[0]), 4, (0, 0, 1), 2)
        cv2.imshow("source", src)
    for p in dst_p_init:
        match_p_dst.append((p[1], p[0]))
        cv2.circle(dst, (p[1], p[0]), 4, (0, 1, 0), 2)
        cv2.imshow("Allen Cortex Map", dst)
    n_points = 2 * n_pairs
    while (len(match_p_src)+len(match_p_dst)) < n_points:
        print(f"{len(match_p_dst)} out of {n_pairs} matching pairs where selected")
        print('Press any key when finished marking the points!! ')
        cv2.waitKey(0)

    match_p_src = np.array(match_p_src, dtype=float)
    match_p_dst = np.array(match_p_dst, dtype=float)
    cv2.destroyAllWindows()

    return match_p_src, match_p_dst


class MatchingPointSelector:
    def __init__(self, image_src, image_dst, match_p_src=None, match_p_dst=None, n_matching_pairs=17):
        self.image_src = image_src
        self.image_dst = image_dst
        self.n_matching_pairs = n_matching_pairs
        self.dst_n_rows, self.dst_n_cols = image_dst.shape
        self.src_n_rows, self.src_n_cols = image_src.shape
        self.match_p_src = match_p_src
        self.match_p_dst = match_p_dst

        self.src_cols = None
        self.src_rows = None
        self.image_warp = self.warp_image()

        self.fig = plt.figure()
        self.ax = plt.gca()
        plt.imshow(self.image_warp)

        self.ax_accept = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.b_accept = Button(self.ax_accept, 'Accept')
        self.b_accept.on_clicked(self.accept)

        self.ax_repeat = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.b_repeat = Button(self.ax_repeat, 'Repeat')
        self.b_repeat.on_clicked(self.repeat)
        plt.show()

    def warp_image(self):
        if self.match_p_src is None or self.match_p_dst is None:
            image_src = self.image_src
            image_dst = self.image_dst
            self.match_p_src = [[0, 0], [0, self.src_n_cols], [self.src_n_rows, 0], [self.src_n_rows, self.src_n_cols]]
            self.match_p_dst = [[0, 0], [0, self.dst_n_cols], [self.dst_n_rows, 0], [self.dst_n_rows, self.dst_n_cols]]
            self.match_p_src, self.match_p_dst = select_matching_points(image_src, image_dst, self.n_matching_pairs,
                                                                        self.match_p_src, self.match_p_dst)

        tform = PiecewiseAffineTransform()
        tform.estimate(self.match_p_src, self.match_p_dst)
        warp_coor = warp_coords(tform.inverse, (self.dst_n_rows, self.dst_n_cols))
        self.src_cols = np.reshape(warp_coor[0], (self.dst_n_rows * self.dst_n_cols, 1))
        self.src_rows = np.reshape(warp_coor[1], (self.dst_n_rows * self.dst_n_cols, 1))

        image_warp = map_coordinates(self.image_src, [self.src_cols, self.src_rows])
        image_warp = np.reshape(image_warp, (self.dst_n_rows, self.dst_n_cols))
        self.match_p_src = None
        self.match_p_dst = None
        return image_warp

    def accept(self, event):
        plt.close()

    def repeat(self, event):
        self.fig.set_visible(not self.fig.get_visible())
        plt.draw()

        self.image_warp = self.warp_image()

        self.fig.set_visible(not self.fig.get_visible())
        plt.draw()
        self.ax.imshow(self.image_warp)


