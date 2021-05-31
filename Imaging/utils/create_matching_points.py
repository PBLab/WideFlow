import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from scipy.ndimage import map_coordinates
from skimage.transform import PiecewiseAffineTransform, warp_coords


def select_matching_points(src, dst, n_pairs):

    def select_point(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN and len(match_p_src) < n_pairs and param == 1:
            match_p_src.append((x, y))
            cv2.circle(src, (x, y), 4, (30255, 30255, 30255), 2)
            cv2.imshow("source", src)
        elif event == cv2.EVENT_LBUTTONDOWN and len(match_p_dst) < n_pairs and param == 2:
            match_p_dst.append((x, y))
            cv2.circle(dst, (x, y), 4, (255, 255, 255), 2)
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


class ApprovalFigure:
    def __init__(self, image_src, image_dst, match_p_src=None, match_p_dst=None, n_matching_pairs=17):
        self.image_src = image_src
        self.image_dst = image_dst
        self.n_matching_pairs = n_matching_pairs
        self.n_rows, self.n_cols = image_dst.shape
        self.match_p_src = match_p_src
        self.match_p_dst = match_p_dst

        self.dst_cols = None
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
            self.match_p_src, self.match_p_dst = select_matching_points(image_src, image_dst, self.n_matching_pairs)

        tform = PiecewiseAffineTransform()
        tform.estimate(self.match_p_src, self.match_p_dst)
        warp_coor = warp_coords(tform.inverse, (self.n_rows, self.n_cols))
        self.src_cols = np.reshape(warp_coor[0], (self.n_rows * self.n_cols, 1))
        self.src_rows = np.reshape(warp_coor[1], (self.n_rows * self.n_cols, 1))

        image_warp = map_coordinates(self.image_src, [self.src_cols, self.src_rows])
        image_warp = np.reshape(image_warp, (self.n_rows, self.n_cols))
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



# from utils.load_tiff import load_tiff
# import h5py
#
#
# vid_path = "C:\\Users\\motar\\PycharmProjects\\WideFlow\\data\\A_thy1\\A_thy1_ch1_32frames.ome.tif"
# vid = load_tiff(vid_path)
# image = vid[0, :, :]
#
# cortex_file_path = "C:\\Users\\motar\\PycharmProjects\\WideFlow\\data\\cortex_map\\allen_2d_cortex.h5"
# with h5py.File(cortex_file_path) as f:
#     c_map = np.transpose(f["map"][()])
# n_rows, n_cols = c_map.shape
#
# # match_p_src = np.array(
# #             [[401., 264.], [182., 438.], [191., 834.], [395., 822.], [453., 750.], [518., 827.], [756., 820.],
# #              [744., 443.], [573., 259.], [448., 254.], [455., 501.], [436., 389.], [450., 622.]])
# # match_p_dst = np.array(
# #             [[155., 6.], [13., 118.], [17., 287.], [121., 287.], [167., 237.], [214., 287.], [326., 286.], [324., 114.],
# #              [242., 13.], [182., 5.], [167., 124.], [169., 64.], [167., 181.]])
#
# appf = ApprovalFigure(image, c_map * np.random.random((n_rows, n_cols)))#, match_p_src=match_p_src, match_p_dst=match_p_dst)
#
# src_cols = appf.src_cols
# src_rows = appf.src_rows