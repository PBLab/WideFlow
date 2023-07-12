import numpy as np
from skimage.transform import AffineTransform
from scipy.ndimage import affine_transform

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from utils.draggable_point import DraggablePoint


class InteractiveAffineTransform:
    def __init__(self, src_img, map, trans_points_pos=None):
        self.src_img = src_img
        self.map = map
        self.trans_points_pos = trans_points_pos

        self.src_nrows, self.src_ncols = self.src_img.shape
        self.map_nrows, self.map_ncols = self.map.shape

        # initiate gui figures
        self.fig_src, self.ax_src = plt.subplots()
        self.fig_src.suptitle('Source Image')
        mng = self.fig_src.canvas.manager
        mng.window.wm_geometry('+1+1')  # set figure position
        self.ax_src.imshow(self.src_img)

        self.ax_accept = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.b_accept = Button(self.ax_accept, 'Accept')
        self.b_accept.on_clicked(self.accept)

        self.ax_reset = plt.axes([0.85, 0.05, 0.1, 0.075])
        self.b_reset = Button(self.ax_reset, 'Reset')
        self.b_reset.on_clicked(self.reset)

        self.fig_dst, self.ax_dst = plt.subplots()
        self.fig_dst.suptitle('Affine Transformed Image')
        mng = self.fig_dst.canvas.manager
        mng.window.wm_geometry(f'+{self.src_nrows + 10}+1')

        self.td_offset = [
            [40, 40],
            [-40, 40],
            [-40, -40]
        ]
        self.tform = AffineTransform()
        self.src_cols = None
        self.dst_cols = None
        self.trans_points_pos, self.fixed_points_pos, self.draggable_point = self.initiate_transform_points()
        self.update_transform()

        self.cidrelease = self.fig_src.canvas.mpl_connect('button_release_event', self.on_release)
        plt.show()

    def initiate_transform_points(self):
        fixed_points_pos = np.array([
            [0, 0],
            [self.map_nrows, 0],
            [self.map_nrows, self.map_ncols]
        ], dtype=np.float64())

        if self.trans_points_pos is None:
            trans_points_pos = np.array([
                [0, 0],
                [self.src_nrows, 0],
                [self.src_nrows, self.src_ncols]
            ], dtype=np.float64())
        else:
            trans_points_pos = np.array(self.trans_points_pos)

        circ_rad = np.min((self.src_nrows, self.src_ncols)) / 20
        circ_patches = {
            'point1': patches.Circle((trans_points_pos[0, 1] + self.td_offset[0][1], trans_points_pos[0, 0] + self.td_offset[0][0]), circ_rad, fc='r', alpha=0.5),
            'point2': patches.Circle((trans_points_pos[1, 1] + self.td_offset[1][1], trans_points_pos[1, 0] + self.td_offset[1][0]), circ_rad, fc='r', alpha=0.5),
            'point3': patches.Circle((trans_points_pos[2, 1] + self.td_offset[2][1], trans_points_pos[2, 0] + self.td_offset[2][0]), circ_rad, fc='r', alpha=0.5)
        }

        draggable_point = {}
        for key, point in circ_patches.items():
            self.ax_src.add_patch(point)
            dp = DraggablePoint(point)
            dp.connect()
            draggable_point[key] = dp

        return trans_points_pos, fixed_points_pos, draggable_point

    def on_release(self, event):
        for i, (key, point) in enumerate(self.draggable_point.items()):
            if not np.allclose(self.trans_points_pos[i, 1] - self.td_offset[i][1], point.point.center[0]) \
                    or not np.allclose(self.trans_points_pos[i, 0] - self.td_offset[i][0], point.point.center[1]):
                self.trans_points_pos[i] = (point.point.center[1] - self.td_offset[i][0], point.point.center[0] - self.td_offset[i][1])
                self.update_transform()

    def disconnect(self):
        self.fig_src.canvas.mpl_disconnect(self.cidrelease)

    def update_transform(self):
        self.tform.estimate(self.trans_points_pos, self.fixed_points_pos)

        image_warp = affine_transform(self.src_img, self.tform._inv_matrix, output_shape=(self.map_nrows, self.map_ncols))
        self.ax_dst.imshow(image_warp * (1 - self.map))
        self.fig_dst.canvas.draw()

    def accept(self, event):
        plt.close(self.fig_src)
        plt.close(self.fig_dst)

    def reset(self, event):
        for _, dg in self.draggable_point.items():
            dg.disconnect()
            dg.point.remove()
        self.trans_points_pos = None
        self.trans_points_pos, self.fixed_points_pos, self.draggable_point = self.initiate_transform_points()
        self.update_transform()


