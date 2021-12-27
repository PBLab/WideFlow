import numpy as np
from skimage.transform import AffineTransform, warp_coords
from scipy.ndimage import map_coordinates

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from wideflow.utils.draggable_point import DraggablePoint


class InteractiveAffineTransform:
    def __init__(self, src_img, map, ):
        self.src_img = src_img
        self.map = map

        self.src_nrows, self.src_ncols = src_img.shape
        self.map_nrows, self.map_ncols = map.shape

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

        self.tform = AffineTransform()
        self.src_cols = None
        self.dst_cols = None
        self.draggable_point, self.fixed_points_pos, self.trans_points_pos = self.initiate_transform_points()
        self.update_transform()

        self.cidmotion = self.fig_src.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)
        plt.show()

    def initiate_transform_points(self):
        fixed_points_pos = np.array([
            [0, 0],
            [self.map_nrows, 0],
            [self.map_nrows, self.map_ncols]
        ])

        trans_points_pos = np.array([
            [0, 0],
            [self.src_nrows, 0],
            [self.src_nrows, self.src_ncols]
        ])

        circ_rad = np.min((self.src_nrows, self.src_ncols)) / 20
        circ_patches = {
            'point1': patches.Circle((trans_points_pos[0, 1], trans_points_pos[0, 0]), circ_rad, fc='r', alpha=0.5),
            'point2': patches.Circle((trans_points_pos[1, 1], trans_points_pos[1, 0]), circ_rad, fc='r', alpha=0.5),
            'point3': patches.Circle((trans_points_pos[2, 1], trans_points_pos[2, 0]), circ_rad, fc='r', alpha=0.5)
        }

        draggable_point = {}
        for key, point in circ_patches.items():
            self.ax_src.add_patch(point)
            dp = DraggablePoint(point)
            dp.connect()
            draggable_point[key] = dp

        return draggable_point, fixed_points_pos, trans_points_pos

    def on_mouse_motion(self, event):
        for i, (key, point) in enumerate(self.draggable_point.items()):
            if self.trans_points_pos[i, 1] != point.point.center[0] or self.trans_points_pos[i, 0] != point.point.center[1]:
                self.trans_points_pos[i] = (point.point.center[1], point.point.center[0])
                self.update_transform()
                break

    def disconnect(self):
        self.fig_src.canvas.mpl_disconnect(self.cidmotion)

    def update_transform(self):
        # TODO: why roll axis (switch between x, y coordinates)?
        self.tform.estimate(np.roll(self.trans_points_pos, 1, axis=1), np.roll(self.fixed_points_pos, 1, axis=1))
        warp_coor = warp_coords(self.tform.inverse, (self.map_nrows, self.map_ncols))
        self.src_cols = np.reshape(warp_coor[0], (self.map_nrows * self.map_ncols, 1))
        self.src_rows = np.reshape(warp_coor[1], (self.map_nrows * self.map_ncols, 1))

        image_warp = map_coordinates(self.src_img, [self.src_cols, self.src_rows])
        image_warp = np.reshape(image_warp, (self.map_nrows, self.map_ncols))
        image_warp = image_warp * (1 - self.map)

        self.ax_dst.imshow(image_warp)
        self.fig_dst.canvas.draw()
        # plt.pause(0.02)

    def accept(self, event):
        plt.close(self.fig_src)
        plt.close(self.fig_dst)

    def reset(self, event):
        for _, dg in self.draggable_point.items():
            dg.disconnect()
            dg.point.remove()
        self.draggable_point, self.fixed_points_pos, self.trans_points_pos = self.initiate_transform_points()
        self.update_transform()


# import h5py
# from wideflow.utils.load_tiff import load_tiff
# cortex_file_path = '/data/Rotem/Wide Field/WideFlow/data/cortex_map/allen_2d_cortex.h5'
# with h5py.File(cortex_file_path, 'r') as f:
#     cortex_mask = np.transpose(f["mask"][()])
#     cortex_map = np.transpose(f["map"][()])
#
# image_file_path = '/data/Rotem/WideFlow prj/extras/tests/cropped_wf_image.ome.tif'
# image = load_tiff(image_file_path)
# iat = InteractiveAffineTransform(image, cortex_map)
#
# print(iat.tform.params)
