from __future__ import print_function
import numpy as np
import random
import matplotlib.pyplot as plt


def plot_optical_flow(vid, flow, frame, scale, rSize, invert_y=False):
    [nt, ny, nx] = vid.shape
    uy = flow[:, :, :, 0]
    ux = flow[:, :, :, 1]
    u = uy[frame - 1, :, :]
    v = ux[frame - 1, :, :]
    v = -v

    # X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))
    X, Y = np.mgrid[0: nx: 1, 0: ny: 1]
    fig, ax = plt.subplots()
    ax.imshow(vid[frame, :, :], vmin=np.amin(vid[frame, :, :]), vmax=np.amax(vid[frame, :, :]))
    ax.quiver(X[::rSize, ::rSize], Y[::rSize, ::rSize], u[::rSize, ::rSize], v[::rSize, ::rSize], scale=scale)
    ax.set_title(f'frame {frame} optical flow')
    if invert_y:
        plt.gca().invert_yaxis()
    plt.show()


def plot_stack_flow(vid, flow, slice_factor, scale, rSize, invert_y=False):
    [nt, ny, nx] = vid.shape


class IndexTracker(object):
    def __init__(self, ax, vid, flow, streamlines, qv_spacer=1, qv_scale=150):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.vid = vid[1:, :, :]
        flow[:, :, :, 1] = -flow[:, :, :, 1]
        self.flow = flow
        self.slices, rows, cols = self.vid.shape

        for i, sl in enumerate(streamlines):
            n = np.size(sl, 0)
            if n < self.slices:
                pad = np.full((self.slices - n, 2), None, dtype=sl.dtype if n>1 else np.float64)
                streamlines[i] = np.concatenate((streamlines[i], pad), axis=0)
        self.slines = np.stack(streamlines, axis=2)

        self.n_slines = np.size(self.slines, 2)
        self.colors = []
        for i in range(self.n_slines):
            self.colors.append('#%06X' % random.randint(0, 0xFFFFFF))
        self.qv_spacer = qv_spacer
        self.qv_scale = qv_scale

        [nt, ny, nx] = vid.shape
        X, Y = np.mgrid[0:nx:1, 0:ny:1]
        self.ind = 0
        self.im = ax.imshow(self.vid[self.ind, :, :], vmin=np.amin(self.vid), vmax=np.amax(self.vid))
        self.qv = ax.quiver(X[::self.qv_spacer, ::self.qv_spacer], Y[::self.qv_spacer, ::self.qv_spacer],
                            self.flow[self.ind, ::self.qv_spacer, ::self.qv_spacer, 0], self.flow[self.ind, ::self.qv_spacer, ::self.qv_spacer, 1],
                            scale=qv_scale)
        self.sc = ax.scatter(self.slines[self.ind, 0, :], self.slines[self.ind, 1, :], color=self.colors)

        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.vid[self.ind, :, :])
        self.qv.set_UVC(self.flow[self.ind, ::self.qv_spacer, ::self.qv_spacer, 0],
                        self.flow[self.ind, ::self.qv_spacer, ::self.qv_spacer, 1])
        self.sc.set_offsets(np.transpose([self.slines[self.ind, 0, :], self.slines[self.ind, 1, :]]))
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


# import pathlib
# from utils.load_matlab_vector_field import load_matlab_OF
# from utils.load_tiff import load_tiff
# from utils.analyze_OF import get_temporal_streamline
#
# vid_path = str(pathlib.Path('C:/') / 'Users' / 'motar' / 'PycharmProjects' / 'WideFlow' / 'data' / 'ImgSeq.tif')
# gt_path = str(pathlib.Path('C:/') / 'Users' / 'motar' / 'PycharmProjects' / 'WideFlow' / 'data' / 'ofamm_results.mat')
#
# vid = load_tiff(vid_path)
# gt_flow = load_matlab_OF(gt_path)
#
# nt, ny, nx, _ = gt_flow.shape
# gt_flowc = gt_flow[:,10:,:,:]
# vidc = vid[:,10:,:]
#
# tsline = get_temporal_streamline(gt_flow, [20, 100])
# tsline2 = get_temporal_streamline(gt_flow, [20, 100])
# tsline3 = get_temporal_streamline(gt_flow, [100, 20])
# tsline4 = get_temporal_streamline(gt_flow, [100, 100])
# tsline5 = get_temporal_streamline(gt_flow, [80, 80])
# tsline6 = get_temporal_streamline(gt_flow, [40, 40])
# tsline7 = get_temporal_streamline(gt_flow, [40, 80])
# tsline8 = get_temporal_streamline(gt_flow, [80, 40])
#
#
# fig, ax = plt.subplots(1, 1)
# tracker = IndexTracker(ax, vid, gt_flow,
#                        [tsline[:, 1:], tsline2[:, 1:], tsline3[:, 1:], tsline4[:, 1:], tsline5[:, 1:], tsline6[:, 1:], tsline7[:, 1:], tsline8[:, 1:]],
#                        qv_spacer=3, qv_scale=200)
# # tracker = IndexTracker(ax, vid, gt_flow, [tsline[:, 1:], ], qv_spacer=3, qv_scale=200)
#
# fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
# plt.show()