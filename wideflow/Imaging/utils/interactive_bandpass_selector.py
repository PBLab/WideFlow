import numpy as np
from skimage.color import rgb2gray

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import RectangleSelector
from utils.matplotlib_rectangle_selector_events import *


class InteractiveBandPassSelector:
    def __init__(self, src_img):
        self.src_img = src_img
        self.src_fft = np.fft.fftshift((np.fft.fft2(self.src_img)))
        self.src_bp = self.src_img.copy()
        self.src_nrows, self.src_ncols = self.src_img.shape

        # initiate gui figures
        self.fig_src, self.ax_src = plt.subplots()
        self.fig_src.suptitle('Image Fourier-Transform')
        mng = self.fig_src.canvas.manager
        mng.window.wm_geometry('+1+1')  # set figure position
        self.ax_src.imshow(rgb2gray(np.log(abs(self.src_fft))), cmap='gray')

        self.ax_accept = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.b_accept = Button(self.ax_accept, 'Accept')
        self.b_accept.on_clicked(self.accept)

        self.ax_reset = plt.axes([0.85, 0.05, 0.1, 0.075])
        self.b_reset = Button(self.ax_reset, 'Reset')
        self.b_reset.on_clicked(self.reset)

        self.fig_dst, self.ax_dst = plt.subplots()
        self.fig_dst.suptitle('Band-Pass Image')
        mng = self.fig_dst.canvas.manager
        mng.window.wm_geometry(f'+{self.src_nrows + 10}+1')
        self.ax_dst.imshow(self.src_bp)

        self.toggle_selector = None
        self.bbox_list = []

        self.cidrelease = self.fig_src.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidclick = self.fig_src.canvas.mpl_connect('button_press_event', self.on_click)

        plt.show()

    def on_click(self, event):
        self.toggle_selector = RectangleSelector(self.ax_src, onselect, drawtype='box')
        self.fig_src.canvas.mpl_connect('key_press_event', self.toggle_selector)
        self.toggle_selector.press(event)

    def on_release(self, event):
        bbox = self.toggle_selector._rect_bbox
        if bbox[1] > 1 and bbox[3] > 1:
            bbox = (int(bbox[0]), int(bbox[0] + bbox[2]), int(bbox[1]), int(bbox[1] + bbox[3]))
            self.bbox_list.append(bbox)
            self.update_band_pass_image(bbox)

        self.toggle_selector = None

    def disconnect(self):
        self.fig_src.canvas.mpl_disconnect(self.cidrelease)
        self.fig_src.canvas.mpl_disconnect(self.cidclick)

    def update_band_pass_image(self, bbox):
        self.src_fft[bbox[2]: bbox[3], bbox[0]: bbox[1]] = 1
        self.src_bp = abs(np.fft.ifft2(self.src_fft))
        self.draw()

    def draw(self):
        self.ax_src.imshow(rgb2gray(np.log(abs(self.src_fft))), cmap='gray')
        self.ax_dst.imshow(self.src_bp)
        self.fig_dst.canvas.draw()
        self.fig_src.canvas.draw()

    def accept(self, event):
        plt.close(self.fig_src)
        plt.close(self.fig_dst)

    def reset(self, event):
        self.src_bp = self.src_img.copy()
        self.src_fft = np.fft.fftshift((np.fft.fft2(self.src_img)))
        self.bbox_list = []
        self.draw()


# path = '/data/Rotem/WideFlow prj/2683/20220206_neurofeedback/regression_coeff_map.npy'
# regression_map = np.load(path)
# ibp = InteractiveBandPassSelector(regression_map[0])
# for bbox in ibp.bbox_list:
#     print(bbox)