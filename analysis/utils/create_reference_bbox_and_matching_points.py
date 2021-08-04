from Imaging.utils.create_matching_points import *
from Imaging.utils.roi_select import *

from utils.load_tiff import load_tiff

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import h5py

import cv2


# ############## change path to the folder the reference image is at, and binning factor according the one used whole imaging ############## #
reference_base_path = 'Z:/Rotem/WideFlow prj/3422/'
binning = (2, 2)
# ########################################################################################################################################## #

reference_path = reference_base_path + 'reference_image.tif'
reference_image = load_tiff(reference_path)
# select roi
fig, ax = plt.subplots()
ax.imshow(reference_image)
toggle_selector = RectangleSelector(ax, onselect, drawtype='box')
fig.canvas.mpl_connect('key_press_event', toggle_selector)
plt.show()
bbox = toggle_selector._rect_bbox
if np.sum(bbox) > 1:
    # convert to PyVcam format
    bbox = (int(bbox[0]), int(bbox[0] + bbox[2]), int(bbox[1]), int(bbox[1] + bbox[3]))

# select the roi and bin before selecting the matching points
reference_image = reference_image[bbox[2]: bbox[3], bbox[0]: bbox[1]]
shape = reference_image.shape
new_shape = (int(shape[0] / binning[0]), int(shape[1] / binning[1]))
reference_image = cv2.resize(reference_image, (new_shape[1], new_shape[0]))

allen_atlas_path = "C:\\Users\\motar\\PycharmProjects\\WideFlow\\data\\cortex_map\\allen_2d_cortex.h5"
with h5py.File(allen_atlas_path, 'r') as f:
    cortex_map = np.transpose(f["map"][()])

mps = MatchingPointSelector(reference_image, cortex_map * np.random.random(cortex_map.shape), None, None, 25)
match_p_src = mps.match_p_src.tolist()
match_p_dst = mps.match_p_dst.tolist()

with open(reference_base_path + 'bbox.txt', 'w+') as f:
    f.write(f'x_min: {bbox[0]}, x_max:{bbox[1]}, y_min:{bbox[2]}, y_max:{bbox[3]}')

with open(reference_base_path + 'matching_points.txt', 'w+') as f:
    f.write(f'match_p_src: {match_p_src}\n')
    f.write(f'match_p_dst: {match_p_dst}\n')
