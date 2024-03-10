# After running parcellation full process run this in the console to see an image of the parcellation

from utils.load_rois_data import load_rois_data
import matplotlib.pyplot as plt
from utils.paint_roi import paint_roi
import numpy as np
from io import BytesIO
import base64
import PIL

mouse_id = '54MRL'
mouse_base_path = f'/data/Lena/WideFlow_prj/{mouse_id}/'  # path to the directory where to save parcellation rois data and map
rois_data_path = mouse_base_path + 'functional_parcellation_rois_dict.h5'
#rois_data_path = mouse_base_path + 'functional_parcellation_cortex_map.h5'
#20221122_MR_CRC3functional_parcellation_rois_dict.h5
#FLfunctional_parcellation_rois_dict_CRC3.h5

roi_list = load_rois_data(rois_data_path)
fig, ax = plt.subplots()
rois_dict=(roi_list)
cortex_map = np.zeros((297,168))
rois_names = list(roi_list.keys())
rois_vals=None
ax=None
annotate=False

fig, ax = plt.subplots()


if rois_vals is None:
    rois_vals = {}
    for ind, roi_name in enumerate(rois_names):
        rois_vals[roi_name] = ind
    # for roi_name in rois_names:
    #     rois_vals[roi_name] = 1

elif len(rois_names) != len(rois_vals):
    raise Exception('The number of elements in rois_names and rois_vals should be the same')

paint_map = np.zeros((cortex_map.shape))
paint_map[:] = cortex_map[:]
# if ax is not None:
#     ax.imshow(paint_map)
for roi_name in rois_names:
    # roi_pixels_list = rois_dict[roi_name]["PixelIdxList"]
    roi_pixels_list = rois_dict[roi_name]["outline"]
    pixels_inds = np.unravel_index(roi_pixels_list, (cortex_map.shape[1], cortex_map.shape[0]))
    paint_map[pixels_inds[1], pixels_inds[0]] = rois_vals[roi_name]
    # if annotate:
    #     ax.annotate(roi_name, (rois_dict[roi_name]["Centroid"][1], rois_dict[roi_name]["Centroid"][0]))

#im = None
# if ax is not None:
#     im = ax.imshow(paint_map)
#ax, im, paint_map = paint_roi (roi_list, np.zeros((297,168)),list(roi_list.keys()), ax=ax, annotate=False)
# to show all rois: list(roi_list.keys())
# to show specific rois: ['roi_', 'roi_']
#['roi_15', 'roi_86','roi_92','roi_88','roi_66','roi_67','roi_69' ]
#MNL metric_ROI - 57


#fig.suptitle(f'{mouse_id}')
#plt.imshow(paint_map)
#plt.savefig('image_try1.svg', format='svg')
import copy
paint_map2= copy.deepcopy(paint_map)
paint_map2[paint_map2>0] = 2
paint_map2[paint_map2==0] = 1
paint_map2[paint_map2==2] = 0
im = ax.imshow(paint_map2, cmap='gray')
#PIL.Image.('/data/Lena/dff.svg')
#plt.show()

# # Save the plot to an in-memory BytesIO object
# svg_buffer = BytesIO()
# plt.savefig(svg_buffer, format='svg', bbox_inches='tight', pad_inches=0.0, transparent=True)
# plt.close()
#
# # Get the SVG content as a string
# svg_content = svg_buffer.getvalue().decode('utf-8')


# Optionally, save the SVG content to a file
# with open(f'/data/Lena/WideFlow_prj/Figs_for_paper/parcellation_{mouse_id}_outlineTRY4.svg', 'w') as svg_file:
#     svg_file.write(svg_content)

#plt.show()
#
#plt.rcParams['svg.fonttype'] = 'none'  # or 'path' or 'none'
plt.savefig( f'/data/Lena/WideFlow_prj/Figs_for_paper/parcellation_{mouse_id}_outlineTRY6.svg',format='svg',dpi=500, transparent=True)

#fig.suptitle(f'{mouse_id}')
