import numpy as np
from skimage.transform import resize
from skimage.morphology import skeletonize
from scipy.ndimage.filters import maximum_filter1d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import h5py
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.optimize import curve_fit
from scipy.stats import wilcoxon

from utils.load_tiff import load_tiff
from utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict

from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
from analysis.utils.peristimulus_time_response import calc_pstr
from utils.load_config import load_config
from utils.load_rois_data import load_rois_data
from analysis.plots import plot_traces, wf_imshow
from analysis.utils.rois_proximity import calc_rois_proximity
from utils.paint_roi import paint_roi
from scipy.stats import ttest_rel

def find_closest_key(dict1, dict2):
    result = {}

    for key1, value1 in dict1.items():
        closest_key = None
        min_difference = float('inf')

        for key2, value2 in dict2.items():
            difference = abs(value1[0] - value2[0]) + abs(value1[1] - value2[1])

            if difference < min_difference:
                min_difference = difference
                closest_key = key2

        result[key1] = closest_key

    return result


base_path = '/data/Lena/WideFlow_prj'
#dates_vec = ['20230604','20230614']
mice_id = ['21ML','31MN','54MRL','63MR','64ML']
#colors = ['cyan', 'orange', 'purple', 'chartreuse', 'magenta'] #21'cyan',24'blue',31'orange',46'green',54'purple', 63'chartreuse', 64'magenta'
#sessions_vec = ['spont_mockNF_NOTexcluded_closest','NF4']

roi_lists = {}
centroid_lists = {}
for mouse in mice_id:
    roi_lists[mouse] = load_rois_data(f'/data/Lena/WideFlow_prj/{mouse}/functional_parcellation_rois_dict.h5')
    centroid_lists[mouse] = {}
    for key in roi_lists[mouse]:
        centroid_lists[mouse][key] = roi_lists[f'{mouse}'][key]['Centroid']

a=5

closest_rois = {}
for mouse in mice_id:
    closest_rois[mouse] = find_closest_key(centroid_lists['54MRL'], centroid_lists[mouse])



a=5
a=5