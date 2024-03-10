import h5py

from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import AnovaRM
from utils.load_rois_data import load_rois_data


def find_dist_outlines(vector1,vector2):
    dists = []
    for value1 in vector1:
        closest_value = min(vector2, key=lambda value2: abs(value1 - value2))
        dists.append(abs(value1-closest_value))

    dists = [item for sublist in dists for item in sublist]
    return dists


#To read the resulting dict:
# with h5py.File(closest_dict_path, 'r') as hf:
#     closest_dict = {key: [item.decode('utf-8') for item in value] for key, value in hf.items()}



base_path = '/data/Lena/WideFlow_prj'

mice_id = ['21ML','31MN','54MRL','63MR','64ML']



for mouse_id in mice_id:
    print(f'{mouse_id}')

    functional_rois_dict_path = f'{base_path}/{mouse_id}/functional_parcellation_rois_dict.h5'
    functional_rois_dict = load_rois_data(functional_rois_dict_path)

    closest = {}
    for key in functional_rois_dict:
        closest[key] = []
        for key2 in functional_rois_dict:
            dists = find_dist_outlines(functional_rois_dict[key]["outline"], functional_rois_dict[key2]["outline"])
            count = dists.count(1)
            if count > 1:
                closest[key].append(str(key2))

    file_path = f'{base_path}/{mouse_id}/closest_dict.h5'

    with h5py.File(file_path, 'w') as f:
        # for key, dict in closest.items():
        #     grp = f.create_group(key)
        #     for key, val in closest.items():
        #         grp.create_dataset(key, data=val)
        for key, value in closest.items():
            f.create_dataset(key, data=value)


    a=5




