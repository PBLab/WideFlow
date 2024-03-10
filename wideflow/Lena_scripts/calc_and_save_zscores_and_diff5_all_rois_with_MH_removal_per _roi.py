from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import AnovaRM
from utils.load_rois_data import load_rois_data
import h5py
from utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict
from datetime import datetime

def calc_z_score(x):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    return (x - x_mean) / (x_std + np.finfo(np.float32).eps)

def calc_diff(x, delta_t):
    x = np.pad(x, [[0, 0], [delta_t, 0]])
    return x[:, delta_t:] - x[:, :-delta_t]

#This needs to be run on sessions were no MH was used in the post_session_procedure to be able to access all ROIs.


base_path = '/data/Lena/WideFlow_prj'
dataset_path = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'

#dates_vec = ['20230615', '20230618', '20230619', '20230620', '20230621', '20230622']
#dates_vec = ['20230604', '20230618', '20230619', '20230620', '20230621', '20230622']
#dates_vec = [ '20230604', '20230612', '20230613', '20230614', '20230615']
dates_vec = ['20230604',
             '20230608','20230611','20230612', '20230613', '20230614', '20230615'
             ]

mice_id = [ #'21ML'
    '31MN','54MRL','63MR','64ML'
           ]

#sessions_vec = ['spont_mockNF_ROI2_excluded_closest','NF21', 'NF22', 'NF23', 'NF24', 'NF25']
#sessions_vec = ['NF1', 'NF2', 'NF3', 'NF4', 'NF5']
sessions_vec = [ 'spont_mockNF_NOTexcluded_closest',
                'CRC4','NF1', 'NF2', 'NF3', 'NF4', 'NF5'
                ]
#sessions_vec = ['spont_mockNF_ROI2_excluded_closest', 'NF1_mock_ROI2','NF2_mock_ROI2','NF3_mock_ROI2','NF4_mock_ROI2', 'NF5_mock_ROI2']
#sessions_vec = ['NF5', 'NF21_mock_ROI1','NF22_mock_ROI1','NF23_mock_ROI1','NF24_mock_ROI1', 'NF25_mock_ROI1']



for mouse_id in mice_id:
    for date, session_name in zip(dates_vec, sessions_vec):
        session_id = f'{date}_{mouse_id}_{session_name}'
        if session_name == 'CRC4' and mouse_id == '63MR':
            session_id = '20230607_63MR_CRC3'
        print(f'starting {session_id} at {datetime.now()}')


        zscores = {}
        diff5_zscores = {}
        diff5 = {}

        functional_rois_dict_path = f'{base_path}/{mouse_id}/functional_parcellation_rois_dict.h5'
        closest_dict_path = f'{base_path}/{mouse_id}/closest_dict.h5'
        functional_rois_dict = load_rois_data(functional_rois_dict_path)
        with h5py.File(closest_dict_path, 'r') as hf:
            closest_dict = {key: [item.decode('utf-8') for item in value] for key, value in hf.items()}


        for key in functional_rois_dict.keys():
            functional_rois_dict_temp = load_rois_data(functional_rois_dict_path)

            if session_name == 'CRC4':
                dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'
            else:
                dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
            data = {}
            with h5py.File(dataset_path_noMH, 'r') as f:
                decompose_h5_groups_to_dict(f, data, f'/{mouse_id}/{session_id}/')
            traces_roi = data['rois_traces']['channel_0']
            for val in closest_dict[key]:
                del functional_rois_dict_temp[val]
                del traces_roi[val]

            traces_mat = np.array(list(traces_roi.values()))
            keys_list = list(traces_roi.keys())
            zscores_mat = calc_z_score(traces_mat)
            zscores[key] = zscores_mat[keys_list.index(key), :]

            diff5_mat = calc_diff(traces_mat, 5)
            diff5[key] = diff5_mat[keys_list.index(key), :]
            diff5_zscores_mat = calc_z_score(diff5_mat)
            diff5_zscores[key] = diff5_zscores_mat[keys_list.index(key), :]
            a=5

        with h5py.File(dataset_path_noMH, 'a') as f:
            mouse_grp = f[mouse_id]
            session_grp = mouse_grp[session_id]
            if 'post_session_analysis_LK2' in session_grp.keys():
                del session_grp['post_session_analysis_LK2']

            eval_grp = session_grp.create_group('post_session_analysis_LK2')
            #################################################################################
            if 'zsores_MH' not in eval_grp.keys():
                zscores_MH_grp = eval_grp.create_group('zsores_MH')
            else:
                zscores_MH_grp = eval_grp['zsores_MH']

            for key, value in zscores.items():
                zscores_MH_grp.create_dataset(key, data=value)



            if 'zsores_MH_diff5' not in eval_grp.keys():
                zscores_MH_diff5_grp = eval_grp.create_group('zsores_MH_diff5')
            else:
                zscores_MH_diff5_grp = eval_grp['zsores_MH_diff5']

            for key, value in diff5_zscores.items():
                zscores_MH_diff5_grp.create_dataset(key, data=value)



            if 'diff5' not in eval_grp.keys():
                diff5_grp = eval_grp.create_group('diff5')
            else:
                diff5_grp = eval_grp['diff5']

            for key, value in diff5.items():
                diff5_grp.create_dataset(key, data=value)




a=5