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

def calc_z_score_exclude_top5(x):
    z = np.empty_like(x)
    for t in range(x.shape[1]):
        col = x[:, t]
        threshold = np.percentile(col, 95)
        include_mask = col < threshold
        mu = np.mean(col[include_mask])
        sigma = np.std(col[include_mask]) + np.finfo(np.float32).eps
        z[:, t] = (col - mu) / sigma
    return z

def calc_z_score_exclude_top10(x):
    z = np.empty_like(x)
    for t in range(x.shape[1]):
        col = x[:, t]
        threshold = np.percentile(col, 90)
        include_mask = col < threshold
        mu = np.mean(col[include_mask])
        sigma = np.std(col[include_mask]) + np.finfo(np.float32).eps
        z[:, t] = (col - mu) / sigma
    return z

def calc_z_score_exclude_top15(x):
    z = np.empty_like(x)
    for t in range(x.shape[1]):
        col = x[:, t]
        threshold = np.percentile(col, 85)
        include_mask = col < threshold
        mu = np.mean(col[include_mask])
        sigma = np.std(col[include_mask]) + np.finfo(np.float32).eps
        z[:, t] = (col - mu) / sigma
    return z

# def calc_z_score_exclude_top15_eval_style(x):
#     """
#     Compute z-scores per ROI and per timepoint using frame-wise exclusion of the top 15% ROIs.
#     Matches 'evaluate()' normalization logic.
#     x: array of shape (n_rois, n_timepoints)
#     """
#     n_rois, n_timepoints = x.shape
#     z = np.empty_like(x)
#
#     for t in range(n_timepoints):
#         col = x[:, t]
#         # Exclude top 15% of ROI values at this timepoint
#         num_exclude = int(0.15 * n_rois)
#         if num_exclude > 0:
#             exclude_idx = np.argpartition(col, -num_exclude)[-num_exclude:]
#             mask = np.ones(n_rois, dtype=bool)
#             mask[exclude_idx] = False
#             filtered = col[mask]
#         else:
#             filtered = col
#
#         mu = np.mean(filtered)
#         sigma = np.std(filtered) + np.finfo(np.float32).eps
#         z[:, t] = (col - mu) / sigma
#
#     return z

#
# def calc_z_score_exclude_top15_eval_style(x, metric_list=None):
#     """
#     Compute z-scores per ROI per timepoint using 'evaluate()'-style normalization:
#     - Exclude top 15% of ROIs per frame
#     - Compute mean and std from remaining ROIs
#     - Return per-ROI z-scores and optionally per-frame scalar results
#     x: ndarray, shape (n_rois, n_timepoints)
#     metric_list: list of ROI indices defining the subset used for the main result (optional)
#     """
#     n_rois, n_timepoints = x.shape
#     z = np.empty_like(x)
#     eval_result = np.empty(n_timepoints) if metric_list is not None else None
#
#     for t in range(n_timepoints):
#         col = x[:, t]
#         num_exclude = int(0.15 * n_rois)
#         if num_exclude > 0:
#             exclude_idx = np.argpartition(col, -num_exclude)[-num_exclude:]
#             mask = np.ones(n_rois, dtype=bool)
#             mask[exclude_idx] = False
#             filtered = col[mask]
#         else:
#             filtered = col
#
#         mu = np.mean(filtered)
#         sigma = np.std(filtered) + np.finfo(np.float32).eps
#         z[:, t] = (col - mu) / sigma
#
#         if metric_list is not None:
#             eval_result[t] = (np.mean(col[metric_list]) - mu) / sigma
#
#     return z, eval_result


#This needs to be run on sessions were no MH was used in the post_session_procedure to be able to access all ROIs.


#base_path = '/data/Lena/WideFlow_prj'
base_path = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj'
#dataset_path = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'
#dataset_path = '/data/Lena/WideFlow_prj/Results/results_exp2.1.h5'
#dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/results_exp2.1.h5'
#dataset_path_noMH ='/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'
#dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/sessions_exp3.h5'
#dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/results_exp3.h5'
#dataset_path_noMH = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_exp3.3.h5'
#dataset_path_noMH = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/sessions_exp3.h5'
#dataset_path_noMH = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_exp3.4.h5'
# dataset_path_noMH = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_exp3.5.h5'
#dataset_path_noMH = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_exp3.5_parc_NEW2.h5'
#dataset_path_noMH = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_exp3.4_full_parcellations.h5'
# dataset_path_noMH = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_tests.h5'
dataset_path_noMH = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_exp4.h5'



#dates_vec = ['20230615', '20230618', '20230619', '20230620', '20230621', '20230622']
#dates_vec = ['20230604', '20230618', '20230619', '20230620', '20230621', '20230622']
#dates_vec = [ '20230604', '20230612', '20230613', '20230614', '20230615']
# dates_vec = [#'20241121','20241123','20241124','20241125'
#     '20241203'
#
#     # ,'20241129'
#              ]

# dates_vec = ['20250911','20250911','20250911']
# sessions_vec = ['NF_control_p1','NF_control_p2','NF_control_p3']
# dates_vec = ['20250802','20250802','20250802','20250803','20250803','20250803']
# sessions_vec = ['NF3_p1','NF3_p2','NF3_p3','NF4_p1','NF4_p2','NF4_p3']
dates_vec = ['20251202','20251202','20251202']
sessions_vec = ['spont_p1','spont_p2','spont_p3']
# dates_vec = ['20251123','20251123','20251123','20251123',
#              '20251123','20251123','20251123','20251123',
#              '20251123','20251123','20251123','20251123',
#              '20251123','20251123','20251123','20251123']
# sessions_vec = ['try_oldcode_diff5_no15removal', 'try_oldcode_diff5_yes15removal', 'try_oldcode_diff10_no15removal', 'try_oldcode_diff10_yes15removal',
#                 'try_oldcode_diff5_no15removal_fake', 'try_oldcode_diff5_yes15removal_fake', 'try_oldcode_diff10_no15removal_fake', 'try_oldcode_diff10_yes15removal_fake',
#                 'try_newcode_diff5_no15removal', 'try_newcode_diff5_yes15removal', 'try_newcode_diff10_no15removal', 'try_newcode_diff10_yes15removal',
#                 'try_newcode_diff5_no15removal_fake', 'try_newcode_diff5_yes15removal_fake', 'try_newcode_diff10_no15removal_fake', 'try_newcode_diff10_yes15removal_fake']

mice_id = [ #'21ML'
    #'31MN','54MRL'
    #,'63MR','64ML'
     #'187FN'
     # '203MN'
     #  ,'204FR'
     #   ,'206FRL'
     #    ,'211MRR'
     #    ,'218MN'
  # '245FRL',
    # '246FN',
    #   '248FL',
   #  '252MR',
   #   '256FLL',
   #  '257FR'
    # '228MN',
   #  '258FL',
   #  '259FRL',
   #   '260FN',
    # '261MR',
    # '263MRL'
    '277FRL'
    ]

#sessions_vec = ['spont_mockNF_ROI2_excluded_closest','NF21', 'NF22', 'NF23', 'NF24', 'NF25']
#sessions_vec = ['NF1', 'NF2', 'NF3', 'NF4', 'NF5']
# sessions_vec = [ 'spont_mockNF_NOTexcluded_closest',
#                 'CRC4','NF1', 'NF2', 'NF3', 'NF4', 'NF5'
#                 ]
#sessions_vec = ['spont_mockNF_ROI2_excluded_closest', 'NF1_mock_ROI2','NF2_mock_ROI2','NF3_mock_ROI2','NF4_mock_ROI2', 'NF5_mock_ROI2']
#sessions_vec = ['NF5', 'NF21_mock_ROI1','NF22_mock_ROI1','NF23_mock_ROI1','NF24_mock_ROI1', 'NF25_mock_ROI1']
# sessions_vec = ['spont','CRC1','CRC2','CRC3'
#                 ,'CRC4'
#                 ]
# sessions_vec = [
#                 'NF5'
#
#                 ]


for mouse_id in mice_id:
    for date, session_name in zip(dates_vec, sessions_vec):
        session_id = f'{date}_{mouse_id}_{session_name}'
        if session_name == 'CRC4' and mouse_id == '63MR':
            session_id = '20230607_63MR_CRC3'
        print(f'starting {session_id} at {datetime.now()}')


        zscores = {}
        diff5_zscores = {}
        diff5 = {}
        diff10_zscores = {}
        diff10 = {}
        diff20_zscores = {}
        diff20 = {}
        diff30_zscores = {}
        diff30 = {}
        #diff5_zscores_exc_top5 = {}
        diff5_zscores_exc_top10 = {}
        diff5_zscores_exc_top15 = {}
        diff5_zscores_exc_top17 = {}
        diff10_zscores_exc_top5 = {}
        diff10_zscores_exc_top10 = {}
        diff10_zscores_exc_top15 = {}
        diff10_zscores_exc_top15_NEW = {}
        diff10_zscores_exc_top15_NEW2 = {}
        diff10_zscores_exc_top15_NEW3 = {}
        diff10_zscores_exc_top15_NEW4 = {}

        functional_rois_dict_path = f'{base_path}/{mouse_id}/functional_parcellation_rois_dict_NEW2.h5'
        closest_dict_path = f'{base_path}/{mouse_id}/closest_dict_NEW2.h5'
        functional_rois_dict = load_rois_data(functional_rois_dict_path)
        with h5py.File(closest_dict_path, 'r') as hf:
            closest_dict = {key: [item.decode('utf-8') for item in value] for key, value in hf.items()}

        # functional_rois_dict['roi_69_67'] = []
        # closest_dict['roi_69_67'] = ['roi_66', 'roi_64', 'roi_39','roi_29','roi_37',
        #                          'roi_55']
        a=5



        for key in functional_rois_dict.keys():
            functional_rois_dict_temp = load_rois_data(functional_rois_dict_path)
            #functional_rois_dict_temp['roi_69_67'] = []

            # if session_name == 'CRC4':
            #     dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'
            # else:
            #     #dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
            #     dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/results_exp2.1.h5'

            data = {}
            with h5py.File(dataset_path_noMH, 'r') as f:
                decompose_h5_groups_to_dict(f, data, f'/{mouse_id}/{session_id}/')
            traces_roi = data['rois_traces']['channel_0']
            #traces_roi['roi_69_67'] = (np.array(traces_roi['roi_69']) + np.array(traces_roi['roi_67'])) / 2
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
            #diff5_zscores_mat_exc_top5 = calc_z_score_exclude_top5(diff5_mat)
            #diff5_zscores_exc_top5[key] = diff5_zscores_mat_exc_top5[keys_list.index(key), :]
            # diff5_zscores_mat_exc_top10 = calc_z_score_exclude_top10(diff5_mat)
            # diff5_zscores_exc_top10[key] = diff5_zscores_mat_exc_top10[keys_list.index(key), :]
            # diff5_zscores_mat_exc_top17 = calc_z_score_exclude_top17(diff5_mat)
            # diff5_zscores_exc_top17[key] = diff5_zscores_mat_exc_top17[keys_list.index(key), :]
            diff5_zscores_mat_exc_top15 = calc_z_score_exclude_top15(diff5_mat)
            diff5_zscores_exc_top15[key] = diff5_zscores_mat_exc_top15[keys_list.index(key), :]



            diff10_mat = calc_diff(traces_mat, 10)
            diff10[key] = diff10_mat[keys_list.index(key), :]
            diff10_zscores_mat = calc_z_score(diff10_mat)
            diff10_zscores[key] = diff10_zscores_mat[keys_list.index(key), :]
            # diff10_zscores_mat_exc_top5 = calc_z_score_exclude_top5(diff10_mat)
            # diff10_zscores_exc_top5[key] = diff10_zscores_mat_exc_top5[keys_list.index(key), :]


            diff10_zscores_mat_exc_top15 = calc_z_score_exclude_top15(diff10_mat)
            diff10_zscores_exc_top15[key] = diff10_zscores_mat_exc_top15[keys_list.index(key), :]
            diff10_zscores_exc_top15_NEW4[key] = diff10_zscores_mat_exc_top15[keys_list.index(key), :]
            # # diff10_zscores_mat_eval = calc_z_score_exclude_top15_eval_style(diff10_mat)
            # # diff10_zscores_exc_top15_NEW[key] = diff10_zscores_mat_eval[keys_list.index(key), :]
            # diff10_zscores_mat_eval, _ = calc_z_score_exclude_top15_eval_style(diff10_mat)
            # diff10_zscores_exc_top15_NEW2[key] = diff10_zscores_mat_eval[keys_list.index(key), :]

            a=5

            diff20_mat = calc_diff(traces_mat, 20)
            diff20[key] = diff20_mat[keys_list.index(key), :]
            diff20_zscores_mat = calc_z_score(diff20_mat)
            diff20_zscores[key] = diff20_zscores_mat[keys_list.index(key), :]
            a=5


            diff30_mat = calc_diff(traces_mat, 30)
            diff30[key] = diff30_mat[keys_list.index(key), :]
            diff30_zscores_mat = calc_z_score(diff30_mat)
            diff30_zscores[key] = diff30_zscores_mat[keys_list.index(key), :]
            a=5


        with h5py.File(dataset_path_noMH, 'a') as f:
            mouse_grp = f[mouse_id]
            session_grp = mouse_grp[session_id]
            # if 'post_session_analysis_LK2' in session_grp.keys():
            #     del session_grp['post_session_analysis_LK2']
            #
            # eval_grp = session_grp.create_group('post_session_analysis_LK2')
            if 'post_session_analysis_LK2' not in session_grp:
                eval_grp = session_grp.create_group('post_session_analysis_LK2')
            else:
                eval_grp = session_grp['post_session_analysis_LK2']
            #################################################################################
            if 'zsores_MH' not in eval_grp.keys():
                zscores_MH_grp = eval_grp.create_group('zsores_MH')
            else:
                zscores_MH_grp = eval_grp['zsores_MH']

            for key, value in zscores.items():
                #zscores_MH_grp.create_dataset(key, data=value)
                if key not in zscores_MH_grp:
                    zscores_MH_grp.create_dataset(key, data=value)




            if 'zsores_MH_diff5' not in eval_grp.keys():
                zscores_MH_diff5_grp = eval_grp.create_group('zsores_MH_diff5')
            else:
                zscores_MH_diff5_grp = eval_grp['zsores_MH_diff5']

            for key, value in diff5_zscores.items():
                #zscores_MH_diff5_grp.create_dataset(key, data=value)
                if key not in zscores_MH_diff5_grp:
                    zscores_MH_diff5_grp.create_dataset(key, data=value)


            # if 'zsores_MH_diff5_exc_top5' not in eval_grp.keys():
            #     zscores_MH_diff5_exc_top5_grp = eval_grp.create_group('zsores_MH_diff5_exc_top5')
            # else:
            #     zscores_MH_diff5_exc_top5_grp = eval_grp['zsores_MH_diff5_exc_top5']
            #
            # for key, value in diff5_zscores_exc_top5.items():
            #     zscores_MH_diff5_exc_top5_grp.create_dataset(key, data=value)



            # if 'zsores_MH_diff5_exc_top10' not in eval_grp.keys():
            #     zscores_MH_diff5_exc_top10_grp = eval_grp.create_group('zsores_MH_diff5_exc_top10')
            # else:
            #     zscores_MH_diff5_exc_top10_grp = eval_grp['zsores_MH_diff5_exc_top10']
            #
            # for key, value in diff5_zscores_exc_top10.items():
            #     #zscores_MH_diff5_exc_top10_grp.create_dataset(key, data=value)
            #     if key not in zscores_MH_diff5_exc_top10_grp:
            #         zscores_MH_diff5_exc_top10_grp.create_dataset(key, data=value)
            #
            #
            if 'zsores_MH_diff5_exc_top15' not in eval_grp.keys():
                zscores_MH_diff5_exc_top15_grp = eval_grp.create_group('zsores_MH_diff5_exc_top15')
            else:
                zscores_MH_diff5_exc_top15_grp = eval_grp['zsores_MH_diff5_exc_top15']

            for key, value in diff5_zscores_exc_top15.items():
                #zscores_MH_diff5_exc_top15_grp.create_dataset(key, data=value)
                if key not in zscores_MH_diff5_exc_top15_grp:
                    zscores_MH_diff5_exc_top15_grp.create_dataset(key, data=value)




            if 'zsores_MH_diff10' not in eval_grp.keys():
                zscores_MH_diff10_grp = eval_grp.create_group('zsores_MH_diff10')
            else:
                zscores_MH_diff10_grp = eval_grp['zsores_MH_diff10']

            for key, value in diff10_zscores.items():
                #zscores_MH_diff10_grp.create_dataset(key, data=value)
                if key not in zscores_MH_diff10_grp:
                    zscores_MH_diff10_grp.create_dataset(key, data=value)





            # if 'zsores_MH_diff10_exc_top5' not in eval_grp.keys():
            #     zscores_MH_diff10_exc_top5_grp = eval_grp.create_group('zsores_MH_diff10_exc_top5')
            # else:
            #     zscores_MH_diff10_exc_top5_grp = eval_grp['zsores_MH_diff10_exc_top5']
            #
            # for key, value in diff10_zscores_exc_top5.items():
            #     #zscores_MH_dif105_exc_top5_grp.create_dataset(key, data=value)
            #     if key not in zscores_MH_diff10_exc_top5_grp:
            #         zscores_MH_diff10_exc_top5_grp.create_dataset(key, data=value)
            #
            #
            #
            #
            #
            # if 'zsores_MH_diff10_exc_top10' not in eval_grp.keys():
            #     zscores_MH_diff10_exc_top10_grp = eval_grp.create_group('zsores_MH_diff10_exc_top10')
            # else:
            #     zscores_MH_diff10_exc_top10_grp = eval_grp['zsores_MH_diff10_exc_top10']
            #
            # for key, value in diff10_zscores_exc_top10.items():
            #     #zscores_MH_dif105_exc_top10_grp.create_dataset(key, data=value)
            #     if key not in zscores_MH_diff10_exc_top10_grp:
            #         zscores_MH_diff10_exc_top10_grp.create_dataset(key, data=value)



            if 'zsores_MH_diff10_exc_top15' not in eval_grp.keys():
                zscores_MH_diff10_exc_top15_grp = eval_grp.create_group('zsores_MH_diff10_exc_top15')
            else:
                zscores_MH_diff10_exc_top15_grp = eval_grp['zsores_MH_diff10_exc_top15']

            for key, value in diff10_zscores_exc_top15.items():
                #zscores_MH_dif105_exc_top15_grp.create_dataset(key, data=value)
                if key not in zscores_MH_diff10_exc_top15_grp:
                    zscores_MH_diff10_exc_top15_grp.create_dataset(key, data=value)


            if 'zsores_MH_diff10_exc_top15_NEW' not in eval_grp.keys():
                zscores_MH_diff10_exc_top15_NEW_grp = eval_grp.create_group('zsores_MH_diff10_exc_top15_NEW')
            else:
                zscores_MH_diff10_exc_top15_NEW_grp = eval_grp['zsores_MH_diff10_exc_top15_NEW']

            for key, value in diff10_zscores_exc_top15_NEW.items():
                #zscores_MH_dif105_exc_top15_NEW_grp.create_dataset(key, data=value)
                if key not in zscores_MH_diff10_exc_top15_NEW_grp:
                    zscores_MH_diff10_exc_top15_NEW_grp.create_dataset(key, data=value)


            if 'zsores_MH_diff10_exc_top15_NEW2' not in eval_grp.keys():
                zscores_MH_diff10_exc_top15_NEW2_grp = eval_grp.create_group('zsores_MH_diff10_exc_top15_NEW2')
            else:
                zscores_MH_diff10_exc_top15_NEW2_grp = eval_grp['zsores_MH_diff10_exc_top15_NEW2']

            for key, value in diff10_zscores_exc_top15_NEW2.items():
                #zscores_MH_dif105_exc_top15_NEW2_grp.create_dataset(key, data=value)
                if key not in zscores_MH_diff10_exc_top15_NEW2_grp:
                    zscores_MH_diff10_exc_top15_NEW2_grp.create_dataset(key, data=value)



            if 'zsores_MH_diff10_exc_top15_NEW4' not in eval_grp.keys():
                zscores_MH_diff10_exc_top15_NEW4_grp = eval_grp.create_group('zsores_MH_diff10_exc_top15_NEW4')
            else:
                zscores_MH_diff10_exc_top15_NEW4_grp = eval_grp['zsores_MH_diff10_exc_top15_NEW4']

            for key, value in diff10_zscores_exc_top15_NEW4.items():
                #zscores_MH_dif105_exc_top15_NEW4_grp.create_dataset(key, data=value)
                if key not in zscores_MH_diff10_exc_top15_NEW4_grp:
                    zscores_MH_diff10_exc_top15_NEW4_grp.create_dataset(key, data=value)



            if 'zsores_MH_diff20' not in eval_grp.keys():
                zscores_MH_diff20_grp = eval_grp.create_group('zsores_MH_diff20')
            else:
                zscores_MH_diff20_grp = eval_grp['zsores_MH_diff20']

            for key, value in diff20_zscores.items():
                #zscores_MH_diff20_grp.create_dataset(key, data=value)
                if key not in zscores_MH_diff20_grp:
                    zscores_MH_diff20_grp.create_dataset(key, data=value)



            if 'zsores_MH_diff30' not in eval_grp.keys():
                zscores_MH_diff30_grp = eval_grp.create_group('zsores_MH_diff30')
            else:
                zscores_MH_diff30_grp = eval_grp['zsores_MH_diff30']

            for key, value in diff30_zscores.items():
                #zscores_MH_diff30_grp.create_dataset(key, data=value)
                if key not in zscores_MH_diff30_grp:
                    zscores_MH_diff30_grp.create_dataset(key, data=value)



            if 'diff5' not in eval_grp.keys():
                diff5_grp = eval_grp.create_group('diff5')
            else:
                diff5_grp = eval_grp['diff5']

            for key, value in diff5.items():
                #diff5_grp.create_dataset(key, data=value)
                if key not in diff5_grp:
                    diff5_grp.create_dataset(key, data=value)



            if 'diff10' not in eval_grp.keys():
                diff10_grp = eval_grp.create_group('diff10')
            else:
                diff10_grp = eval_grp['diff10']

            for key, value in diff10.items():
                #diff10_grp.create_dataset(key, data=value)
                if key not in diff10_grp:
                    diff10_grp.create_dataset(key, data=value)



            if 'diff20' not in eval_grp.keys():
                diff20_grp = eval_grp.create_group('diff20')
            else:
                diff20_grp = eval_grp['diff20']

            for key, value in diff20.items():
                #diff20_grp.create_dataset(key, data=value)
                if key not in diff20_grp:
                    diff20_grp.create_dataset(key, data=value)



            if 'diff30' not in eval_grp.keys():
                diff30_grp = eval_grp.create_group('diff30')
            else:
                diff30_grp = eval_grp['diff30']

            for key, value in diff30.items():
                #diff30_grp.create_dataset(key, data=value)
                if key not in diff30_grp:
                    diff30_grp.create_dataset(key, data=value)




a=5