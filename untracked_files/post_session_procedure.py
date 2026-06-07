from wideflow.utils.load_config import load_config
from core.session.mock_neurofeedback_session import PostAnalysisNeuroFeedbackSession
from run_convert_dat_to_tif import run_converter
from tqdm import tqdm

# base_path = '/data/Rotem/WideFlow prj'
#base_path = '/data/Lena/WideFlow_prj'
# base_path = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj'
base_path = '/storage/DataCrunching/Lena'


#dates_vec = ['20230604','20230604','20230604','20230604','20230604']
dates_vec = [

# # '20251202',
# '20251202',
# '20251202',

'20260602',
'20260602',
'20260602',

# '20251205',
# '20251205',
# '20251205',
#
# '20251205',
# '20251205',
# '20251205',
#
# '20251205',
# '20251205',
# '20251205'

             ]

mouse_id_vec = [#'218MN'
    # ,'218MN','218MN','218MN'
    # '187FN'
    # ,'203MN'
    # ,'204FR'
    # ,'206FRL'
    # ,'211MRR'
    #'218MN'

     #'229FR'
    #, '232FN'
    #  ,'232FN'
    # '241FRLL'
    #   , '226MR'
    #    '228MN'
    #  ,'228MN'
    #'241FRLL'
     #  ,'241FRLL'
   #'241FRLL'
     # '226MR'
    # , '232FN'
    # , '241FRLL'
    # '245FRL',
    # '246FN',
    #  '248FL',
    #  '252MR',
    # '256FLL',
    #   '257FR',
# '248FL',
#     '248FL',
# '187FN',
#
#
# '228MN',
# '228MN',
# '228MN',
#
# '258FL',
# '258FL',
# '258FL',
#
# '259FRL',
# '259FRL',
# '259FRL',
#
# '260FN',
# '260FN',
# '260FN',
#
# '261MR',
# '261MR',
# '261MR',
#
# '263MRL',
# '263MRL',
# '263MRL'

# '266FR',
# '266FR',
# '266FR',
#
# '276FL',
# '276FL',
# '276FL',
#
# '277FRL',
# '277FRL',
# '277FRL',
#
# '281MRL',
# '281MRL',
# '281MRL'

'331FN',
'331FN',
'331FN'
                ]
session_name_vec = [
                    #'20250616_226MR_NF11_diff10'
                    #  '20250616_229FR_NF11_diff10'
                       #'20250616_232FN_NF11_diff10'
                    #  ,'20250624_226MR_NF3.2'
                      #'20250617_228MN_NF2'
    #                    ,'20250624_232FN_NF3.2'
    # , '20250625_226MR_NF4.2'
    # , '20250625_229FR_NF4.2'
    #  '20250720_241FRLL_NF3.3'
    # , '20250721_226MR_NF4.3'
    # , '20250721_228MN_NF4.3'
     #'20250718_232FN_NF1.3_new_eval'
    #'20250727_257FR_spont',
# #
# #
# '20251205_266FR_NF1_p1',
# '20251205_266FR_NF1_p2',
# '20251205_266FR_NF1_p3',
#
# '20251205_276FL_NF1_p1',
# '20251205_276FL_NF1_p2',
# '20251205_276FL_NF1_p3',
#
# '20251205_277FRL_NF1_p1',
# '20251205_277FRL_NF1_p2',
# '20251205_277FRL_NF1_p3',
#
# '20251205_281MRL_NF1_p1',
# '20251205_281MRL_NF1_p2',
# '20251205_281FRL_NF1_p3',


'20260602_331FN_spont_p1',
'20260602_331FN_spont_p2',
'20260602_331FN_spont_p3',
]


for date, mouse_id, session_name in zip(dates_vec, mouse_id_vec, session_name_vec):   #LK to change back to single mouse remove this line and create single variables for mouse_id and session_name
    print(f'{session_name}') #added for loop by LK
    session_path = base_path + '/' + date + '/' + mouse_id + '/' + session_name

    config = load_config(f'{session_path}/session_config.json')
    # fix config paths
    config["base_path"] = base_path
    config["date"] = date
    config["registration_config"]["matching_point_path"] = f'{session_path}/matching_points.txt'
    # config["supplementary_data_config"]["rois_dict_path"] = f'{config["base_path"]}/{mouse_id}/functional_parcellation_rois_dict_left_hemi.h5'
    config["supplementary_data_config"]["rois_dict_path"] = f'{config["base_path"]}/{mouse_id}/functional_parcellation_rois_dict_NEW_ROI1.h5'
    # config["supplementary_data_config"]["mask_path"] = "/claustrum-storage/pblab_shared_data/Lena/WideFlow/data/cortex_map/allen_2d_cortex.h5"
    config["supplementary_data_config"]["mask_path"] = "/storage/DataCrunching/Lena/WideFlow/data/cortex_map/allen_2d_cortex.h5"
    #20221122_MR_CRC3functional_parcellation_rois_dict.h5
    #FLfunctional_parcellation_rois_dict_CRC3.h5
    #20221122_{mouse_id}_CRC3functional_parcellation_rois_dict

    #For correction of sessions April 2023
    config["session_name"] = session_name
    config["mouse_id"] = mouse_id
    #config["acquisition_config"]["num_of_frames"] = 50000


    #for spont
    # config["acquisition_config"]["metric_roi"] = ['roi_41']
    # config["supplementary_data_config"]["closest_rois"] = []

    #to remove "Mexican hat" from session that ran with "mexican hat
    config["supplementary_data_config"]["closest_rois"] = []

    # ##for CRC or spont session to run as mock NF
    config["analysis_pipeline_config"]["args"]["metric_args"] = ["ROIDiff", ["roi_22_34"], 10]
    # config["feedback_config"]["update_frames"] = [1000,70000]
    # config["feedback_config"]["eval_frames"] = 20000
    # config["feedback_config"]["update_every"] = 10
    # config["feedback_config"]["metric_threshold"] = 1.0
    # config["feedback_config"]["percentile"] = 95
    config["acquisition_config"]["metric_roi"] = ['roi_22_34']
    config["supplementary_data_config"]["closest_rois"] = [] #don't put closest here, because you want z-scores metrics
                                                        # for them as well, so you can compare to all ROIs


#    run_converter(session_path)
    sess = PostAnalysisNeuroFeedbackSession(config)
    #a=5
    sess.session_preparation()
    sess.run_session_pipeline()
