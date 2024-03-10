from wideflow.utils.load_config import load_config
from core.session.mock_neurofeedback_session import PostAnalysisNeuroFeedbackSession
from run_convert_dat_to_tif import run_converter

# base_path = '/data/Rotem/WideFlow prj'
base_path = '/data/Lena/WideFlow_prj'
#dates_vec = ['20230604','20230604','20230604','20230604','20230604']
dates_vec = ['20230605']

mouse_id_vec = [#'64ML','64ML','64ML','64ML','64ML'
                '64ML'
                # #'24MLL',
                # '31MN',
                # #'46FLL',
                # '54MRL',
                # '63MR',
                # '64ML'
                ]
session_name_vec = [
                    # '20221129_MR_NF2_tiffs',
                    # '20221130_MR_NF3_tiffs',
                    # '20221201_MR_NF4_tiffs',
                    # '20221218_MR_NF6_tiffs',
                    # '20221219_MR_NF7_tiffs',
                    # '20221222_MR_NF9_tiffs',
                    # '20221223_MR_NF10_tiffs',
                    # '20230102_MR_NF12_tiffs',
                    # '20230103_MR_NF13_tiffs',
                    # '20230104_MR_NF14_tiffs',
                    # '20230116_MR_NF16_tiffs'
                    #'20221128_MR_NF1_tiffs'
                    # '20221128_MNL_NF1_tiffs',
                    # '20221202_MNL_NF5_tiffs',
                    # '20221221_MNL_NF8_tiffs',
                    # '20221223_MNL_NF10_tiffs',
                    # '20230102_MNL_NF12_tiffs',
                    # '20230103_MNL_NF13_tiffs',
                    # '20230104_MNL_NF14_tiffs',
                    # '20230116_MNL_NF16_tiffs',
                    # '20230117_MNL_NF17_tiffs',
                    # '20230118_MNL_NF18_tiffs',
                    # '20230119_MNL_NF19_tiffs',
                    # '20230122_MNL_NF20_tiffs',
                    # 'MNL_NF21_tiffs',
                    # 'MNL_NF22_tiffs',
                    # 'MNL_NF23_tiffs'
                    # '20230611_64ML_NF1','20230612_64ML_NF2','20230613_64ML_NF3',
                    # '20230614_64ML_NF4','20230615_64ML_NF5'
                    # '20230622_24MLL_NF25',
                    # '20230622_31MN_NF25',
                    # '20230622_46FLL_NF25',
                    # '20230622_54MRL_NF25',
                    # '20230622_63MR_NF25',
                    # '20230622_64ML_NF25'
                    #'20230604_21ML_spont',
                    # '20230604_21ML_spont_mockNF_NOTexcluded_closest',
                    # '20230604_31MN_spont_mockNF_NOTexcluded_closest',
                    # '20230604_54MRL_spont_mockNF_NOTexcluded_closest',
                    # '20230604_63MR_spont_mockNF_NOTexcluded_closest',
                    # '20230604_64ML_spont_mockNF_NOTexcluded_closest' #FOR SPONT SESSIONS - remember to change metric ROI according to the mouse!!!!!!
                    '20230605_64ML_CRC1'
                    # '20230606_64ML_CRC2',
                    # '20230607_64ML_CRC3',
                    # '20230608_64ML_CRC4'


]


for date, mouse_id, session_name in zip(dates_vec,mouse_id_vec, session_name_vec):    #LK to change back to single mouse remove this line and create single variables for mouse_id and session_name
    print(f'{session_name}') #added for loop by LK
    session_path = base_path + '/' + date + '/' + mouse_id + '/' + session_name

    config = load_config(f'{session_path}/session_config.json')
    # fix config paths
    config["base_path"] = base_path
    config["date"] = date
    config["registration_config"]["matching_point_path"] = f'{session_path}/matching_points.txt'
    # config["supplementary_data_config"]["rois_dict_path"] = f'{config["base_path"]}/{mouse_id}/functional_parcellation_rois_dict_left_hemi.h5'
    config["supplementary_data_config"]["rois_dict_path"] = f'{config["base_path"]}/{mouse_id}/functional_parcellation_rois_dict.h5'
    config["supplementary_data_config"]["mask_path"] = "/data/Rotem/Wide Field/WideFlow/data/cortex_map/allen_2d_cortex.h5"
    #20221122_MR_CRC3functional_parcellation_rois_dict.h5
    #FLfunctional_parcellation_rois_dict_CRC3.h5
    #20221122_{mouse_id}_CRC3functional_parcellation_rois_dict

    #For correction of sessions April 2023
    #config["session_name"] = session_name
    #config["acquisition_config"]["num_of_frames"] = 50000


    #for spont
    #config["acquisition_config"]["metric_roi"] = ['roi_135']
    #config["supplementary_data_config"]["closest_rois"] = []

    # #to remove "Mexican hat" from session that ran with "mexican hat
    # config["supplementary_data_config"]["closest_rois"] = []

    ##for CRC session to run as mock NF
    config["analysis_pipeline_config"]["args"]["metric_args"] = ["ROIDiff", ["roi_72"], 5]
    config["feedback_config"]["update_frames"] = [1000,60000]
    config["feedback_config"]["eval_frames"] = 20000
    config["feedback_config"]["update_every"] = 10
    config["feedback_config"]["metric_threshold"] = 2.8
    config["feedback_config"]["percentile"] = 95
    config["acquisition_config"]["metric_roi"] = ['roi_72']
    config["supplementary_data_config"]["closest_rois"] = []

#    run_converter(session_path)
    sess = PostAnalysisNeuroFeedbackSession(config)
    #a=5
    sess.session_preparation()
    sess.run_session_pipeline()
