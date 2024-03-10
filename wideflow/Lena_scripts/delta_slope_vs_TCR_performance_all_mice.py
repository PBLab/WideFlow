from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import pingouin as pg
import h5py
from utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict
from skimage.morphology import skeletonize
from utils.load_rois_data import load_rois_data
from analysis.utils.rois_proximity import calc_rois_proximity

def calc_rois_corr(rois_dict, data, data_chosen_roi):
    rois_corr = {}
    for i, roi_key in enumerate (rois_dict.keys()):
        corr=(np.corrcoef(data[roi_key],data_chosen_roi)[0,1])
        rois_corr[roi_key] = corr
    return rois_corr



base_path = '/data/Lena/WideFlow_prj'
#dates_vec = ['20230615', '20230618', '20230619', '20230620', '20230621', '20230622']
#dates_vec_roi1 = ['20230604', '20230611', '20230612', '20230613', '20230614', '20230615']
dates_vec_roi1 = ['20230604', '20230615']
mice_id = ['21ML','31MN','54MRL', '63MR', '64ML']
colors = ['cyan', 'orange', 'purple', 'chartreuse', 'magenta'] #21'cyan',24'blue',31'orange',46'green',54'purple', 63'chartreuse', 64'magenta'
mice_and_roi = 'All_mice_exp_24_46_sessNF5'
spont_sess_length_frames = 50000
NF_sess_length_frames = 65000
#sessions_vec_roi1 = ['spont_mockNF_NOTexcluded_closest','NF1', 'NF2', 'NF3', 'NF4', 'NF5']
sessions_vec_roi1 = ['spont_mockNF_NOTexcluded_closest','NF5']
#sessions_vec = ['spont_mockNF_ROI2_excluded_closest', 'NF1_mock_ROI2','NF2_mock_ROI2','NF3_mock_ROI2','NF4_mock_ROI2', 'NF5_mock_ROI2']
#sessions_vec = ['NF5', 'NF21_mock_ROI1','NF22_mock_ROI1','NF23_mock_ROI1','NF24_mock_ROI1', 'NF25_mock_ROI1']
set_threshold = 1.3
indexes_vec = [134, 105, 85, 52, 71 ]#(those are the indexes, the actual ROI numbers are this +1)

title = 'best_delta_slope_vs_best_TCR_scatter_NF5'

results_path = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'


#crossings ROI1
crossings_roi1 = np.zeros((len(mice_id),len(sessions_vec_roi1)))
slopes = []
for mouse_id, metric_index in zip(mice_id, indexes_vec):
    for date, session_name in zip(dates_vec_roi1, sessions_vec_roi1):
        session_id = f'{date}_{mouse_id}_{session_name}'
        timestamp, cue, metric_result, threshold, serial_readout = extract_from_metadata_file(f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt')


        #TCR
        count = 0
        for value in metric_result:
            if value > set_threshold:
                count += 1

        crossings_roi1[mice_id.index(mouse_id),sessions_vec_roi1.index(session_name)] = count

        #Slopes
        data = {}
        with h5py.File(results_path, 'r') as f:
            decompose_h5_groups_to_dict(f, data, f'/{mouse_id}/{session_id}/')

        functional_cortex_map_path = f'{base_path}/{mouse_id}/functional_parcellation_cortex_map.h5'
        functional_rois_dict_path = f'{base_path}/{mouse_id}/functional_parcellation_rois_dict.h5'
        with h5py.File(functional_cortex_map_path, 'r') as f:
            functional_cortex_mask = f["mask"][()]
            functional_cortex_map = f["map"][()]
        functional_cortex_mask = functional_cortex_mask[:, :168]
        functional_cortex_map = functional_cortex_map[:, :168]
        functional_cortex_map = skeletonize(functional_cortex_map)
        functional_rois_dict = load_rois_data(functional_rois_dict_path)

        metric_corr = {}
        rois_proximity = {}
        corr_all_rois = {}
        traces = data['rois_traces']['channel_0']

        for i, (key, val) in enumerate(functional_rois_dict.items()):
            # metric_corr[key] = np.corrcoef(pstr_cat[key], pstr_cat[metric_roi])[0, 1]  # correlation with metric ROI
            # dff_corr[key] = np.corrcoef (sessions_data[sess_id]['post_session_analysis']['dff']['traces'][i], sessions_data[sess_id]['post_session_analysis']['dff']['traces'][105])[0,1]
            # metric_corr[key] = np.corrcoef(metric_trace, sessions_data[sess_id]['post_session_analysis']['dff']['traces'][i])[0, 1]
            metric_corr[key] = np.corrcoef(traces[key],traces[f'roi_{metric_index+1}'])[0, 1]

            corr_all_rois[key] = calc_rois_corr(functional_rois_dict,traces,traces[key])
            rois_proximity[key] = calc_rois_proximity(functional_rois_dict, key)


        rois_proximity_metric = calc_rois_proximity(functional_rois_dict, f'roi_{metric_index+1}')

        x_data = np.array(list(rois_proximity_metric.values()))
        y_data = np.array(list(metric_corr.values()))
        coefficients = np.polyfit(x_data, y_data, 1)
        slopes.append(coefficients[0])

#crossings[3,3] = (crossings[3,2]+crossings[3,4])/2
first_column_roi1 = crossings_roi1[:, 0]
first_column_roi1 = (NF_sess_length_frames/spont_sess_length_frames)*first_column_roi1
crossings_roi1[:,0] = first_column_roi1
norm_crossings_roi1 = crossings_roi1 - first_column_roi1[:, np.newaxis]
norm_crossings_roi1 = [[element / first_column_roi1[i] for element in row] for i, row in enumerate(norm_crossings_roi1)]

#maxes TCR ROI1
maxes = np.zeros ((len(mice_id),1))
for i in range(len(mice_id)):
    maxes[i,0] = np.max(norm_crossings_roi1[i])


#maxes delta slopes
max_delta_slopes = []
slopes_sessions = [slopes[i:i+len(sessions_vec_roi1)] for i in range(0, len(slopes), len(sessions_vec_roi1))]
for i in range(len(mice_id)):
    max = np.max([np.abs(slopes_sessions[i][1] - slopes_sessions[i][0])])
                  #np.abs(slopes_sessions[2][i])-np.abs(slopes_sessions[0][i])])
                  #np.abs(slopes_sessions[3][i])-np.abs(slopes_sessions[0][i]),
                  #np.abs(slopes_sessions[4][i])-np.abs(slopes_sessions[0][i]),
                  #np.abs(slopes_sessions[5][i])-np.abs(slopes_sessions[0][i])])
    max_delta_slopes.append(max)

a=5


for i,mouse,color_name in zip(range(len(mice_id)),mice_id,colors):
    plt.scatter(maxes[i,0],max_delta_slopes [i],label = f'{mouse}', color = color_name)

#plt.plot(sessions_vec, mean, color='black', linestyle='--', label='Mean')
#plt.suptitle(f'{mice_and_roi}_fix_crossings_{set_threshold}')
plt.grid()
plt.ylabel ('Delta slope')
plt.xlabel ('TCR')
plt.legend()

#plt.show()
plt.savefig(f'{base_path}/Figs_for_paper/{title}.svg',format='svg',dpi=500)


a=5