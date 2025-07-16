import numpy as np
import matplotlib.pyplot as plt
from wideflow.analysis.utils.extract_from_metadata_file import extract_from_metadata_file
from wideflow.utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict
import h5py


base_path = '/data/Lena/WideFlow_prj'
dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/results_exp3.h5'


dates_vec = ['20250622']
sessions_vec = ['NF2.2']

mice_id = [ #'21ML'
    #'31MN','54MRL'
    #,'63MR','64ML'
     #'187FN'
     # '203MN'
     #  ,'204FR'
     #   ,'206FRL'
     #    ,'211MRR'
     #    ,'218MN'
    '226MR',
    #'228MN',
    # '229FR',
     #'232FN',
    #'241FRLL'
    ]
metric_index = [ #somatosensory target ROI
    #134,  # 21
    #105,  # 31
    # 85,  # 54
    # 52,  # 63
    #71,  # 64
     #56,  #187
    # 41,  #203
    #69,  #204
    #  50,  #206
      #53,  #211
     # 46  #218
    35, #226
    #46, #228
   # 40, #229
   #46, #232
    #67, #241
     ]#(those are the indexes, the actual ROI numbers are this +1)[134, 105, 85, 52, 71 ]

session_id = f'{dates_vec[0]}_{mice_id[0]}_{sessions_vec[0]}'

d3 = {}
with h5py.File(f'{dataset_path_noMH}','r') as f:
        decompose_h5_groups_to_dict(f, d3, f'/{mice_id[0]}/{session_id}/post_session_analysis_LK2/')


[timestamp, cue, metric_result, threshold, serial_readout,trial_number] = extract_from_metadata_file(f'{base_path}/{dates_vec[0]}/{mice_id[0]}/{session_id}/metadata.txt')
cue1 = cue[::2]

# m232_roi47_diff5_org = d3['zsores_MH_diff5'][f'roi_{str(metric_index[0] + 1)}']
m232_roi47_diff5_org = d3['zsores_MH_diff5'][f'roi_36']
#m232_roi47_diff10_exc_top10 = d3['zsores_MH_diff10_exc_top10']['roi_36']
m232_roi47_diff10_exc_top15 = d3['zsores_MH_diff10_exc_top15'][f'roi_36']
m232_roi47_diff10_exc_top15_roi471 = d3['zsores_MH_diff10_exc_top15'][f'roi_361']
# m232_roi47_diff5_top15 = d3['zsores_MH_diff5_exc_top15']['roi_36']
# m232_roi47_diff5_top20 = d3['zsores_MH_diff5_exc_top20']['roi_36']
# m232_roi47_diff5_top10 = d3['zsores_MH_diff5_exc_top10']['roi_36']


#plt.plot(m232_roi4_diff10_top5, color = 'green', label='diff10_exc_top5')
plt.plot(9*np.array(cue1), color = 'black', label='rewards in session')
plt.plot(m232_roi47_diff10_exc_top15_roi471, color='green', alpha = 0.7, label='diff10_exc_top15_roi47+49')
#plt.plot(m232_roi47_diff10_exc_top15, color='yellow', alpha = 0.7, label='diff10_exc_top15')
plt.plot(m232_roi47_diff5_org, color = 'purple', alpha = 0.5, label='diff5_org')
#plt.plot(5.7 * np.ones_like(m232_roi4_diff5_top10))
plt.plot(3.5 * np.ones_like(m232_roi47_diff10_exc_top15), label='threshold for diff5', color = 'purple')
plt.plot(4.9 * np.ones_like(m232_roi47_diff10_exc_top15), label = 'threshold for diff10 exc top 15_roi47+49',color='green')
#plt.plot(5.3 * np.ones_like(m232_roi47_diff10_exc_top15), label = 'threshold for diff10 exc top 15', color='yellow')
plt.legend()
plt.show(block='false')
