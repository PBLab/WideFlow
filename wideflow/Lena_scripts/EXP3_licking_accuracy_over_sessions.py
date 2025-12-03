import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
from analysis.utils.peristimulus_time_response import calc_sdf
from analysis.plots import *
from scipy.ndimage.filters import maximum_filter1d
from scipy.stats import ttest_rel


#
# base_path = '/data/Lena/WideFlow_prj'
#
#
# mice_id = [
#     #'21ML'
#     '31MN'
#     ,'54MRL'
#     #,'63MR'
#     #,'64ML'
#     ,'187FN'
#     ,'203MN'
#     ,'204FR'
#     ,'206FRL'
#     ,'211MRR'
#     #,'218MN'
# ]
# #dates = ['20230605','20230608','20230605','20230608','20230605','20230608']
# sessions_names = {
#     #mice_id[0]: ['20230606_21ML_CRC2', '20230608_21ML_CRC4'],
#                   mice_id[0]: ['20230606_31MN_CRC2', '20230608_31MN_CRC4'],
#                   mice_id[1]: ['20230605_54MRL_CRC1', '20230608_54MRL_CRC4'],
#                   #mice_id[3]: ['20230605_63MR_CRC1', '20230607_63MR_CRC3'],
#                   #mice_id[4]: ['20230605_64ML_CRC1', '20230608_64ML_CRC4']
# mice_id[2]: ['20241123_187FN_CRC1', '20241126_187FN_CRC4'],
# mice_id[3]: ['20241123_203MN_CRC1', '20241125_203MN_CRC3'],
# mice_id[4]: ['20241123_204FR_CRC1', '20241125_204FR_CRC3'],
# mice_id[5]: ['20241123_206FRL_CRC1', '20241126_206FRL_CRC4'],
# mice_id[6]: ['20241123_211MRR_CRC1', '20241126_211MRR_CRC4'],
# #mice_id[7]: ['20241123_218MN_CRC1', '20241126_218MN_CRC4'],
# }
#
# sessions_dates = {
#     #mice_id[0]: ['20230606_21ML_CRC2', '20230608_21ML_CRC4'],
#                   mice_id[0]: ['20230606', '20230608'],
#                   mice_id[1]: ['20230605', '20230608'],
#                   #mice_id[3]: ['20230605_63MR_CRC1', '20230607_63MR_CRC3'],
#                   #mice_id[4]: ['20230605_64ML_CRC1', '20230608_64ML_CRC4']
#                 mice_id[2]: ['20241123', '20241126'],
#                 mice_id[3]: ['20241123', '20241125'],
#                 mice_id[4]: ['20241123', '20241125'],
#                 mice_id[5]: ['20241123', '20241126'],
#                 mice_id[6]: ['20241123', '20241126'],
#                 #mice_id[7]: ['20241123', '20241126'],
# }
# # sessions_names = ['20230605_54MRL_CRC1', '20230608_54MRL_CRC4','20230605_63MR_CRC1', '20230607_63MR_CRC3',
# #                   '20230605_64ML_CRC1', '20230608_64ML_CRC4']
#
# session_meta = {}
# for mouse_id in mice_id:
#     session_meta[mouse_id] = []
#     for s, (sess_name, sess_date) in enumerate(zip(sessions_names[mouse_id],sessions_dates[mouse_id])):
#         [timestamp, cue, metric_result, threshold, serial_readout] = extract_from_metadata_file(f'{base_path}/{sess_date}/{mouse_id}/{sess_name}/metadata.txt')
from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter1d
from scipy import stats
import pandas as pd
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import AnovaRM

base_path_qnap = '/data/Lena/WideFlow_prj'
base_path = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj'

#  statistics hyperparameters
sdf_win = [100, 500]
sdfx_win = [200, 200]
frames_win = 400
beta = 1

mice_id = [
    '245FRL',
    '246FN',
    '228MN',
    '259FRL',
    '252MR',
    '257FR',
    '258FL',
    '260FN',
    '261MR',
    # '248FL',
    # '256FLL',
    # '263MRL'
]
control_indexes = [0,1,2,3]
NF_indexes = [4,5,6,7,8,9, 10, 11]
mice_and_roi = 'EXP3.5_mice_248_256_263_removed'

colors = ['cyan', 'orange', 'purple', 'chartreuse', 'magenta','green','blue','red','mediumslateblue','olive', 'dodgerblue','maroon'] #21'cyan',24'blue',31'orange',46'green',54'purple', 63'chartreuse', 64'magenta'
lines = ['dashed','dashed','dashed','dashed','solid','solid','solid','solid','solid','solid','solid','solid']
#sessions_vec = ['spont_mockNF_ROI2_excluded_closest','NF21', 'NF22', 'NF23', 'NF24', 'NF25']
sessions_vec = [#'NF1_p1', 'NF1_p2', 'NF1_p3',
                'NF2_p1', 'NF2_p2', 'NF2_p3',
                'NF3_p1', 'NF3_p2', 'NF3_p3',
                'NF4_p1', 'NF4_p2', 'NF4_p3',
                'NF5_p1', 'NF5_p2', 'NF5_p3',
                'NF6_p1', 'NF6_p2', 'NF6_p3',]
    #,'NF_control_p1','NF_control_p2','NF_control_p3']#sessions_vec = ['spont_mockNF_ROI2_excluded_closest', 'NF1_mock_ROI2','NF2_mock_ROI2','NF3_mock_ROI2','NF4_mock_ROI2', 'NF5_mock_ROI2']
#sessions_vec = ['NF5', 'NF21_mock_ROI1','NF22_mock_ROI1','NF23_mock_ROI1','NF24_mock_ROI1', 'NF25_mock_ROI1']
#set_threshold = 0.5


#success_rates = np.zeros((len(mice_id),len(sessions_vec)))

session_meta = {}
for mouse_id in mice_id:
    if mouse_id == '248FL' or mouse_id == '257FR':
        dates_vec = [#'20250731', '20250731', '20250731',
                     '20250801', '20250801', '20250801', '20250802', '20250802','20250802', '20250803', '20250803','20250803',
                      '20250804', '20250804','20250804', '20250805', '20250805','20250805']#,'20250806', '20250806','20250806']
        sessions_vec = [# 'NF1_p1', 'NF1_p2', 'NF1_p3',
                         'NF2_p1', 'NF2_p2', 'NF2_p3', 'NF3_p1', 'NF3_p2', 'NF3_p3', 'NF4_p1', 'NF4_p2', 'NF4_p3', 'NF5_p1',
                         'NF5_p2', 'NF5_p3' , 'NF6_p1', 'NF6_p2', 'NF6_p3']
            # ,'NF_control_p1','NF_control_p2','NF_control_p3'  ]
    elif mouse_id == '252MR':
        dates_vec = [#'20250731', '20250731', '20250731',
                     '20250801', '20250801', '20250801', '20250802', '20250802', '20250802',
                     '20250804', '20250804', '20250804', '20250805', '20250805', '20250805','20250806', '20250806', '20250806']#,'20250810', '20250810', '20250810']
        sessions_vec = [#'NF1_p1', 'NF1_p2', 'NF1_p3',
                        'NF2_p1', 'NF2_p2', 'NF2_p3', 'NF3_p1', 'NF3_p2', 'NF3_p3',
                        'NF5_p1', 'NF5_p2', 'NF5_p3', 'NF6_p1', 'NF6_p2', 'NF6_p3','NF7_p1', 'NF7_p2', 'NF7_p3']#,'NF_control_p1','NF_control_p2','NF_control_p3']

    elif mouse_id == '245FRL' or mouse_id=='246FN':
        dates_vec = [#'20250731', '20250731', '20250731',
                     '20250731', '20250731', '20250731', '20250801', '20250801', '20250801','20250801', '20250801', '20250801',
                     '20250803', '20250803','20250803', '20250804', '20250804','20250804']#,'20250805', '20250805','20250805']
        sessions_vec = [#'NF1_p1', 'NF1_p2', 'NF1_p3',
                        'NF1_p1', 'NF1_p2', 'NF1_p3', 'NF2_p1', 'NF2_p2', 'NF2_p3','NF2_p1', 'NF2_p2', 'NF2_p3', 'NF4_p1', 'NF4_p2', 'NF4_p3', 'NF5_p1',
                         'NF5_p2', 'NF5_p3']#,'NF_control_p1','NF_control_p2','NF_control_p3']

    elif (mouse_id == '258FL' or mouse_id=='260FN' or mouse_id=='261MR' or mouse_id=='263MRL'
          or mouse_id=='228MN' or mouse_id=='259FRL'):
        dates_vec = [#'20250905','20250905','20250905',
                     '20250906','20250906','20250906','20250907','20250907','20250907','20250908'
                     , '20250908','20250908','20250909','20250909','20250909','20250910','20250910','20250910']#,'20250805', '20250805','20250805']
        sessions_vec = [# 'NF1_p1', 'NF1_p2', 'NF1_p3',
                         'NF2_p1', 'NF2_p2', 'NF2_p3', 'NF3_p1', 'NF3_p2', 'NF3_p3', 'NF4_p1', 'NF4_p2', 'NF4_p3', 'NF5_p1',
                         'NF5_p2', 'NF5_p3' , 'NF6_p1', 'NF6_p2', 'NF6_p3']



    else:
        dates_vec = [#'20250731', '20250731', '20250731',
                     '20250731', '20250731', '20250731', '20250801', '20250801', '20250801', '20250802', '20250802','20250802',
                     '20250803', '20250803','20250803', '20250804', '20250804','20250804']#,'20250805', '20250805','20250805']
        sessions_vec = [#'NF1_p1', 'NF1_p2', 'NF1_p3',
                        'NF1_p1', 'NF1_p2', 'NF1_p3', 'NF2_p1', 'NF2_p2', 'NF2_p3', 'NF3_p1', 'NF3_p2', 'NF3_p3', 'NF4_p1', 'NF4_p2', 'NF4_p3', 'NF5_p1',
                         'NF5_p2', 'NF5_p3']#,'NF_control_p1','NF_control_p2','NF_control_p3']

    # for date, session_name in zip(dates_vec, sessions_vec):
    #     session_id = f'{date}_{mouse_id}_{session_name}'
    session_meta[mouse_id] = []
    for s, (date, session_name) in enumerate(zip(dates_vec, sessions_vec)):
        session_id = f'{date}_{mouse_id}_{session_name}'
        print (f'{session_id}')
        timestamp, cue, metric_result, threshold, serial_readout, trial_number = extract_from_metadata_file(f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt')

        dt = np.mean(np.diff(timestamp))
        cue = np.array(cue)
        serial_readout = 1 - np.array(serial_readout)
        serial_readout = maximum_filter1d(serial_readout, 5)
        #sdf = np.mean(calc_sdf(cue, serial_readout, sdf_win, 2), axis=0)

        serial_readoutx = copy.copy(serial_readout)
        for i in range(len(cue)):
            if cue[i]:
                serial_readoutx[i:i + frames_win*3] = 0
        serial_readoutc = serial_readout - serial_readoutx
        sdfx = np.mean(calc_sdf(serial_readoutx, serial_readoutx, sdfx_win, 2), axis=0)

        # autocorr = np.mean(calc_sdf(np.ones((len(serial_readoutx), )), serial_readout, sdfx_win, 2), axis=0)
        autocorr = np.correlate(serial_readoutx, serial_readoutx, 'same')
        autocorr = autocorr[int(len(autocorr) / 2) - 200: int(len(autocorr) / 2) + 200]

        session_meta[mouse_id].append({"timestamp": timestamp, "cue": cue, "metric_result": metric_result,
                                       "threshold": threshold, "serial_readout": serial_readout, "dt": dt,
                                       "autocorr": autocorr})

        n_samples = len(cue)
        lick_frames = np.sum(cue) * frames_win

        tp = 0  # mouse lick when it should lick
        fp = 0  # mouse lick when it shouldn't lick
        tn = 0  # mouse don't lick when it shouldn't lick
        fn = 0  # mouse don't lick when it should lick

        tpi = np.zeros((n_samples,))
        fpi = np.zeros((n_samples,))
        tni = np.zeros((n_samples,))
        fni = np.zeros((n_samples,))
        for i in range(n_samples):
            if 1 in cue[np.max((0, i-frames_win)): i]:  # in licking period
                if serial_readout[i]:
                    tp += 1
                    tpi[i] = 1
                else:
                    fn += 1
                    fni[i] = 1
            else:  # in quiet period
                if serial_readout[i]:
                    fp += 1
                    fpi[i] = 1
                else:
                    tn += 1
                    tni[i] = 1

        if (tp+fn) == 0:
            true_positive_rate = 0
        else:
            true_positive_rate = tp / (tp + fn)

        if (tp+fn) == 0:
            false_negative_rate = 1
        else:
            false_negative_rate = fn / (tp + fn)

        #true_positive_rate = tp / (tp + fn)
        true_negative_rate = tn / (tn + fp)
        false_positive_rate = fp / (tn + fp)
        #false_negative_rate = fn / (tp + fn)
        false_discovery_rate = fp / (fp + tp)
        licking_rate = (tp+fp)/n_samples


        precision = tp / (tp + fp)
        #recall = tp / (tp + fn)
        #recall_inv = tn / (fp + tn)
        #f1_score = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
        accuracy = (tp+tn)/(tp+tn+fp+fn)

        session_meta[mouse_id][s]["tpr"] = true_positive_rate
        session_meta[mouse_id][s]["tp"] = tp
        session_meta[mouse_id][s]["tnr"] = true_negative_rate
        session_meta[mouse_id][s]["tn"] = tn
        session_meta[mouse_id][s]["fpr"] = false_positive_rate
        session_meta[mouse_id][s]["fp"] = fp
        session_meta[mouse_id][s]["fnr"] = false_negative_rate
        session_meta[mouse_id][s]["fn"] = fn
        session_meta[mouse_id][s]["fdr"] = false_discovery_rate
        session_meta[mouse_id][s]["lr"] = licking_rate

        session_meta[mouse_id][s]["tpi"] = tpi
        session_meta[mouse_id][s]["tni"] = tni
        session_meta[mouse_id][s]["fpi"] = fpi
        session_meta[mouse_id][s]["fni"] = fni

        session_meta[mouse_id][s]["precision"] = precision
        session_meta[mouse_id][s]["accuracy"] = accuracy
        #session_meta[mouse_id][s]["recall"] = recall
        #session_meta[mouse_id][s]["recall_inv"] = recall_inv
        #session_meta[mouse_id][s]["F1 score"] = f1_score
        #session_meta[mouse_id][s]["random F1 score"] = lick_frames / n_samples
    a=5


a=5

# sessions_vec1 = [#'NF1_p1', 'NF1_p2', 'NF1_p3',
#                 'NF2_p1', 'NF2_p2', 'NF2_p3',
#                 'NF3_p1', 'NF3_p2', 'NF3_p3',
#                 'NF4_p1', 'NF4_p2', 'NF4_p3',
#                 'NF5_p1', 'NF5_p2', 'NF5_p3',
#                 'NF6_p1', 'NF6_p2', 'NF6_p3']
sessions_vec1 = ['NF1','NF2','NF3','NF4','NF5']
# mice_id1 = [x for x in mice_id for _ in range(3)]
# colors = [x for x in colors for _ in range(3)]
# lines = [x for x in lines for _ in range(3)]



###FDR
fdr_matrix = {
    mouse_id: [session['fdr'] for session in sessions]
    for mouse_id, sessions in session_meta.items()
}
df_fdr = pd.DataFrame.from_dict(fdr_matrix, orient='index')
df_avg_fdr = df_fdr.apply(
    lambda row: np.mean(row.values.reshape(-1, 3), axis=1),
    axis=1, result_type='expand'
)
##Norm to NF1 - comment out following line if norm not needed
df_avg_fdr = (df_avg_fdr.sub(df_avg_fdr.iloc[:, 0], axis=0)).div(df_avg_fdr.iloc[:, 0], axis=0)

sum_NF_fdr = []
for a,b,c,d,e in zip (list(df_avg_fdr.loc['252MR']),list(df_avg_fdr.loc['257FR']),list(df_avg_fdr.loc['258FL']),list(df_avg_fdr.loc['260FN']),
                        list(df_avg_fdr.loc['261MR'])):
        # ,list(df_avg_fdr.loc['248FL'])
        # ,list(df_avg_fdr.loc['256FLL']),list(df_avg_fdr.loc['263MRL'])):
    sum_NF_fdr.append(a+b+c+d+e)
mean_NF_fdr = [num /5 for num in sum_NF_fdr]

sum_control_fdr = []
for a,b,c,d in zip (list(df_avg_fdr.loc['245FRL']),list(df_avg_fdr.loc['246FN']), list(df_avg_fdr.loc['228MN']), list(df_avg_fdr.loc['259FRL'])):
    sum_control_fdr.append(a+b+c+d)

mean_control_fdr = [num /4 for num in sum_control_fdr]


####TPR
tpr_matrix = {
    mouse_id: [session['tpr'] for session in sessions]
    for mouse_id, sessions in session_meta.items()
}
df_tpr = pd.DataFrame.from_dict(tpr_matrix, orient='index')
df_avg_tpr = df_tpr.apply(
    lambda row: np.mean(row.values.reshape(-1, 3), axis=1),
    axis=1, result_type='expand'
)
##Norm to NF1 - comment out following line if norm not needed
df_avg_tpr = (df_avg_tpr.sub(df_avg_tpr.iloc[:, 0], axis=0)).div(df_avg_tpr.iloc[:, 0], axis=0)

sum_NF_tpr = []
for a,b,c,d,e in zip (list(df_avg_tpr.loc['252MR']),list(df_avg_tpr.loc['257FR']),list(df_avg_tpr.loc['258FL']),list(df_avg_tpr.loc['260FN']),
                        list(df_avg_tpr.loc['261MR'])):
        # ,list(df_avg_fdr.loc['248FL'])
        #                 ,list(df_avg_fdr.loc['256FLL']),list(df_avg_fdr.loc['263MRL'])):
    sum_NF_tpr.append(a+b+c+d+e)

mean_NF_tpr = [num /5 for num in sum_NF_tpr]

sum_control_tpr = []
for a,b,c,d in zip (list(df_avg_tpr.loc['245FRL']),list(df_avg_tpr.loc['246FN']), list(df_avg_tpr.loc['228MN']), list(df_avg_tpr.loc['259FRL'])):
    sum_control_tpr.append(a+b+c+d)

mean_control_tpr = [num /4 for num in sum_control_tpr]


###ACCURACY
accu_matrix = {
    mouse_id: [session['accuracy'] for session in sessions]
    for mouse_id, sessions in session_meta.items()
}
df_accu = pd.DataFrame.from_dict(accu_matrix, orient='index')
df_avg_accu = df_accu.apply(
    lambda row: np.mean(row.values.reshape(-1, 3), axis=1),
    axis=1, result_type='expand'
)
##Norm to NF1 - comment out following line if norm not needed
df_avg_accu = (df_avg_accu.sub(df_avg_accu.iloc[:, 0], axis=0)).div(df_avg_accu.iloc[:, 0], axis=0)


sum_NF_accu = []
for a,b,c,d,e in zip (list(df_avg_accu.loc['252MR']),list(df_avg_accu.loc['257FR']),list(df_avg_accu.loc['258FL']),list(df_avg_accu.loc['260FN']),
                        list(df_avg_accu.loc['261MR'])):
        # ,list(df_avg_fdr.loc['248FL'])
        #                     ,list(df_avg_fdr.loc['256FLL']),list(df_avg_fdr.loc['263MRL'])):
    sum_NF_accu.append(a+b+c+d+e)

mean_NF_accu = [num /5 for num in sum_NF_accu]

sum_control_accu = []
for a,b,c,d in zip (list(df_avg_accu.loc['245FRL']),list(df_avg_accu.loc['246FN']), list(df_avg_accu.loc['228MN']), list(df_avg_accu.loc['259FRL'])):
    sum_control_accu.append(a+b+c+d)

mean_control_accu = [num /4 for num in sum_control_accu]




###PRECISION
preci_matrix = {
    mouse_id: [session['precision'] for session in sessions]
    for mouse_id, sessions in session_meta.items()
}
df_preci = pd.DataFrame.from_dict(preci_matrix, orient='index')
df_avg_preci = df_preci.apply(
    lambda row: np.mean(row.values.reshape(-1, 3), axis=1),
    axis=1, result_type='expand'
)
##Norm to NF1 - comment out following line if norm not needed
df_avg_preci = (df_avg_preci.sub(df_avg_preci.iloc[:, 0], axis=0)).div(df_avg_preci.iloc[:, 0], axis=0)


sum_NF_preci = []
for a,b,c,d,e in zip (list(df_avg_preci.loc['252MR']),list(df_avg_preci.loc['257FR']),list(df_avg_preci.loc['258FL']),list(df_avg_preci.loc['260FN']),
                        list(df_avg_preci.loc['261MR'])):
        # ,list(df_avg_fdr.loc['248FL'])
        #                 ,list(df_avg_fdr.loc['256FLL']),list(df_avg_fdr.loc['263MRL'])):
    sum_NF_preci.append(a+b+c+d+e)

mean_NF_preci = [num /5 for num in sum_NF_preci]

sum_control_preci = []
for a,b,c,d in zip (list(df_avg_preci.loc['245FRL']),list(df_avg_preci.loc['246FN']), list(df_avg_preci.loc['228MN']), list(df_avg_preci.loc['259FRL'])):
    sum_control_preci.append(a+b+c+d)

mean_control_preci = [num /4 for num in sum_control_preci]


######PLOTTING
f = plt.figure(constrained_layout=True, figsize=(16, 8))
gs = f.add_gridspec(2,2)


ax_fdr = f.add_subplot(gs[0, 0])
for i, mouse_id1,color_name, linestyle in zip(list(range(0,len(mice_id))), mice_id,colors, lines):
    ax_fdr.plot(sessions_vec1, list(df_avg_fdr.loc[f'{mouse_id1}']), label = f'{mouse_id1}', color = color_name, linestyle = linestyle)
ax_fdr.plot(sessions_vec1, mean_NF_fdr, color='black', linestyle='solid', label='Mean', linewidth=5)
ax_fdr.plot(sessions_vec1, mean_control_fdr, color='grey', linestyle='--', label='Mean control', linewidth = 5)

ax_fdr.set_title(f'FDR')
ax_fdr.grid ()
ax_fdr.legend()


ax_tpr = f.add_subplot(gs[0, 1])
for i, mouse_id1,color_name, linestyle in zip(list(range(0,len(mice_id))), mice_id,colors, lines):
    ax_tpr.plot(sessions_vec1, list(df_avg_tpr.loc[f'{mouse_id1}']), label = f'{mouse_id1}', color = color_name, linestyle = linestyle)
ax_tpr.plot(sessions_vec1, mean_NF_tpr, color='black', linestyle='solid', label='Mean', linewidth=5)
ax_tpr.plot(sessions_vec1, mean_control_tpr, color='grey', linestyle='--', label='Mean control', linewidth = 5)

ax_tpr.set_title(f'TPR')
ax_tpr.grid ()
#ax_tpr.legend()


ax_accu = f.add_subplot(gs[1, 0])
for i, mouse_id1,color_name, linestyle in zip(list(range(0,len(mice_id))), mice_id,colors, lines):
    ax_accu.plot(sessions_vec1, list(df_avg_accu.loc[f'{mouse_id1}']), label = f'{mouse_id1}', color = color_name, linestyle = linestyle)
ax_accu.plot(sessions_vec1, mean_NF_accu, color='black', linestyle='solid', label='Mean', linewidth=5)
ax_accu.plot(sessions_vec1, mean_control_accu, color='grey', linestyle='--', label='Mean control', linewidth = 5)

ax_accu.set_title(f'Accuracy')
ax_accu.grid ()
#ax_accu.legend()


ax_preci = f.add_subplot(gs[1, 1])
for i, mouse_id1,color_name, linestyle in zip(list(range(0,len(mice_id))), mice_id,colors, lines):
    ax_preci.plot(sessions_vec1, list(df_avg_preci.loc[f'{mouse_id1}']), label = f'{mouse_id1}', color = color_name, linestyle = linestyle)
ax_preci.plot(sessions_vec1, mean_NF_preci, color='black', linestyle='solid', label='Mean', linewidth=5)
ax_preci.plot(sessions_vec1, mean_control_preci, color='grey', linestyle='--', label='Mean control', linewidth = 5)

ax_preci.set_title(f'Precision')
ax_preci.grid ()
#ax_preci.legend()

# plt.show()
plt.savefig(f'{base_path_qnap}/figs_for_paper_EXP3.5/FDR_TPR_Accuracy_Precision_over_sessions_{mice_and_roi}.svg',format='svg',dpi=200)

########Addind results for 21ML CRC4 Experiment 2 7.1.2024####################

#Those values were calculated manually after counting the licks in the behavioral recording:
# tp=3598
# fn=6396
# fp=3505
# tn=46500
#
# n_samples = 25
# lick_frames = 25 * frames_win
#
# true_positive_rate = tp / (tp + fn)
# true_negative_rate = tn / (tn + fp)
# false_positive_rate = fp / (tn + fp)
# false_negative_rate = fn / (tp + fn)
# false_discovery_rate = fp / (fp + tp)
#
# precision = tp / (tp + fp)
# recall = tp / (tp + fn)
# recall_inv = tn / (fp + tn)
# f1_score = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
#
# session_meta['21ML'][1]["tpr"] = true_positive_rate
# session_meta['21ML'][1]["tp"] = tp
# session_meta['21ML'][1]["tnr"] = true_negative_rate
# session_meta['21ML'][1]["tn"] = tn
# session_meta['21ML'][1]["fpr"] = false_positive_rate
# session_meta['21ML'][1]["fp"] = fp
# session_meta['21ML'][1]["fnr"] = false_negative_rate
# session_meta['21ML'][1]["fn"] = fn
# session_meta['21ML'][1]["fdr"] = false_discovery_rate
##########################
# session_meta[mouse_id][s]["tpi"] = tpi
# session_meta[mouse_id][s]["tni"] = tni
# session_meta[mouse_id][s]["fpi"] = fpi
# session_meta[mouse_id][s]["fni"] = fni

# session_meta['21ML'][1]["precision"] = precision
# session_meta['21ML'][1]["recall"] = recall
# session_meta['21ML'][1]["recall_inv"] = recall_inv
# session_meta['21ML'][1]["F1 score"] = f1_score
# session_meta['21ML'][1]["random F1 score"] = lick_frames / n_samples
#

#######End of 21ML CRC4###############################


###################################### plot results ######################################
# f = plt.figure(constrained_layout=True, figsize=(16, 8))
# gs = f.add_gridspec(4, 3)
#
# bar_width = 0.6
# font_size = 13
# c_response = ['royalblue', 'crimson','green']
# x = [2*bar_width, 3.5*bar_width, 5*bar_width, 6.5*bar_width, 8*bar_width, 9.5*bar_width]
#
# # tpr fpr bar plots ----------------------------------------------------------------------------
# ax_bar_up = f.add_subplot(gs[:2, 2])
# #mouse_colors = plt.cm.viridis(np.linspace(0, 1, len(mice_id)))  # Use a colormap for distinct colors
# mouse_colors = ['cyan', 'orange', 'purple', 'chartreuse', 'magenta','blue','red','olivedrab','grey','green','aquamarine'] #21'cyan',24'blue',31'orange',46'green',54'purple', 63'chartreuse', 64'magenta'
#
# # #y1 = [session_meta[mice_id[-1]][1]["tpr"], session_meta[mice_id[-1]][0]["tpr"]]
# # y1 = [(session_meta[mice_id[0]][1]["tpr"]+session_meta[mice_id[1]][1]["tpr"]+session_meta[mice_id[2]][1]["tpr"]+session_meta[mice_id[3]][1]["tpr"]+session_meta[mice_id[4]][1]["tpr"])/5,
# #      (session_meta[mice_id[0]][0]["tpr"]+session_meta[mice_id[1]][0]["tpr"]+session_meta[mice_id[2]][0]["tpr"]+session_meta[mice_id[3]][0]["tpr"]+session_meta[mice_id[4]][0]["tpr"])/5]
# # #std_y1 = [0.0372, 0.0778]
# # b1 = ax_bar_up.barh([x[1], x[3]], y1, height=bar_width, color=[c_response[0], c_response[0]], alpha=0.8)
# # #ax_bar_up.errorbar(x=(x[1],x[3]), y=y1, yerr=std_y1,fmt='o', color='black')
#
# # Extract TPR values for each subject (pre and post)
# y_post = [session_meta[m][1]["tpr"] for m in mice_id]  # Post-training
# y_pre = [session_meta[m][0]["tpr"] for m in mice_id]   # Pre-training
#
# # Define x-coordinates for pre and post bars
# x_pre = x[1]
# x_post = x[0]
#
# # Compute mean for bars
# y1 = [ sum(y_pre) / len(y_pre),sum(y_post) / len(y_post)]
#
# # Plot bars
# b1 = ax_bar_up.barh([x_pre, x_post], y1, height=bar_width, color=[c_response[0], c_response[0]], alpha=0.8)
#
# # Add individual subject connecting lines
# for i in range(len(mice_id)):
#     ax_bar_up.plot([y_pre[i], y_post[i]], [x_pre, x_post], color=mouse_colors[i], marker='o', alpha=0.6)
#
# ax_bar_up.set_yticks([])
# ax_bar_up.set_ylabel('post-training               pre-training', fontsize=font_size)
# ax_bar_up.grid(axis='x')
# ax_bar_up.set_xlim([0, 1])
# ax_bar_up.tick_params(axis='x', colors=c_response[0])
#
#
# ax_bar_up_t = ax_bar_up.twiny()
# # #y2 = [session_meta[mice_id[-1]][1]["fdr"], session_meta[mice_id[-1]][0]["fdr"]]
# # y2 = [(session_meta[mice_id[0]][1]["fdr"]+session_meta[mice_id[1]][1]["fdr"]+session_meta[mice_id[2]][1]["fdr"]+session_meta[mice_id[3]][1]["fdr"]+session_meta[mice_id[4]][1]["fdr"])/5,
# #      (session_meta[mice_id[0]][0]["fdr"]+session_meta[mice_id[1]][0]["fdr"]+session_meta[mice_id[2]][0]["fdr"]+session_meta[mice_id[3]][0]["fdr"]+session_meta[mice_id[4]][0]["fdr"])/5]
# # b2 = ax_bar_up_t.barh([x[0], x[2]], y2, height=bar_width, color=[c_response[1], c_response[1]], alpha=0.8)
# # Extract TPR values for each subject (pre and post)
# y_post2 = [session_meta[m][1]["fdr"] for m in mice_id]  # Post-training
# y_pre2 = [session_meta[m][0]["fdr"] for m in mice_id]   # Pre-training
#
# # Define x-coordinates for pre and post bars
# x_pre2 = x[3]
# x_post2 = x[2]
#
# # Compute mean for bars
# y2 = [ sum(y_pre2) / len(y_pre2), sum(y_post2) / len(y_post2)]
#
# # Plot bars
# b2 = ax_bar_up.barh([x_pre2, x_post2], y2, height=bar_width, color=[c_response[1], c_response[1]], alpha=0.8)
#
# # Create a dictionary to store handles for unique mice
# legend_handles = {}
# # Add individual subject connecting lines
# for i in range(len(mice_id)):
#     line, = ax_bar_up.plot([y_pre2[i], y_post2[i]], [x_pre2, x_post2], color=mouse_colors[i],label=f'{mice_id[i]}', marker='o', alpha=0.6)
#     if mice_id[i] not in legend_handles:
#         legend_handles[mice_id[i]] = line
#
# #ax_bar_up.legend(loc="upper right", fontsize=10)
# ax_bar_up_t.tick_params(axis='x', colors=c_response[1])
# ax_bar_up_t.set_xlim([0, 1])
#
# # ax_bar_up.legend([b1[0], b2[0]], ['sensitivity', 'false discovery rate', 'licking rate'], loc='lower right')
#
# ax_bar_up_t1 = ax_bar_up_t.twiny()
# y_post3 = [session_meta[m][1]["lr"] for m in mice_id]  # Post-training
# y_pre3 = [session_meta[m][0]["lr"] for m in mice_id]   # Pre-training
#
# # Define x-coordinates for pre and post bars
# x_pre3 = x[5]
# x_post3 = x[4]
#
# # Compute mean for bars
# y3 = [ sum(y_pre3) / len(y_pre3), sum(y_post3) / len(y_post3)]
#
# # Plot bars
# b3 = ax_bar_up.barh([x_pre3, x_post3], y3, height=bar_width, color=[c_response[2], c_response[2]], alpha=0.8)
#
# # Add individual subject connecting lines
# for i in range(len(mice_id)):
#     ax_bar_up.plot([y_pre3[i], y_post3[i]], [x_pre3, x_post3], color=mouse_colors[i],label=f'{mice_id[i]}', marker='o', alpha=0.6)
# ax_bar_up_t1.set_xlim([0, 1])
# ax_bar_up.legend([b1[0], b2[0],b3[0]], ['sensitivity', 'false discovery rate', 'licking rate'], loc='lower right')
# #ax_bar_up.legend(legend_handles.values(), legend_handles.keys(), loc="upper right", fontsize=10)
#
# # session stim lick timing ------------------------------------------------------------
# ax0 = f.add_subplot(gs[0, :2])
# plot_reward_response(ax0, session_meta[mice_id[3]][0]['cue'],
#             session_meta[mice_id[3]][0]['tpi'],
#             c_response=c_response[0])
# plot_reward_response(ax0, session_meta[mice_id[3]][0]['cue'],
#             session_meta[mice_id[3]][0]['fpi'],
#             c_response=c_response[1])
#
# ax0.set_ylabel('pre-training', fontsize=font_size)
# ax0.set_yticks([])
# ax0.set_xticks([])
#
# # ax0.set_title("Mouse #1", loc='left')
#
# ax1 = f.add_subplot(gs[1, :2])
# plot_reward_response(ax1, session_meta[mice_id[3]][1]['cue'],
#             session_meta[mice_id[3]][1]['tpi'],
#             c_response=c_response[0])
# plot_reward_response(ax1, session_meta[mice_id[3]][1]['cue'],
#             session_meta[mice_id[3]][1]['fpi'],
#             c_response=c_response[1])
# ax1.set_ylabel('post-training', fontsize=font_size)
# ax1.set_yticks([])
# ax1.set_xticks([])
#
# # ax0.axis('off')
# # ax1.axis('off')
# ax0.spines['left'].set_visible(False)
# ax0.spines['top'].set_visible(False)
# ax0.spines['right'].set_visible(False)
# ax0.spines['bottom'].set_visible(False)
# ax1.spines['left'].set_visible(False)
# ax1.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)
# ax1.spines['bottom'].set_visible(False)
#
# # cross mice statistics
# sdf_mean_pre = np.mean(
#     np.vstack((session_meta[mice_id[1]][0]['sdf'], session_meta[mice_id[2]][0]['sdf'],session_meta[mice_id[3]][0]['sdf'],session_meta[mice_id[4]][0]['sdf']))
#     , axis=0
# )
# sdf_mean_post = np.mean(
#     np.vstack((session_meta[mice_id[1]][1]['sdf'], session_meta[mice_id[2]][1]['sdf'], session_meta[mice_id[3]][1]['sdf'], session_meta[mice_id[4]][1]['sdf']))
#     , axis=0
# )
#
# sdfx_mean_pre = np.mean(
#     np.vstack(( session_meta[mice_id[1]][0]['sdfx'], session_meta[mice_id[2]][0]['autocorr'], session_meta[mice_id[3]][0]['autocorr'], session_meta[mice_id[4]][0]['autocorr']))
#     , axis=0
# )
# sdfx_mean_post = np.mean(
#     np.vstack(( session_meta[mice_id[1]][1]['sdfx'], session_meta[mice_id[2]][1]['autocorr'], session_meta[mice_id[3]][1]['autocorr'], session_meta[mice_id[4]][1]['autocorr']))
#     , axis=0
# )
# # sdf plots_______________________________________________________________
# ax_sdf = f.add_subplot(gs[2, :2])
#
# t1 = np.arange(-sdf_win[0], sdf_win[1], 1) * session_meta[mice_id[-1]][0]['dt'] * 1000
# ax_sdf.plot(t1, sdf_mean_pre, color='green')
# t2 = np.arange(-sdf_win[0], sdf_win[1], 1) * session_meta[mice_id[-1]][1]['dt'] * 1000
# ax_sdf.plot(t2, sdf_mean_post, color='blue')
# ax_sdf.vlines(0, 0, 1, linestyle='--', color='k')
# ax_sdf.set_ylabel('SDF [a.u]', fontsize=font_size)
# ax_sdf.legend(['before training', 'after training'])
# ax_sdf.spines['top'].set_visible(False)
# ax_sdf.spines['right'].set_visible(False)
#
# ax_sdfx = f.add_subplot(gs[3, :2])
#
# t1 = np.arange(-sdfx_win[0], sdfx_win[1], 1) * session_meta[mice_id[-1]][0]['dt'] * 1000
# ax_sdfx.plot(t1, sdfx_mean_pre, color='green')
# t2 = np.arange(-sdfx_win[0], sdfx_win[1], 1) * session_meta[mice_id[-1]][1]['dt'] * 1000
# ax_sdfx.plot(t2, sdfx_mean_post, color='blue')
# ax_sdfx.vlines(0, 0, 700, linestyle='--', color='k')
# ax_sdfx.set_xlabel('Time[ms]', fontsize=font_size)
# ax_sdfx.set_ylabel('Auto Correlation [a.u]', fontsize=font_size)
# ax_sdfx.legend(['before training', 'after training'])
# ax_sdfx.spines['top'].set_visible(False)
# ax_sdfx.spines['right'].set_visible(False)
#
# # confusion matrix plot__________________________________________________
# ax_conf = f.add_subplot(gs[2:, 2:])
# # cross_mice_stats = {}
# # cross_mice_stats['pre'], cross_mice_stats['post'] = {}, {}
# # cross_mice_stats['pre']['fp'] = (session_meta['31MN'][0]['fp'] +session_meta['54MRL'][0]['fp'] + session_meta['63MR'][0]['fp'] + session_meta['64ML'][0]['fp']) / 4
# # cross_mice_stats['pre']['tp'] = ( session_meta['31MN'][0]['tp'] +session_meta['54MRL'][0]['tp'] + session_meta['63MR'][0]['tp'] + session_meta['64ML'][0]['tp']) / 4
# # cross_mice_stats['pre']['fn'] = (session_meta['31MN'][0]['fn'] +session_meta['54MRL'][0]['fn'] + session_meta['63MR'][0]['fn'] + session_meta['64ML'][0]['fn']) / 4
# # cross_mice_stats['pre']['tn'] = (session_meta['31MN'][0]['tn'] +session_meta['54MRL'][0]['tn'] + session_meta['63MR'][0]['tn'] + session_meta['64ML'][0]['tn']) / 4
# #
# # cross_mice_stats['post']['fp'] = ( session_meta['31MN'][1]['fp'] +session_meta['54MRL'][1]['fp'] + session_meta['63MR'][1]['fp'] + session_meta['64ML'][1]['fp']) / 4
# # cross_mice_stats['post']['tp'] = ( session_meta['31MN'][1]['tp'] +session_meta['54MRL'][1]['tp'] + session_meta['63MR'][1]['tp'] + session_meta['64ML'][1]['tp']) / 4
# # cross_mice_stats['post']['fn'] = (session_meta['31MN'][1]['fn'] +session_meta['54MRL'][1]['fn'] + session_meta['63MR'][1]['fn'] + session_meta['64ML'][1]['fn']) / 4
# # cross_mice_stats['post']['tn'] = (session_meta['31MN'][1]['tn'] +session_meta['54MRL'][1]['tn'] + session_meta['63MR'][1]['tn'] + session_meta['64ML'][1]['tn']) / 4
# #
# # cross_mice_stats['pre']['fpr'] = (session_meta['31MN'][0]['fpr'] +session_meta['54MRL'][0]['fpr'] + session_meta['63MR'][0]['fpr'] + session_meta['64ML'][0]['fpr']) / 4
# # cross_mice_stats['pre']['tpr'] = ( session_meta['31MN'][0]['tpr'] +session_meta['54MRL'][0]['tpr'] + session_meta['63MR'][0]['tpr'] + session_meta['64ML'][0]['tpr']) / 4
# # cross_mice_stats['pre']['fnr'] = (session_meta['31MN'][0]['fnr'] +session_meta['54MRL'][0]['fnr'] + session_meta['63MR'][0]['fnr'] + session_meta['64ML'][0]['fnr']) / 4
# # cross_mice_stats['pre']['tnr'] = ( session_meta['31MN'][0]['tnr'] +session_meta['54MRL'][0]['tnr'] + session_meta['63MR'][0]['tnr'] + session_meta['64ML'][0]['tnr']) / 4
# #
# # cross_mice_stats['post']['fpr'] = (session_meta['31MN'][1]['fpr'] +session_meta['54MRL'][1]['fpr'] + session_meta['63MR'][1]['fpr'] + session_meta['64ML'][1]['fpr']) / 4
# # cross_mice_stats['post']['tpr'] = (session_meta['31MN'][1]['tpr'] +session_meta['54MRL'][1]['tpr'] + session_meta['63MR'][1]['tpr'] + session_meta['64ML'][1]['tpr']) / 4
# # cross_mice_stats['post']['fnr'] = ( session_meta['31MN'][1]['fnr'] +session_meta['54MRL'][1]['fnr'] + session_meta['63MR'][1]['fnr'] + session_meta['64ML'][1]['fnr']) / 4
# # cross_mice_stats['post']['tnr'] = (session_meta['31MN'][1]['tnr'] +session_meta['54MRL'][1]['tnr'] + session_meta['63MR'][1]['tnr'] + session_meta['64ML'][1]['tnr']) / 4
# #
# # conf_count = np.array([[cross_mice_stats['pre']['tp'], cross_mice_stats['post']['tp'], cross_mice_stats['pre']['fp'], cross_mice_stats['post']['fp']],
# #                      [cross_mice_stats['pre']['fn'], cross_mice_stats['post']['fn'], cross_mice_stats['pre']['tn'], cross_mice_stats['post']['tn']]])
# # conf_mat = np.array([[cross_mice_stats['pre']['tpr'], cross_mice_stats['post']['tpr'], cross_mice_stats['pre']['fpr'], cross_mice_stats['post']['fpr']],
# #                      [cross_mice_stats['pre']['fnr'], cross_mice_stats['post']['fnr'], cross_mice_stats['pre']['tnr'], cross_mice_stats['post']['tnr']]])
# # # conf_count = np.array([[session_meta[mice_id[0]][0]['tp'], session_meta[mice_id[0]][1]['tp'], session_meta[mice_id[0]][0]['fp'], session_meta[mice_id[0]][1]['fp']],
# # #                      [session_meta[mice_id[0]][0]['fn'], session_meta[mice_id[0]][1]['fn'], session_meta[mice_id[0]][0]['tn'], session_meta[mice_id[0]][1]['tn'],]])
# # # conf_mat = np.array([[session_meta[mice_id[0]][0]['tpr'], session_meta[mice_id[0]][1]['tpr'], session_meta[mice_id[0]][0]['fpr'], session_meta[mice_id[0]][1]['fpr']],
# # #                      [session_meta[mice_id[0]][0]['fnr'], session_meta[mice_id[0]][1]['fnr'], session_meta[mice_id[0]][0]['tnr'], session_meta[mice_id[0]][1]['tnr'],]])
# #
# # group_names = ['True Pos', 'False Pos', 'False Neg', 'True Neg']
# # group_counts = ["{0:0.0f}".format(value) for value in conf_count.flatten()]
# # group_percentages = ["{0:.2%}".format(value) for value in conf_mat.flatten()]
# # # labels = [f"{v2}\n{v3}" for v2, v3 in
# # #           zip(group_counts, group_percentages)]
# # labels = group_percentages
# # labels = np.asarray(labels).reshape(2, 4)
# # sns.heatmap(conf_mat, annot=labels, fmt='', cmap='Blues', ax=ax_conf)
# # # ax.set_title('Confusion Matrix\n\n')
# # ax_conf.set_xlabel('Actual Licks', fontsize=font_size)
# # ax_conf.set_ylabel('Prediction Licks', fontsize=font_size)
# # ## Ticket labels - List must be in alphabetical order
# # ax_conf.xaxis.set_ticks([1, 3])
# # ax_conf.xaxis.set_ticklabels(['True', 'False'], fontsize=font_size)
# # ax_conf.yaxis.set_ticklabels(['True', 'False'], fontsize=font_size)
# # ax_conf.annotate(group_names[0], xy=(0.5, 0.2), c='y', fontsize=font_size)
# # ax_conf.annotate(group_names[1], xy=(2.5, 0.2), c='y', fontsize=font_size)
# # ax_conf.annotate(group_names[2], xy=(0.5, 1.2), c='y', fontsize=font_size)
# # ax_conf.annotate(group_names[3], xy=(2.5, 1.2), c='y', fontsize=font_size)
# # ax_conf.arrow(0.9, 0.7, 0.2, 0.0, head_width=0.06)
# # ax_conf.arrow(0.9, 1.7, 0.2, 0.0, head_width=0.06)
# # ax_conf.arrow(2.9, 0.7, 0.2, 0.0, head_width=0.06)
# # ax_conf.arrow(2.9, 1.7, 0.2, 0.0, head_width=0.06)
#
# plt.show()
# # plt.savefig(f'{base_path}/Figs_for_paper/CRC_performance_21corr_31_54_63_64.svg',format='svg',dpi=200)
#
#
# ###################################### statistics #####################################
# # tpr_t0 = [session_meta['21ML'][0]['tpr'],session_meta['31MN'][0]['tpr'],session_meta['54MRL'][0]['tpr'], session_meta['63MR'][0]['tpr'], session_meta['64ML'][0]['tpr']]
# # tpr_t1 = [session_meta['21ML'][1]['tpr'],session_meta['31MN'][1]['tpr'],session_meta['54MRL'][1]['tpr'], session_meta['63MR'][1]['tpr'], session_meta['64ML'][1]['tpr']]
# tpr_t0 = [session_meta['31MN'][0]['tpr'],session_meta['54MRL'][0]['tpr'],
#           session_meta['187FN'][0]['tpr'], session_meta['203MN'][0]['tpr'],
#           session_meta['204FR'][0]['tpr'], session_meta['206FRL'][0]['tpr'],
#           session_meta['211MRR'][0]['tpr']#, session_meta['218MN'][0]['tpr']
#           ]
# tpr_t1 = [session_meta['31MN'][1]['tpr'],session_meta['54MRL'][1]['tpr'],
#           session_meta['187FN'][1]['tpr'], session_meta['203MN'][1]['tpr'],
#           session_meta['204FR'][1]['tpr'], session_meta['206FRL'][1]['tpr'],
#           session_meta['211MRR'][1]['tpr']#, session_meta['218MN'][1]['tpr']
#           ]
# tpr_t0_avg = np.mean(np.array(tpr_t0))
# tpr_t1_avg = np.mean(np.array(tpr_t1))
# tpr_tstats, tpr_pval = ttest_rel(tpr_t0, tpr_t1, alternative='less')
#
# # fdr_t0 = [session_meta['21ML'][0]['fdr'],session_meta['31MN'][0]['fdr'],session_meta['54MRL'][0]['fdr'], session_meta['63MR'][0]['fdr'], session_meta['64ML'][0]['fdr']]
# # fdr_t1 = [session_meta['21ML'][1]['fdr'],session_meta['31MN'][1]['fdr'],session_meta['54MRL'][1]['fdr'], session_meta['63MR'][1]['fdr'], session_meta['64ML'][1]['fdr']]
# fdr_t0 = [session_meta['31MN'][0]['fdr'],session_meta['54MRL'][0]['fdr'],
#           session_meta['187FN'][0]['fdr'], session_meta['203MN'][0]['fdr'],
#           session_meta['204FR'][0]['fdr'], session_meta['206FRL'][0]['fdr'],
#           session_meta['211MRR'][0]['fdr']#, session_meta['218MN'][0]['fdr']
#           ]
# #fdr_t1 = [session_meta['21ML'][1]['fdr'],session_meta['31MN'][1]['fdr'],session_meta['54MRL'][1]['fdr'], session_meta['63MR'][1]['fdr'], session_meta['64ML'][1]['fdr']]
# fdr_t1 = [session_meta['31MN'][1]['fdr'],session_meta['54MRL'][1]['fdr'],
#           session_meta['187FN'][1]['fdr'], session_meta['203MN'][1]['fdr'],
#           session_meta['204FR'][1]['fdr'], session_meta['206FRL'][1]['fdr'],
#           session_meta['211MRR'][1]['fdr']#, session_meta['218MN'][1]['fdr']
#           ]
#
# fdr_t0_avg = np.mean(np.array(fdr_t0))
# fdr_t1_avg = np.mean(np.array(fdr_t1))
# fdr_tstats, fdr_pval = ttest_rel(fdr_t0, fdr_t1,alternative='greater')
#
#
# lr_t0 = [session_meta['31MN'][0]['lr'],session_meta['54MRL'][0]['lr'],
#           session_meta['187FN'][0]['lr'], session_meta['203MN'][0]['lr'],
#           session_meta['204FR'][0]['lr'], session_meta['206FRL'][0]['lr'],
#           session_meta['211MRR'][0]['lr']#, session_meta['218MN'][0]['lr']
#          ]
# #fdr_t1 = [session_meta['21ML'][1]['fdr'],session_meta['31MN'][1]['fdr'],session_meta['54MRL'][1]['fdr'], session_meta['63MR'][1]['fdr'], session_meta['64ML'][1]['fdr']]
# lr_t1 = [session_meta['31MN'][1]['lr'],session_meta['54MRL'][1]['lr'],
#           session_meta['187FN'][1]['lr'], session_meta['203MN'][1]['lr'],
#           session_meta['204FR'][1]['lr'], session_meta['206FRL'][1]['lr'],
#           session_meta['211MRR'][1]['lr']#, session_meta['218MN'][1]['lr']
#          ]
#
# lr_t0_avg = np.mean(np.array(lr_t0))
# lr_t1_avg = np.mean(np.array(lr_t1))
# lr_tstats, lr_pval = ttest_rel(lr_t0, lr_t1,alternative='less')
#
#
# # print results
# print(f'sensitivity: pre {tpr_t0_avg}  post {tpr_t1_avg}    one-sided paired ttest p-value: {tpr_pval}')
# print(f'false discovery rate: pre {fdr_t0_avg}  post {fdr_t1_avg}    one-sided paired ttest p-value: {fdr_pval}')
# print(f'licking rate: pre {lr_t0_avg}  post {lr_t1_avg}    one-sided paired ttest p-value: {lr_pval}')
