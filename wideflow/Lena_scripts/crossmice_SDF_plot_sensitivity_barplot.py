import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
from analysis.utils.peristimulus_time_response import calc_sdf
from analysis.plots import *
from scipy.ndimage.filters import maximum_filter1d
from scipy.stats import ttest_rel

#  statistics hyperparameters
sdf_win = [100, 500]
sdfx_win = [200, 200]
frames_win = 400
beta = 1

base_path = '/data/Lena/WideFlow_prj'

mice_id = ['MNL', #'MR'
           ]
sessions_names = {mice_id[0]: ['20221121_MNL_CRC2', '20221122_MNL_CRC3'],
                  #mice_id[1]: ['20221120_MR_CRC1', '20221121_MR_CRC2'],
                  #mice_id[2]: ['20220215_CueRewardCoupling', '20220217_CueRewardCoupling']
                  }

session_meta = {}
for mouse_id in mice_id:
    session_meta[mouse_id] = []
    for s, sess_name in enumerate(sessions_names[mouse_id]):
        [timestamp, cue, metric_result, threshold, serial_readout] = extract_from_metadata_file(f'{base_path}/{mouse_id}/{sess_name}/metadata.txt')
        dt = np.mean(np.diff(timestamp))
        cue = np.array(cue)
        serial_readout = 1 - np.array(serial_readout)
        serial_readout = maximum_filter1d(serial_readout, 5)
        sdf = np.mean(calc_sdf(cue, serial_readout, sdf_win, 2), axis=0)

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
                                       "sdf": sdf, "sdfx": sdfx, "autocorr": autocorr})

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

        true_positive_rate = tp / (tp + fn)
        true_negative_rate = tn / (tn + fp)
        false_positive_rate = fp / (tn + fp)
        false_negative_rate = fn / (tp + fn)
        false_discovery_rate = fp / (fp + tp)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        recall_inv = tn / (fp + tn)
        f1_score = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

        session_meta[mouse_id][s]["tpr"] = true_positive_rate
        session_meta[mouse_id][s]["tp"] = tp
        session_meta[mouse_id][s]["tnr"] = true_negative_rate
        session_meta[mouse_id][s]["tn"] = tn
        session_meta[mouse_id][s]["fpr"] = false_positive_rate
        session_meta[mouse_id][s]["fp"] = fp
        session_meta[mouse_id][s]["fnr"] = false_negative_rate
        session_meta[mouse_id][s]["fn"] = fn
        session_meta[mouse_id][s]["fdr"] = false_discovery_rate

        session_meta[mouse_id][s]["tpi"] = tpi
        session_meta[mouse_id][s]["tni"] = tni
        session_meta[mouse_id][s]["fpi"] = fpi
        session_meta[mouse_id][s]["fni"] = fni

        session_meta[mouse_id][s]["precision"] = precision
        session_meta[mouse_id][s]["recall"] = recall
        session_meta[mouse_id][s]["recall_inv"] = recall_inv
        session_meta[mouse_id][s]["F1 score"] = f1_score
        session_meta[mouse_id][s]["random F1 score"] = lick_frames / n_samples

###################################### plot results ######################################

f = plt.figure(constrained_layout=True, figsize=(16, 8))
gs = f.add_gridspec(4, 3)

bar_width = 0.6
font_size = 13
c_response = ['royalblue', 'crimson']
x = [2*bar_width, 3.5*bar_width, 5*bar_width, 6.5*bar_width]

# tpr fpr bar plots ----------------------------------------------------------------------------
ax_bar_up = f.add_subplot(gs[:2, 2])
y1 = [session_meta[mice_id[0]][1]["tpr"], session_meta[mice_id[0]][0]["tpr"]]
b1 = ax_bar_up.barh([x[1], x[3]], y1, height=bar_width, color=[c_response[0], c_response[0]], alpha=0.8)
ax_bar_up.set_yticks([])
ax_bar_up.set_ylabel('post-training               pre-training', fontsize=font_size)
ax_bar_up.grid(axis='x')
ax_bar_up.set_xlim([0.1, 1])
ax_bar_up.tick_params(axis='x', colors=c_response[0])

ax_bar_up_t = ax_bar_up.twiny()
y2 = [session_meta[mice_id[0]][1]["fdr"], session_meta[mice_id[0]][0]["fdr"]]
b2 = ax_bar_up_t.barh([x[0], x[2]], y2, height=bar_width, color=[c_response[1], c_response[1]], alpha=0.8)

ax_bar_up_t.tick_params(axis='x', colors=c_response[1])

ax_bar_up.legend([b1[0], b2[0]], ['sensitivity', 'false discovery rate'], loc='upper right')
#f.savefig("/home/elenakreines/WideFlow/wideflow/Lena_scripts/MR_MNL_postCRC2")

# cross mice statistics
sdf_mean_pre = np.mean(
    np.vstack((session_meta[mice_id[0]][0]['sdf'], #session_meta[mice_id[1]][0]['sdf']
               ))
    , axis=0
)
sdf_mean_post = np.mean(
    np.vstack((session_meta[mice_id[0]][1]['sdf'], #session_meta[mice_id[1]][1]['sdf']
               ))
    , axis=0
)

sdfx_mean_pre = np.mean(
    np.vstack((session_meta[mice_id[0]][0]['sdfx'], #session_meta[mice_id[1]][0]['sdfx']
               ))
    , axis=0
)
sdfx_mean_post = np.mean(
    np.vstack((session_meta[mice_id[0]][1]['sdfx'], #session_meta[mice_id[1]][1]['sdfx']
               ))
    , axis=0
)

# sdf plots_______________________________________________________________
ax_sdf = f.add_subplot(gs[2, :2])

t1 = np.arange(-sdf_win[0], sdf_win[1], 1) * session_meta[mice_id[0]][0]['dt'] * 1000
ax_sdf.plot(t1, sdf_mean_pre, color='green')
t2 = np.arange(-sdf_win[0], sdf_win[1], 1) * session_meta[mice_id[0]][1]['dt'] * 1000
ax_sdf.plot(t2, sdf_mean_post, color='blue')
ax_sdf.vlines(0, 0, 1, linestyle='--', color='k')
ax_sdf.set_ylabel('SDF [a.u]', fontsize=font_size)
ax_sdf.legend(['before training', 'after training'])
ax_sdf.spines['top'].set_visible(False)
ax_sdf.spines['right'].set_visible(False)

# ax_sdfx = f.add_subplot(gs[3, :2])
#
# t1 = np.arange(-sdfx_win[0], sdfx_win[1], 1) * session_meta[mice_id[0]][0]['dt'] * 1000
# ax_sdfx.plot(t1, sdfx_mean_pre, color='green')
# t2 = np.arange(-sdfx_win[0], sdfx_win[1], 1) * session_meta[mice_id[0]][1]['dt'] * 1000
# ax_sdfx.plot(t2, sdfx_mean_post, color='blue')
# ax_sdfx.vlines(0, 0, 700, linestyle='--', color='k')
# ax_sdfx.set_xlabel('Time[ms]', fontsize=font_size)
# ax_sdfx.set_ylabel('Auto Correlation [a.u]', fontsize=font_size)
# ax_sdfx.legend(['before training', 'after training'])
# ax_sdfx.spines['top'].set_visible(False)
# ax_sdfx.spines['right'].set_visible(False)

f.savefig("/home/elenakreines/WideFlow/wideflow/Lena_scripts/MNL_CRC2vsCRC3")
