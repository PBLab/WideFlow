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

mice_id = [ '21ML','31MN','54MRL','63MR','64ML']
#dates = ['20230605','20230608','20230605','20230608','20230605','20230608']
sessions_names = {mice_id[0]: ['20230606_21ML_CRC2', '20230608_21ML_CRC4'],
                  mice_id[1]: ['20230606_31MN_CRC2', '20230608_31MN_CRC4'],
                  mice_id[2]: ['20230605_54MRL_CRC1', '20230608_54MRL_CRC4'],
                  mice_id[3]: ['20230605_63MR_CRC1', '20230607_63MR_CRC3'],
                  mice_id[4]: ['20230605_64ML_CRC1', '20230608_64ML_CRC4']}
# sessions_names = ['20230605_54MRL_CRC1', '20230608_54MRL_CRC4','20230605_63MR_CRC1', '20230607_63MR_CRC3',
#                   '20230605_64ML_CRC1', '20230608_64ML_CRC4']

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



########Addind results for 21ML CRC4 Experiment 2 7.1.2024####################

#Those values were calculated manually after counting the licks in the behavioral recording:
tp=3598
fn=6396
fp=3505
tn=46500

n_samples = 25
lick_frames = 25 * frames_win

true_positive_rate = tp / (tp + fn)
true_negative_rate = tn / (tn + fp)
false_positive_rate = fp / (tn + fp)
false_negative_rate = fn / (tp + fn)
false_discovery_rate = fp / (fp + tp)

precision = tp / (tp + fp)
recall = tp / (tp + fn)
recall_inv = tn / (fp + tn)
f1_score = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

session_meta['21ML'][1]["tpr"] = true_positive_rate
session_meta['21ML'][1]["tp"] = tp
session_meta['21ML'][1]["tnr"] = true_negative_rate
session_meta['21ML'][1]["tn"] = tn
session_meta['21ML'][1]["fpr"] = false_positive_rate
session_meta['21ML'][1]["fp"] = fp
session_meta['21ML'][1]["fnr"] = false_negative_rate
session_meta['21ML'][1]["fn"] = fn
session_meta['21ML'][1]["fdr"] = false_discovery_rate

# session_meta[mouse_id][s]["tpi"] = tpi
# session_meta[mouse_id][s]["tni"] = tni
# session_meta[mouse_id][s]["fpi"] = fpi
# session_meta[mouse_id][s]["fni"] = fni

session_meta['21ML'][1]["precision"] = precision
session_meta['21ML'][1]["recall"] = recall
session_meta['21ML'][1]["recall_inv"] = recall_inv
session_meta['21ML'][1]["F1 score"] = f1_score
session_meta['21ML'][1]["random F1 score"] = lick_frames / n_samples


#######End of 21ML CRC4###############################


###################################### plot results ######################################
f = plt.figure(constrained_layout=True, figsize=(16, 8))
gs = f.add_gridspec(4, 3)

bar_width = 0.6
font_size = 13
c_response = ['royalblue', 'crimson']
x = [2*bar_width, 3.5*bar_width, 5*bar_width, 6.5*bar_width]

# tpr fpr bar plots ----------------------------------------------------------------------------
ax_bar_up = f.add_subplot(gs[:2, 2])
#y1 = [session_meta[mice_id[-1]][1]["tpr"], session_meta[mice_id[-1]][0]["tpr"]]
y1 = [(session_meta[mice_id[0]][1]["tpr"]+session_meta[mice_id[1]][1]["tpr"]+session_meta[mice_id[2]][1]["tpr"]+session_meta[mice_id[3]][1]["tpr"]+session_meta[mice_id[4]][1]["tpr"])/5,
     (session_meta[mice_id[0]][0]["tpr"]+session_meta[mice_id[1]][0]["tpr"]+session_meta[mice_id[2]][0]["tpr"]+session_meta[mice_id[3]][0]["tpr"]+session_meta[mice_id[4]][0]["tpr"])/5]
#std_y1 = [0.0372, 0.0778]
b1 = ax_bar_up.barh([x[1], x[3]], y1, height=bar_width, color=[c_response[0], c_response[0]], alpha=0.8)
#ax_bar_up.errorbar(x=(x[1],x[3]), y=y1, yerr=std_y1,fmt='o', color='black')
ax_bar_up.set_yticks([])
ax_bar_up.set_ylabel('post-training               pre-training', fontsize=font_size)
ax_bar_up.grid(axis='x')
ax_bar_up.set_xlim([0, 1])
ax_bar_up.tick_params(axis='x', colors=c_response[0])


ax_bar_up_t = ax_bar_up.twiny()
#y2 = [session_meta[mice_id[-1]][1]["fdr"], session_meta[mice_id[-1]][0]["fdr"]]
y2 = [(session_meta[mice_id[0]][1]["fdr"]+session_meta[mice_id[1]][1]["fdr"]+session_meta[mice_id[2]][1]["fdr"]+session_meta[mice_id[3]][1]["fdr"]+session_meta[mice_id[4]][1]["fdr"])/5,
     (session_meta[mice_id[0]][0]["fdr"]+session_meta[mice_id[1]][0]["fdr"]+session_meta[mice_id[2]][0]["fdr"]+session_meta[mice_id[3]][0]["fdr"]+session_meta[mice_id[4]][0]["fdr"])/5]
b2 = ax_bar_up_t.barh([x[0], x[2]], y2, height=bar_width, color=[c_response[1], c_response[1]], alpha=0.8)

ax_bar_up_t.tick_params(axis='x', colors=c_response[1])
ax_bar_up_t.set_xlim([0, 1])

ax_bar_up.legend([b1[0], b2[0]], ['sensitivity', 'false discovery rate'], loc='upper right')

# session stim lick timing ------------------------------------------------------------
ax0 = f.add_subplot(gs[0, :2])
plot_reward_response(ax0, session_meta[mice_id[-1]][0]['cue'],
            session_meta[mice_id[-1]][0]['tpi'],
            c_response=c_response[0])
plot_reward_response(ax0, session_meta[mice_id[-1]][0]['cue'],
            session_meta[mice_id[-1]][0]['fpi'],
            c_response=c_response[1])

ax0.set_ylabel('pre-training', fontsize=font_size)
ax0.set_yticks([])
ax0.set_xticks([])

# ax0.set_title("Mouse #1", loc='left')

ax1 = f.add_subplot(gs[1, :2])
plot_reward_response(ax1, session_meta[mice_id[-1]][1]['cue'],
            session_meta[mice_id[-1]][1]['tpi'],
            c_response=c_response[0])
plot_reward_response(ax1, session_meta[mice_id[-1]][1]['cue'],
            session_meta[mice_id[-1]][1]['fpi'],
            c_response=c_response[1])
ax1.set_ylabel('post-training', fontsize=font_size)
ax1.set_yticks([])
ax1.set_xticks([])

# ax0.axis('off')
# ax1.axis('off')
ax0.spines['left'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

# cross mice statistics
sdf_mean_pre = np.mean(
    np.vstack((session_meta[mice_id[1]][0]['sdf'], session_meta[mice_id[2]][0]['sdf'],session_meta[mice_id[3]][0]['sdf'],session_meta[mice_id[4]][0]['sdf']))
    , axis=0
)
sdf_mean_post = np.mean(
    np.vstack((session_meta[mice_id[1]][1]['sdf'], session_meta[mice_id[2]][1]['sdf'], session_meta[mice_id[3]][1]['sdf'], session_meta[mice_id[4]][1]['sdf']))
    , axis=0
)

sdfx_mean_pre = np.mean(
    np.vstack(( session_meta[mice_id[1]][0]['sdfx'], session_meta[mice_id[2]][0]['autocorr'], session_meta[mice_id[3]][0]['autocorr'], session_meta[mice_id[4]][0]['autocorr']))
    , axis=0
)
sdfx_mean_post = np.mean(
    np.vstack(( session_meta[mice_id[1]][1]['sdfx'], session_meta[mice_id[2]][1]['autocorr'], session_meta[mice_id[3]][1]['autocorr'], session_meta[mice_id[4]][1]['autocorr']))
    , axis=0
)
# sdf plots_______________________________________________________________
ax_sdf = f.add_subplot(gs[2, :2])

t1 = np.arange(-sdf_win[0], sdf_win[1], 1) * session_meta[mice_id[-1]][0]['dt'] * 1000
ax_sdf.plot(t1, sdf_mean_pre, color='green')
t2 = np.arange(-sdf_win[0], sdf_win[1], 1) * session_meta[mice_id[-1]][1]['dt'] * 1000
ax_sdf.plot(t2, sdf_mean_post, color='blue')
ax_sdf.vlines(0, 0, 1, linestyle='--', color='k')
ax_sdf.set_ylabel('SDF [a.u]', fontsize=font_size)
ax_sdf.legend(['before training', 'after training'])
ax_sdf.spines['top'].set_visible(False)
ax_sdf.spines['right'].set_visible(False)

ax_sdfx = f.add_subplot(gs[3, :2])

t1 = np.arange(-sdfx_win[0], sdfx_win[1], 1) * session_meta[mice_id[-1]][0]['dt'] * 1000
ax_sdfx.plot(t1, sdfx_mean_pre, color='green')
t2 = np.arange(-sdfx_win[0], sdfx_win[1], 1) * session_meta[mice_id[-1]][1]['dt'] * 1000
ax_sdfx.plot(t2, sdfx_mean_post, color='blue')
ax_sdfx.vlines(0, 0, 700, linestyle='--', color='k')
ax_sdfx.set_xlabel('Time[ms]', fontsize=font_size)
ax_sdfx.set_ylabel('Auto Correlation [a.u]', fontsize=font_size)
ax_sdfx.legend(['before training', 'after training'])
ax_sdfx.spines['top'].set_visible(False)
ax_sdfx.spines['right'].set_visible(False)

# confusion matrix plot__________________________________________________
ax_conf = f.add_subplot(gs[2:, 2:])
# cross_mice_stats = {}
# cross_mice_stats['pre'], cross_mice_stats['post'] = {}, {}
# cross_mice_stats['pre']['fp'] = (session_meta['31MN'][0]['fp'] +session_meta['54MRL'][0]['fp'] + session_meta['63MR'][0]['fp'] + session_meta['64ML'][0]['fp']) / 4
# cross_mice_stats['pre']['tp'] = ( session_meta['31MN'][0]['tp'] +session_meta['54MRL'][0]['tp'] + session_meta['63MR'][0]['tp'] + session_meta['64ML'][0]['tp']) / 4
# cross_mice_stats['pre']['fn'] = (session_meta['31MN'][0]['fn'] +session_meta['54MRL'][0]['fn'] + session_meta['63MR'][0]['fn'] + session_meta['64ML'][0]['fn']) / 4
# cross_mice_stats['pre']['tn'] = (session_meta['31MN'][0]['tn'] +session_meta['54MRL'][0]['tn'] + session_meta['63MR'][0]['tn'] + session_meta['64ML'][0]['tn']) / 4
#
# cross_mice_stats['post']['fp'] = ( session_meta['31MN'][1]['fp'] +session_meta['54MRL'][1]['fp'] + session_meta['63MR'][1]['fp'] + session_meta['64ML'][1]['fp']) / 4
# cross_mice_stats['post']['tp'] = ( session_meta['31MN'][1]['tp'] +session_meta['54MRL'][1]['tp'] + session_meta['63MR'][1]['tp'] + session_meta['64ML'][1]['tp']) / 4
# cross_mice_stats['post']['fn'] = (session_meta['31MN'][1]['fn'] +session_meta['54MRL'][1]['fn'] + session_meta['63MR'][1]['fn'] + session_meta['64ML'][1]['fn']) / 4
# cross_mice_stats['post']['tn'] = (session_meta['31MN'][1]['tn'] +session_meta['54MRL'][1]['tn'] + session_meta['63MR'][1]['tn'] + session_meta['64ML'][1]['tn']) / 4
#
# cross_mice_stats['pre']['fpr'] = (session_meta['31MN'][0]['fpr'] +session_meta['54MRL'][0]['fpr'] + session_meta['63MR'][0]['fpr'] + session_meta['64ML'][0]['fpr']) / 4
# cross_mice_stats['pre']['tpr'] = ( session_meta['31MN'][0]['tpr'] +session_meta['54MRL'][0]['tpr'] + session_meta['63MR'][0]['tpr'] + session_meta['64ML'][0]['tpr']) / 4
# cross_mice_stats['pre']['fnr'] = (session_meta['31MN'][0]['fnr'] +session_meta['54MRL'][0]['fnr'] + session_meta['63MR'][0]['fnr'] + session_meta['64ML'][0]['fnr']) / 4
# cross_mice_stats['pre']['tnr'] = ( session_meta['31MN'][0]['tnr'] +session_meta['54MRL'][0]['tnr'] + session_meta['63MR'][0]['tnr'] + session_meta['64ML'][0]['tnr']) / 4
#
# cross_mice_stats['post']['fpr'] = (session_meta['31MN'][1]['fpr'] +session_meta['54MRL'][1]['fpr'] + session_meta['63MR'][1]['fpr'] + session_meta['64ML'][1]['fpr']) / 4
# cross_mice_stats['post']['tpr'] = (session_meta['31MN'][1]['tpr'] +session_meta['54MRL'][1]['tpr'] + session_meta['63MR'][1]['tpr'] + session_meta['64ML'][1]['tpr']) / 4
# cross_mice_stats['post']['fnr'] = ( session_meta['31MN'][1]['fnr'] +session_meta['54MRL'][1]['fnr'] + session_meta['63MR'][1]['fnr'] + session_meta['64ML'][1]['fnr']) / 4
# cross_mice_stats['post']['tnr'] = (session_meta['31MN'][1]['tnr'] +session_meta['54MRL'][1]['tnr'] + session_meta['63MR'][1]['tnr'] + session_meta['64ML'][1]['tnr']) / 4
#
# conf_count = np.array([[cross_mice_stats['pre']['tp'], cross_mice_stats['post']['tp'], cross_mice_stats['pre']['fp'], cross_mice_stats['post']['fp']],
#                      [cross_mice_stats['pre']['fn'], cross_mice_stats['post']['fn'], cross_mice_stats['pre']['tn'], cross_mice_stats['post']['tn']]])
# conf_mat = np.array([[cross_mice_stats['pre']['tpr'], cross_mice_stats['post']['tpr'], cross_mice_stats['pre']['fpr'], cross_mice_stats['post']['fpr']],
#                      [cross_mice_stats['pre']['fnr'], cross_mice_stats['post']['fnr'], cross_mice_stats['pre']['tnr'], cross_mice_stats['post']['tnr']]])
# # conf_count = np.array([[session_meta[mice_id[0]][0]['tp'], session_meta[mice_id[0]][1]['tp'], session_meta[mice_id[0]][0]['fp'], session_meta[mice_id[0]][1]['fp']],
# #                      [session_meta[mice_id[0]][0]['fn'], session_meta[mice_id[0]][1]['fn'], session_meta[mice_id[0]][0]['tn'], session_meta[mice_id[0]][1]['tn'],]])
# # conf_mat = np.array([[session_meta[mice_id[0]][0]['tpr'], session_meta[mice_id[0]][1]['tpr'], session_meta[mice_id[0]][0]['fpr'], session_meta[mice_id[0]][1]['fpr']],
# #                      [session_meta[mice_id[0]][0]['fnr'], session_meta[mice_id[0]][1]['fnr'], session_meta[mice_id[0]][0]['tnr'], session_meta[mice_id[0]][1]['tnr'],]])
#
# group_names = ['True Pos', 'False Pos', 'False Neg', 'True Neg']
# group_counts = ["{0:0.0f}".format(value) for value in conf_count.flatten()]
# group_percentages = ["{0:.2%}".format(value) for value in conf_mat.flatten()]
# # labels = [f"{v2}\n{v3}" for v2, v3 in
# #           zip(group_counts, group_percentages)]
# labels = group_percentages
# labels = np.asarray(labels).reshape(2, 4)
# sns.heatmap(conf_mat, annot=labels, fmt='', cmap='Blues', ax=ax_conf)
# # ax.set_title('Confusion Matrix\n\n')
# ax_conf.set_xlabel('Actual Licks', fontsize=font_size)
# ax_conf.set_ylabel('Prediction Licks', fontsize=font_size)
# ## Ticket labels - List must be in alphabetical order
# ax_conf.xaxis.set_ticks([1, 3])
# ax_conf.xaxis.set_ticklabels(['True', 'False'], fontsize=font_size)
# ax_conf.yaxis.set_ticklabels(['True', 'False'], fontsize=font_size)
# ax_conf.annotate(group_names[0], xy=(0.5, 0.2), c='y', fontsize=font_size)
# ax_conf.annotate(group_names[1], xy=(2.5, 0.2), c='y', fontsize=font_size)
# ax_conf.annotate(group_names[2], xy=(0.5, 1.2), c='y', fontsize=font_size)
# ax_conf.annotate(group_names[3], xy=(2.5, 1.2), c='y', fontsize=font_size)
# ax_conf.arrow(0.9, 0.7, 0.2, 0.0, head_width=0.06)
# ax_conf.arrow(0.9, 1.7, 0.2, 0.0, head_width=0.06)
# ax_conf.arrow(2.9, 0.7, 0.2, 0.0, head_width=0.06)
# ax_conf.arrow(2.9, 1.7, 0.2, 0.0, head_width=0.06)

#plt.show()
plt.savefig(f'{base_path}/Figs_for_paper/CRC_performance_21corr_31_54_63_64.svg',format='svg',dpi=200)


###################################### statistics #####################################
tpr_t0 = [session_meta['21ML'][0]['tpr'],session_meta['31MN'][0]['tpr'],session_meta['54MRL'][0]['tpr'], session_meta['63MR'][0]['tpr'], session_meta['64ML'][0]['tpr']]
tpr_t1 = [session_meta['21ML'][1]['tpr'],session_meta['31MN'][1]['tpr'],session_meta['54MRL'][1]['tpr'], session_meta['63MR'][1]['tpr'], session_meta['64ML'][1]['tpr']]
tpr_t0_avg = np.mean(np.array(tpr_t0))
tpr_t1_avg = np.mean(np.array(tpr_t1))
tpr_tstats, tpr_pval = ttest_rel(tpr_t0, tpr_t1, alternative='less')

fdr_t0 = [session_meta['21ML'][0]['fdr'],session_meta['31MN'][0]['fdr'],session_meta['54MRL'][0]['fdr'], session_meta['63MR'][0]['fdr'], session_meta['64ML'][0]['fdr']]
fdr_t1 = [session_meta['21ML'][1]['fdr'],session_meta['31MN'][1]['fdr'],session_meta['54MRL'][1]['fdr'], session_meta['63MR'][1]['fdr'], session_meta['64ML'][1]['fdr']]
fdr_t0_avg = np.mean(np.array(fdr_t0))
fdr_t1_avg = np.mean(np.array(fdr_t1))
fdr_tstats, fdr_pval = ttest_rel(fdr_t0, fdr_t1,alternative='greater')

# print results
print(f'sensitivity: pre {tpr_t0_avg}  post {tpr_t1_avg}    one-sided paired ttest p-value: {tpr_pval}')
print(f'false discovery rate: pre {fdr_t0_avg}  post {fdr_t1_avg}    one-sided paired ttest p-value: {fdr_pval}')
