import matplotlib.pyplot as plt
import numpy as np

from wideflow.analysis.utils.extract_from_metadata_file import extract_from_metadata_file
from wideflow.analysis.plots import plot_session
from wideflow.analysis.plots import plot_reward_response

base_path = '/data/Lena/WideFlow_prj'
date = '20250619'
mouse_id = '232FN'
session_id = f'{date}_{mouse_id}_NF4'
#session_id = f'20230618_54MRL_NF21_mocknF_ROI1'
#base_path = '/data/Lena/WideFlow_prj/MNL/20230123_MNL_NF21'
#base_path= '/data/Lena/WideFlow_prj/20230608/20230608_21ML_mockNF_NOTexclude_closest'
[timestamp, cue, metric_result, threshold, serial_readout,trial_number] = extract_from_metadata_file(f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt')
#[timestamp, cue, metric_result, threshold, serial_readout] = extract_from_metadata_file(f'{base_path}/metadata.txt')
#/data/Lena/WideFlow_prj/MR/20230117_MR_NF17/metadata.txt
serial_readout_correct = [1-x for x in serial_readout]
#SR0 = np.zeros(60000)

fig, ax = plt.subplots()
#
plot_session(ax, metric_result, cue, serial_readout_correct,threshold, dt=0.038/60, fig=fig)
#plot_session(ax, metric_result[:50000], cue[:50000],SR0[:50000] , threshold[:50000], dt=0.038/60, fig=fig)
#ax.legend(['metric', 'rewards timing', 'licking timing'])

#fig.suptitle('20230608_21ML_mockNF_NOT_exclude_closest')
#fig.suptitle(f'{session_id}')


#plt.figure(figsize=(10, 3))
#plt.plot(trial_number, drawstyle='steps-post')  # To emphasize step changes
# plt.plot(np.array(cue)*28, color = 'black')

########### Under here - lines for plotting with trials
# plot_reward_response(ax, cue, serial_readout_correct)
# #plt.plot(threshold, color = 'blue')
# #plt.plot(threshold2[:15000], color='red', alpha=0.5)
# #plt.plot(metric_result, color='green', alpha=0.4)
# #plt.plot(metric_result2[:15000], color='purple', alpha=0.4)
# #plt.plot(np.array(cue)*1.5, color = 'black')
# #plt.plot(serial_readout_correct)
# #plt.plot(np.array(cue2[:15000]),color='red', alpha=0.7)
# #ax.plot(trial_number, color='grey', alpha=0.3)
# # Loop through vector to find changes
# for i in range(1, len(timestamp)):
#     if trial_number[i] != trial_number[i - 1]:
#         # Draw dashed vertical line
#         ax.axvline(x=i, ymin = 0.1, ymax=0.9, linestyle='--', color='green')
#         # Add text label of new value
#         ax.text(i, 0.92, str(int(trial_number[i])), ha='center', va='bottom')


plt.legend()
#ax.legend(['metric', 'rewards timing', 'licking timing'])
plt.title(f'{mouse_id}_{session_id}')

#plt.show()


#plt.savefig(f'/data/Lena/WideFlow_prj/Figures_Rotem/{mouse_id}_{session_id}')
#plt.savefig(f'{base_path}/{mouse_id}/simple_figs/{session_id}')
plt.savefig(f'{base_path}/Figures_exp3/{session_id}_trials.png', format = 'png')
#plt.show()


#plot_reward_response(ax, cue, serial_readout_correct, ymin=0, ymax=1, t=np.array(timestamp), c_reward='k', c_response='b', fig=fig)
#plt.plot(2+np.array(cue))
#plt.plot(np.array(threshold))
#plt.plot(np.array(metric_result))

