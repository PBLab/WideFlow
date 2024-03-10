import matplotlib.pyplot as plt
import numpy as np

from wideflow.analysis.utils.extract_from_metadata_file import extract_from_metadata_file
from wideflow.analysis.plots import plot_session

base_path = '/data/Lena/WideFlow_prj'
date = '20230608'
mouse_id = '21ML'
session_id = f'{date}_{mouse_id}_CRC4'
#session_id = f'20230618_54MRL_NF21_mocknF_ROI1'
#base_path = '/data/Lena/WideFlow_prj/MNL/20230123_MNL_NF21'
#base_path= '/data/Lena/WideFlow_prj/20230608/20230608_21ML_mockNF_NOTexclude_closest'
[timestamp, cue, metric_result, threshold, serial_readout] = extract_from_metadata_file(f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt')
#[timestamp, cue, metric_result, threshold, serial_readout] = extract_from_metadata_file(f'{base_path}/metadata.txt')
#/data/Lena/WideFlow_prj/MR/20230117_MR_NF17/metadata.txt
serial_readout_correct = [1-x for x in serial_readout]
#SR0 = np.zeros(60000)

fig, ax = plt.subplots()

plot_session(ax, metric_result, cue, serial_readout_correct,threshold, dt=0.038/60, fig=fig)
#plot_session(ax, metric_result[:50000], cue[:50000],SR0[:50000] , threshold[:50000], dt=0.038/60, fig=fig)
#ax.legend(['metric', 'rewards timing', 'licking timing'])

#fig.suptitle('20230608_21ML_mockNF_NOT_exclude_closest')
fig.suptitle(f'{session_id}')


#plt.savefig(f'/data/Lena/WideFlow_prj/Figures_Rotem/{mouse_id}_{session_id}')
#plt.savefig(f'{base_path}/{mouse_id}/simple_figs/{session_id}')
plt.show()


#plot_reward_response(ax, cue, serial_readout_correct, ymin=0, ymax=1, t=np.array(timestamp), c_reward='k', c_response='b', fig=fig)
#plt.plot(2+np.array(cue))
#plt.plot(np.array(threshold))
#plt.plot(np.array(metric_result))

