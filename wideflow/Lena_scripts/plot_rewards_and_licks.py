import matplotlib.pyplot as plt
import numpy as np

from wideflow.analysis.utils.extract_from_metadata_file import extract_from_metadata_file


mouse_id = '281MRL'
date = '20251203'
session_id = 'CRC_p1'
[timestamp, cue, metric_result, threshold, serial_readout, trial_number] = extract_from_metadata_file(f'/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/{date}/'
                                                                                        f'{mouse_id}/{date}_{mouse_id}_{session_id}/metadata.txt')

#[timestamp, cue, metric_result, threshold, serial_readout] = extract_from_metadata_file(f'/data/Rotem/WideFlow prj/{mouse_id}/20220217_CueRewardCoupling/metadata.txt')
serial_readout_correct = [1-x for x in serial_readout]

from wideflow.analysis.plots import plot_reward_response
fig, ax = plt.subplots()


plot_reward_response(ax, cue, serial_readout_correct, ymin=0, ymax=0.1, t=np.array(timestamp), c_reward='k', c_response='b', fig=fig)

# plt.show()
fig.suptitle(f'{date}_{mouse_id}_{session_id}')
fig.savefig(f'/data/Lena/WideFlow_prj/exp4_simple_figs/{date}_{mouse_id}_{session_id}')