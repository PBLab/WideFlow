import matplotlib.pyplot as plt
import numpy as np

from wideflow.analysis.utils.extract_from_metadata_file import extract_from_metadata_file


mouse_id = '21ML'
[timestamp, cue, metric_result, threshold, serial_readout] = extract_from_metadata_file(f'/data/Lena/WideFlow_prj/20230608/'
                                                                                        f'{mouse_id}/20230608_{mouse_id}_CRC4/metadata.txt')

#[timestamp, cue, metric_result, threshold, serial_readout] = extract_from_metadata_file(f'/data/Rotem/WideFlow prj/{mouse_id}/20220217_CueRewardCoupling/metadata.txt')
serial_readout_correct = [1-x for x in serial_readout]

from wideflow.analysis.plots import plot_reward_response
fig, ax = plt.subplots()


plot_reward_response(ax, cue, serial_readout_correct, ymin=0, ymax=1, t=np.array(timestamp), c_reward='k', c_response='b', fig=fig)

plt.show()
#fig.suptitle(f'20230608_{mouse_id}_CRC4')
#fig.savefig(f'/data/Lena/WideFlow_prj/{mouse_id}/simple_figs/20230608_{mouse_id}_CRC4')