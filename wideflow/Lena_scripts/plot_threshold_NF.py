import matplotlib.pyplot as plt
import numpy as np

from wideflow.analysis.utils.extract_from_metadata_file import extract_from_metadata_file

from wideflow.config import DATA_STAGING_PATH  #added by Claude 20260906
# [timestamp, cue, metric_result, threshold, serial_readout] = extract_from_metadata_file('/data/Lena/WideFlow_prj/MR/20230117_MR_NF17/metadata.txt')
[timestamp, cue, metric_result, threshold, serial_readout] = extract_from_metadata_file(DATA_STAGING_PATH + '/MR/20230117_MR_NF17/metadata.txt')  #added by Claude 20260906
#a = np.sum(cue)
#b = np.mean(metric_result)
serial_readout_correct = [1-x for x in serial_readout]

plt.plot(np.array(threshold))
plt.grid()
# plt.savefig("/data/Lena/WideFlow_prj/MR/simple_figs/MR_NF7_threshold_20221219")
plt.savefig(DATA_STAGING_PATH + '/MR/simple_figs/MR_NF7_threshold_20221219')  #added by Claude 20260906
