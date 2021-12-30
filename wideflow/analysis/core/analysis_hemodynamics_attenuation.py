import numpy as np
from wideflow.analysis.utils import calc_regression_map


def hemodynamics_attenuation(wf_data, regression_coeff_map, hemo_correct_ch, capacity, reg_smooth_fac):
    # create regression map if none is provided
    if regression_coeff_map is None:
        regression_coeff_map = calc_regression_map(wf_data['channel_0'], wf_data['channel_1'], reg_smooth_fac)

    # correct intrinsic channel with the regression map
    wf_data[hemo_correct_ch[1]][0] = np.multiply(regression_coeff_map[0], wf_data[hemo_correct_ch[1]][0]) + \
                                     regression_coeff_map[1]

    # subtract between the channels
    wf_data[hemo_correct_ch[0]][:capacity] = \
        wf_data[hemo_correct_ch[0]][:capacity] - wf_data[hemo_correct_ch[1]][:capacity]
    sub_mean = np.mean(wf_data[hemo_correct_ch[0]][:capacity], axis=0)
    wf_data[hemo_correct_ch[0]][:capacity] = wf_data[hemo_correct_ch[0]][:capacity] - sub_mean
    for i in range(capacity, len(wf_data[hemo_correct_ch[0]])):
        wf_data[hemo_correct_ch[1]][i] = np.multiply(regression_coeff_map[0], wf_data[hemo_correct_ch[1]][i]) + \
                                    regression_coeff_map[1]

        wf_data[hemo_correct_ch[0]][i] = wf_data[hemo_correct_ch[0]][i] - wf_data[hemo_correct_ch[1]][i]
        sub_mean = np.mean(wf_data[hemo_correct_ch[0]][i-capacity: i], axis=0)
        wf_data[hemo_correct_ch[0]][i] = wf_data[hemo_correct_ch[0]][i] - sub_mean

    return regression_coeff_map
