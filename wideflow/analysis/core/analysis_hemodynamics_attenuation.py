import numpy as np


def hemodynamics_attenuation(wf_data, regression_coeff_map, hemo_correct_ch, capacity):
    wf_data[hemo_correct_ch[1]][0] = np.multiply(regression_coeff_map[0], wf_data[hemo_correct_ch[1]][0]) + \
                                     regression_coeff_map[1]

    wf_data[hemo_correct_ch[0]][0] = wf_data[hemo_correct_ch[0]][0] - wf_data[hemo_correct_ch[1]][0]
    sub_mean = wf_data[hemo_correct_ch[0]][0]
    wf_data[hemo_correct_ch[0]][0] = wf_data[hemo_correct_ch[0]][0] - sub_mean
    for i in range(1, len(wf_data[hemo_correct_ch[0]])):
        wf_data[hemo_correct_ch[1]][i] = np.multiply(regression_coeff_map[0], wf_data[hemo_correct_ch[1]][i]) + \
                                    regression_coeff_map[1]


        wf_data[hemo_correct_ch[0]][i] = wf_data[hemo_correct_ch[0]][i] - wf_data[hemo_correct_ch[1]][i]
        sub_mean = np.mean(wf_data[hemo_correct_ch[0]][np.max([0, i-capacity]): i], axis=0)
        wf_data[hemo_correct_ch[0]][i] = wf_data[hemo_correct_ch[0]][i] - sub_mean

    return wf_data

