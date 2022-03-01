import matplotlib.pyplot as plt
import numpy as np

from analysis.utils.peristimulus_time_response import calc_pstr


def plot_reward_response(ax, rewards, responses, ymin=0, ymax=1):
    rewards_inds = np.array(rewards)
    rewards_inds = [i for i in range(len(rewards_inds)) if rewards_inds[i]]
    responses_inds = (1 - np.array(responses))
    responses_inds = [i for i in range(len(responses_inds)) if responses_inds[i]]

    ax.vlines(rewards_inds, ymin=ymin, ymax=ymax, color='k', linewidth=1)
    ax.vlines(responses_inds, ymin=0.5*ymin, ymax=0.5*ymax, color='b', linewidth=0.2)


def plot_session(ax, metric, rewards, responses, threshold, dt):

    t = np.arange(0, dt * len(metric), dt) / 60  # convert to minutes
    ax.plot(metric, color='g', linewidth=0.5, alpha=0.5)
    ax.plot(threshold, color='r', linewidth=0.2)
    plot_reward_response(ax, rewards, responses, ymin=np.min(metric), ymax=np.max(metric))

    ax.set_xticklabels(np.int32(np.linspace(0, t[-1], 10)))
    ax.set_ylabel('Metric')
    ax.set_xlabel('Time [minutes]')


def plot_pstr(ax, rois_pstr_dict, dt, bold_list=[]):
    nargs = len(rois_pstr_dict)
    hsv_tuples = [(x * 1.0 / nargs, 0.5, 0.5) for x in range(nargs)]

    delta_t2 = len(rois_pstr_dict[list(rois_pstr_dict.keys())[0]])
    delta_t = np.floor(delta_t2/2)
    dt = dt * 1000  # convert to milliseconds
    t = np.linspace(-delta_t * dt, (delta_t + 1) * dt, delta_t2)

    legend = []
    rois_pstr_mat = np.array(list(rois_pstr_dict.values()))
    pstrs_mean = np.mean(rois_pstr_mat, axis=0)
    pstrs_std = np.std(rois_pstr_mat, axis=0)
    for i, (key, pstr) in enumerate(rois_pstr_dict.items()):
        legend.append(key)
        if key in bold_list:
            # ax.plot(t, pstr, color=hsv_tuples[i], linewidth=2)
            ax.plot(t, pstr, color='red', linewidth=2)
        else:
            # ax.plot(t, pstr, color=hsv_tuples[i], linewidth=0.5)
            ax.plot(t, pstr, color='blue', linewidth=0.5)

    ax.plot(t, pstrs_mean, linestyle='dashed', color='black')
    ax.fill_between(t, pstrs_mean - pstrs_std, pstrs_mean + pstrs_std, color='gray', alpha=0.2)
    ax.axvline(x=0, ymin=0, ymax=np.max(rois_pstr_mat), color='k')

    ax.set_title('Peristimulus Time Response')
    ax.set_ylabel("pstr")
    ax.set_xlabel("Time [ms]")
    # ax.legend(legend, ncol=np.max((int(nargs/10), 1)))


def plot_box_plot(ax, *args, **kwargs):
    '''

    Args:
        ax: pyplot figure axis
        *args: numpy 2d arrays - columns corresponds to data for all x-values,
        rows contain data for specific x-value.
        Each arg will add box plot to all x-values  (number of columns must be the same for all args)
        **kwargs: pyplot key-value pairs - axis attributes

    Returns:

    '''
    nargs = len(args)
    hsv_tuples = [(x * 1.0 / nargs, 0.5, 0.5) for x in range(nargs)]
    nx = args[0].shape[1]  # nx should be the same for each arg
    x = np.arange(nx)
    width = (1 / nargs) / 2

    bp = []
    # plot boxes
    for i, data in enumerate(args):
        if data.shape[0] == 1:
            notch = False
        else:
            notch = False
        bp.append([None] * nx)
        for j in range(nx):
            bp[i][j] = ax.boxplot(data[:, j], widths=width, positions=[x[j] + i * width], notch=notch, patch_artist=True,
                        boxprops=dict(facecolor=hsv_tuples[i]), medianprops=dict(color="black"))

    # plot trend lines
    bp_med = []
    for i in range(nargs):
        bp_med.append([None] * nx)
        for j in range(nx):
            bp_med[i][j] = bp[i][j]['medians'][0].get_ydata()[0]

    bp_med = np.array(bp_med)
    for i in range(nargs):
        ax.plot(x + i * width, bp_med[i, :], color=hsv_tuples[i], linestyle='--')
        ax.set_xticks(x + width)

    # add axis attributes
    for key, val in kwargs.items():
        attr = getattr(ax, key)
        if isinstance(val, dict):
            attr(**val)
        else:
            attr(val)

