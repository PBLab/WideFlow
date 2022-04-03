import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.ndimage import convolve
import copy

from analysis.utils.generate_color_list import generate_gradient_color_list


def plot_reward_response(ax, rewards, responses, ymin=0, ymax=1, t=None, c_reward='k', c_response='b'):
    if t is None:
        t = np.arange(len(rewards))
    rewards_inds = np.array(rewards)
    rewards_inds = [i for i in range(len(rewards_inds)) if rewards_inds[i]]
    responses_inds = np.array(responses)
    responses_inds = [i for i in range(len(responses_inds)) if responses_inds[i]]

    ax.vlines(t[np.ix_(rewards_inds)], ymin=ymin, ymax=ymax, color=c_reward, linewidth=1)
    ax.vlines(t[np.ix_(responses_inds)], ymin=0.5*ymin, ymax=0.5*ymax, color=c_response, linewidth=0.2)


def plot_session(ax, metric, rewards, responses, threshold, dt):

    t = np.arange(0, dt * len(metric), dt)  # convert to minutes
    ax.plot(t, metric, color='g', linewidth=0.5, alpha=0.5)
    ax.plot(t, threshold, color='r', linewidth=0.2)
    plot_reward_response(ax, rewards, responses, ymin=np.min(metric), ymax=np.max(metric), t=t)

    ax.set_xticks(np.int32(np.linspace(0, t[-1], 10)))
    ax.set_xticklabels(np.int32(np.linspace(0, t[-1], 10)))
    ax.set_ylabel('Z-score')
    ax.set_xlabel('Time [minutes]')


def plot_pstr(ax, rois_pstr_dict, dt, bold_list=[], proximity_dict={}):
    delta_t2 = len(rois_pstr_dict[list(rois_pstr_dict.keys())[0]])
    delta_t = np.floor(delta_t2/2)
    dt = dt * 1000  # convert to milliseconds
    t = np.linspace(-delta_t * dt, (delta_t + 1) * dt, delta_t2)

    plot_traces(ax, rois_pstr_dict, t, bold_list, proximity_dict)
    rois_pstr_mat = np.array(list(rois_pstr_dict.values()))
    ax.axvline(x=0, ymin=0, ymax=np.max(rois_pstr_mat), color='k')

    ax.set_title('Peristimulus Time Response')
    ax.set_ylabel("PSTR")
    ax.set_xlabel("Time [ms]")
    # ax.legend(legend, ncol=np.max((int(nargs/10), 1)))


def plot_traces(ax, rois_traces_dict, dt, bold_list=[], proximity_dict={}, **kwargs):
    nargs = len(rois_traces_dict)
    n_samples = len(rois_traces_dict[list(rois_traces_dict.keys())[0]])

    # color_list = generate_gradient_color_list(nargs, "blue", "red")
    cmap = copy.deepcopy(plt.cm.get_cmap('turbo'))
    color_list = cmap.colors

    proximity_dict_cp = proximity_dict.copy()
    if len(proximity_dict_cp) == 0:
        for key in rois_traces_dict:
            if key in bold_list:
                proximity_dict_cp[key] = 0
            else:
                proximity_dict_cp[key] = nargs - 1
    else:
        proximity_vals = np.array(list(proximity_dict_cp.values()))
        proximity_max, proximity_min = np.max(proximity_vals), np.min(proximity_vals)
        for key, val in proximity_dict_cp.items():
            # val = int(((val - proximity_min) / proximity_max) * (nargs-1))
            val = int(((val - proximity_min) / proximity_max) * (255))
            proximity_dict_cp[key] = val

    if type(dt) is not type(np.array([])):
        t = np.linspace(0, n_samples * dt, n_samples)
    else:
        t = dt

    legend = []
    rois_traces_mat = np.array(list(rois_traces_dict.values()))
    traces_mean = np.mean(rois_traces_mat, axis=0)
    traces_std = np.std(rois_traces_mat, axis=0)
    traces_median = np.median(rois_traces_mat, axis=0)
    for i, (key, trace) in enumerate(rois_traces_dict.items()):
        if key not in bold_list:
            ax.plot(t, trace, color=color_list[proximity_dict_cp[key]], linewidth=0.5)
            legend.append(key)

    for i, (key, trace) in enumerate(rois_traces_dict.items()):  # plot bold_list traces last
        if key in bold_list:
            ax.plot(t, trace, color=color_list[proximity_dict_cp[key]], linewidth=2)
            legend.append(key)

    ax.plot(t, traces_mean, linestyle='dashed', color='black')
    ax.plot(t, traces_median, linestyle='dotted', color='black')
    ax.fill_between(t, traces_mean - traces_std, traces_mean + traces_std, color='gray', alpha=0.2)

    # add axis attributes
    for key, val in kwargs.items():
        attr = getattr(ax, key)
        if isinstance(val, dict):
            attr(**val)
        else:
            attr(val)


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
    # hsv_tuples = [(x * 1.0 / nargs, 0.5, 0.5) for x in range(nargs)]
    cmap = copy.deepcopy(plt.cm.get_cmap('plasma'))
    c_list = cmap.colors[::int(256/nargs)]
    for i in range(len(c_list)):
        c_list[i].append(0.6)
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
                        boxprops=dict(facecolor=c_list[i]), medianprops=dict(color="black"), showfliers=False)

    # plot trend lines
    bp_med = []
    for i in range(nargs):
        bp_med.append([None] * nx)
        for j in range(nx):
            bp_med[i][j] = bp[i][j]['medians'][0].get_ydata()[0]

    bp_med = np.array(bp_med)
    for i in range(nargs):
        ax.plot(x + i * width, bp_med[i, :], color=c_list[i], linestyle='--')
        ax.set_xticks(x + width/2)

    # add axis attributes
    for key, val in kwargs.items():
        attr = getattr(ax, key)
        if isinstance(val, dict):
            attr(**val)
        else:
            attr(val)

    if 'legend' in list(kwargs.keys()):
        leg = ax.get_legend()
        hl_dict = {handle.get_label(): handle for handle in leg.legendHandles}
        for i in range(nargs):
            hl_dict[f'_line{i}'].set_color(c_list[i])


def wf_imshow(ax, image, mask=None, map=None, conv_ker=None, show_cb=True, cm_name='turbo', vmin=None, vmax=None, cb_side='right'):

    imagec = image.copy()
    if conv_ker is not None:
        imagec = convolve(image, conv_ker)

    if mask is not None:
        imagec[mask == 0] = None

    if map is not None:
        imagec[map == 1] = None

    cmap = copy.deepcopy(plt.cm.get_cmap(cm_name))#.reversed()
    im = ax.imshow(imagec, cmap=cmap, vmin=vmin, vmax=vmax)
    # plt.axis('off')

    if show_cb:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(cb_side, size="10%", pad=0.1)
        plt.colorbar(im, cax=cax)

    return im, imagec