import matplotlib.pyplot as plt
import colorsys
import numpy as np


def plot_box_plot(ax, *args, **kwargs):
    '''

    Args:
        ax: pyplot figure axis for ploting
        *args: numpy 2d arrays - columns corresponds to data for specific x-values. Each arg will add box plot to all
                                 x-values
        **kwargs: pyplot key-value pairs - axis attributes

    Returns:

    '''
    nargs = len(args)
    hsv_tuples = [(x * 1.0 / nargs, 0.5, 0.5) for x in range(nargs)]
    nx = args[0].shape[1]  # nx should be the same for each arg
    x = np.arange(nx)
    width = (1 / nargs) / 2

    bp = []
    # plot box plots
    for i, data in enumerate(args):
        if data.shape[0] == 1:
            notch = False
        else:
            notch = True
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

    # add axis methods
    for key, val in kwargs.items():
        attr = getattr(ax, key)
        attr(val)

