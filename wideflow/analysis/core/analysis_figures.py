import matplotlib.pyplot as plt
import numpy as np


def plot_figures(results_path, cue, serial_readout):
    plt.figure(figsize=(30.0, 10.0))
    plt.plot(cue)
    plt.plot((1 - np.array(serial_readout)) * 0.5)
    plt.legend(["cues", "response"])
    plt.title("cues and responses")

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.savefig(results_path + 'cues_responses_plot.png', bbox_inches='tight')
    plt.close()