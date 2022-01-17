import pickle


def plot_pickled_figure(path):
    figx = pickle.load(open(path, 'rb'))
    figx.show()
    return figx
