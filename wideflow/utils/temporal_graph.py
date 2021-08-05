import numpy as np
from wideflow.utils.analyze_OF import get_roi_stream_connectivity, find_streamline_travel_list, get_temporal_streamline
import matplotlib.pyplot as plt
import random



class TimeGraph():
    def __init__(self, roi_list, flow_shape, graph=None, flow=None, exclude_adjacent_rois=False):
        """

        :param roi_lsit:
        :param graph_dict:
        :param interval:
        """
        self.roi_list = roi_list
        self.n_nodes = len(self.roi_list)
        self.time_graph = [] if graph is None else [graph, ]
        self.graph_state = graph or {}
        # self.interval = interval or np.size(flow, 0) or None
        self.flow = flow
        self.shape = flow_shape
        self.exclude_adjacent_rois = exclude_adjacent_rois

    def update_state(self, flow):
        self.flow = flow
        for key in self.roi_list:
            self.graph_state[key] = self.get_node_edges(key)
        self.time_graph.append(self.graph_state)

    def get_node_edges(self, node_key):
        edges = np.zeros((len(self.roi_list), 1))
        for pos in self.roi_list[node_key]["outline"]:
            sl = get_temporal_streamline(self.flow, pos)
            travel_list = np.unique(find_streamline_travel_list(sl[:,:2], self.roi_list, self.shape)) - 1
            if len(travel_list):
                edges[travel_list] += 1
        # if node_key=='roi_9' or node_key=='roi_55' or node_key=='roi_40':
        #     print(f' {node_key}:\n {edges.transpose()}')
        return edges/len(self.roi_list[node_key]["outline"])

    def plot_time_dependent_reachability(self):
        T = len(self.time_graph)
        t = np.arange(T)
        HSV_tuples = [(x * 1.0 / self.n_nodes, 0.5, 0.5) for x in range(self.n_nodes)]
        random.shuffle(HSV_tuples)
        x, y = [], []
        for roi in self.roi_list:
            y.append(self.roi_list[roi]['Centroid'][0])
            x.append(self.roi_list[roi]['Centroid'][1])

        for i, g in enumerate(self.time_graph):
            plt.figure()
            plt.scatter(x, y, color=HSV_tuples)
            ind = 0
            for src_key, val in g.items():
                src_node = self.roi_list[src_key]['Centroid']
                for dst_idx, edge in enumerate(val):
                    if edge > 0:
                        if self.exclude_adjacent_rois and dst_idx in self.roi_list[f'roi_{dst_idx+1}']['Adjacent_rois_Idx']:
                            continue
                        dst_node = self.roi_list[f'roi_{dst_idx+1}']['Centroid']
                        plt.plot([src_node[1], dst_node[1]], [src_node[0], dst_node[0]], color=HSV_tuples[ind])
                ind += 1
        plt.show()

    def exclude_morphological_adjacent(self): # exclude morphological adjacent rois from the travel list
        pass




import pathlib
from utils.load_matlab_vector_field import load_extended_rois_list
from utils.load_matlab_vector_field import load_matlab_OF
flow_path = str(pathlib.Path('C:/') / 'Users' / 'motar' / 'PycharmProjects' / 'WideFlow' / 'data' / 'OFAMM' / 'ofamm_results.mat')
roi_path = str(pathlib.Path('C:/') / 'Users' / 'motar' / 'PycharmProjects' / 'WideFlow' / 'data' / 'mock_data' / 'mock_rois_extended_2.h5')

flow = load_matlab_OF(flow_path)
roi_list = load_extended_rois_list(roi_path)
# left_roi_list = {}
# for key, val in roi_list.items():
#     if int(key.split('_')[1]) <= 28:
#         left_roi_list[key] = val

interval = 30
tg = TimeGraph(roi_list, flow_shape=flow.shape[1:3], exclude_adjacent_rois=True)
for i in range(10, 100, 50):
    tg.update_state(flow[20:20+interval, :, :, :])

tg.plot_time_dependent_reachability()
z=3

