import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Generate some example data
data = np.random.rand(10, 2)


# Create a clustermap with the desired threshold
cluster_map = sns.clustermap(data, cmap="mako", row_cluster=True, col_cluster=True,
                              row_linkage=None, col_linkage=None,
                              row_threshold=0.5, col_threshold=0.5)

# Adjust the plot
plt.setp(cluster_map.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)  # Rotate y-axis labels
plt.show()
