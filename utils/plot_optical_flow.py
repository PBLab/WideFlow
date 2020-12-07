import numpy as np
import h5py
import pathlib
import matplotlib.pyplot as plt
from skimage import io


def plot_optical_flow(vid, flow, frame, scale, rSize, invert_y=False):
    [nt, ny, nx] = vid.shape
    uy = flow[:, :, :, 0]
    ux = flow[:, :, :, 1]
    u = uy[frame - 1, :, :]
    v = ux[frame - 1, :, :]
    v = -v  # TODO: figure out why we need to invert v

    for i in range(np.size(u, 0)):
        for j in range(np.size(u, 1)):
            if np.floor(i / rSize) != i / rSize or np.floor(j / rSize) != j / rSize:
                u[i, j] = 0
                v[i, j] = 0

    Y, X = np.mgrid[0:ny:1, 0:nx:1]
    plt.figure()
    plt.imshow(vid[frame, :, :], vmin=np.amin(vid[frame, :, :]), vmax=np.amax(vid[frame, :, :]))
    plt.quiver(Y, X, u, v, scale=scale)
    plt.title(f'frame {frame} optical flow')
    if invert_y:
        plt.gca().invert_yaxis()
    plt.show()


