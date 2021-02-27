import numpy as np
from numba import cuda
from skimage.transform import rescale, resize

@cuda.jit
def dff(image, baseline):
    return (image - baseline) / (baseline + np.finfo(np.float32).eps)


@cuda.jit
def imresize(image, resize_image, H):
    pass


@cuda.jit
def bilinear_interpolation(dst, src):
    pass
