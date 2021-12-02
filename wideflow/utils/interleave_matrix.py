import numpy as np


def interleave_matrix(matrix, interleave_factor):
    '''

    Args:
        matrix: 2D numpy-ndarray
        interleave_factor: number of interleave duplicates to create
        dim: dimension for interleaving

    Returns:
        numpy-ndarray of size (..., ndarray.shape[dim] * ndarray, ...)

    '''
    ndims = matrix.ndim
    shape = matrix.shape
    dtype = matrix.dtype

    if interleave_factor > 1:
        if ndims == 1:
            shape = (shape[0] * interleave_factor, )
            interleaved_matrix = np.ndarray(shape, dtype=dtype)
            for dup in range(interleave_factor):
                interleaved_matrix[dup::interleave_factor] = matrix

        if ndims == 2:
            shape = (shape[0], shape[1] * interleave_factor)
            interleaved_matrix = np.ndarray(shape, dtype=dtype)
            for i, row in enumerate(matrix):
                for dup in range(interleave_factor):
                    interleaved_matrix[i, dup::interleave_factor] = row

        return interleaved_matrix
    else:
        return matrix