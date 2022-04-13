import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def swap_3dmatrix_elements(mat):
    nt, ny, nx = mat.shape
    swap_ind = np.random.permutation(np.arange(ny * nx))
    flat = np.reshape(mat, [nt, ny*nx])
    flat_swap = np.zeros(flat.shape, dtype=np.float32)
    for i in range(ny * nx):
        flat_swap[:, i] = flat[:, swap_ind[i]]

    return np.reshape(flat_swap, [nt, ny, nx])


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1, 2)
               .reshape(-1, nrows, ncols))


def overlapped_blockshaped(arr, nrows, ncols):
    m, n = arr.shape
    arr = np.pad(arr, [[0, nrows-1], [0, ncols-1]])
    ol_block = np.zeros((m*n, nrows, ncols))
    cols_cond = np.mod(np.arange(n*m), ncols)
    rows_cond = np.zeros(n)
    for i in range(1, nrows):
        rows_cond = np.concatenate([rows_cond, np.zeros(n)+i])
    rows_cond = np.tile(rows_cond, int(m/nrows))
    for j in range(ncols):
        for i in range(nrows):
            blockij = blockshaped(arr[i:m+i, j:n+j], nrows, ncols)
            mask = np.logical_and(cols_cond==j, rows_cond==i)
            ol_block[mask] = blockij

    return ol_block


def roi_outline_from_pixels_indices(pixels_indices, shape, order='C'):
    """

    :param pixels_indices: a list of indices
    :param shape: shape of the 2d matrix from where pixels_indices where drawn from
    :param order: specify the subscript format
    :return:
    """

    pixels_indices = np.array(pixels_indices)
    roi_id = np.unravel_index(pixels_indices, shape=shape, order=order)
    ((top, left), (down, right)) = roi_top_left_bottom_right(pixels_indices, shape, order)

    img = np.zeros(shape=shape)
    img[roi_id[0], roi_id[1]] = 1
    croped_img = img[top: down, left: right]
    croped_img = np.pad(croped_img, ((1, 1), (1, 1)))
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv.erode(croped_img, kernel, iterations=1)
    margin = croped_img - erosion
    margin = margin[1:-1, 1:-1]
    img[top: down, left: right] = margin
    temp = np.where(img == 1)
    outline = np.array([temp[0], temp[1]]).transpose()
    # exclude outline points that coincide with the image boundaries
    ex = outline[:, 0] == 0
    outline = np.delete(outline, (ex), axis=0)
    ex = outline[:, 0] == shape[0]
    outline = np.delete(outline, (ex), axis=0)
    ex = outline[:, 1] == 0
    outline = np.delete(outline, (ex), axis=0)
    ex = outline[:, 1] == shape[1]
    outline = np.delete(outline, (ex), axis=0)
    return outline


def roi_top_left_bottom_right(pixels_indices, shape, order):
    """
    :param pixels_indices: a list of indices in Fortran format
    :param shape: shape of the 2d matrix from where pixels_indices where drawn from
    :return: tuple ((top, left), (down, right))
    """

    roi_id = np.unravel_index(pixels_indices, shape=shape, order=order)
    top = np.amin(roi_id[0])
    down = np.amax(roi_id[0]) + 1
    left = np.amin(roi_id[1])
    right = np.amax(roi_id[1]) + 1
    return ((top, left), (down, right))


def add_properties_to_roi_list(rois_dict, shape, order='F'):
    for roi_name, roi_dict in rois_dict.items():
        rois_dict[roi_name]["top_left_bottom_right"] = roi_top_left_bottom_right(roi_dict["PixelIdxList"], shape, order)
        rois_dict[roi_name]["outline"] = roi_outline_from_pixels_indices(rois_dict[roi_name]["PixelIdxList"], shape, order)
    return rois_dict


def add_morphological_adjacent_rois_to_roi_list(roi_list, shape):
    # works only for overlapped outlines!
    for key1, val1 in roi_list.items():
        outline_flat1 = np.ravel_multi_index((val1['outline'][:, 0], val1['outline'][:, 1]), shape)
        adj_roi = []
        for key2, val2 in roi_list.items():
            outline_flat2 = np.ravel_multi_index((val2['outline'][:, 0], val2['outline'][:, 1]), shape)
            if any(x in outline_flat1 for x in outline_flat2):
                adj_roi.append(val2['Index'])

        roi_list[key1]['Adjacent_rois_Idx'] = adj_roi
        roi_list[key1]['Adjacent_rois_Idx'].remove(val1['Index'])

    return roi_list


def extract_rois_data(image, roi_dict):
    shape = image.shape
    rois_data = [0] * len(roi_dict)
    i = 0
    for key, val in roi_dict.items():
        pixels_inds = np.unravel_index(val['PixelIdxList'], shape)
        rois_data[i] = np.mean(image[pixels_inds])
        i += 1

    return rois_data


def mse(x, y):
    return np.mean(np.square(x - y))