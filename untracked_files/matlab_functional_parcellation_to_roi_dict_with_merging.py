from mat4py import loadmat
import h5py
import numpy as np
import re

def get_8_neighbors(p):
    x, y = p
    return [(x + dx, y + dy) for dx in [-1, 0, 1]
                            for dy in [-1, 0, 1]
                            if not (dx == 0 and dy == 0)]

def linear_to_xy(indices, height, width):
    y, x = np.unravel_index(indices, (height, width))
    # Convert each coordinate pair to a tuple (x, y)
    return [(int(xi), int(yi)) for xi, yi in zip(x, y)]

def xy_to_linear(coords, width):
    # Convert (x, y) tuples to linear indices
    return np.array([y * width + x for (x, y) in coords])

def convert_and_merge_rois(rois_struct_path, rois_dict_path, merge_roi_name_1, merge_roi_name_2):
    data = loadmat(rois_struct_path)
    rois_struct = data['ROI_list']
    rois_dict = {}

    merged_pixels = []
    outlines = []
    merged_centroid_sum = np.zeros(2)
    merged_area = 0
    merged_name = f"{merge_roi_name_1}_{merge_roi_name_2}"

    merge_indices = []

    # Collect outlines and pixels for merging, and store others in dict
    for i in range(len(rois_struct['name'])):
        name = rois_struct['name'][i]
        if name in [merge_roi_name_1, merge_roi_name_2]:
            merge_indices.append(i)
            pixels = np.array(rois_struct['pixel_list'][i]) - 1  # zero-based
            outline = np.array(rois_struct['boundary_list'][i]) - 1  # zero-based
            centroid = np.array(rois_struct['centerPos'][i]) - 1
            merged_pixels.append(pixels)
            outlines.append(outline)
            merged_centroid_sum += centroid * len(pixels)
            merged_area += len(pixels)
        else:
            #roi_key = f'roi_{len(rois_dict)+1:02d}'
            # extract numeric part of the name
            match = re.search(r'\d+', name)
            if not match:
                raise ValueError(f"Cannot extract ROI number from name: {name}")
            roi_number = int(match.group())
            roi_key = f'roi_{roi_number:02d}'
            rois_dict[roi_key] = {
                'Index': len(rois_dict),
                'Area': len(rois_struct['pixel_list'][i]),
                'Centroid': np.array(rois_struct['centerPos'][i]) - 1,
                'PixelIdxList': np.array(rois_struct['pixel_list'][i]) - 1,
                'outline': np.array(rois_struct['boundary_list'][i]) - 1,
                'top_left_bottom_rigth': [],
                'name': name
            }

    if len(merge_indices) != 2:
        raise ValueError("Both specified ROI names must exist and be unique in the input.")

    # For merged ROIs, convert pixel linear indices to (x,y)
    outline_1_linear = np.array(rois_struct['boundary_list'][merge_indices[0]]) - 1
    outline_2_linear = np.array(rois_struct['boundary_list'][merge_indices[1]]) - 1

    outline_1_xy = linear_to_xy(outline_1_linear,337, 297)
    outline_2_xy = linear_to_xy(outline_2_linear, 337, 297)

    outline_1_set = set(outline_1_xy)
    outline_2_set = set(outline_2_xy)

    # Identify internal seam pixels: pixels with >= 2 neighbors in the other outline
    shared_from_1 = {p for p in outline_1_set if sum((n in outline_2_set) for n in get_8_neighbors(p)) >= 2}
    shared_from_2 = {p for p in outline_2_set if sum((n in outline_1_set) for n in get_8_neighbors(p)) >= 2}

    # Prune seam pixels from each outline
    merged_outline_set = (outline_1_set - shared_from_1) | (outline_2_set - shared_from_2)
    merged_outline = np.array(list(merged_outline_set))
    merged_outline_linear = xy_to_linear(merged_outline, width=297)

    # Concatenate pixel lists from both ROIs (union of all pixels)
    merged_pixels = np.concatenate(merged_pixels, axis=0)

    merged_centroid = merged_centroid_sum / merged_area

    #merged_key = f'roi_{len(rois_dict)+1:02d}'
    match1 = re.search(r'\d+', merge_roi_name_1)
    match2 = re.search(r'\d+', merge_roi_name_2)
    if not (match1 and match2):
        raise ValueError("ROI names must contain numeric suffixes")

    num1 = int(match1.group())
    num2 = int(match2.group())
    merged_key = f'roi_{num1}_{num2}'
    merged_name = f"{merge_roi_name_1}_{merge_roi_name_2}"

    rois_dict[merged_key] = {
        'Index': len(rois_dict),
        'Area': merged_area,
        'Centroid': merged_centroid,
        'PixelIdxList': merged_pixels,
        'outline': merged_outline_linear,
        'top_left_bottom_rigth': [],
        'name': merged_name
    }

    # Save to HDF5 file
    with h5py.File(rois_dict_path, 'w') as f:
        for roi_key, roi_dict in rois_dict.items():
            grp = f.create_group(roi_key)
            for key, val in roi_dict.items():
                grp.create_dataset(key, data=val)