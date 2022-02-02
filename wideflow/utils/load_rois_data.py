import h5py


def load_rois_data(file_path, shift=0):
    with h5py.File(file_path, 'r') as f:
        roi_list = {}
        for key, grp in f.items():
            roi_list[key] = {}
            roi_list[key]['Index'] = int(key.split('_')[1])
            roi_list[key]['Area'] = grp['Area'][()]
            roi_list[key]['Centroid'] = grp['Centroid'][()]
            roi_list[key]['PixelIdxList'] = grp['PixelIdxList'][()] - shift  # shift=1 to convert from matlab to python
            roi_list[key]['outline'] = grp['outline'][()]
            roi_list[key]['top_left_bottom_rigth'] = grp['top_left_bottom_rigth'][()]
            roi_list[key]['name'] = grp['name'][()]

    # keep keys order as index order
    roi_list = {k: v for k, v in sorted(roi_list.items(), key=lambda item: item[1]['Index'])}
    return roi_list