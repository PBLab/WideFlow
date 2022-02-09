def write_bbox_file(path, bbox):
    with open(path, 'w') as f:
        f.writelines(f"x_min: {bbox[0]}, x_max: {bbox[1]}, y_min: {bbox[2]}, y_max: {bbox[3]}")
