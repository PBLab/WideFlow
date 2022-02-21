def write_bbox_file(path, bbox):
    with open(path, 'w') as f:
        f.writelines(f"x_min: {bbox[0]}, y_min: {bbox[1]}, x_width: {bbox[2]}, y_height: {bbox[3]}")
