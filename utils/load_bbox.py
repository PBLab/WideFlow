

def load_bbox(path):
    '''
    text file should be in the following format:
        "y_min:100, x_min:100, y_max:512, x_max:512"
    '''
    with open(path) as f:
        text = f.readline()

    coortext = text.split(',')
    coordinates = []
    for coor in coortext:
        coordinates.append(int(coor.split(':')[1]))

    return coordinates