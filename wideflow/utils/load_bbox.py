

def load_bbox(path):
    '''
    text file should be in the following format:
        "x_min:100, y_min:100, x_width:512, y_height:512"
    '''
    with open(path) as f:
        text = f.readline()

    coortext = text.split(',')
    coordinates = []
    for coor in coortext:
        coordinates.append(int(coor.split(':')[1]))

    return coordinates