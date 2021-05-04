

def load_config(path, handler=None):
    import os
    import json

    if not os.path.exists(path):
        raise RuntimeError('could not find path: ' + path)

    if os.path.isdir(path):
        path = os.path.join(path, 'config.json')

    with open(path, 'r') as f:
        data = f.read()
        config = json.loads(data)

    return config


def crop(img, xrange, yrange):
    return img[yrange[0]: yrange[1], xrange[0]: xrange[1]]


