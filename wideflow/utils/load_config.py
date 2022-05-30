import os
import json


def load_config(path, handler=None):
    if not os.path.exists(path):
        raise RuntimeError('could not find path: ' + path)

    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.endswith('json'):
                path = os.path.join(path, file)

    with open(path, 'r') as f:
        data = f.read()
        config = json.loads(data)

    return config




