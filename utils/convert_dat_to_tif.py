from tifffile import TiffWriter
import numpy as np


def convert_dat_to_tif(path, nbytes, shape, type, nframes):
    frames_offset = nbytes * shape[0]
    for i in range(int(np.ceil(nframes / shape[0]))):
        with TiffWriter(path[:-4] + '_' + str(i) + '.tif') as tif:
            fr_data = np.reshape(np.fromfile(path,
                                             dtype=np.dtype(type),
                                             count=np.prod(shape),
                                             offset=frames_offset * i),
                                 shape)
            tif.write(fr_data, contiguous=True)