from tifffile import TiffWriter
import h5py
import numpy as np


def convert_dat_to_tif(path, nbytes, tiff_shape, type, nframes):
    frames_offset = nbytes * tiff_shape[0]
    last_tif_shape = (nframes % tiff_shape[0], tiff_shape[1], tiff_shape[2])
    for i in range(int(np.floor(nframes / tiff_shape[0]))):
         with TiffWriter(path[:-4] + '_' + str(i) + '.tif') as tif:
            fr_data = np.reshape(np.fromfile(path,
                                             dtype=np.dtype(type),
                                             count=np.prod(tiff_shape),
                                             offset=frames_offset * i),
                                 tiff_shape)
            tif.write(fr_data, contiguous=True)

    if last_tif_shape[0]:
        if np.floor(nframes / tiff_shape[0]) == 0:
            i = -1
        with TiffWriter(path[:-4] + '_' + f'{i+1:3d}' + '.tif') as tif:
            fr_data = np.reshape(np.fromfile(path,
                                             dtype=np.dtype(type),
                                             count=np.prod(last_tif_shape),
                                             offset=frames_offset * (i+1)),
                                 last_tif_shape)
            tif.write(fr_data, contiguous=True)


def convert_h5_to_tif(path, tiff_shape):

    with h5py.File(path, 'r') as f:
        h5_shape = f['wf_metadata/shape'][()]

        nframes = h5_shape[0]
        last_tif_shape = (nframes % tiff_shape[0], tiff_shape[1], tiff_shape[2])

        for i in range(int(np.floor(nframes / tiff_shape[0]))):
            start_ind = i * tiff_shape[0]
            end_ind = (i + 1) * tiff_shape[0]
            with TiffWriter(path[:-4] + '_' + str(i) + '.tif') as tif:
                fr_data = f['wf_raw_data'][start_ind:end_ind]
                tif.write(fr_data, contiguous=True)

        if last_tif_shape[0]:
            if not np.floor(nframes / tiff_shape[0]):
                i = -1
            start_ind = (i + 1) * tiff_shape[0]
            with TiffWriter(path[:-4] + '_' + str(i + 1) + '.tif') as tif:
                fr_data = f['wf_raw_data'][start_ind:]
                tif.write(fr_data, contiguous=True)