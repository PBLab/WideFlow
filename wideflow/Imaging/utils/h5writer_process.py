from multiprocessing import shared_memory
import numpy as np
import h5py


class MemoryHandler:
    def __init__(self, query, path, data_shape, data_type='uint16'):
        self.query = query
        self.path = path
        self.data_shape = data_shape
        self.data_type = data_type

    def __call__(self, shared_mem_name):
        existing_shm = shared_memory.SharedMemory(name=shared_mem_name)
        frame = np.ndarray(shape=self.data_shape[-2:], dtype=np.dtype(self.data_type), buffer=existing_shm.buf)

        with h5py.File(self.path, 'w') as f:
            f.create_group('wf_metadata')
            f.create_dataset('wf_metadata/shape', data=self.data_shape)
            f.create_dataset('wf_metadata/type', data=self.data_type)

            vid_dset = f.create_dataset('wf_raw_data', size=self.data_shape, dtype='int16')
            frame_counter = 0
            while True:
                if self.query.empty():
                    continue

                q = self.query.get()
                if q == "flush":
                    vid_dset[frame_counter, :, :] = frame
                    frame_counter += 1
                elif q == "terminate":
                    break
