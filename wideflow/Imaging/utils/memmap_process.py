import sys
if int(sys.version[2]) >= 8:
    from multiprocessing import shared_memory
else:
    from multiprocess import shared_memory
import numpy as np


class MemoryHandler:
    def __init__(self, query, path, data_shape, dtype='uint16'):
        self.query = query
        self.path = path
        self.data_shape = data_shape
        self.dtype = dtype

    def __call__(self, shared_mem_name):
        vid_mem = np.memmap(self.path, dtype=np.dtype(self.dtype), mode='w+', shape=self.data_shape)
        existing_shm = shared_memory.SharedMemory(name=shared_mem_name)
        frame = np.ndarray(shape=self.data_shape[-2:], dtype=np.dtype(self.dtype), buffer=existing_shm.buf)

        frame_counter = 0
        while True:
            if self.query.empty():
                continue

            q = self.query.get()
            if q == "flush":
                vid_mem[frame_counter] = frame
                vid_mem.flush()
                frame_counter += 1
            elif q == "terminate":
                del vid_mem
                break
