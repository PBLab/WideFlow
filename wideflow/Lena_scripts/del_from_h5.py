import numpy as np
import h5py
import os

path = '/data/Lena/WideFlow_prj/Results/results2.h5'
with h5py.File(path, "w") as myfile:
    # data = np.random.rand(int(1e6))
    # myfile.create_dataset("MyDataSet", data=data)
    print(os.path.getsize(path))
#
# with h5py.File(path, "a") as myfile:
#     del myfile['default']
#     # try:
#     #     myfile["MyDataSet"].value
#     # except KeyError as err:
#     #     # print(err)
#     #     pass
#
# print(os.path.getsize(path))