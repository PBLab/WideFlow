import numpy as np
import matplotlib.pyplot as plt


image_size = [128, 256, 512, 1024]
buffer_size = [16, 32, 64]

acquisition_time = [20, 25, 30, 35]
processing_time16 = [20, 30, 45, 60]
processing_time32 = [0, 1, 2, 3]
processing_time64 = [0, 1, 2, 3]

f, ax = plt.subplots()
ax.plot(image_size, acquisition_time, c='k', linestyle='-')
ax.plot(image_size, processing_time16, c='k', linestyle='-')