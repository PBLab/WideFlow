import numpy as np
import matplotlib.pyplot as plt
from utils.load_tiff import load_tiff
import skimage
import h5py

base_path = '/data/Rotem/WideFlow prj'
dataset_path = f'{base_path}/results/sessions_20220320.h5'

moouse_id = '2601'
session_id = '20220323_neurofeedback'
session_path = f'{base_path}/{moouse_id}/{session_id}'

wf_data = load_tiff(f'{session_path}/wf_raw_data_10.tif')