import torch
from torch.utils.data import Dataset

import numpy as np
import tiffcapture as tc
from skimage import transform

from random import randrange
import os
from os import listdir
from os.path import isfile, join


class Seq2SeqDataSet(Dataset):
    def __init__(self, dirs_list, n_frames, transforms=None):
        super(Seq2SeqDataSet, self).__init__()

        self.dirs_list = dirs_list
        self.files_list = self.create_files_list()
        self.n_frames = n_frames
        self.transforms = transforms

    def __getitem__(self, index: int):
        vid_pth = self.files_list[index]
        tiff = tc.opentiff(vid_pth)

        N_frames = tiff.length
        init_frame = randrange(0, N_frames - self.n_frames)
        tiff.seek(init_frame)

        f0 = tiff.read()
        vid = np.empty((f0[1].shape[0], f0[1].shape[1], self.n_frames))
        vid[:, :, 0] = f0[1]
        for i in range(1, self.n_frames):
            vid[:, :, i] = tiff.read()[1]

        if self.transforms is not None:
            vid = self.transforms(vid)

        return vid

    def __len__(self):
        return len(self.files_list)

    def create_files_list(self):
        files_list = []
        for dir in self.dirs_list:
            files = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f)) and os.path.splitext(f)[-1] == '.tif']
            files_list += files

        return files_list


class Rescale():
    """Rescale all images in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, vid):

        h, w, t = vid.shape
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        tvid = np.empty((new_h, new_w, t))
        for i, img in enumerate(np.moveaxis(vid, 2, 0)):
            tvid[:, :, i] = transform.resize(img, (new_h, new_w))

        return tvid


class ToTensor():
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, vid):

        # swap time axis because
        # numpy image: H x W x T
        # torch image with 1 channel: 1 X T X H X W

        return torch.Tensor(np.expand_dims(vid.transpose((2, 0, 1)), 0))
