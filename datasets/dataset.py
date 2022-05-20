import torch
from torch.utils.data import Dataset
import numpy as np
import h5py


class H5Dataset(Dataset):

    def __init__(self, h5py_filename, dataset_name,  batch_size=1, transform=None):
        super().__init__()
        h5file = h5py.File(h5py_filename, mode='r')
        self.pointclouds = h5file[dataset_name]
        self.transform = transform
        self.batch_size = batch_size

    def __len__(self):
        return (self.pointclouds.shape[0] // self.batch_size) * self.batch_size

    def __getitem__(self, idx: int):
        item = {
            'pos': torch.FloatTensor(self.pointclouds[idx]),
        }

        if self.transform is not None:
            item = self.transform(item)

        return item