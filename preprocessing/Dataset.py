"""Define Torch Dataset_original and Dataloader"""
import os
import numpy as np
from typing import Tuple, List
from pathlib import Path
from compute_path import get_seg_data_path, SegMode, SNR, Density

import torch
from torch.utils.data import Dataset
from torchvision import transforms


from typing import Any
from torchvision.transforms import functional as F
import random

# dataset
class VirusDataset(Dataset):
    def __init__(self, snr: SNR, densities:List[Density], mode: SegMode, dir_name, transform=None):
        self.snr = snr
        self.densities = densities
        self.segMode = mode
        self.transform = transform

        dts_path_training = os.path.join(dir_name)
        self.file_list = os.listdir(dts_path_training)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # return due numpy array di dimensioni 10, 512, 512
        data = np.load(self.file_list[index])
        img = data['image'] / 255.0
        target = data['target'] / 255.0
        if self.transform:
            img, target = self.transform(img, target)
        return img, target