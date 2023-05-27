"""Define Torch Dataset_original and Dataloader"""
import os

import torch
import numpy as np
from typing import Any
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from utils.definitions import DTS_TRAIN_PATH
import random


# dataset
class VirusDataset(Dataset):
    def __init__(self, dir_path: str, transform=None):
        self.dir_path = dir_path
        self.transform = transform
        self.files = os.listdir(self.dir_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = np.load(os.path.join(self.dir_path, self.files[index]), allow_pickle=True)
        img = data['img'] / 255.0
        target = data['target'] / 255.0
        if self.transform:
            img, target = self.transform(img, target)
        return {'img': img, 'target': target}
