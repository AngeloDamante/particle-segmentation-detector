"""Utility functions for processing phase"""
import math

import torch
import os
from preprocessing.Dataset import VirusDataset
from torch.utils.data import DataLoader


class CheckpointSaver:
    def __init__(self, file_name: str = ""):
        self.file_name = file_name
        self.loss_value = None

    def get_file_name(self):
        """Filename getter

        :return:
        """
        return self.file_name

    def set_file_name(self, file_name):
        """Change file_name

        :param file_name:
        :return:
        """
        self.file_name = file_name

    def is_present(self):
        """Verify if checkpoint was already saved

        :return:
        """
        return os.path.exists(self.file_name)

    def save_checkpoint(self, loss_value: float, dict_model, dict_opt):
        """Save Checkpoints

        :param loss_value:
        :param dict_model:
        :param dict_opt:
        :return:
        """
        if math.isnan(loss_value): return
        if self.loss_value is None: self.loss_value = loss_value
        if self.loss_value > loss_value:
            print("=> Saving checkpoint")
            checkpoint = {"state_dict": dict_model, "optimizer": dict_opt}
            torch.save(checkpoint, self.file_name)
            self.loss_value = loss_value


def get_loaders(train_path, val_path, batch_size, train_transforms, val_transforms, num_workers, pin_memory):
    """Get Loaders

    :param train_path:
    :param val_path:
    :param batch_size:
    :param train_transforms:
    :param val_transforms:
    :param num_workers:
    :param pin_memory:
    :return:
    """
    # training
    train_dts = VirusDataset(train_path, train_transforms)
    train_dtl = DataLoader(train_dts,
                           shuffle=True,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           pin_memory=pin_memory)

    # validation
    val_dts = VirusDataset(val_path, val_transforms)
    val_dtl = DataLoader(val_dts,
                         shuffle=False,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         pin_memory=pin_memory)

    return train_dtl, val_dtl
