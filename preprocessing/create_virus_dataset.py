import logging
import os
import numpy as np
from tqdm import tqdm
from preprocessing.Segmenter import Segmenter
from utils.Types import mapSNR, mapDensity
from preprocessing.analyser import make_npy
from utils.definitions import DTS_COMPLETE, TIME_INTERVAL
from utils.compute_path import get_data_path, get_seg_data_path
from utils.logger import configure_logger

configure_logger(logging.INFO)


def make_data_files(kernel: int, sigma: float):
    """Make npy original and segmented files

    :param kernel:
    :param sigma:
    :return:
    """
    o_seg = Segmenter(kernel=kernel, sigma=sigma)
    for snr in mapSNR.values():
        for density in mapDensity.values():
            logging.info(f' SNR = {snr.value}, Density = {density.value} ')
            make_npy(snr, density)
            o_seg.create_dataset(snr, density, save_img=True)


def make_complete_dataset():
    """To produce npz files with img, target, snr, density attributes

    :return:
    """
    for snr in tqdm(mapSNR.values()):
        for density in mapDensity.values():
            directory = os.path.join(DTS_COMPLETE, f'{snr.value}_{density.value}')
            os.makedirs(directory, exist_ok=True)
            for t in range(TIME_INTERVAL):
                img = np.load(get_data_path(snr, density, t))
                target = np.load(get_seg_data_path(snr, density, t))
                data_name = f't_{str(t).zfill(3)}'
                np.savez(os.path.join(directory, data_name), img=img, target=target, snr=snr, density=density, t=t)


def split_dataset(perc: int):
    """Split dataset into training, testing, validation directories

        Input percentage  is the value for training. The rest of the percentage
        is divided equally between testing and validation.

    :param perc: percentage of split for training
    :return:
    """
    # TODO
    pass


# make_data_files(kernel=5, sigma=0.5)
make_complete_dataset()
