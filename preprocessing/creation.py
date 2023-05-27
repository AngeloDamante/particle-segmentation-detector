"""Functions to create data for working dataset"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing.segmentation import segment_data
from utils.Types import SNR, Density
from utils.definitions import DEPTH, TIME_INTERVAL
from utils.compute_path import get_data_path, get_slice_path, compute_name


def slices_to_npy(snr: SNR, density: Density, slices_dir: str, dest_dir: str) -> None:
    """Compact Slices into npy file

    :param snr:
    :param density:
    :param slices_dir:
    :param dest_dir:
    :return:
    """
    for t in range(TIME_INTERVAL):
        img_list = []
        for z in range(DEPTH):
            im = cv2.imread(get_slice_path(snr, density, t, z, root=slices_dir), cv2.IMREAD_GRAYSCALE)
            img_list.append(im)
        img_3d = np.stack(img_list, axis=2)
        dest_path = os.path.join(dest_dir, compute_name(snr, density))
        os.makedirs(dest_path, exist_ok=True)
        np.save(os.path.join(dest_path, compute_name(snr, density, t)), img_3d)


def make_raw_data(snr: SNR, density: Density, kernel: int, sigma: float, dest_dir: str) -> None:
    """Make Raw data with the following format saved in npz

        [
            img:np.ndarray,
            target:np.ndarray,
            gth:List[Particle],
            snr:SNR,
            density:Density,
            t:int
        ]

    :param snr:
    :param density:
    :param kernel:
    :param sigma:
    :param dest_dir:
    :return:
    """
    for t in tqdm(range(TIME_INTERVAL)):
        target, gth = segment_data(snr, density, t, kernel, sigma)
        img_3d = np.load(get_data_path(snr, density, t))
        dir_name = compute_name(snr, density)
        data_name = compute_name(snr, density, t)
        np.savez(os.path.join(dest_dir, dir_name, data_name), img=img_3d, target=target, gth=gth, snr=snr, density=density, t=t)

