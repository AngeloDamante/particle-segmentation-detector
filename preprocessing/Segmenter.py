"""Define class segmenter to make segmentation dataset"""

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy import signal
from typing import List, Tuple
from preprocessing.analyser import extract_particles, query_particles
from preprocessing.Particle import Particle
from utils.Types import SNR, Density
from utils.compute_path import get_gth_xml_path
from utils.definitions import DTS_DIR, DTS_SEG_IMG, DTS_SEG_DATA

C, H, W, D = 1, 512, 512, 10
SIZE_VOL = (C, H, W, D)
FINAL_TIME = 100
MAX_DEPTH = 10


class Segmenter:
    def __init__(self, sigma: float = 1.0, kernel: int = 5, value: int = 255):
        """ Constructor

        :param sigma:
        :param kernel:
        :param value:
        """
        self.sigma: float = sigma
        self.kernel: int = kernel
        self.value: int = value
        self.size = (H, W, D)

    def update_size(self, size: tuple):
        """Change size of 3d img

        :param size:
        :return:
        """
        self.size = size

    def update_values(self, sigma: float = None, kernel: int = None, value: int = None):
        """Update main values for both techniques

        :param sigma:
        :param kernel:
        :param value:
        :return:
        """
        if sigma is not None: self.sigma = sigma
        if kernel is not None: self.kernel = kernel
        if value is not None: self.value = value

    def create_dataset(self,
                       snr: SNR,
                       density: Density,
                       time_interval: int = FINAL_TIME,
                       save_img: bool = False) -> Tuple[List[np.ndarray], List[Particle]]:
        """Create Dataset

        :param snr:
        :param density:
        :param time_interval:
        :param save_img:
        :return:
        """
        # take particles
        path_gth = get_gth_xml_path(snr, density)
        particles = extract_particles(path_gth)

        # initialize
        dts_img = []
        dts_gth = []

        # make dirs
        os.makedirs(DTS_SEG_IMG, exist_ok=True)
        os.makedirs(DTS_SEG_DATA, exist_ok=True)

        for t in tqdm(range(time_interval)):
            particles_t = query_particles(particles, (lambda pa, time=t: True if pa.t == time else False))
            img_3d = np.zeros(shape=(self.size[0], self.size[1], self.size[2]))  # (C,H,W,D)
            img_3d = self._make_gauss(img_3d, particles_t)
            dts_img.append(img_3d)
            dts_gth.append(particles_t)
            self._save_data(img_3d, t, f'{snr.value}_{str(density.value)}', save_img)
        return dts_img, dts_gth

    def _make_gauss(self, img_3d: np.ndarray, particles: List[Particle]) -> np.ndarray:
        """Make segmentation with convolution using gaussian filter

        :param img_3d:
        :param particles:
        :return: np.ndarray
        """
        interval = (-self.kernel // 2 + 1, self.kernel // 2 + 1, 1)
        x, y, z = np.meshgrid(np.arange(*interval), np.arange(*interval), np.arange(*interval))
        kernel = np.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * self.sigma ** 2))

        for p in particles:
            center = (round(p.x), round(p.y), np.clip(round(p.z), 0, 9))
            img_3d[center] = self.value

        filtered_left = signal.convolve(img_3d, kernel, mode="same").astype(np.uint8)
        filtered_mirror = np.rot90(filtered_left, k=1, axes=(0, 1))
        filtered = np.flipud(filtered_mirror)
        return filtered

    def _save_data(self, img_3d: np.ndarray, time: int, directory: str, save_img: bool):
        """ Save data

        :param img_3d:
        :param time:
        :param directory:
        :param save_img:
        :return:
        """
        # create dir data
        dir_data = os.path.join(DTS_SEG_DATA, directory)
        os.makedirs(dir_data, exist_ok=True)

        # save npy
        np.save(os.path.join(dir_data, f't_{str(time).zfill(3)}'), img_3d)

        # create img dir
        if not save_img: return
        dir_img = os.path.join(DTS_SEG_IMG, directory)
        os.makedirs(dir_img, exist_ok=True)

        # save slices
        for depth in range(self.size[2]):
            img_name = f't_{str(time).zfill(3)}_z_{str(depth).zfill(2)}.tiff'
            img = Image.fromarray(img_3d[:, :, depth].astype(np.uint8))
            img.save(os.path.join(dir_img, img_name))
