"""Define class segmenter to make segmentation dataset"""

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy import signal
from typing import List, Tuple
from preprocessing.analyser import extract_particles, query_particles
from preprocessing.Particle import Particle
from utils.Types import SegMode, SNR, Density
from utils.compute_path import get_gth_xml_path
from utils.definitions import DTS_DIR



C, H, W, D = 1, 512, 512, 10
SIZE_VOL = (C, H, W, D)
FINAL_TIME = 100
MAX_DEPTH = 10


class Segmenter:
    def __init__(self, sigma: float = 1.0, kernel: int = 5, radius: int = 1, value: int = 255):
        """ Constructor

        :param sigma:
        :param kernel:
        :param radius:
        :param value:
        """
        self.sigma: float = sigma
        self.kernel: int = kernel
        self.radius: int = radius
        self.value: int = value
        self.size = (H, W, D)

    def update_size(self, size: tuple):
        """Change size of 3d img

        :param size:
        :return:
        """
        self.size = size

    def update_values(self, sigma: float = None, kernel: int = None, radius: int = None, value: int = None):
        """Update main values for both techniques

        :param sigma:
        :param kernel:
        :param radius:
        :param value:
        :return:
        """
        if sigma is not None: self.sigma = sigma
        if kernel is not None: self.kernel = kernel
        if radius is not None: self.radius = radius
        if value is not None: self.value = value

    def create_dataset(self,
                       mode: SegMode,
                       snr: SNR,
                       density: Density,
                       time_interval: int = FINAL_TIME,
                       save_img: bool = False) -> Tuple[List[np.ndarray], List[Particle]]:
        """Create Dataset

        :param snr:
        :param density:
        :param mode:
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

        # make dir
        dir_name = os.path.join(DTS_DIR, f'segmentation_{str(mode.value)}')
        os.makedirs(dir_name, exist_ok=True)

        for t in tqdm(range(time_interval)):
            particles_t = query_particles(particles, (lambda pa, time=t: True if pa.t == time else False))
            img_3d = np.zeros(shape=(self.size[0], self.size[1], self.size[2]))  # (C,H,W,D)

            if mode.value == SegMode.sphere.value:
                img_3d = self._make_sphere(img_3d, particles_t)
            if mode.value == SegMode.gauss.value:
                img_3d = self._make_gauss(img_3d, particles_t)

            dts_img.append(img_3d)
            dts_gth.append(particles_t)
            self._save_data(img_3d, t, os.path.join(dir_name, f'{snr.value}_{str(density.value)}'), save_img)
        return dts_img, dts_gth

    def _make_sphere(self, img_3d: np.ndarray, particles: List[Particle]) -> np.ndarray:
        """ Make segmentation with white spheres centered in particles coord

        :param img_3d:
        :param particles:
        :return:
        """
        x, y, z = np.meshgrid(np.arange(img_3d.shape[0]), np.arange(img_3d.shape[1]), np.arange(img_3d.shape[2]))
        for p in particles:
            center = (round(p.x), round(p.y), np.clip(round(p.z), 0, MAX_DEPTH - 1))
            distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
            mask = (distance <= self.radius)
            img_3d[mask] = self.value
        return img_3d

    def _make_gauss(self, img_3d: np.ndarray, particles: List[Particle]) -> np.ndarray:
        """Make segmentation with convolution using gaussian filter

        :param img_3d:
        :param particles:
        :return: np.ndarray
        """
        interval = (-self.kernel//2+1, self.kernel//2+1, 1)

        x = np.arange(*interval)
        y = np.arange(*interval)
        z = np.arange(*interval)
        xx, yy, zz = np.meshgrid(x, y, z)
        kernel = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * self.sigma ** 2))

        for p in particles:
            center = (round(p.x), round(p.y), np.clip(round(p.z), 0, 9))
            img_3d[center] = self.value

        filtered_left = signal.convolve(img_3d, kernel, mode="same").astype(np.uint8)

        # mask
        filtered_left[filtered_left > 0] = self.value

        filtered_mirror = np.rot90(filtered_left, k=1, axes=(0,1))
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
        dir_data = f'{directory}_data'
        os.makedirs(dir_data, exist_ok=True)

        # save npy
        np.save(os.path.join(dir_data, f't_{str(time).zfill(3)}'), img_3d)

        # create img dir
        if not save_img: return
        dir_img = f'{directory}_img'
        os.makedirs(dir_img, exist_ok=True)

        # save slices
        for depth in range(self.size[2]):
            img_name = f't_{str(time).zfill(3)}_z_{str(depth).zfill(2)}.tiff'
            img = Image.fromarray(img_3d[:, :, depth].astype(np.uint8))
            img.save(os.path.join(dir_img, img_name))
