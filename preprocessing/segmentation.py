"""Define functions to create segmentated dataset"""

import numpy as np
from scipy import signal
from typing import List, Tuple
from preprocessing.analyser import extract_particles, query_particles
from utils.Types import SNR, Density, Particle
from utils.compute_path import get_gth_xml_path
from utils.definitions import SIZE, DEPTH


def segment_data(snr: SNR, density: Density, t: int, kernel: int, sigma: float) -> Tuple[np.ndarray, List[Particle]]:
    """Segment dataset with desired SNR and density

    :param snr:
    :param density:
    :param t:
    :param kernel:
    :param sigma:
    :return:
    """
    # take particles
    path_gth = get_gth_xml_path(snr, density)
    particles = extract_particles(path_gth)
    particles_t = query_particles(particles, (lambda pa, time=t: True if pa.t == time else False))

    # compute segmaps
    img_3d = np.zeros(shape=(SIZE[0], SIZE[1], SIZE[2]))  # (H,W,D)
    img_3d = gauss_conv(img_3d, particles_t, kernel, sigma)
    return img_3d, particles_t


def gauss_conv(img_3d: np.ndarray, particles: List[Particle], kernel: int, sigma: float) -> np.ndarray:
    """Make segmentation with convolution using gaussian filter

    :param img_3d:
    :param particles:
    :param kernel:
    :param sigma:
    :return: np.ndarray
    """
    interval = (-kernel // 2 + 1, kernel // 2 + 1, 1)
    x, y, z = np.meshgrid(np.arange(*interval), np.arange(*interval), np.arange(*interval))
    kernel = np.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))

    for p in particles:
        center = (round(p.x), round(p.y), np.clip(round(p.z), 0, DEPTH - 1))
        img_3d[center] = 255  # white point

    filtered = signal.convolve(img_3d, kernel, mode="same")
    filtered_mirror = np.rot90(filtered, k=1, axes=(0, 1))
    filtered = np.flipud(filtered_mirror)
    filtered = np.resize(filtered, (512, 512, 10))
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)
    return filtered
