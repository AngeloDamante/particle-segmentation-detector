"""Define functions to create segmentated dataset"""

import numpy as np
import cv2
from scipy import signal
from typing import List, Tuple
from preprocessing.analyser import extract_particles, query_particles
from utils.Types import SNR, Density, Particle
from utils.compute_path import get_gth_xml_path
from utils.definitions import SIZE, DEPTH

WHITE_POINT = 255


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

    # potremmo fare 100 slice per ridurre l'errore l'asse z di un fattore 100 e poi interpolare le slice (basta un resize)

    # compute segmaps
    img_3d = np.zeros(shape=(SIZE[0], SIZE[1], SIZE[2] * 10))  # (H,W,D)
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
        # center = (round(p.x), round(p.y), np.clip(round(p.z), 0, DEPTH - 1))
        center = (round(p.x), round(p.y), np.clip(round(p.z * 10), 0, DEPTH * 10 - 1))
        img_3d[center] = WHITE_POINT

    filtered_left = np.clip(signal.convolve(img_3d, kernel, mode="same"), 0, 255).astype(np.uint8)
    filtered_mirror = np.rot90(filtered_left, k=1, axes=(0, 1))
    filtered = np.flipud(filtered_mirror)

    # FIXME
    from scipy.interpolate import interp1d
    # ntime, nheight_in, nlat, nlon = (10, 20, 30, 40)
    heights = np.linspace(0, 1, DEPTH * 10)
    # t_in = np.random.normal(size=(ntime, nheight_in, nlat, nlon))
    f_out = interp1d(heights, filtered, axis=2)
    nheight_out = DEPTH
    new_heights = np.linspace(0, 1, nheight_out)
    filtered_interp = f_out(new_heights)

    # import pdb
    # pdb.set_trace()

    return filtered_interp
