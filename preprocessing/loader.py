"""Functions to load slices and volume with their gth"""

from typing import Tuple, List, Callable
import torch
from torchvision import transforms
from PIL import Image
from preprocessing.Particle import Particle, SNR, Density
from preprocessing.analyser import get_img_path

def get_slice(snr: SNR, density: Density, t: int, depth: int) -> Image:
    """Get image from identifying tuple

    :param snr:
    :param density:
    :param t:
    :param depth:
    :return: PIL image
    """
    idx = (snr, density, t, depth)
    path_file_img = get_img_path(*idx)

    # load image
    img = Image.open(path_file_img)
    return img.copy()

# def get_3d_image(snr: SNR, density: Density, t: int) -> Tuple[Image, List[Particle]]:
    # pass