import os
from utils.Types import SNR, Density
from utils.definitions import DTS_VIRUS, DTS_GTH

def get_img_path(snr: SNR, density: Density, t: int, depth: int) -> str:
    """Compute slice path for identifying tuple(snr, density, t, depth)

    :param snr:
    :param density:
    :param t:
    :param depth:
    :return: path
    """
    name = f'VIRUS_{snr.value}_{density.value}'
    path_file_img = os.path.join(DTS_VIRUS, name, f'{name}_t{str(t).zfill(3)}_z{str(depth).zfill(2)}.tif')
    return path_file_img


def get_gth_path(snr: SNR, density: Density) -> str:
    """Compute path for file.xml with gth

    :param snr:
    :param density:
    :return: path
    """
    name = f'VIRUS_{snr.value}_{density.value}'
    path_file_gth = os.path.join(DTS_GTH, f'{name}.xml')
    return path_file_gth
