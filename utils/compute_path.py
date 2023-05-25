"""Define method to compute paths for extraction phases"""

import os
from utils.Types import SNR, Density
from utils.definitions import DTS_VIRUS, DTS_GTH, DTS_DIR, DTS_SEG_DATA, DTS_SEG_IMG


def get_slice_path(snr: SNR, density: Density, t: int, depth: int) -> str:
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


def get_gth_xml_path(snr: SNR, density: Density) -> str:
    """Compute path for file.xml with gth

    :param snr:
    :param density:
    :return: path
    """
    name = f'VIRUS_{snr.value}_{density.value}'
    path_file_gth = os.path.join(DTS_GTH, f'{name}.xml')
    return path_file_gth


def get_seg_data_path(snr: SNR, density: Density, t: int) -> str:
    """Compute path for npy data

    :param snr:
    :param density:
    :param t:
    :param mode:
    :return:
    """
    data_path = os.path.join(DTS_SEG_DATA, f'{snr.value}_{density.value}', f't_{str(t).zfill(3)}.npy')
    return data_path

def get_seg_slice_path(snr: SNR, density: Density, t: int, z: int) -> str:
    """Compute segmented Slice path

    :param snr:
    :param density:
    :param t:
    :param z:
    :return:
    """
    slice_path = os.path.join(DTS_SEG_IMG, f'{snr.value}_{density.value}', f't_{str(t).zfill(3)}_z_{str(z).zfill(2)}.tiff')
    return slice_path