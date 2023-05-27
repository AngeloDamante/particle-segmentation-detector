"""Define method to compute paths for data"""

import os
from utils.Types import SNR, Density
from utils.definitions import DEFAULT_SLICES_PATH, DEFAULT_DATA_PATH, DEFAULT_GTH_PATH


def compute_name(snr: SNR, density: Density, t: int = None, z: int = None) -> str:
    """Compute General name for data

    :param snr:
    :param density:
    :param t:
    :param z:
    :return:
    """
    z_name, t_name = "", ""
    if t is not None: t_name = f'_t{str(t).zfill(3)}'
    if z is not None: z_name = f'_z{str(z).zfill(2)}'
    return f'{snr.value}_{density.value}{t_name}{z_name}'


def get_slice_path(snr: SNR, density: Density, t: int, z: int, root: str = None) -> str:
    """Computd path for slice with input directory

    :param snr:
    :param density:
    :param t:
    :param z:
    :param root: if not present, this function compute relative path
    :return:
    """
    if not root: root = DEFAULT_SLICES_PATH
    file_name = f'{compute_name(snr, density, t, z)}.tif'
    return os.path.join(root, compute_name(snr, density), file_name)


def get_data_path(snr: SNR, density: Density, t: int, is_npz: bool = False, root: str = None) -> str:
    """Compute path for slice with input directory

    :param snr:
    :param density:
    :param t:
    :param is_npz:
    :param root: if not present, this function compute relative path
    :return:
    """
    if not root: root = DEFAULT_DATA_PATH
    ext = 'npz' if is_npz else 'npy'
    file_name = f'{compute_name(snr, density, t)}.{ext}'
    return os.path.join(root, compute_name(snr, density), file_name)


def get_gth_xml_path(snr: SNR, density: Density, root: str = None) -> str:
    """Compute path for file.xml with gth

    :param snr:
    :param density:
    :param root:
    :return: path
    """
    if not root: root = DEFAULT_GTH_PATH
    name = compute_name(snr, density)
    path_file_gth = os.path.join(root, f'{name}.xml')
    return path_file_gth
