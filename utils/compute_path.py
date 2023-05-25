"""Define method to compute paths for extraction phases

Dataset
├── Challenge
│   ├── ground_truth -> get_gth_xml_path()
│   │   ├── file.xml
│   │   └── ...
│   ├── VIRUS
│   │   ├── VIRUS_snr_1_density_high -> get_slice_path()
│   │   └── ...
│   └── VIRUS_npy
│       ├── snr_1_density_high -> get_data_path()
│       └── ...
└── Segmaps
│    ├── Data
│    │   ├── snr_1_density_high -> get_seg_data_path()
│    │   └── ...
│    └── Images
│        ├── snr_1_density_high -> get_seg_slice_path()
│        └── ...
└── DTS
    ├── snr_1_density_high -> get_npz_data_path()
    └── ...
"""

import os
from utils.Types import SNR, Density
from utils.definitions import DTS_VIRUS, DTS_GTH, DTS_SEG_DATA, DTS_SEG_IMG, DTS_DATA, DTS_COMPLETE


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
    """Compute path for npy seg data

    :param snr:
    :param density:
    :param t:
    :return:
    """
    seg_data_path = os.path.join(DTS_SEG_DATA, f'{snr.value}_{density.value}', f't_{str(t).zfill(3)}.npy')
    return seg_data_path


def get_seg_slice_path(snr: SNR, density: Density, t: int, z: int) -> str:
    """Compute segmented Slice path

    :param snr:
    :param density:
    :param t:
    :param z:
    :return:
    """
    slice_path = os.path.join(DTS_SEG_IMG, f'{snr.value}_{density.value}',
                              f't_{str(t).zfill(3)}_z_{str(z).zfill(2)}.tiff')
    return slice_path


def get_data_path(snr: SNR, density: Density, t: int) -> str:
    """Compute path for npy data

    :param snr:
    :param density:
    :param t:
    :return:
    """
    data_path = os.path.join(DTS_DATA, f'{snr.value}_{density.value}', f't_{str(t).zfill(3)}.npy')
    return data_path

def get_npz_data_path(snr:SNR, density:Density, t: int) -> str:
    """Compute npz data path

    :param snr:
    :param density:
    :param t:
    :return:
    """
    npz_data_path = os.path.join(DTS_COMPLETE, f'{snr.value}_{density.value}', f't_{str(t).zfill(3)}.npz')
    return npz_data_path