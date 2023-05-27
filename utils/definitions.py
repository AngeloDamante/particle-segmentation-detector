""" Main Definitions for project """

import os
from pathlib import Path
from utils.Types import SNR, Density

# sequence values
HIGH, WIDTH, DEPTH = 512, 512, 10
SIZE = (HIGH, WIDTH, DEPTH)
TIME_INTERVAL = 100

# main dirs
ROOT_DIR = Path(__file__).absolute().parent.parent
DTS_DIR = os.path.join(ROOT_DIR, "Dataset")

# default dirs (data provides from challenge)
DEFAULT_DATA_PATH = os.path.join(DTS_DIR, "Challenge", "Data")
DEFAULT_SLICES_PATH = os.path.join(DTS_DIR, "Challenge", "Slices")
DEFAULT_GTH_PATH = os.path.join(DTS_DIR, "Challenge", "ground_truth")

# dataset
DTS_RAW_PATH = os.path.join(DTS_DIR, "Raw")
DTS_TRAIN_PATH = os.path.join(DTS_DIR, "Train")
DTS_TEST_PATH = os.path.join(DTS_DIR, "Test")
DTS_VALIDATION_PATH = os.path.join(DTS_DIR, "Validation")

# dict to parsing
mapSNR = {'snr_1': SNR.TYPE_1, 'snr_2': SNR.TYPE_2, 'snr_4': SNR.TYPE_4, 'snr_7': SNR.TYPE_7}
mapDensity = {'density_low': Density.LOW, 'density_mid': Density.MID, 'density_high': Density.HIGH}
