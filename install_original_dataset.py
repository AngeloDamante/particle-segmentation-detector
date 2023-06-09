"""Run this script to download and install original Dataset provided by Challenge"""

import os
import logging
import tarfile
from preprocessing.creation import slices_to_npy
from utils.logger import configure_logger
from utils.definitions import (
    mapSNR,
    mapDensity,
    DEFAULT_SLICES_PATH,
    DEFAULT_DATA_PATH,
    DTS_DIR
)

configure_logger(log_lvl=logging.INFO)

ARCHIVE_NAME = 'Challenge_dts.tar.xz'
ARCHIVE_PATH = os.path.join(DTS_DIR, ARCHIVE_NAME)
ARCHIVE_LINK = 'https://github.com/AngeloDamante/particle-ViT-segmentation/releases/download/v1.0/Challenge_dts.tar.xz'

# installation phase
logging.info("Install Original Dataset")
os.system(f'curl -Lk {ARCHIVE_LINK} > {ARCHIVE_PATH}')

# extraction phase
logging.info(f"Extracting Original Dataset in {DTS_DIR}")
with tarfile.open(ARCHIVE_PATH, 'r') as f:
    f.extractall(DTS_DIR, numeric_owner=True)
os.remove(ARCHIVE_PATH)

# make npy phase
logging.info(f"Make Raw Data from original dataset in {DEFAULT_DATA_PATH}")
for snr in mapSNR.values():
    for density in mapDensity.values():
        logging.info(f'processing: snr = {snr.value}, density = {density.value}')
        slices_to_npy(snr, density, DEFAULT_SLICES_PATH, DEFAULT_DATA_PATH)
logging.info('[DONE]')
