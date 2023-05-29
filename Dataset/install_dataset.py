import os
import logging
import platform
from utils.definitions import mapSNR, mapDensity, DEFAULT_SLICES_PATH, DEFAULT_DATA_PATH, DTS_DIR
from preprocessing.creation import slices_to_npy
from utils.logger import configure_logger

ARCHIVE_NAME = 'Challenge_dts.tar.xz'
ARCHIVE_PATH = os.path.join(DTS_DIR, ARCHIVE_NAME)
ARCHIVE_LINK = 'https://github.com/AngeloDamante/particle-ViT-segmentation/releases/download/v1.0/Challenge_dts.tar.xz'

configure_logger(logging.INFO)
logging.info(f"This script is running on {platform.system()}")

# install phase
logging.info("Install Original Dataset")
os.system(f'curl -Lk {ARCHIVE_LINK} > {ARCHIVE_PATH}')
os.system(f'tar -xJv -f {ARCHIVE_PATH} --directory {DTS_DIR}')

# clean phase
if platform.system() == 'Windows':
    os.system(f'del {ARCHIVE_PATH}')
else:
    os.system(f'rm {ARCHIVE_PATH}')

# make npy
logging.info(f"Make Data from original dataset in {DEFAULT_DATA_PATH}")
for snr in mapSNR.values():
    for density in mapDensity.values():
        logging.info(f'processing: snr = {snr.value}, density = {density.value}')
        slices_to_npy(snr, density, DEFAULT_SLICES_PATH, DEFAULT_DATA_PATH)

logging.info('[DONE]')