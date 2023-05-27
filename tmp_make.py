from tqdm import tqdm
import logging
from preprocessing.creation import slices_to_npy, make_raw_data
from utils.definitions import mapSNR, mapDensity, DEFAULT_DATA_PATH, DEFAULT_SLICES_PATH, DTS_RAW_PATH
from utils.logger import configure_logger

configure_logger(logging.INFO)

# logging.info("make original DATA")
# for snr in tqdm(mapSNR.values()):
#     for density in mapDensity.values():
#         slices_to_npy(snr, density, DEFAULT_SLICES_PATH, DEFAULT_DATA_PATH)

logging.info("make RAW data")
for snr in mapSNR.values():
    for density in mapDensity.values():
        logging.info(f'snr = {snr.value}, density = {density.value}')
        make_raw_data(snr, density, kernel=3, sigma=0.5, dest_dir=DTS_RAW_PATH)