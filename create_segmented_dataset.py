"""Run this script to create segmented dataset"""

import logging
import os
import numpy as np
from preprocessing.analyser import comparison_seg
from utils.logger import configure_logger
from utils.compute_path import get_data_path, compute_name
from preprocessing.creation import make_raw_data
from utils.definitions import (
    DTS_RAW_PATH,
    DTS_ANALYZE_PATH,
    mapDensity,
    mapSNR,
    DEPTH,
    KERNEL,
    SIGMA
)

configure_logger(logging.INFO)


def main():
    logging.info('[ CREATING RAW DATASET ]')
    for snr in mapSNR.values():
        for density in mapDensity.values():
            logging.info(f'processing: snr = {snr.value}, density = {density.value}')
            make_raw_data(snr, density, KERNEL, SIGMA, dest_dir=DTS_RAW_PATH)
    logging.info('[ DONE ]')

    logging.info(f'[ SAVING SLICES IN  {DTS_ANALYZE_PATH}]')
    for snr in mapSNR.values():
        for density in mapDensity.values():
            for t in range(DEPTH):
                data = np.load(get_data_path(snr, density, t=t, is_npz=True, root=DTS_RAW_PATH))
                slices_dir = os.path.join(DTS_ANALYZE_PATH, compute_name(snr, density, t))
                comparison_seg(data['img'], data['target'], save_dir=slices_dir)
    logging.info('[ DONE ]')


if __name__ == '__main__':
    main()
