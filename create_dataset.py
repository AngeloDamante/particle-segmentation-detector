import logging
import os
import shutil
import argparse
import json

import numpy as np

from utils.Types import SNR, Density
from preprocessing.analyser import comparison_seg
from utils.logger import configure_logger
from utils.compute_path import get_data_path, compute_name
from preprocessing.creation import make_raw_data
from utils.definitions import (
    DTS_DIR,
    DTS_TRAIN_PATH,
    DTS_TEST_PATH,
    DTS_VALIDATION_PATH,
    DTS_RAW_PATH,
    DTS_ANALYZE_PATH,
    TIME_INTERVAL,
    CONFIG_DIR,
    mapDensity,
    mapSNR,
    DEPTH
)

configure_logger(logging.INFO)


def split_dataset(snr: SNR, density: Density, p_train: int = 80, is_test=True):
    """Split dataset into training, testing, validation directories

        Input percentage  is the value for training. The rest of the percentage
        is divided equally between testing and validation.

    :param is_test:
    :param p_train: percentage of splitting for training
    :param density:
    :param snr:
    :return:
    """
    p_test = 0
    if not is_test:
        p_val = TIME_INTERVAL - p_train
    else:
        p_test = (TIME_INTERVAL - p_train) // 2
        p_val = p_test
    os.makedirs(DTS_TRAIN_PATH, exist_ok=True)
    os.makedirs(DTS_TEST_PATH, exist_ok=True)
    os.makedirs(DTS_VALIDATION_PATH, exist_ok=True)

    for time in range(p_train):
        path = get_data_path(snr, density, t=time, is_npz=True, root=DTS_RAW_PATH)
        shutil.copy2(path, DTS_TRAIN_PATH)

    for time in range(p_test):
        t = p_train + time
        path = get_data_path(snr, density, t=t, is_npz=True, root=DTS_RAW_PATH)
        shutil.copy2(path, DTS_TEST_PATH)

    for time in range(p_val):
        t = p_train + p_test + time
        path = get_data_path(snr, density, t=t, is_npz=True, root=DTS_RAW_PATH)
        shutil.copy2(path, DTS_VALIDATION_PATH)


def json_parser(path_file: str):
    """Json parser to take settings

    :param path_file:
    :return:
    """
    dataset, training_set = [], []
    train_settings = {}
    with open(path_file, 'r') as json_file:
        json_data = json.load(json_file)
        if 'dataset' in json_data.keys(): dataset = json_data['dataset']
        if 'training_set' in json_data.keys(): training_set = json_data['training_set']
        if 'train_settings' in json_data.keys(): train_settings = json_data['train_settings']
    return dataset, training_set, train_settings


def main():
    # parsing
    parser = argparse.ArgumentParser()
    # default_config_file = os.path.join(CONFIG_DIR, 'make_raw_dataset.json')
    # default_config_file = os.path.join(CONFIG_DIR, 'first_train.json')
    # default_config_file = os.path.join(CONFIG_DIR, 'high_train.json')
    default_config_file = os.path.join(CONFIG_DIR, 'make_high_dataset.json')
    parser.add_argument("-C", "--config", type=str, default=default_config_file, help=f"name of config in {CONFIG_DIR}")
    args = parser.parse_args()
    config_file = args.config

    # integrity check
    if not config_file.endswith('json'):
        raise TypeError('config file must be a json file')
    if not os.path.isfile(config_file):
        raise ValueError('file not found')

    # take setting from input json
    dataset, training_set, train_settings = json_parser(os.path.join(CONFIG_DIR, config_file))

    # make datset phase
    if dataset:
        logging.info('[ DATASET RAW CREATION ]')
        shutil.rmtree(DTS_RAW_PATH, ignore_errors=True)
        for dts in dataset:
            snr = mapSNR[dts['snr']]
            density = mapDensity[dts['density']]
            logging.info(f'[ RAW ]: snr = {snr.value},  density = {density.value}')
            logging.info(f"[ MAKE SEGMENTATION DATASET WITH kernel = {dts['kernel']} and sigma = {dts['sigma']}]")
            make_raw_data(snr, density, dts['kernel'], dts['sigma'], dest_dir=DTS_RAW_PATH)

            if dts['save_slices']:
                logging.info('[ SAVING SLICES ]')
                for t in range(DEPTH):
                    data = np.load(get_data_path(snr, density, t=t, is_npz=True, root=DTS_RAW_PATH))
                    slices_dir = os.path.join(DTS_ANALYZE_PATH, compute_name(snr, density, t))
                    comparison_seg(data['img'], data['target'], slices_dir)

    # make training dir
    if training_set and train_settings:
        logging.info('[ TRAINING DATASET CREATION ]')
        shutil.rmtree(DTS_TRAIN_PATH, ignore_errors=True)
        shutil.rmtree(DTS_TEST_PATH, ignore_errors=True)
        shutil.rmtree(DTS_VALIDATION_PATH, ignore_errors=True)
        for tr in training_set:
            snr = mapSNR[tr['snr']]
            density = mapDensity[tr['density']]
            logging.info(f'[ TRAIN ]: snr = {snr.value},  density = {density.value}')
            split_dataset(snr, density, train_settings['p_split'], is_test=train_settings['is_test'])


if __name__ == '__main__':
    main()
