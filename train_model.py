"""Training model"""

import os
import shutil
import argparse
import logging
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from models.unet import UNET
from models.vit import SegFormer
from preprocessing.Dataset import train_transform, val_transform
from processing.utilities import get_loaders, CheckpointSaver
from processing.functions import train_fn, val_fn
from utils.logger import configure_logger
from utils.Types import SNR, Density
from utils.compute_path import get_data_path
from utils.json_parser import json_parser_train
from utils.definitions import (
    TIME_INTERVAL,
    DTS_TRAIN_PATH,
    DTS_TEST_PATH,
    DTS_VALIDATION_PATH,
    DTS_RAW_PATH,
    CONFIG_DIR,
    DEPTH,
    DEVICE,
    mapSNR,
    mapDensity,
    RESULTS_UNET_CHECKPOINT,
    RESULTS_VIT_CHECKPOINT
)

configure_logger(logging.INFO)


def split_dataset(snr: SNR, density: Density, p_train: int = 80, is_test=False):
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


def main():
    # parsing
    parser = argparse.ArgumentParser()
    # default_config_file = 'snr_7_density_low_train.json'
    default_config_file = 'snr_7_density_low_train_vit.json'
    parser.add_argument("-C", "--config", type=str, default=default_config_file, help=f"name of config in {CONFIG_DIR}")
    args = parser.parse_args()
    config_file = os.path.join(CONFIG_DIR, args.config)

    # extracting settings
    logging.info('[ EXTRACTING JSON SETTINGS ]')
    settings, training_dts, hyperparameters = json_parser_train(config_file)
    train_name = settings['training_name']
    p_split = settings['p_split']

    # make dataset for training
    logging.info('[ TRAINING DATASET CREATION ]')
    shutil.rmtree(DTS_TRAIN_PATH, ignore_errors=True)
    shutil.rmtree(DTS_TEST_PATH, ignore_errors=True)
    shutil.rmtree(DTS_VALIDATION_PATH, ignore_errors=True)
    for dts in training_dts:
        snr = mapSNR[dts['snr']]
        density = mapDensity[dts['density']]
        logging.info(f'[ TRAIN ]: snr = {snr.value},  density = {density.value}')
        split_dataset(snr, density, p_split)

    # chose model
    logging.info(f'[ CHOSEN MODEL = {settings["model"]} ]')
    model = nn.Module()
    checkpoint_saver = CheckpointSaver()
    if settings['model'] == 'unet':
        model = UNET(in_channels=DEPTH, out_channels=DEPTH).to(DEVICE)
        os.makedirs(RESULTS_UNET_CHECKPOINT, exist_ok=True)
        checkpoint_saver.set_file_name(os.path.join(RESULTS_UNET_CHECKPOINT, f'{train_name}.pth.tar'))
    elif settings['model'] == 'vit':
        os.makedirs(RESULTS_VIT_CHECKPOINT, exist_ok=True)
        checkpoint_saver.set_file_name(os.path.join(RESULTS_VIT_CHECKPOINT, f'{train_name}.pth.tar'))
        model = SegFormer(
            in_channels=10,
            widths=[64, 128, 256, 512],
            depths=[3, 4, 6, 3],
            all_num_heads=[1, 2, 4, 8],
            patch_size=[7, 3, 3, 3],
            overlap_sizes=[4, 2, 2, 2],
            reduction_ratios=[8, 4, 2, 1],
            mlp_expansions=[4, 4, 4, 4],
            decoder_channels=256,
            scale_factors=[8, 4, 2, 1],
            out_channels=10
        ).to(DEVICE)

    # define all elements for training
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters['lr'])
    scaler = torch.cuda.amp.GradScaler()

    # import loader for train and validation
    train_loader, val_loader = get_loaders(
        train_path=DTS_TRAIN_PATH,
        val_path=DTS_VALIDATION_PATH,
        batch_size=hyperparameters['batch'],
        train_transforms=train_transform,
        val_transforms=val_transform,
        num_workers=hyperparameters['num_workers'],
        pin_memory=hyperparameters['pin_memory']
    )

    # load model only if is present
    if hyperparameters['load_model'] and checkpoint_saver.is_present():
        logging.info(f'[ LOAD MODEL {checkpoint_saver.get_file_name()}]')
        print("=> Loading checkpoint")
        checkpoint = torch.load(checkpoint_saver.get_file_name())
        model.load_state_dict(checkpoint["state_dict"])
        val_fn(val_loader, model, loss_fn, device=DEVICE)

    # start training
    logging.info('[ TRAINING STARTED ]')
    num_epochs = hyperparameters['num_epochs']
    for epoch in range(num_epochs):
        print(f'----------------------------------------> epoch = {epoch + 1}/{num_epochs}')
        # train
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # validation
        val_loss = val_fn(val_loader, model, loss_fn, device=DEVICE)

        # save model
        checkpoint_saver.save_checkpoint(val_loss, model.state_dict(), optimizer.state_dict())


if __name__ == '__main__':
    main()
