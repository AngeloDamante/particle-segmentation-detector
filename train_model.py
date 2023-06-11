"""Training model"""

import os
import shutil
import argparse
import logging
from tqdm import tqdm
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models.unet import UNET
from models.vit import SegFormer
from preprocessing.Dataset import train_transform, val_transform
from processing.utilities import get_loaders, CheckpointSaver
from processing.functions import train_fn, val_fn, inference_fn
from utils.logger import configure_logger
from utils.Types import SNR, Density
from utils.compute_path import get_data_path
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
    EXPERIMENTS_PATH
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


def json_parser(path_file: str):
    """Json parser to take settings for training

    :param path_file:
    :return:
    """
    if not os.path.exists(path_file): return {}, {}, {}
    with open(path_file, 'r') as json_file:
        json_data = json.load(json_file)
        settings = json_data['settings']
        training_dts = json_data['training_dts']
        hyperparameters = json_data['hyperparameters']
    return settings, training_dts, hyperparameters


def main():
    # parsing
    parser = argparse.ArgumentParser()
    default_config_file = os.path.join(CONFIG_DIR, 'train_unet.json')
    parser.add_argument("-C", "--config", type=str, default=default_config_file, help="absolute path of config file")
    args = parser.parse_args()
    config_file = args.config

    # extracting settings
    logging.info('[ EXTRACTING JSON DATA ]')
    settings, training_dts, hyperparameters = json_parser(config_file)
    train_name = settings['training_name']
    p_split = settings['p_split']
    experiment_dir = os.path.join(EXPERIMENTS_PATH, train_name)

    # create writer for tensorboard
    logging.info(f"[ EXPERIMENT {train_name} SAVING ON {experiment_dir} ]")
    os.makedirs(experiment_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=experiment_dir, filename_suffix=train_name)
    shutil.copy2(config_file, os.path.join(experiment_dir, f'{train_name}.json'))

    # make dataset for training
    logging.info('[ TRAINING DATASET CREATION ]')
    shutil.rmtree(DTS_TRAIN_PATH, ignore_errors=True)
    shutil.rmtree(DTS_TEST_PATH, ignore_errors=True)
    shutil.rmtree(DTS_VALIDATION_PATH, ignore_errors=True)
    for dts in training_dts:
        snr = mapSNR[dts['snr']]
        density = mapDensity[dts['density']]
        logging.info(f'[ DATASET: snr = {snr.value},  density = {density.value} ]')
        split_dataset(snr, density, p_split, is_test=True)

    # chose model
    logging.info(f'[ CHOSEN MODEL = {settings["model"]} ]')
    model = nn.Module()
    if settings['model'] == 'unet':
        model = UNET(in_channels=DEPTH, out_channels=DEPTH).to(DEVICE)
    elif settings['model'] == 'vit':
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
    checkpoint_saver = CheckpointSaver(file_name=os.path.join(experiment_dir, f'{train_name}.pth.tar'))
    if hyperparameters['load_model'] and checkpoint_saver.is_present():
        logging.info(f'[ LOAD MODEL {checkpoint_saver.get_file_name()}]')
        print("=> Loading checkpoint")
        checkpoint = torch.load(checkpoint_saver.get_file_name())
        model.load_state_dict(checkpoint["state_dict"])
        val_fn(val_loader, model, loss_fn, device=DEVICE)

    # start training
    logging.info(f'[ TRAINING STARTED with {DEVICE} device ]')
    num_epochs = hyperparameters['num_epochs']
    for epoch in range(num_epochs):
        print(f'----------------------------------------> epoch = {epoch + 1}/{num_epochs}')
        # train
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        writer.add_scalar("Loss/train", train_loss, epoch)

        # validation
        val_loss, dice_score = val_fn(val_loader, model, loss_fn, device=DEVICE)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Dice_score", dice_score, epoch)

        # save model
        checkpoint_saver.save_checkpoint(val_loss, model.state_dict(), optimizer.state_dict())

    # tensorboard
    writer.close()
    logging.info('[ TRAINING FINISHED ]')

    # inference - testing
    logging.info("[ TESTING TRAINED MODEL ]")
    img_dir = os.path.join(experiment_dir, 'Images')
    checkpoint = torch.load(checkpoint_saver.get_file_name())
    model.load_state_dict(checkpoint["state_dict"])
    for dts in training_dts:
        for t in tqdm(range(settings['p_split'], TIME_INTERVAL)):
            inference_fn(model, mapSNR[dts['snr']], mapDensity[dts['density']], t=t, save_dir=img_dir)
    logging.info("[ DONE ]")


if __name__ == '__main__':
    main()
