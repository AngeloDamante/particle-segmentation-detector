"""Use model to do inference"""

import os
import torch.cuda
import torch.nn as nn
import logging
import argparse
from processing.functions import inference
from models.unet import UNET
from utils.logger import configure_logger
from utils.json_parser import json_parser_inference
from utils.definitions import (
    CONFIG_DIR,
    DEPTH,
    DEVICE,
    RESULTS_UNET_CHECKPOINT,
    RESULTS_UNET_IMAGES,
    TIME_INTERVAL,
    mapSNR,
    mapDensity
)

configure_logger(logging.INFO)


def main():
    # parsing
    parser = argparse.ArgumentParser()
    default_config_file = 'inference.json'
    parser.add_argument("-C", "--config", type=str, default=default_config_file, help=f"name of config in {CONFIG_DIR}")
    args = parser.parse_args()
    config_file = os.path.join(CONFIG_DIR, args.config)

    # extracting
    logging.info("[ EXTRACTING SETTINGS FROM JSON ]")
    model, params, dataset = json_parser_inference(config_file)
    logging.info(f"model = {model}, params = {params}")

    # model chosen
    net_model = nn.Module()
    img_dir = ""
    if model == "unet":
        net_model = UNET(in_channels=DEPTH, out_channels=DEPTH).to(DEVICE)
        checkpoint = torch.load(os.path.join(RESULTS_UNET_CHECKPOINT, params))
        net_model.load_state_dict(checkpoint['state_dict'])
        img_dir = RESULTS_UNET_IMAGES
    elif model == "vit":
        pass  # TODO

    # inference
    logging.info("[ INFERENCE STARTING ]")
    for dts in dataset:
        for t in range(TIME_INTERVAL):
            inference(net_model, mapSNR[dts['snr']], mapDensity[dts['density']], t, img_dir)
    logging.info("[ DONE ]")


if __name__ == '__main__':
    main()
