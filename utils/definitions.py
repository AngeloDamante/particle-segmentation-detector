""" Main Definitions for project """

import os
from pathlib import Path
from utils.Types import SNR, Density
import torch

# dataset values
HIGH, WIDTH, DEPTH = 512, 512, 10
SIZE = (HIGH, WIDTH, DEPTH)
TIME_INTERVAL = 100
KERNEL = 7
SIGMA = 1.5

# Default Hyperparameter
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8  # 16, 32, 64
NUM_EPOCHS = 200
NUM_WORKERS = 8
PIN_MEMORY = True
LOAD_MODEL = True

# main dirs
ROOT_DIR = Path(__file__).absolute().parent.parent
DTS_DIR = os.path.join(ROOT_DIR, "Dataset")
CONFIG_DIR = os.path.join(ROOT_DIR, "Config")

# default dirs (data provides from challenge)
DEFAULT_DATA_PATH = os.path.join(DTS_DIR, "Challenge", "Data")
DEFAULT_SLICES_PATH = os.path.join(DTS_DIR, "Challenge", "Slices")
DEFAULT_GTH_PATH = os.path.join(DTS_DIR, "Challenge", "ground_truth")

# dataset
DTS_RAW_PATH = os.path.join(DTS_DIR, "Raw")
DTS_TRAIN_PATH = os.path.join(DTS_DIR, "Train")
DTS_TEST_PATH = os.path.join(DTS_DIR, "Test")
DTS_VALIDATION_PATH = os.path.join(DTS_DIR, "Validation")
DTS_ANALYZE_PATH = os.path.join(DTS_DIR, "Analyze")

# net comparison
RESULTS_PATH = os.path.join(ROOT_DIR, "Results")
RESULTS_UNET_CHECKPOINT = os.path.join(RESULTS_PATH, "Unet", "Checkpoints")
RESULTS_UNET_IMAGES = os.path.join(RESULTS_PATH, "Unet", "Images")
RESULTS_VIT_CHECKPOINT = os.path.join(RESULTS_PATH, "Vit", "Checkpoints")
RESULTS_VIT_IMAGES = os.path.join(RESULTS_PATH, "Vit", "Images")

# dict to parsing
mapSNR = {'snr_1': SNR.TYPE_1, 'snr_2': SNR.TYPE_2, 'snr_4': SNR.TYPE_4, 'snr_7': SNR.TYPE_7}
mapDensity = {'density_low': Density.LOW, 'density_mid': Density.MID, 'density_high': Density.HIGH}
