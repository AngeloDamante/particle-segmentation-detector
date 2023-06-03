import os
import numpy as np
import torch.cuda
from preprocessing.analyser import save_slices
from utils.Types import SNR, Density
from utils.compute_path import get_data_path, compute_name
from techniques.unet.model import UNET
from utils.definitions import (
    DEPTH,
    UNET_RESULTS_IMAGES,
    UNET_RESULTS_CHECKPOINTS,
    IMG_HEIGHT,
    IMG_WIDTH
)


def test_fn(model: torch.nn.Module, snr: SNR, density: Density, t: int):
    model.eval()
    img = torch.from_numpy(np.load(get_data_path(snr, density, t))).float()
    img = torch.reshape(img, (DEPTH, IMG_HEIGHT, IMG_WIDTH))
    img = img.unsqueeze(0)

    with torch.no_grad():
        preds = torch.sigmoid(model(img))
        preds = (preds > 0.5).float()
        save_slices(preds, UNET_RESULTS_IMAGES, f'{compute_name(snr, density, t)}.png')


def main():
    # image settings
    snr = SNR.TYPE_7
    density = Density.LOW
    t = 0

    # model settings
    checkpoint_name = 'day_03_06_2023_time_18_49_30.pth.tar'
    model = UNET(in_channels=DEPTH, out_channels=DEPTH)
    data_model = torch.load(os.path.join(UNET_RESULTS_CHECKPOINTS, checkpoint_name))
    model.load_state_dict(data_model['state_dict'])

    test_fn(model, snr, density, t)


if __name__ == '__main__':
    main()
