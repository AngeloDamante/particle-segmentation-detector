"""Use model to make segmentation map from input image"""

import os
import numpy as np
import torch.cuda
from preprocessing.analyser import comparison
from utils.Types import SNR, Density
from utils.compute_path import get_data_path, compute_name
from techniques.unet.model import UNET
from utils.definitions import (
    DEPTH,
    UNET_RESULTS_IMAGES,
    UNET_RESULTS_CHECKPOINTS,
    IMG_HEIGHT,
    IMG_WIDTH,
    DTS_RAW_PATH
)


def use_net(model: torch.nn.Module, snr: SNR, density: Density, t: int):
    # TODO use ToTensor
    # TODO normalize!
    data = np.load(get_data_path(snr, density, t, is_npz=True, root=DTS_RAW_PATH))
    x = data['img']
    y = data['target']
    x = np.divide(x, 255.0, dtype=np.float32)
    x = torch.from_numpy(x)
    x = torch.permute(x, (2, 0, 1)).unsqueeze(0)

    with torch.no_grad():
        preds = torch.sigmoid(model(x)).squeeze(0)
        y_hat = torch.permute(preds, (1, 2, 0)).numpy() * 255

        # TODO need something to save slices
        comparison(y_hat, y, os.path.join(UNET_RESULTS_IMAGES, f'{compute_name(snr, density, t)}'))
        # save_slices(preds, UNET_RESULTS_IMAGES, f'{compute_name(snr, density, t)}.png')


def main():
    # image settings
    snr, density, t = SNR.TYPE_7, Density.LOW, 0

    # model settings
    checkpoint_name = 'day_04_06_2023_time_22_58_05.pth.tar'
    model = UNET(in_channels=DEPTH, out_channels=DEPTH)
    data_model = torch.load(os.path.join(UNET_RESULTS_CHECKPOINTS, checkpoint_name))
    model.load_state_dict(data_model['state_dict'])

    # model processing
    use_net(model, snr, density, t)


if __name__ == '__main__':
    main()
