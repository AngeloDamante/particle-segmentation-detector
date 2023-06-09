"""Functions to train and validate models"""

import os
import torch
from tqdm import tqdm
import numpy as np
from utils.definitions import DEVICE
from preprocessing.analyser import comparison_pred
from utils.Types import SNR, Density
from utils.compute_path import get_data_path, compute_name
from utils.definitions import DTS_RAW_PATH


def train_fn(loader, model, optimizer, loss_fn, scaler):
    """Function for training

    :param loader:
    :param model:
    :param optimizer:
    :param loss_fn:
    :param scaler:
    :return:
    """
    pbar = tqdm(enumerate(loader), total=len(loader))
    for batch_idx, item in pbar:
        data = item['img'].to(device=DEVICE)
        targets = item['target'].float().to(device=DEVICE)

        optimizer.zero_grad()

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        pbar.set_description(f"Processing {batch_idx + 1}/{len(loader)}, train_loss = {loss.item()}")


def val_fn(loader, model, loss, device="cuda") -> float:
    """ Validate Function

    :param loader:
    :param model:
    :param loss:
    :param device:
    :return:
    """
    dice_score = 0.0
    val_loss = 0.0

    # switch to evaluation mode
    model.eval()

    with torch.no_grad(), torch.cuda.amp.autocast():
        for data in loader:
            x = data['img'].to(device)
            y = data['target'].to(device)

            preds = model(x)
            val_loss += loss(preds, y).item()
            preds = torch.sigmoid(preds)

            # IoU equivalent for segmentation
            # TODO serve qualcosa per verificare che le particelle trovate siano quelle della gth
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

        print(f'val loss: {val_loss / len(loader)}')
        print(f"Dice score: {dice_score / len(loader)}")

    # switch to train mode
    model.train()
    return val_loss / len(loader)


def inference(model: torch.nn.Module, snr: SNR, density: Density, t: int, save_dir: str):
    """To make inference about data

    :param model:
    :param snr:
    :param density:
    :param t:
    :param save_dir:
    :return:
    """
    data = np.load(get_data_path(snr, density, t, is_npz=True, root=DTS_RAW_PATH))
    x = data['img']
    x_orig = x.copy()
    y = data['target']
    x = np.divide(x, 255.0, dtype=np.float32)
    x = torch.from_numpy(x)
    x = torch.permute(x, (2, 0, 1)).unsqueeze(0)

    with torch.no_grad():
        preds = torch.sigmoid(model(x)).squeeze(0)
        y_hat = torch.permute(preds, (1, 2, 0)).numpy() * 255
        comparison_pred(x_orig, y, y_hat, os.path.join(save_dir, f'{compute_name(snr, density, t)}'))
