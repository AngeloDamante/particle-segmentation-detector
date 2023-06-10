"""Functions to train and validate models"""

import os
import torch
from tqdm import tqdm
import numpy as np
from typing import Tuple
from utils.definitions import DEVICE
from preprocessing.analyser import comparison_pred
from utils.Types import SNR, Density
from utils.compute_path import get_data_path, compute_name
from utils.definitions import DTS_RAW_PATH


def compute_dice_score(target: torch.Tensor, preds: torch.Tensor, threshold: float) -> float:
    """Compute dice score for binary mask

        (2*TP) / (2*TP - FP - FN)

    :param target:
    :param preds:
    :param threshold:
    :return:
    """
    preds_bin = (preds > threshold).type(torch.uint8)
    y_bin = (target > threshold).type(torch.uint8)
    return (2 * (preds_bin * y_bin).sum()) / ((preds_bin + y_bin).sum() + 1e-8)


def train_fn(loader, model, optimizer, loss_fn, scaler) -> float:
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
        pbar.set_description(f"Processing {batch_idx + 1}/{len(loader)}, train_loss = {loss.item():.5f}")
    return loss


def val_fn(loader, model, loss, device="cuda") -> Tuple[float, float]:
    """ Validate Function

    :param loader:
    :param model:
    :param loss:
    :param device:
    :return:
    """
    dice_score = 0.0
    val_loss = 0.0
    threshold = 0.3

    model.eval() # switch to evaluation mode
    with torch.no_grad(), torch.cuda.amp.autocast():
        for data in loader:
            x = data['img'].to(device)
            y = data['target'].to(device)

            preds = model(x)
            val_loss += loss(preds, y).item()
            preds = torch.sigmoid(preds)

            dice_score += compute_dice_score(y, preds, threshold)

        print(f'val loss: {(val_loss / len(loader)):.5f}')
        print(f"Dice score: {(dice_score / len(loader)):.5f}")

    model.train() # switch to train mode
    return val_loss / len(loader), (dice_score / len(loader))


def inference_fn(model: torch.nn.Module, snr: SNR, density: Density, t: int, save_dir: str):
    """To make inference about data, in other words, use model

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
    x = torch.from_numpy(x).to(DEVICE)
    x = torch.permute(x, (2, 0, 1)).unsqueeze(0)

    with torch.no_grad():
        preds = torch.sigmoid(model(x))
        preds = preds.to('cpu')
        y_hat = torch.permute(preds.squeeze(0), (1, 2, 0)).numpy() * 255
        comparison_pred(x_orig, y, y_hat, os.path.join(save_dir, f'{compute_name(snr, density, t)}'))
