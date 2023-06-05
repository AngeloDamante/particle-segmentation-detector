"""Define utility for network"""

import os
import torch
import torchvision
from preprocessing.Dataset import VirusDataset
from torch.utils.data import DataLoader
from utils.definitions import (
    DTS_TEST_PATH,
    DTS_TRAIN_PATH,
    DTS_VALIDATION_PATH
)


def get_loaders(train_path, val_path, batch_size, train_transforms, val_transforms, num_workers, pin_memory):
    """Get Loaders

    :param train_path:
    :param val_path:
    :param batch_size:
    :param train_transforms:
    :param val_transforms:
    :param num_workers:
    :param pin_memory:
    :return:
    """
    # training
    train_dts = VirusDataset(train_path, train_transforms)
    train_dtl = DataLoader(train_dts,
                           shuffle=True,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           pin_memory=pin_memory)

    # validation
    val_dts = VirusDataset(val_path, val_transforms)
    val_dtl = DataLoader(val_dts,
                         shuffle=False,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         pin_memory=pin_memory)

    return train_dtl, val_dtl


def val_fn(loader, model, loss, device="cuda"):
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
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

            # TODO serve qualcosa per verificare che le particelle trovate siano quelle della gth
            # TODO la dice score forse non fa per noi

        print(f'val loss: {val_loss / len(loader)}')
        print(f"Dice score: {dice_score / len(loader)}")

    # switch to train mode
    model.train()


def save_preds_as_imgs(loader, model, folder, device="cuda"):
    model.eval()
    for idx, data in enumerate(loader):
        x = data['img'].to(device)
        y = data['target'].to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))  # FIXME (1,1,10)
        torchvision.utils.save_image(preds, os.path.join(folder, f'pred_{idx}.png'))
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
    model.train()
