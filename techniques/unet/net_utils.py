"""Define utility for network"""

import os
import torch
import torchvision
from preprocessing.Dataset import VirusDataset
from torch.utils.data import DataLoader
from utils.definitions import (
    DTS_TEST_PATH,
    DTS_TRAIN_PATH,
    DTS_VALIDATION_PATH,
    DEPTH
)


class CheckpointSaver:
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.loss_value = None

    def save_checkpoint(self, loss_value: float, dict_model, dict_opt):
        """Save Checkpoints

        :param loss_value:
        :param dict_model:
        :param dict_opt:
        :return:
        """
        if self.loss_value is None: self.loss_value = loss_value
        if self.loss_value > loss_value:
            print("=> Saving checkpoint")
            checkpoint = {"state_dict": dict_model, "optimizer": dict_opt}
            torch.save(checkpoint, self.file_name)
            self.loss_value = loss_value


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
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

            # TODO serve qualcosa per verificare che le particelle trovate siano quelle della gth
            # TODO la dice score forse non fa per noi

        print(f'val loss: {val_loss / len(loader)}')
        print(f"Dice score: {dice_score / len(loader)}")

    # switch to train mode
    model.train()
    return val_loss / len(loader)


def save_preds_as_imgs(loader, model, folder, device="cuda"):  # FIXME
    model.eval()
    for idx, data in enumerate(loader):
        x = data['img'].to(device)
        y = data['target'].to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
        for batch_id in range(x.shape[0]):
            my_preds = preds[batch_id, :, :, :].unsqueeze(1)
            my_target = y[batch_id, :, :, :].unsqueeze(1)
            torchvision.utils.save_image(my_preds, os.path.join(folder, f'pred_{idx}.png'), nrow=5)
            torchvision.utils.save_image(my_target, f"{folder}{idx}.png", nrow=5)
    model.train()
