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


# def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
#     print("=> Saving checkpoint")
#     torch.save(state, filename)

#
# def load_checkpoint(checkpoint, model):
#     print("=> Loading checkpoint")
#     model.load_state_dict(checkpoint["state_dict"])


def check_accuracy(loader, model, device="cuda"):
    """Check Accuracy

    :param loader:
    :param model:
    :param device:
    :return:
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        for data in loader:
            x = data['img'].to(device)
            y = data['target'].to(device)
            # x = x.to(device)
            # y = y.to(device).unsqueeze(1)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            num_correct += (preds == y.round()).sum()  # TODO
            num_pixels += torch.numel(preds)

            # IoU equivalent for segmentation
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with accuracy {num_correct / num_pixels * 100:.2f}")
    print(f"Dice score: {dice_score / len(loader)}")

    # switch to train mode
    model.train()