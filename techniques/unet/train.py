"""Implementation of UNET Training"""
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from techniques.unet.model import UNET
from techniques.unet.net_utils import (
    #     load_checkpoint,
    #     save_checkpoint,
    get_loaders,
    #     check_accuracy,
    #     save_predictions_as_imgs
)

from utils.definitions import (
    DTS_TRAIN_PATH,
    DTS_VALIDATION_PATH,
    DTS_TEST_PATH,
    DEPTH
)

from preprocessing.Dataset import T

# Hyperparameter
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMG_HEIGHT = 512
IMG_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = False


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix()


def main():
    model = UNET(in_channels=DEPTH, out_channels=DEPTH).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()  # not binary
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        train_path=DTS_TRAIN_PATH,
        val_path=DTS_VALIDATION_PATH,
        batch_size=BATCH_SIZE,
        train_transforms=T,
        val_transforms=T,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)


if __name__ == '__main__':
    main()
