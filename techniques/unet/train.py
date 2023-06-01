"""Implementation of UNET Training"""
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from techniques.unet.model import UNET
from techniques.unet.net_utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs
)

from utils.definitions import (
    DTS_TRAIN_PATH,
    DTS_VALIDATION_PATH,
    DTS_TEST_PATH,
    DTS_ANALYZE_PATH,
    DEPTH,
    LEARNING_RATE,
    DEVICE,
    BATCH_SIZE,
    NUM_EPOCHS,
    NUM_WORKERS,
    IMG_HEIGHT,
    IMG_WIDTH,
    PIN_MEMORY,
    LOAD_MODEL
)

from preprocessing.Dataset import T


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

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)  # TODO filename
    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    for _ in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(val_loader, model, folder=DTS_ANALYZE_PATH, device=DEVICE)


if __name__ == '__main__':
    main()
