from pickle import TRUE
import time
from sympy import Limit
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import UNET
from dataset import ImgDataset
import albumentations as A
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2


LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.mps.is_available() else 'cpu')
BATCH_SIZE = 8
NUM_EPOCHS = 100
NUM_WORKERS = 4
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_DATA_PATH = 'data/train'
VAL_DATA_PATH = 'data/val'
WEIGHT_DECAY = 0.001

def train(model, train_loader, val_loader, optimizer):
    loop = tqdm(train_loader, dynamic_ncols=True)

    for inputs, targets in loop:
        model.train()

        # torch.cuda.synchronize()
        # start_time = time.time()
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        with torch.amp.autocast(DEVICE, dtype=torch.bfloat16): # TODO: 
            _, loss = model(inputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end_time = time.time()
        torch.cuda.synchronize()
        # dt = (end_time - start_time)*1000
        loop.set_postfix(loss=f"{loss.item(): .6f}")


def main():
    torch.set_float32_matmul_precision('high')
    model = UNET().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.Rotate(limit=10, p=0.7),
        A.HorizontalFlip(),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ], additional_targets={'image1': 'image'})

    val_transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

    train_dataset = ImgDataset(TRAIN_DATA_PATH, transform=train_transform)
    val_dataset = ImgDataset(VAL_DATA_PATH, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # # TODO: Compile
    # model.compile()

    for epoch in range(NUM_EPOCHS):
        train(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer)


if __name__ == '__main__':
    main()