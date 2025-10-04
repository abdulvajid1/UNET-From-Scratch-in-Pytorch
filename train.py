import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import UNET
from dataset import get_dataloader


def eval(model, optimizer)

def train(model, optimizer):

    # TODO : Add autocast bfloat training

    progress_bar = tqdm(dataloader)
    model.train()
    for inputs, targets in progress_bar:
        _ ,loss = model(inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

def main():
    model = UNET()
    loader = get_dataloader()
    optimizer = optim.Adam(model.parameters())
    num_epoches = 100

    for epoch in num_epoches:
        train(model, optimizer, loader)
