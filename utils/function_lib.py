import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np


def train_loop(dataloader, model, loss_fn, optimizer, scheduler, device):
    datalen = len(dataloader.dataset)
    model.train()
    for batch, (X, Y) in enumerate(dataloader):
        optimizer.zero_grad(set_to_none=True)
        
        # Compute prediction and loss
        pred = model(X.to(device)).squeeze()
        # Y = Y.type(torch.long)
        loss = loss_fn(pred, Y.to(device))

        # Backpropagation
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(X)
        if batch % 100 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{datalen:>5d}]")


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        imagesize = 0
        for X, Y in dataloader:
            imagesize = torch.numel(Y)
            pred = model(X.to(device)).squeeze()
            # Y = Y.type(torch.long)
            Y = Y.to(device)
            test_loss += loss_fn(pred, Y).item()

    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss

def deprocess_img(image, mask, out_size):
    trans_image = transforms.Compose([
        transforms.Resize(out_size, transforms.InterpolationMode.BILINEAR, antialias=True)
    ])
    trans_mask = transforms.Compose([
        transforms.Resize(out_size, transforms.InterpolationMode.NEAREST)
    ])
    image = trans_image(image)
    image = image * 0.2 + 0.2
    
    mask = trans_mask(mask)
    return image, mask
    

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=min(len(imgs), 4), nrows=(len(imgs)-1)//4+1, squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[i//4, i %4].imshow(np.asarray(img))
        axs[i//4, i %4].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def show_cpu(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=min(len(imgs), 4), nrows=(len(imgs)-1)//4+1, squeeze=False)
    for i, img in enumerate(imgs):
        img = F.to_pil_image(img)
        axs[i//4, i %4].imshow(np.asarray(img))
        axs[i//4, i %4].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
