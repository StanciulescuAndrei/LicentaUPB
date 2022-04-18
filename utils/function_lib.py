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
    
    mask = trans_mask(mask)
    return image, mask
    

def show_model_seg(model, dataset, idx):
    model.eval()
    input_tensor, mask = dataset.__getitem__(idx)
    input_batch = input_tensor.repeat(2, 1, 1, 1)
    # input_batch = input_tensor
    pred = model(input_batch)

    final_func = nn.Sigmoid()
    pred = final_func(pred).squeeze()
    pred = torch.nn.functional.interpolate(pred.unsqueeze(0), size=512, mode="bilinear", align_corners=False).squeeze()
    output_predictions = torch.round(pred[0])

    print(output_predictions.shape)
    output_predictions = output_predictions.detach().squeeze()
    output_predictions = np.round(output_predictions)
    mask = mask.cpu().squeeze()

    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(3)])[:, None] * 1.1 * palette
    colors = (colors % 255).numpy().astype("uint8")
    r = Image.fromarray(output_predictions.byte().cpu().numpy())
    r.putpalette(colors)

    r_mask = Image.fromarray(mask.byte().cpu().numpy())
    r_mask.putpalette(colors)

    print(torch.max(output_predictions))
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(r)
    plt.subplot(1, 3, 2)
    plt.imshow(r_mask)
    plt.subplot(1, 3, 3)
    if input_tensor.size()[0] > 1:
        plt.imshow(np.squeeze(input_tensor[1, :, :].numpy()))
    else:
        plt.imshow(np.squeeze(input_tensor.numpy()))
    plt.show()

