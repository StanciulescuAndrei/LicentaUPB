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
import nibabel as nib


class LiverDataset(Dataset):
    """Liver segmentation dataset."""

    def __init__(self, root_dir, transform_image=None, transform_mask=None):
        self.root_dir = root_dir
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        self.img_files = sorted(glob.glob(root_dir + 'images/*.ct'))
        self.mask_files = sorted(glob.glob(root_dir + 'masks/*.ct'))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        input_image = np.fromfile(img_path, np.float16)
        input_image = np.squeeze(input_image)
        edge_len = np.sqrt(len(input_image)).astype(np.int32)
        if self.transform_image:
            input_image = self.transform_image(np.array(input_image).reshape([edge_len, edge_len]).astype(np.float32))
        input_image = input_image.type(torch.float)
        input_image = input_image

        mask_path = self.mask_files[idx]
        label = np.fromfile(mask_path, np.float16)
        label = np.squeeze(label)
        label[label > 1.0] = 1.0 # segmentam doar ficat
        if self.transform_mask:
            label = self.transform_mask(np.array(label).reshape([edge_len, edge_len]).astype(np.float32))
        label = torch.squeeze(label)

        return input_image, label
    