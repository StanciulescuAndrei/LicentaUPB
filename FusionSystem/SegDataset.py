import torch
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np


class SegDataset(Dataset):
    """Liver segmentation dataset."""

    def __init__(self, root_dir, model_list=['DenseNet201', 'Inception', 'ResNet152', 'ResNeXt101']):
        self.root_dir = root_dir
        self.segmentations = {}
        self.model_list = model_list
        for model in model_list:
            self.segmentations[model] = sorted(glob.glob(root_dir + model + '/*.ct'))
        self.mask_files = sorted(glob.glob(root_dir + 'masks/*.ct'))

    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, idx):
        seg_stack = np.zeros((len(self.model_list), 256, 256))
        stack = 0
        for model in self.model_list:
            print(model)
            seg_stack[stack, :, :] = np.fromfile(self.segmentations[model][idx], bool).squeeze().reshape([256, 256]).astype(np.float32)
            stack = stack + 1
        seg_stack = torch.tensor(seg_stack)

        mask_path = self.mask_files[idx]
        label = np.fromfile(mask_path, np.float16)
        label = np.squeeze(label)
        label[label > 1.0] = 1.0
        label = np.array(label).reshape([256, 256]).astype(np.float32)
        label = torch.squeeze(torch.tensor(label))

        return seg_stack, label
