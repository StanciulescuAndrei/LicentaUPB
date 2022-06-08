from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from utils.DiceLoss import DiceLoss
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Incarcare imagini presegmentate
seg_folder = 'G:/MachineLearning/liver-database/validation/processed/'
models = ['DenseNet201', 'Inception', 'ResNet152', 'ResNeXt101', 'Fusion']
# seg = np.zeros((5, 4776, 512, 512), dtype=bool)
for i in range(len(models)):
    full_path = seg_folder + models[i]

    seg_files = sorted(glob.glob(full_path + '/*.ct'))
    lits_seg = sorted(glob.glob('G:/MachineLearning/liver-database/validation/masks/*.ct'))
    ct_files = sorted(glob.glob('G:/MachineLearning/liver-database/validation/images/*.ct'))

    for j in tqdm(range(len(seg_files))): # len(seg_files)
        seg = np.fromfile(seg_files[j], dtype=np.int8).squeeze().reshape([512, 512])
        lits = np.fromfile(lits_seg[j], dtype=np.float16).squeeze().reshape([512, 512])
        lits[lits > 1.0] = 1.0
        ct = np.fromfile(ct_files[j], dtype=np.float16).squeeze().reshape([512, 512])
        plt.figure(figsize=(20, 16))
        plt.subplot(1, 3, 1)
        plt.imshow(seg, cmap='jet')
        plt.title('Model segmentation')
        plt.xlabel('px')
        plt.ylabel('px')

        plt.subplot(1, 3, 2)
        plt.imshow(lits, cmap='jet')
        plt.title('Ground truth')
        plt.xlabel('px')
        plt.ylabel('px')

        plt.subplot(1, 3, 3)
        plt.imshow(ct, cmap='binary_r')
        plt.title('CT image')
        plt.xlabel('px')
        plt.ylabel('px')
        plt.savefig('G:/MachineLearning/liver-database/outputs-formatted/' + models[i] + '/' + ("%04d.png" % j))
        plt.close()

