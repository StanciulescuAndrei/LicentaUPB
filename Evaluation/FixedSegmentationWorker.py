from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from utils.DiceLoss import DiceLoss
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt

# Incarcare imagini presegmentate
seg_folder = 'G:/MachineLearning/liver-database/outputs/'
models = ['DenseNet201', 'Inception', 'ResNet152', 'ResNeXt101']
seg = np.zeros((4, 4776, 512, 512), dtype=bool)
for i in range(len(models)):
    full_path = seg_folder + models[i]
    seg_files = glob.glob(full_path + '/*.ct')
    for j in range(len(seg_files)):
        seg[i, j, :, :] = np.fromfile(seg_files[j], dtype=np.int8).squeeze().reshape([512, 512]).astype(bool)
    print(f"Done loading {models[i]}")

seg = torch.tensor(seg)

# Incarcare ground truth
lits_seg = glob.glob('G:/MachineLearning/liver-database/validation/masks/*.ct')
ground_truth = np.zeros((4776, 512, 512), dtype=bool)
for j in range(len(lits_seg)):
    ground_truth[j, :, :] = np.fromfile(lits_seg[j], dtype=np.float16).squeeze().reshape([512, 512]).astype(bool)
print(f"Done loading ground truth")
ground_truth = torch.tensor(ground_truth)

dice_loss = DiceLoss()

# Mai intai evaluam DICE per case

for i in range(len(models)):
    dice = 1.0 - dice_loss(ground_truth, seg[i, :, :, :].squeeze())
    print(f"Dice loss average per case for {models[i]}: {dice}")

for i in range(len(models)):
    dice = 0
    for s in range(ground_truth.size(dim=0)):
        dice += (1.0 - dice_loss(ground_truth[s, :, :], seg[i, s, :, :]))
    dice = dice / ground_truth.size(dim=0)
    print(f"Dice loss per slice average for {models[i]}: {dice}")

# Facem diferentele de segmentari
num_models = len(models)
diff_dice = torch.zeros(num_models, num_models)
for i in range(num_models):
    for j in range(num_models):
        diff_dice[i, j] = 1.0 - dice_loss(seg[i, :, :, :], seg[j, :, :, :])

diff_dice = diff_dice.cpu().numpy()
fig, ax = plt.subplots()
im = ax.imshow(diff_dice)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(models)), labels=models)
ax.set_yticks(np.arange(len(models)), labels=models)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(models)):
    for j in range(len(models)):
        text = ax.text(j, i, diff_dice[i, j],
                       ha="center", va="center", color="w")

ax.set_title("DICE similarity between model segmentations")
fig.tight_layout()
plt.show()