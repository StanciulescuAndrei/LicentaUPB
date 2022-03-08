from PIL import Image
import glob
import shutil
import os
import numpy as np
import nibabel as nib

root_dir = "G:/MachineLearning/training-batch-1/"
image_files = sorted(glob.glob(root_dir + '*.nii'))
counts = np.zeros([3, 1])
for i in range(len(image_files)):
    image = nib.load(image_files[i])
    data = image.get_fdata()
    counts[0] = counts[0] + np.sum(np.round(data) == 0)
    counts[1] = counts[1] + np.sum(np.round(data) == 1)
    counts[2] = counts[2] + np.sum(np.round(data) == 2)

counts[0] = counts[0] / counts[0]
counts[1] = counts[1] / counts[0]
counts[2] = counts[2] / counts[0]

print(f"Bakground: {counts[0]}, Liver: {counts[1]}, Tumor: {counts[2]}")

# num_img = len(image_files)
# means = np.zeros(num_img)
# stds = np.zeros(num_img)
# for i in range(num_img):
#     image = nib.load(image_files[i])
#     data = image.get_fdata()
#     means[i] = np.mean(data)
#
# means = np.squeeze(means)
# mean = means.mean()
#
# for i in range(num_img):
#     image = nib.load(image_files[i])
#     data = image.get_fdata()
#     stds[i] = np.mean(abs(data - mean)**2)
#
# stds = np.squeeze(stds)
# std = np.sqrt(stds.mean())
#
# print(f"Mean: {mean}\n\rSTD: {std}")