from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import sys
import os

os.environ['CUDA_VISIBLE_DEVICES']='1'

from ModelBuilder.ResNet152_DeepLab import ResNet152DeepLab
from ModelBuilder.AlexNet_DeepLab import AlexNetDeepLab
from ModelBuilder.ResNeXt101_DeepLab import ResNeXt101DeepLab
from ModelBuilder.ClassifierHead import DeepLabHead

from utils.function_lib import *
from utils.LiverDataset import *
from utils.DiceLoss import *

transform_image = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2], std=[0.2])
    ])
transform_mask = transforms.Compose(
    [
        transforms.ToTensor()
    ])

path = 'liver-database/'
save_path = 'models/'

dataset_learn = LiverDataset(path + 'training/', transform_image=transform_image, transform_mask=transform_mask)
dataloader_learn = torch.utils.data.DataLoader(dataset_learn, batch_size=256, shuffle=True, num_workers=32, pin_memory=True)

dataset_test = LiverDataset(path + 'testing/', transform_image=transform_image, transform_mask=transform_mask)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=256, shuffle=True, num_workers=32, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'AlexNet'
model = AlexNetDeepLab()

optimizer = optim.SGD([
    {'params': model.parameters()}
], lr=0.05, momentum=0.9)
sched = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.93)
loss_fcn = nn.BCEWithLogitsLoss().to(device)

if True:
    for param in model.parameters():
        param.requires_grad = True
    torch.save(model, save_path + 'AlexNet_0.pt')
    epoch = 1
else:
    checkpoint = torch.load(save_path + model_name + '_checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    sched.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']

torch.cuda.empty_cache()
model = model.to(device)

while epoch < 501:
    print(f"Epoch {epoch}\n-------------------------------")

    train_loop(dataloader_learn, model, loss_fcn, optimizer, sched, device)
    validation_loss = test_loop(dataloader_learn, model, loss_fcn, device)
    sched.step()

    if epoch % 5 == 0:
        torch.save(model, save_path + model_name + '_' + str(epoch) + '.pt')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': sched.state_dict()
    }, save_path + model_name + '_checkpoint.pt')

    epoch = epoch + 1

print("Done!")