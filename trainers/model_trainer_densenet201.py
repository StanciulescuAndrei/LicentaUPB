from torch.utils.data import Dataset
import os

os.environ['CUDA_VISIBLE_DEVICES']='1'

from ModelBuilder.DenseNet.DenseNet201_DeepLab import DenseNet201DeepLab

from utils.function_lib import *
from utils.LiverDataset import *
from utils.DiceLoss import *

transform_image = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
transform_mask = transforms.Compose(
    [
        transforms.ToTensor()
    ])

path = 'liver-database/'
save_path = 'models/'

dataset_learn = LiverDataset(path + 'training/', transform_image=transform_image, transform_mask=transform_mask)
dataloader_learn = torch.utils.data.DataLoader(dataset_learn, batch_size=32, shuffle=True, num_workers=32, pin_memory=True)

dataset_test = LiverDataset(path + 'testing/', transform_image=transform_image, transform_mask=transform_mask)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=32, shuffle=True, num_workers=32, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'DenseNet201'
model = DenseNet201DeepLab()
model = model.to(device)

optimizer = optim.SGD([
    {'params': model.parameters()}
], lr=0.05, momentum=0.9)
sched = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.93)
loss_fcn = nn.BCEWithLogitsLoss().to(device)

if False:
    for param in model.parameters():
        param.requires_grad = True
    torch.save(model, save_path + 'DenseNet201_0.pt')
    epoch = 1
else:
    checkpoint = torch.load(save_path + model_name + '_checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    sched.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']

while epoch < 201:
    print(f"Epoch {epoch}\n-------------------------------")

    train_loop(dataloader_learn, model, loss_fcn, optimizer, sched, device)
    validation_loss = test_loop(dataloader_learn, model, loss_fcn, device)
    sched.step()

    if epoch % 10 == 0:
        torch.save(model, save_path + model_name + '_' + str(epoch) + '.pt')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': sched.state_dict()
    }, save_path + model_name + '_checkpoint.pt')

    epoch = epoch + 1

print("Done!")
