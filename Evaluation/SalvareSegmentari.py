from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from tqdm import tqdm

from utils.LiverDataset import *
from utils.DiceLoss import *

from ModelBuilder.ResNeXt101_DeepLab import ResNeXt101DeepLab

transform_image = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(256, interpolation=transforms.functional.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2], std=[0.2])
    ])
transform_mask = transforms.Compose(
    [
        transforms.ToTensor()
    ])

path = 'liver-database/'

dataset_test = LiverDataset(path + 'validation/', transform_image=transform_image, transform_mask=transform_mask)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

save_path = 'models/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'ResNeXt101'
model = ResNeXt101DeepLab()
model = model.to(device)

checkpoint = torch.load(save_path + model_name + '_checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])

save_images = 'liver-database/outputs/' + model_name + '/'

iter = 0
model.eval()
print(f"Starting evaluation...")
with torch.no_grad():
    imagesize = 0
    for X, Y in tqdm(dataloader_test):
        imagesize = torch.numel(Y)
        pred = model(X.to(device)).squeeze()
        final_func = nn.Sigmoid()
        pred = final_func(pred)
        pred = torch.nn.functional.interpolate(pred, size=512, mode="bilinear", align_corners=False).squeeze()
        pred = torch.round(pred)
        for i in range(pred.size[0]):
            slc = pred[i, :, :, :]
            slc = slc.numpy()
            slc.astype(np.int8).tofile(save_images + 'output-' + ("-%04d.ct" % iter))
            iter = iter + 1
