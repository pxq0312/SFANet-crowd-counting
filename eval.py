import torch
from torch.utils import data
from dataset import Dataset
from models import Model
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='SHA', type=str, help='dataset')
parser.add_argument('--data_path', default=r'D:\dataset', type=str, help='path to dataset')
parser.add_argument('--save_path', default=r'D:\checkpoint\SFANet', type=str, help='path to save checkpoint')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')

args = parser.parse_args()

test_dataset = Dataset(args.data_path, args.dataset, False)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda:' + str(args.gpu))

model = Model().to(device)

checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint_best.pth'))
model.load_state_dict(checkpoint['model'])

model.eval()
with torch.no_grad():
    mae, mse = 0.0, 0.0
    for i, (images, gt) in enumerate(test_loader):
        images = images.to(device)

        predict, _ = model(images)

        print('predict:{:.2f} label:{:.2f}'.format(predict.sum().item(), gt.item()))
        mae += torch.abs(predict.sum() - gt).item()
        mse += ((predict.sum() - gt) ** 2).item()

    mae /= len(test_loader)
    mse /= len(test_loader)
    mse = mse ** 0.5
    print('MAE:', mae, 'MSE:', mse)
