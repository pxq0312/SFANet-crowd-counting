import torch
from torch import nn
from torch import optim
from torch.utils import data
from dataset import Dataset
from models import Model
from tensorboardX import SummaryWriter
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bs', default=8, type=int, help='batch size')
parser.add_argument('--epoch', default=500, type=int, help='train epochs')
parser.add_argument('--dataset', default='SHA', type=str, help='dataset')
parser.add_argument('--data_path', default=r'D:\dataset', type=str, help='path to dataset')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--load', default=False, action='store_true', help='load checkpoint')
parser.add_argument('--save_path', default=r'D:\checkpoint\SFANet', type=str, help='path to save checkpoint')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--log_path', default='./logs', type=str, help='path to log')

args = parser.parse_args()

train_dataset = Dataset(args.data_path, args.dataset, True)
train_loader = data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
test_dataset = Dataset(args.data_path, args.dataset, False)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda:' + str(args.gpu))

model = Model().to(device)

writer = SummaryWriter(args.log_path)

mseloss = nn.MSELoss(reduction='sum').to(device)
bceloss = nn.BCELoss(reduction='sum').to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

if args.load:
    checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint_latest.pth'))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    best_mae = torch.load(os.path.join(args.save_path, 'checkpoint_best.pth'))['mae']
    start_epoch = checkpoint['epoch'] + 1
else:
    best_mae = 999999
    start_epoch = 0

for epoch in range(start_epoch, start_epoch + args.epoch):
    loss_avg, loss_att_avg = 0.0, 0.0

    for i, (images, density, att) in enumerate(train_loader):
        images = images.to(device)
        density = density.to(device)
        att = att.to(device)
        outputs, attention = model(images)
        print('output:{:.2f} label:{:.2f}'.format(outputs.sum().item() / args.bs, density.sum().item() / args.bs))

        loss = mseloss(outputs, density) / args.bs
        loss_att = bceloss(attention, att) / args.bs * 0.1
        loss_sum = loss + loss_att

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        loss_avg += loss.item()
        loss_att_avg += loss_att.item()

        print("Epoch:{}, Step:{}, Loss:{:.4f} {:.4f}".format(epoch, i, loss_avg / (i + 1), loss_att_avg / (i + 1)))

    writer.add_scalar('loss/train_loss', loss_avg / len(train_loader), epoch)
    writer.add_scalar('loss/train_loss_att', loss_att_avg / len(train_loader), epoch)

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
        print('Epoch:', epoch, 'MAE:', mae, 'MSE:', mse)
        writer.add_scalar('eval/MAE', mae, epoch)
        writer.add_scalar('eval/MSE', mse, epoch)

        state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'mae': mae,
                 'mse': mse}
        torch.save(state, os.path.join(args.save_path, 'checkpoint_latest.pth'))

        if mae < best_mae:
            best_mae = mae
            torch.save(state, os.path.join(args.save_path, 'checkpoint_best.pth'))
    model.train()

writer.close()
