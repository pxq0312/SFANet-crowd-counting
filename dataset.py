from torch.utils import data
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from transforms import Transforms
import glob
from torchvision.transforms import functional


class Dataset(data.Dataset):
    def __init__(self, data_path, dataset, is_train):
        self.is_train = is_train
        self.dataset = dataset
        if is_train:
            is_train = 'train_data'
        else:
            is_train = 'test_data'
        if dataset == 'SHA':
            dataset = os.path.join('ShanghaiTech', 'part_A_final')
        elif dataset == 'SHB':
            dataset = os.path.join('ShanghaiTech', 'part_B_final')

        self.image_list = glob.glob(os.path.join(data_path, dataset, is_train, 'images', '*.jpg'))
        self.label_list = glob.glob(os.path.join(data_path, dataset, is_train, 'new_data', '*.h5'))
        self.image_list.sort()
        self.label_list.sort()

    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert('RGB')
        label = h5py.File(self.label_list[index], 'r')
        density = np.array(label['density'], dtype=np.float32)
        attention = np.array(label['attention'], dtype=np.float32)
        gt = np.array(label['gt'], dtype=np.float32)
        trans = Transforms((0.8, 1.2), (400, 400), 2, (0.5, 1.5), self.dataset)
        if self.is_train:
            image, density, attention = trans(image, density, attention)
            return image, density, attention
        else:
            height, width = image.size[1], image.size[0]
            height = round(height / 16) * 16
            width = round(width / 16) * 16
            image = image.resize((width, height), Image.BILINEAR)

            image = functional.to_tensor(image)
            image = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            return image, gt

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    train_dataset = Dataset(r'D:\dataset', 'SHA', True)
    train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    for image, label, att in train_loader:
        print(image.size())
        print(label.size())
        print(att.size())

        img = np.transpose(image.numpy().squeeze(), [1, 2, 0]) * 0.2 + 0.45
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.subplot(1, 3, 2)
        plt.imshow(label.squeeze(), cmap='jet')
        plt.subplot(1, 3, 3)
        plt.imshow(att.squeeze(), cmap='jet')
        plt.show()
