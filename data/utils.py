import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from glob import glob
import os
from sklearn.model_selection import train_test_split
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, X, Y, transforms=None):
        self.X = X
        self.Y = Y
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = Image.open(self.X[index])
        X = self.transforms(X)
        Y = self.Y[index]
        return X, Y

def getData(root, input_shape=(3, 224, 224)):
    if root is None:
        raise ValueError('Data directory not specified!')

    train_trainsforms = transforms.Compose([
        transforms.Resize([224, 224]),
        # transforms.RandomCrop((320, 240), padding=40),
        transforms.RandomEqualize(),
        transforms.RandomAutocontrast(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    val_trainsforms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    up_imgs = sorted(glob(os.path.join(root, 'Up/*')))[500:]
    left_imgs = sorted(glob(os.path.join(root, 'Left/*')))[500:]
    right_imgs = sorted(glob(os.path.join(root, 'Right/*')))[500:]


    data_all = up_imgs + left_imgs + right_imgs
    # data_all = left_imgs + right_imgs
    print('Length of the whole data is: ', len(data_all))

    labels = [0] * len(up_imgs) + [1] * len(left_imgs) + [2] * len(right_imgs)
    # labels = [0] * len(left_imgs) + [1] * len(right_imgs)

    labels = torch.tensor(labels).long()
    X_train, X_val, y_train, y_val = train_test_split(data_all, labels, test_size=0.1, random_state=5)
    print('Train：', len(X_train), 'Validation：', len(X_val))

    train_dataset = MyDataset(X_train, y_train, train_trainsforms)
    val_dataset = MyDataset(X_val, y_val, val_trainsforms)

    return train_dataset, val_dataset