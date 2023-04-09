from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import torch
class train_dataset(Dataset):
    def __init__(self, x, y, transform):
        self.x_data = x
        self.y_data = y
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.transform(self.x_data[idx])
        y = self.y_data[idx]
        return x, y


class test_dataset(Dataset):
    def __init__(self, x, y, transform):
        self.x_data = x
        self.y_data = y
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.transform(self.x_data[idx])
        y = self.y_data[idx]
        return x, y

class val_dataset(Dataset):
    def __init__(self, x, y, transform):
        self.x_data = x
        self.y_data = y
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.transform(self.x_data[idx])
        y = self.y_data[idx]
        return x, y

def data_split(train_x, test_x, train_y, test_y, seed):
    x = np.concatenate((test_x, train_x), axis=0)
    y = np.concatenate((test_y, train_y), axis=0)
    print(len(y))
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=1 / 5, shuffle=True,
        stratify = y, random_state=40)
    train_x, val_x, train_y, val_y = train_test_split(
        train_x, train_y, test_size=1/4, shuffle=True,
        stratify = train_y,  random_state = seed)
    return train_x, val_x, test_x, train_y, test_y, val_y
