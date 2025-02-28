import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y,_ = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y,1), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        #image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image = torch.from_numpy(image.astype(np.float32))
        image = image.permute(2,0,1)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample
class RandomGenerator1(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        x, y ,_= image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y,1), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        #image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image = torch.from_numpy(image.astype(np.float32))
        image = image.permute(2,0,1)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

def sliding_window(image, window_size, stride):
    windows = []
    positions = []
    img_h, img_w = image.shape[:2]
    window_h, window_w = window_size
    stride_h, stride_w = stride

    for i in range(0, img_h - window_h + 1, stride_h):
        for j in range(0, img_w - window_w + 1, stride_w):
            window = image[i:i + window_h, j:j + window_w]
            windows.append(window)
            positions.append((i, j))  # 记录窗口的位置
    return windows, positions


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split)).readlines()
        # self.data_dir = base_dir
        if self.split == "train":
            self.data_dir = base_dir + "/train_npz"
        else:
            self.data_dir = base_dir + "/test_npz"
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            slice_name = self.sample_list[idx].strip('\n')
            data_path = self.data_dir + "/" + slice_name + '.npz'
            #data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample