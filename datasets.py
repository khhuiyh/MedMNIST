# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 16:35:38 2021

@author: Administrator
"""


import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class MedMNIST(Dataset):
    def __init__(self,split = 'train',transform=None,
                 target_transform=None):
        npz_file = np.load(r'D:\Homework\机器学习工程实践\MNIST\octmnist.npz')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        if self.split == 'train':
            self.img = npz_file['train_images']
            self.label = npz_file['train_labels']
        elif self.split == 'val':
            self.img = npz_file['val_images']
            self.label = npz_file['val_labels']
        elif self.split == 'test':
            self.img = npz_file['test_images']
            self.label = npz_file['test_labels']
    
    def __getitem__(self, index):
        img, target = self.img[index], self.label[index].astype(int)
        img = Image.fromarray(np.uint8(img))
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    
    def __len__(self):
        return self.img.shape[0]
    
