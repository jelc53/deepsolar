from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms, utils
import torchvision.transforms.functional as TF

import numpy as np
import json
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import skimage
import skimage.io
import skimage.transform
from PIL import Image
import time
import os
from os.path import join, exists
import copy
import random
from collections import OrderedDict


class ImageFolderModified(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.idx2dir = []
        self.path_list = []
        for subdir in sorted(os.listdir(self.root_dir)):
            if not os.path.isfile(subdir):
                self.idx2dir.append(subdir)
        for class_idx, subdir in enumerate(self.idx2dir):
            class_dir = os.path.join(self.root_dir, subdir)
            for f in os.listdir(class_dir):
                if 'true_seg' in f:
                    continue
                if f[-4:] in ['.png', '.jpg', 'JPEG', 'jpeg']:
                    self.path_list.append([os.path.join(class_dir, f), class_idx])

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        img_path, class_idx = self.path_list[idx]
        image = Image.open(img_path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        sample = [image, class_idx, img_path]
        return sample


class ImageFolderModifiedEvaluation(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.idx2dir = ['0', '1']
        self.path_list = []
        for subdir1 in sorted(os.listdir(self.root_dir)):
            # if not os.path.isfile(subdir):
            #     self.idx2dir.append(subdir)
            for class_idx, subdir2 in enumerate(self.idx2dir):
                class_dir = os.path.join(self.root_dir, subdir1, subdir2)
                for f in os.listdir(class_dir):
                    if 'true_seg' in f:
                        continue
                    if f[-4:] in ['.png', '.jpg', 'JPEG', 'jpeg']:
                        self.path_list.append([os.path.join(class_dir, f), class_idx])  

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        img_path, class_idx = self.path_list[idx]
        image = Image.open(img_path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        sample = [image, class_idx, img_path]
        return sample

class ImageFolderModifiedClassificationEvaluation(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.idx2dir = ['0', '1']
        self.path_list = []
        for subdir1 in sorted(os.listdir(self.root_dir)):
            # if not os.path.isfile(subdir):
            #     self.idx2dir.append(subdir)
            for class_idx, subdir2 in enumerate(self.idx2dir):
                class_dir = os.path.join(self.root_dir, subdir1, subdir2)
                for f in os.listdir(class_dir):
                    if 'true_seg' in f:
                        continue
                    if f[-4:] in ['.png', '.jpg', 'JPEG', 'jpeg']:
                        self.path_list.append([os.path.join(class_dir, f), class_idx])  

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        img_path, class_idx = self.path_list[idx]
        image = Image.open(img_path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        # sample = [image, class_idx, img_path]
        return image, class_idx


# Create Interpolated Images and Labels from dataset pairs
# lambda = the interpolation weight, (0, 1) where 
# beta = the selection of 
class ImageFolderModifiedLisaTrain(Dataset):
    def __init__(self, root_dir, transform_before, transform_after,
                 alpha=2, beta=2, psel=0.0):
        random.seed(42)
        self.root_dir = root_dir
        self.transform_before = transform_before
        self.transform_after = transform_after
        self.idx2dir = ['0', '1', '1us']
        self.finetune_path_list = []
        self.us_neg_path_list = []
        self.us_pos_path_list = []
        self.alpha = alpha
        self.beta = beta
        self.psel = psel
        for class_idx, subdir in enumerate(self.idx2dir):
            class_dir = os.path.join(self.root_dir, subdir)
            for f in os.listdir(class_dir):
                if 'true_seg' in f:
                    continue
                if f[-4:] in ['.png', '.jpg', 'JPEG', 'jpeg']:
                    if class_idx == 0:
                        self.us_neg_path_list.append([os.path.join(class_dir, f), class_idx])
                    elif class_idx == 1:
                        self.finetune_path_list.append([os.path.join(class_dir, f), class_idx])
                    elif class_idx == 2:
                        self.us_pos_path_list.append([os.path.join(class_dir, f), class_idx])
                    else:
                        raise ValueError("class_idx not recognized: {}".format(class_idx))
        assert len(self.us_neg_path_list) == len(self.us_pos_path_list)
        assert len(self.us_neg_path_list) == len(self.finetune_path_list)
        print(len(self.finetune_path_list))


    def __len__(self):
        return len(self.finetune_path_list)
    
    def __getitem__(self, idx):
        p = random.uniform(0,1)
        l = random.betavariate(self.alpha, self.beta)
        finetune_image = self.transform_before(Image.open(self.finetune_path_list[idx][0]).convert('RGB'))
        # print('finetune image ' + str(idx))
        # print(self.finetune_path_list[idx][0])

        # print(finetune_image.shape)
        if p > self.psel:
            # do inter-domain
            interpolate_image = self.transform_before(Image.open(self.us_pos_path_list[idx][0]).convert('RGB'))
            # print('interpolate image ' + str(idx))
            # print(self.us_pos_path_list[idx][0])
            # print(interpolate_image.shape)
            class_idx = torch.tensor([0., 1.])
        else: 
            # do inter-label
            interpolate_image = self.transform_before(Image.open(self.us_neg_path_list[idx][0]).convert('RGB'))
            class_idx = torch.tensor([1. - l, l]) 
        #     print(interpolate_image.shape)
        # print(l)
        # print(class_idx)
        # print('interpolated image shape')
        image = l * finetune_image + (1. - l) * interpolate_image 

        # print(image.shape)
        image = self.transform_after(image)
        # print(image.shape)

        return image, class_idx


        

        