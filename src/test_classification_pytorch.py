from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms, utils
import torchvision.transforms.functional as TF

from tqdm import tqdm
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
from sklearn.metrics import r2_score
from image_dataset import ImageFolderModifiedClassificationEvaluation

from torch.nn import functional as F
from torchvision.models import Inception3

# Configuration
# directory for loading training/validation/test data 
mode = 'val' # 'eval' or 'val'
data_dir = '/home/ubuntu/deepsolar/data/ds-usa/val'
# old_ckpt_path = '/home/ubuntu/deepsolar/models/deepsolar_pretrained.pth'
old_ckpt_path = '/home/ubuntu/deepsolar/checkpoint/retrain_pytorch/baseline_classification_0_last.tar'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 299
batch_size = 32
threshold = 0.5  # threshold probability to identify am image as positive

def metrics(stats):
    """stats: {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
    return: must be a single number """
    precision = (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FP'] + 0.00001)
    recall = (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FN'] + 0.00001)
    return 0.5*(precision + recall)


def test_model(model, dataloader, metrics, threshold):
    stats = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    model.eval()
    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            prob = F.softmax(outputs, dim=1)
            preds = prob[:, 1] >= threshold

        stats['TP'] += torch.sum((preds == 1) * (labels == 1)).cpu().item()
        stats['TN'] += torch.sum((preds == 0) * (labels == 0)).cpu().item()
        stats['FP'] += torch.sum((preds == 1) * (labels == 0)).cpu().item()
        stats['FN'] += torch.sum((preds == 0) * (labels == 1)).cpu().item()

    metric_value = metrics(stats)
    return stats, metric_value

transform_test = transforms.Compose([
                 transforms.Resize((input_size, input_size)),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                 ])

if __name__ == '__main__':
    # data
    if mode =='eval':
        print('evaluating on test set')
        dataset_test = ImageFolderModifiedClassificationEvaluation(data_dir, transform_test)
    else:
        dataset_test = datasets.ImageFolder(data_dir, transform_test)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=4)
    # model
    model = Inception3(num_classes=2, aux_logits=True, transform_input=False)
    model = model.to(device)
    # load old parameters
    checkpoint = torch.load(old_ckpt_path, map_location=device)
    if old_ckpt_path[-4:] == '.tar':  # it is a checkpoint dictionary rather than just model parameters
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print('Old checkpoint loaded: ' + old_ckpt_path)
    stats, metric_value = test_model(model, dataloader_test, metrics, threshold=threshold)
    precision = (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FP'] + 0.00001)
    recall = (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FN'] + 0.00001)
    print('metric value: '+str(metric_value))
    print('precision: ' + str(round(precision, 4)))
    print('recall: ' + str(round(recall, 4)))
