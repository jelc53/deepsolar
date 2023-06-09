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

from torch.nn import functional as F
from torchvision.models import Inception3

from inception_modified import InceptionSegmentation
from image_dataset import ImageFolderModified, ImageFolderModifiedEvaluation

# Configuration
# directory for loading training/validation/test data
mode = 'eval' # 'eval' or 'val'
#data_dir = '/home/ubuntu/deepsolar/data/bdappv-france/ft_eval'
data_dir = '/home/ubuntu/deepsolar/data/ds-usa/eval'
#'/home/ubuntu/projects/deepsolar/deepsolar_dataset_toy/test'

#basic_params_path = 'checkpoint/bdappv_ft100_w0.001_lr0.001/deepsolar_classification_4_last.tar'
old_ckpt_path = 'checkpoint/bdappv_ft5000_w0.001_lr0.001/deepsolar_seg_level2_1_last.tar'
#old_ckpt_path = 'checkpoint/bdappv_ft100/deepsolar_seg_level2_5.tar'
#old_ckpt_path = '/home/ubuntu/deepsolar/models/deepsolar_seg_pretrained.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 299
batch_size = 1   # must be 1 for testing segmentation
class_threshold = 0.5  # threshold probability to identify am image as positive (originally 0.5)
seg_threshold = 0.57    # threshold to identify a pixel as positive.(originally 0.37)
level = 2

def metrics(stats):
    """stats: {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
    return: must be a single number """
    precision = (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FP'] + 0.00001)
    recall = (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FN'] + 0.00001)
    return 0.5*(precision + recall)

def test_model(model, dataloader, metrics, class_threshold, seg_threshold):
    stats = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    iou = []
    estimated_area = 0
    true_area = 0
    model.eval()
    CAM_list = []
    for inputs, labels, paths in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            _, outputs, CAM = model(inputs, testing=True)   # CAM is a 1 x 35 x 35 activation map
            prob = F.softmax(outputs, dim=1)
            preds = prob[:, 1] >= class_threshold

        CAM = CAM.squeeze(0).cpu().numpy()   # transform tensor into numpy array
        for i in range(preds.size(0)):
            predicted_label = preds[i]
            if predicted_label.cpu().item():
            #if labels[i] == 1:  # oracle classifier
                CAM_list.append((CAM, paths[i]))        # only use the generated CAM if it is predicted to be 1
                CAM_rescaled = (CAM - CAM.min()) / (CAM.max() - CAM.min())    # get predicted area
                CAM_pred = CAM_rescaled > seg_threshold
                pred_pixel_area = np.sum(CAM_pred)
                estimated_area += pred_pixel_area
            else:
                CAM_pred = np.zeros_like(CAM)
                CAM_list.append((CAM_pred, paths[i]))  # otherwise the CAM is a totally black one

            if labels[i] == 1:
                # calculate true area
                img_path = os.path.splitext(paths[i])
                true_seg_path = img_path[0] + '_true_seg' + img_path[1]
                true_seg_img = Image.open(true_seg_path)
                transform = transforms.Compose([transforms.ToTensor()])
                true_seg = transform(true_seg_img)
                true_seg = true_seg.squeeze(0).cpu().numpy()
                true_pixel_area = np.sum(true_seg)
                true_pixel_area = true_pixel_area * (35 * 35) / (true_seg.shape[0] * true_seg.shape[1])
                true_area += true_pixel_area
                CAM_true = (skimage.transform.resize(true_seg, (35, 35)) > 0.5)
            else:
                CAM_true = np.zeros_like(CAM)

            # intersection over union
            intersection = CAM_true * CAM_pred     # logical AND
            union = CAM_true + CAM_pred            # logical OR
            if union.sum() > 0:
                iou.append(intersection.sum() / float(union.sum()))

        # confusion matrix statistics
        stats['TP'] += torch.sum((preds == 1) * (labels == 1)).cpu().item()
        stats['TN'] += torch.sum((preds == 0) * (labels == 0)).cpu().item()
        stats['FP'] += torch.sum((preds == 1) * (labels == 0)).cpu().item()
        stats['FN'] += torch.sum((preds == 0) * (labels == 1)).cpu().item()

    print("accuracy = ", (stats['TP'] + stats['TN'] + 0.00001) / (stats['TP'] + stats['TN'] + stats['FP'] + stats['FN'] + 0.00001))
    print("precision = ",(stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FP'] + 0.00001))
    print("recall = ", (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FN'] + 0.00001))
    metric_value = metrics(stats)
    print("iou = ", np.mean(iou))
    return stats, metric_value, CAM_list, estimated_area, true_area

transform_test = transforms.Compose([
                 transforms.Resize((input_size, input_size)),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                 ])

if __name__ == '__main__':
    # data
    if mode=='eval':
        print('evaluating on test set')
        dataset_test = ImageFolderModifiedEvaluation(data_dir, transform_test)
    else:
        dataset_test = ImageFolderModified(data_dir, transform_test)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)

    # model
    model = InceptionSegmentation(num_outputs=2, level=level)
    model.load_existing_params(old_ckpt_path)

    model = model.to(device)

    stats, metric_value, CAM_list, estimated_area, true_area = test_model(model, dataloader_test, metrics, class_threshold=class_threshold, seg_threshold=seg_threshold)
    precision = (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FP'] + 0.00001)
    recall = (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FN'] + 0.00001)
    print('metric value: '+str(metric_value))
    print('precision: ' + str(round(precision, 4)))
    print('recall: ' + str(round(recall, 4)))
    print('estimated_area: ' + str(estimated_area))
    print('true_area: ' + str(true_area))
    print('ratio (est / true): ' + str(estimated_area / true_area))

    with open('CAM_list.pickle', 'wb') as f:
        pickle.dump(CAM_list, f)

