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

import sys
sys.path.append("/home/ubuntu/deepsolar/src/")

from inception_modified import InceptionSegmentation
from image_dataset import ImageFolderModified, ImageFolderModifiedEvaluation

# Configuration
# directory for loading training/validation/test data
mode = 'val' # 'eval' or 'val'
# data_dir = '/home/ubuntu/deepsolar/data/ds-usa/eval'  #'/home/ubuntu/projects/deepsolar/deepsolar_dataset_toy/test'
data_dir = '/home/ubuntu/deepsolar/data/ds-france/google/ft_5000/val'
# data_dir = '/home/ubuntu/deepsolar/data/ds-france/google/ft_100/val/' 
classification_path = '/home/ubuntu/deepsolar/checkpoint/ft_5000_classification_tune_sweep_best_models/psel_0-6956991453638643_lr_0-00016152257047812214_lr_decay_rate_0-3042236746546522_weight_decay_0-16265604808312772_epoch__8_last.tar'
segmentation_path = '/home/ubuntu/deepsolar/checkpoint/ft_5000_segmentation_level_2_tune_sweep_best_models/psel_0-6956991453638643_lr_0-0002431815776062007_lr_decay_rate_0-3461895018894605_weight_decay_0-07219288691242215_epoch__0_last.tar'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 299
batch_size = 1   # must be 1 for testing segmentation
class_threshold = 0.5  # threshold probability to identify am image as positive
true_cam_threshold = 0.5
seg_threshold = 0.82    # threshold to identify a pixel as positive.
level = 2
cam_filepath = 'CAM_ft_5000_finetune_test.pickle' 

transform_test = transforms.Compose([
                 transforms.Resize((input_size, input_size)),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                 ])

def precision(stats):
    return (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FP'] + 0.00001)
                           
def recall(stats):
    return (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FN'] + 0.00001)

def f1_score(stats):
    prec = precision(stats)
    rec = recall(stats)
    print('precision: ' + str(prec))
    print('recall: ' + str(rec)) 
    print('stats: ')
    print(stats)
    f1 = (prec * rec) / (prec + rec)
    return f1

def metrics(stats):
    """
    Self-defined metrics function to evaluate and compare models
    stats: {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
    return: must be a single number """
    accuracy = (stats['TP'] + stats['TN']) / (stats['TP'] + stats['FP'] + stats['TN'] + stats['FN'])
    return accuracy

import matplotlib.pyplot as plt
def test_model(model, dataloader, metrics, class_threshold, seg_threshold, true_cam_threshold):
    stats = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    estimated_area = 0
    true_area = 0
    model.eval()
    CAM_list = []
    true_CAM_list = []
    iou = []
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
            if labels[i]==1: # oracle to see how much improving classification would improve estimation
            # if predicted_label.cpu().item():
                CAM_list.append((CAM, paths[i]))        # only use the generated CAM if it is predicted to be 1
                CAM_rescaled = (CAM - CAM.min()) / (CAM.max() - CAM.min())    # get predicted area
                CAM_pred = CAM_rescaled > seg_threshold
                pred_pixel_area = np.sum(CAM_pred)
                estimated_area += pred_pixel_area
                
            else:
                CAM_pred = np.zeros_like(CAM)
                CAM_list.append((np.zeros_like(CAM), paths[i]))  # otherwise the CAM is a totally black one

            if labels[i] == 1:
                # calculate true area
                img_path = os.path.splitext(paths[i])
                true_seg_path = img_path[0] + '_true_seg' + img_path[1]
                true_seg_img = Image.open(true_seg_path)
                transform = transforms.Compose([transforms.ToTensor()])
                true_seg = transform(true_seg_img)
                true_seg = true_seg.squeeze(0).cpu().numpy()
                true_pixel_area = np.sum(true_seg)
                true_pixel_area = true_pixel_area * (35. * 35.) / (true_seg.shape[0] * true_seg.shape[1])
                true_area += true_pixel_area
                CAM_true = skimage.transform.resize(true_seg, (35,35))
                CAM_true = CAM_true > true_cam_threshold

            else:
                CAM_true = np.zeros_like(CAM)
                true_CAM_list.append((CAM_true, paths[i]))

        intersection = CAM_true * CAM_pred
        union = CAM_true + CAM_pred
        if union.sum() > 0:
            iou.append(intersection.sum() / float(union.sum()))

        stats['TP'] += torch.sum((preds == 1) * (labels == 1)).cpu().item()
        stats['TN'] += torch.sum((preds == 0) * (labels == 0)).cpu().item()
        stats['FP'] += torch.sum((preds == 1) * (labels == 0)).cpu().item()
        stats['FN'] += torch.sum((preds == 0) * (labels == 1)).cpu().item()

    mean_iou = np.mean(iou)
    metric_value = metrics(stats)
    return stats, mean_iou, CAM_list, true_CAM_list, estimated_area, true_area

def run_eval(segmentation_path, seg_threshold, cam_filepath, data_dir, mode):
    if mode=='eval':
        print('evaluating on test set')
        dataset_test = ImageFolderModifiedEvaluation(data_dir, transform_test)
    else:
        dataset_test = ImageFolderModified(data_dir, transform_test)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)

    # model
    model = InceptionSegmentation(num_outputs=2, level=level)
    model.load_existing_params(segmentation_path)

    model = model.to(device)

    stats, iou, CAM_list, true_CAM_list, estimated_area, true_area = test_model(model, dataloader_test, metrics, class_threshold=class_threshold, seg_threshold=seg_threshold, true_cam_threshold=true_cam_threshold)

    cams = {'true': true_CAM_list,
            'pred': CAM_list}
    # dump CAM to file
    with open(cam_filepath, 'wb') as f:
        pickle.dump(cams, f)
    print("Saved CAMs to " + cam_filepath)

    prec = precision(stats)
    rec = recall(stats)
    acc = metrics(stats)
    area_fractional_error = (estimated_area - true_area) / true_area
    
    print('accuracy: ' + str(acc))
    print('precision: ' + str(round(prec, 4)))
    print('recall: ' + str(round(rec, 4)))
    print('iou: ' + str(iou))
    print('estimated_area: ' + str(estimated_area))
    print('true_area: ' + str(true_area))
    print('error ((est - true) / true): ' + str(area_fractional_error))

    return (acc, prec, rec, iou, area_fractional_error)

NAMES = ['ft_100', 'ft_500', 'ft_1000', 'ft_5000']
SEGMENTATION_PATHS = [
    '/home/ubuntu/deepsolar/checkpoint/ft_100_segmentation_level_2_sweep_best_models/psel_0_lr_0-0004228521209106932_lr_decay_rate_0-1180038312075688_weight_decay_0-2207810660245898_epoch__8_last.tar',
    '/home/ubuntu/deepsolar/checkpoint/ft_500_segmentation_level_2_tune_sweep_best_models/psel_0-8188110871270571_lr_0-0008709800586192774_lr_decay_rate_0-4240342150652191_weight_decay_0-2204940780344005_epoch__9_last.tar',
    '/home/ubuntu/deepsolar/checkpoint/ft_1000_segmentation_level_2_tune_sweep_best_models/psel_0-7246077310839052_lr_0-0006188732400968487_lr_decay_rate_0-33457774028832415_weight_decay_0-05248446740883417_epoch__10_last.tar',
    '/home/ubuntu/deepsolar/checkpoint/ft_5000_segmentation_level_2_tune_sweep_best_models/psel_0-6956991453638643_lr_0-0002431815776062007_lr_decay_rate_0-3461895018894605_weight_decay_0-07219288691242215_epoch__0_last.tar'
]
SEGMENTATION_THRESHOLDS = [0.73, 0.77, 0.86, 0.73]
FR_EVAL_DATA = '/home/ubuntu/deepsolar/data/ds-france/google/ft_eval'
US_EVAL_DATA = '/home/ubuntu/deepsolar/data/ds-usa/eval'

if __name__ == '__main__':
    names = []
    accs = []
    precs = []
    recs = []
    ious = []
    area_fractional_errors = []
    ds_names = []

    for name, path, seg_threshold in zip(NAMES, SEGMENTATION_PATHS, SEGMENTATION_THRESHOLDS):
        for ds_name, data_dir, mode in zip(['fr'], [FR_EVAL_DATA], ['val']):
        # for ds_name, data_dir, mode in zip(['fr', 'us'], [FR_EVAL_DATA, US_EVAL_DATA], ['val', 'eval']):
            print("############################ Evaluating {} on {} eval dataset #######################################".format(name, ds_name))
            cam_filepath = 'CAM_list_oracle_' + name + '_' + ds_name + '_eval.pickle'
            acc, prec, rec, iou, area_fractional_error = run_eval(path, seg_threshold, cam_filepath, data_dir, mode)
            names.append(name)
            accs.append(acc)
            precs.append(prec)
            recs.append(rec)
            ious.append(iou)
            area_fractional_errors.append(area_fractional_error)
            ds_names.append(ds_name)
            
    import pandas as pd

    results = pd.DataFrame(zip(names, ds_names, accs, precs, recs, ious, area_fractional_errors), 
                        columns = ['model', 'eval ds', 'accuracy', 'precision', 'recall', 'iou', 'area_fractional_error'])

    results.to_csv('results_oracle.csv')