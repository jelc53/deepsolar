from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from image_dataset import ImageFolderModifiedValidation
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


# Configuration
# directory for loading training/validation/test data
data_dir = '/home/ubuntu/deepsolar/data/bdappv-france/ft_5000/'  
#'/home/ubuntu/deepsolar/data/ds-usa/'
#'/home/ubuntu/deepsolar/data/bdappv-france/ft_5000/'  
#'/home/ubuntu/projects/deepsolar/deepsolar_dataset_toy'

# path to load basic main branch model, "None" if not loading.
#basic_params_path = '/home/ubuntu/deepsolar/models/deepsolar_pretrained.pth'
basic_params_path = 'checkpoint/bdappv_ft5000_w0.001_lr0.001/deepsolar_classification_5_last.tar'
#basic_params_path = 'checkpoint/bdappv_ft100_w0.001_lr0.001/deepsolar_seg_level1_9_last.tar' 
#'/home/ubuntu/projects/deepsolar/deepsolar_pytorch_pretrained/deepsolar_pretrained.pth'

# path to load old model parameters, "None" if not loading.
level1_ckpt_path = 'checkpoint/bdappv_ft5000_w0.001_lr0.001/deepsolar_seg_level1_7_last.tar' 
old_ckpt_path = '/home/ubuntu/deepsolar/models/deepsolar_seg_pretrained.pth'
#'checkpoint/deepsolar_toy/deepsolar_seg_level1_5.tar'

# directory for saving model/checkpoint
ckpt_save_dir = 'checkpoint/bdappv_ft5000_w0.001_lr0.001'
#'checkpoint/pyt_test'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = 'deepsolar_seg_level2'  # the prefix of the filename for saving model/checkpoint
return_best = True           # whether to return the best model according to the validation metrics
if_early_stop = True         # whether to stop early after validation metrics doesn't improve for definite number of epochs
level = 2                    # train the first level or second level of segmentation branch
input_size = 299              # image size fed into the mdoel (originally 299)
imbalance_rate = 1            # weight given to the positive (rarer) samples in loss function (originally 5)
learning_rate = 0.001          # learning rate (originally 0.01)
weight_decay = 0.001           # l2 regularization coefficient (originally 0.00)
batch_size = 64
num_epochs = 10               # number of epochs to train
lr_decay_rate = 0.7           # learning rate decay rate for each decay step
lr_decay_epochs = 5          # number of epochs for one learning rate decay
early_stop_epochs = 5        # after validation metrics doesn't improve for "early_stop_epochs" epochs, stop the training.
save_epochs = 5              # save the model/checkpoint every "save_epochs" epochs
threshold = 0.5               # threshold probability to identify am image as positive


def RandomRotationNew(image):
    angle = random.choice([0, 90, 180, 270])
    image = TF.rotate(image, angle)
    return image

def only_train(model, trainable_params):
    """trainable_params: The list of parameters and modules that are set to be trainable.
    Set require_grad = False for all those parameters not in the trainable_params"""
    print('Only the following layers:')
    for name, p in model.named_parameters():
        p.requires_grad = False
        for target in trainable_params:
            if target == name or target in name:
                p.requires_grad = True
                print('    ' + name)
                break

def metrics(stats):
    """
    Self-defined metrics function to evaluate and compare models
    stats: {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
    return: must be a single number """
    precision = (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FP'] + 0.00001)
    recall = (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FN'] + 0.00001)
    return 0.5*(precision + recall)


def train_model(model, model_name, dataloaders, criterion, optimizer, metrics, num_epochs, threshold=0.5, training_log=None,
                verbose=True, return_best=True, if_early_stop=True, early_stop_epochs=10, scheduler=None,
                save_dir=None, save_epochs=5):
    since = time.time()
    if not training_log:
        training_log = dict()
        training_log['train_loss_history'] = []
        training_log['val_loss_history'] = []
        training_log['val_metric_value_history'] = []
        training_log['current_epoch'] = -1
    current_epoch = training_log['current_epoch'] + 1

    best_model_wts = copy.deepcopy(model.state_dict())
    best_optimizer_wts = copy.deepcopy(optimizer.state_dict())
    best_log = copy.deepcopy(training_log)

    best_metric_value = -np.inf
    nodecrease = 0  # to count the epochs that val loss doesn't decrease
    early_stop = False

    for epoch in range(current_epoch, current_epoch + num_epochs):
        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            stats = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs, testing=False)
                    loss = criterion(outputs, labels)

                    prob = F.softmax(outputs, dim=1)
                    preds = prob[:, 1] >= threshold

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                stats['TP'] += torch.sum((preds == 1) * (labels == 1)).cpu().item()
                stats['TN'] += torch.sum((preds == 0) * (labels == 0)).cpu().item()
                stats['FP'] += torch.sum((preds == 1) * (labels == 0)).cpu().item()
                stats['FN'] += torch.sum((preds == 0) * (labels == 1)).cpu().item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_metric_value = metrics(stats)
            print(stats)
            print("precision = ", (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FP'] + 0.00001))
            print("recall = ", (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FN'] + 0.00001))
 
            if verbose:
                print('{} Loss: {:.4f} Metrics: {:.4f}'.format(phase, epoch_loss, epoch_metric_value))

            training_log['current_epoch'] = epoch
            if phase == 'val':
                training_log['val_metric_value_history'].append(epoch_metric_value)
                training_log['val_loss_history'].append(epoch_loss)
                # deep copy the model
                if epoch_metric_value > best_metric_value:
                    best_metric_value = epoch_metric_value
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_optimizer_wts = copy.deepcopy(optimizer.state_dict())
                    best_log = copy.deepcopy(training_log)
                    nodecrease = 0
                else:
                    nodecrease += 1
            else:  # train phase
                training_log['train_loss_history'].append(epoch_loss)
                if scheduler != None:
                    scheduler.step()

            if nodecrease >= early_stop_epochs:
                early_stop = True

        if save_dir and epoch % save_epochs == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_log': training_log
            }
            torch.save(checkpoint,
                       os.path.join(save_dir, model_name + '_' + str(training_log['current_epoch']) + '.tar'))

        if if_early_stop and early_stop:
            print('Early stopped!')
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation metric value: {:4f}'.format(best_metric_value))     
    print("precision = ", (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FP'] + 0.00001))
    print("recall = ", (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FN'] + 0.00001))
    print(stats)

    # load best model weights
    if return_best:
        model.load_state_dict(best_model_wts)
        optimizer.load_state_dict(best_optimizer_wts)
        training_log = best_log

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_log': training_log
    }
    torch.save(checkpoint,
               os.path.join(save_dir, model_name + '_' + str(training_log['current_epoch']) + '_last.tar'))

    return model, training_log


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.Lambda(RandomRotationNew),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}


if __name__ == '__main__':
    assert level in [1, 2]
    # data
    image_datasets = {
        'train': ImageFolderModifiedValidation(os.path.join(data_dir, 'train'), data_transforms['train']),
        'val': ImageFolderModifiedValidation(os.path.join(data_dir, 'val'), data_transforms['val'])
    }
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x],
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=4)
        for x in ['train', 'val']
    }

    if not os.path.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)
    # model
    model = InceptionSegmentation(num_outputs=2, level=level)
    if level == 1 and basic_params_path and old_ckpt_path:
        model.load_mismatched_params(basic_params_path, old_ckpt_path)
    elif level == 1 and basic_params_path:
        model.load_basic_params(basic_params_path)
    elif level == 2 and basic_params_path and old_ckpt_path and level1_ckpt_path:
        model.load_mismatched_params(basic_params_path, old_ckpt_path, level1_ckpt_path)
    elif level == 2 and old_ckpt_path:
        model.load_existing_params(old_ckpt_path)

    if level == 1:
        trainable_params = ['convolution1', 'linear1']
    else:
        trainable_params = ['convolution2', 'linear2']
    only_train(model, trainable_params)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
                           weight_decay=weight_decay, amsgrad=True)
    class_weight = torch.tensor([1, imbalance_rate], dtype=torch.float).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_epochs, gamma=lr_decay_rate)

    training_log = None

    _, _ = train_model(model, model_name=model_name, dataloaders=dataloaders_dict, criterion=loss_fn,
                       optimizer=optimizer, metrics=metrics, num_epochs=num_epochs, threshold=threshold,
                       training_log=training_log, verbose=True, return_best=return_best,
                       if_early_stop=if_early_stop, early_stop_epochs=early_stop_epochs,
                       scheduler=scheduler, save_dir=ckpt_save_dir, save_epochs=save_epochs)

