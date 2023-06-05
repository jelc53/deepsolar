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
from image_dataset import *
import wandb


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
                    if phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    prob = F.softmax(outputs, dim=1)
                    preds = prob[:, 1] >= threshold
                    true_labels = labels

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        true_labels = labels[:, 1] >= threshold  


                # statistics
                running_loss += loss.item() * inputs.size(0)
                stats['TP'] += torch.sum((preds == 1) * (true_labels == 1)).cpu().item()
                stats['TN'] += torch.sum((preds == 0) * (true_labels == 0)).cpu().item()
                stats['FP'] += torch.sum((preds == 1) * (true_labels == 0)).cpu().item()
                stats['FN'] += torch.sum((preds == 0) * (true_labels == 1)).cpu().item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_metric_value = metrics(stats)

            if verbose:
                print('{} Loss: {:.4f} Metrics: {:.4f}'.format(phase, epoch_loss, epoch_metric_value))

            training_log['current_epoch'] = epoch
            if phase == 'val':
                wandb.log({'epoch': epoch, 
                           'val_loss': epoch_loss, 
                           'val_f1': f1_score(stats),
                           'val_acc': epoch_metric_value})
                wandb.log({'epoch': epoch,
                           'val_prec': precision(stats), 
                           'val_rec': recall(stats)})
                
                wandb.log({'epoch': epoch,
                           'val_tp': stats['TP'], 
                           'val_fp': stats['FP'],
                           'val_tn': stats['TN'],
                           'val_fn': stats['FN']})
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
                wandb.log({'epoch': epoch, 
                           'train_loss': epoch_loss, 
                           'train_f1': f1_score(stats),
                           'train_acc': epoch_metric_value})

                wandb.log({'epoch': epoch,
                           'val_prec': precision(stats), 
                           'val_rec': recall(stats)})
                
                wandb.log({'epoch': epoch,
                           'train_tp': stats['TP'], 
                           'train_fp': stats['FP'],
                           'train_tn': stats['TN'],
                           'train_fn': stats['FN']})
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

    wandb.log({'best_val_acc': best_metric_value})

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation metric value: {:4f}'.format(best_metric_value))

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

input_size = 299              # image size fed into the model
data_transforms = {
    'train_before_interpolation': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ]),
    'train_after_interpolation': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Lambda(RandomRotationNew),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]),
    'val': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}


# SET PROJECT NAME HERE
PROJECT_NAME = 'cs231n_finetune_classification_lisa_ft_100_debug'

def run_sweep():
    wandb.init(project=PROJECT_NAME)
    # Configuration

    trainable_params = None     # layers or modules set to be trainable. "None" if training all layers
    return_best = True           # whether to return the best model according to the validation metrics
    if_early_stop = True         # whether to stop early after validation metrics doesn't improve for definite number of epochs
    imbalance_rate = 1            # weight given to the positive (rarer) samples in loss function
    batch_size = 64
    num_epochs = 15               # number of epochs to train
    lr_decay_epochs = 5          # number of epochs for one learning rate decay
    early_stop_epochs = 5        # after validation metrics doesn't improve for "early_stop_epochs" epochs, stop the training.
    save_epochs = 5              # save the model/checkpoint every "save_epochs" epochs
    threshold = 0.5               # threshold probability to identify am image as positive

    psel=wandb.config.psel
    learning_rate = wandb.config.lr        # learning rate 
    lr_decay_rate = wandb.config.lr_decay_rate           # learning rate decay rate for each decay step
    weight_decay = wandb.config.weight_decay           # l2 regularization coefficient
    model_name = 'psel_{}_lr_{}_lr_decay_rate_{}_weight_decay_{}_epoch_'.format('-'.join(str(psel).split('.')),
                                                                         '-'.join(str(learning_rate).split('.')), 
                                                                        '-'.join(str(lr_decay_rate).split('.')),
                                                                        '-'.join(str(weight_decay).split('.')))  # the prefix of the filename for saving model/checkpoint

    image_datasets = {'train': ImageFolderModifiedLisaTrain(os.path.join(wandb.config.data_dir,'train'), 
                                                            data_transforms['train_before_interpolation'],
                                                            data_transforms['train_after_interpolation'],
                                                            psel=psel),
                    'val': ImageFolderModifiedValidation(os.path.join(wandb.config.data_dir, 'val'), data_transforms['val'])}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                       shuffle=True, num_workers=0) for x in ['train', 'val']}
    # model
    model = Inception3(num_classes=2, aux_logits=True, transform_input=False)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
                           weight_decay=weight_decay, amsgrad=True)
    class_weight = torch.tensor([1, imbalance_rate], dtype=torch.float).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_epochs, gamma=lr_decay_rate)

    # load old parameters
    if wandb.config.old_ckpt_path:
        checkpoint = torch.load(wandb.config.old_ckpt_path, map_location=device)
        if wandb.config.old_ckpt_path[-4:] == '.tar':  # it is a checkpoint dictionary rather than just model parameters
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            training_log = checkpoint['training_log']
        else:
            model.load_state_dict(checkpoint)
            training_log = None
        print('Old checkpoint loaded: ' + wandb.config.old_ckpt_path)
    else:
        training_log = None

    # fix some layers and make others trainable
    if trainable_params:
        only_train(model, trainable_params)

    _, _ = train_model(model, model_name=model_name, dataloaders=dataloaders_dict, criterion=loss_fn,
                       optimizer=optimizer, metrics=metrics, num_epochs=num_epochs, threshold=threshold,
                       training_log=training_log, verbose=True, return_best=return_best,
                       if_early_stop=if_early_stop, early_stop_epochs=early_stop_epochs,
                       scheduler=scheduler, save_dir=wandb.config.ckpt_save_dir, save_epochs=save_epochs)

if __name__ == '__main__':
    wandb.login()
    sweep_configuration={
        'method': 'bayes',
        'metric': 
        {
            'goal': 'maximize', 
            'name': 'best_val_acc'
            },
        'parameters': 
        { 
            'lr': {'max': 0.005, 'min': 0.000005},
            'psel': {'values': [0.0, 0.5, 1.0]}, 
            'weight_decay': {'min': 0.0, 'max': 0.1},  
            'lr_decay_rate': {'min': 0.6, 'max': 1.0},
        # directory for loading training/validation/test data
        'data_dir' : {'values': ['/home/ubuntu/deepsolar/data/ds-france/google/ft_100/']},  #'/home/ubuntu/projects/deepsolar/deepsolar_dataset_toy'
        # path to load old model/checkpoint, "None" if not loading.
        'old_ckpt_path' : {'values': ['/home/ubuntu/deepsolar/models/deepsolar_pretrained.pth']},
        # directory for saving model/checkpoint
        'ckpt_save_dir' : {'values': ['/home/ubuntu/deepsolar/checkpoint/' + PROJECT_NAME]},
        }  
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=PROJECT_NAME)
    wandb.agent(sweep_id, function=run_sweep, count=25) 

