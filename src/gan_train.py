from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

import os
import random
from torch.nn import functional as F

from inception_modified import InceptionSegmentation
from gan import ZGenerator, run_a_gan


# Configuration
# directory for loading training/validation/test data
data_dir = '/Users/avdh/Desktop/deepsolar/data/ds-usa' #'/home/ubuntu/deepsolar/data/ds-usa/'  #'/home/ubuntu/projects/deepsolar/deepsolar_dataset_toy'
# path to load basic main branch model, "None" if not loading.
basic_params_path = '/Users/avdh/Desktop/deepsolar/models/deepsolar_seg_pretrained.pth' #'/home/ubuntu/deepsolar/models/deepsolar_seg_pretrained.pth'  #'/home/ubuntu/projects/deepsolar/deepsolar_pytorch_pretrained/deepsolar_pretrained.pth'
# path to load old model parameters, "None" if not loading.
old_ckpt_path = None  #'checkpoint/deepsolar_toy/deepsolar_seg_level1_5.tar'
# directory for saving model/checkpoint
ckpt_save_dir = 'checkpoint/gan_test'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = 'deepsolar_seg_level1'  # the prefix of the filename for saving model/checkpoint
return_best = True           # whether to return the best model according to the validation metrics
if_early_stop = True         # whether to stop early after validation metrics doesn't improve for definite number of epochs
level = 1                    # train the first level or second level of segmentation branch
input_size = 299              # image size fed into the mdoel
imbalance_rate = 5            # weight given to the positive (rarer) samples in loss function
learning_rate = 0.01          # learning rate
weight_decay = 0.00           # l2 regularization coefficient
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
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
        for x in ['train', 'val']
    }
    dataloaders_dict_gan = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        ) for x in ['train', 'val']
    }
    if not os.path.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)

    # instantiate model
    model = InceptionSegmentation(num_outputs=1, level=level)
    generator = ZGenerator(out_dim=3*299*299)  # TODO: check in/out dim

    # inception model parameters
    model.load_basic_params(basic_params_path)
    trainable_params = ['convolution1', 'linear1']
    only_train(model, trainable_params)

    # adversarial data augmentation
    D_solver = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))  # TODO: check if needs to match original
    G_solver = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def bce_loss(input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()

    def G_loss(logits_fake):
        loss = None
        N = logits_fake.shape[0]
        loss = bce_loss(logits_fake, torch.ones(N).type(torch.float32))
        return loss

    def D_loss(logits_real, logits_fake):  # TODO: check if needs to match original 
        loss = None
        N = logits_fake.shape[0]
        loss = bce_loss(logits_real, torch.ones(N).type(torch.float32)) + bce_loss(logits_fake, torch.zeros(N).type(torch.float32))
        return loss

    fake_images = run_a_gan(model, generator, D_solver, G_solver, D_loss, G_loss,
                            dataloaders_dict_gan['train'], show_every=250, batch_size=64,
                            noise_size=(3,299,299), num_epochs=10)

    # write fake images to file
    for idx, img in enumerate(fake_images):
        path = os.path.join(data_dir, 'fake')
        img_name = 'fake_{}.png'.format(idx)
        torchvision.utils.save_image(img, os.path.join(path, img_name))

    # add fakes to image datatsets object
    # fake_dataset = datasets.ImageFolder(os.path.join(data_dir, 'fake'), data_transforms['train'])
    # image_datasets['train'] = torch.utils.data.ConcatDataset([image_datasets['train'], fake_dataset])

    # dataloaders_dict = {  # update dataloader
    #     x: torch.utils.data.DataLoader(
    #         image_datasets[x],
    #         batch_size=batch_size,
    #         shuffle=True,
    #         num_workers=4
    #     ) for x in ['train', 'val']
    # }
