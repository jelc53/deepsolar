import torch
import torch.nn as nn

import torch.nn.functional as F
from torchvision.models import Inception3
from collections import namedtuple

_InceptionOuputs = namedtuple('InceptionOuputs', ['logits', 'aux_logits'])


class InceptionGenerator(nn.Module):
    def __init__(self, num_outputs=784):
        super(InceptionGenerator, self).__init__()
        self.inception3 = Inception3_modified(num_classes=2, aux_logits=False, transform_input=False)

        # Sequential container
        # Layers will be added to it in the order they are passed in the constructor
        self.layers = nn.Sequential(
            nn.Conv2d(288, 512, bias=True, kernel_size=3, padding=1),
            nn.Linear(512, 256),  # an affine operation: y = Wx + b
            nn.ReLU(),            # applies the rectified linear unit function element-wise
            nn.Linear(256, num_outputs),  # final linear transformation to output size 28 * 28
        )

    def forward(self, x):
        _, x = self.inception3(x)
        x = self.layers(x)   # Pass the input through the layers
        return x


class ZGenerator(nn.Module):
    def __init__(self, in_channels=3, out_dim=784):  # input_dim=784
        super(ZGenerator, self).__init__()

        # Sequential container
        # Layers will be added to it in the order they are passed in the constructor
        self.layers = nn.Sequential(  # [N, 3, 299, 299]
            # insert logic here to handle input shape!!
            nn.Conv2d(in_channels, 64, bias=True, kernel_size=5, stride=1, padding=2),  # [N, 64, 784]
            nn.MaxPool2d(kernel_size=2, stride=2),  # [N, 32, 392]
            nn.Conv2d(32, 128, bias=True, kernel_size=5, stride=1, padding=2),  # [N, 128, 392]
            nn.MaxPool2d(kernel_size=2, stride=2),  # [N, 64, 196]
            nn.Flatten(),
            nn.Linear(12544, 1024),  # an affine operation: y = Wx + b
            nn.ReLU(),            # applies the rectified linear unit function element-wise
            nn.Linear(1024, out_dim),  # final linear transformation to output size 28 * 28
        )

    def forward(self, x):
        x = self.layers(x)   # Pass the input through the layers
        return x


def run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, loader_train, show_every=250,
              batch_size=128, noise_size=96, num_epochs=10):
    """
    Train a GAN!

    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    """
    images = []
    iter_count = 0
    for epoch in range(num_epochs):
        for x, _ in loader_train:
            if len(x) != batch_size:
                continue
            D_solver.zero_grad()
            real_data = x.type(dtype)
            logits_real = D(2* (real_data - 0.5)).type(dtype)

            g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
            fake_images = G(g_fake_seed).detach()
            logits_fake = D(fake_images.view(batch_size, 1, 28, 28))

            d_total_error = discriminator_loss(logits_real, logits_fake)
            d_total_error.backward()
            D_solver.step()

            G_solver.zero_grad()
            g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
            fake_images = G(g_fake_seed)

            gen_logits_fake = D(fake_images.view(batch_size, 1, 28, 28))
            g_error = generator_loss(gen_logits_fake)
            g_error.backward()
            G_solver.step()

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_total_error.item(),g_error.item()))
                imgs_numpy = fake_images.data.cpu().numpy()
                images.append(imgs_numpy[0:16])

            iter_count += 1

    return images

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


class Inception3_modified(Inception3):
    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        intermediate = x.clone()
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:
            return _InceptionOuputs(x, aux)
        return x, intermediate