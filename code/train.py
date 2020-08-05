"""
Code from MICCAI 2020 https://www.miccai2020.org/en/ paper:
Generative Modelling of 3D in-silico Spongiosa with Controllable Micro-Structural Parameters 
by Emmanuel Iarussi, Felix Thomsen, Claudio Delrieux is licensed under CC BY 4.0. 
To view a copy of this license, visit https://creativecommons.org/licenses/by/4.0

"""
# misc
import numpy as np
import scipy.ndimage.morphology as morph
from tqdm import tqdm 
import ipyvolume as ipv
import matplotlib.pyplot as plt 

import sys
import os
import io
import random
import copy

# network Model (PW-GAN GP 3D)
from progressive_model3d import *
from trabecuar_dataset import BonesPatchDataset
from utils3d import *

# pyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torchvision
import torchvision.utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from progress_bar import printProgressBar

torch.manual_seed(12)

# data path
data_path = 'data/patches_32x32x32'

# hyperparams. Need to set these using command args
device        = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
latent_dims   = 10
num_epochs    = 30
batch_size    = 16
learning_rate = 1e-4
use_gpu       = True
savemodel     = 5

# list of PW-GAN GP params
max_res = 3 # for 32x32x32 output
nch     = 1 # number of channels
bn      = False # Use BatchNorm
ws      = True  # Use WeightScale
pn      = True  # Use PixelNorm
batchSizes = [16, 16, 16, 16]  # list of batch sizes during the training
lambdaGP   = 10                # lambda for gradient penalty
gamma      = 1                 # gamma for gradient penalty
n_iter = 5      # number of epochs to train before changing the progress
e_drift = 0.001 # epsilon drift for discriminator loss

# train
def train():
    # transforms (no rotations yet)
    img_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # dataset
    file_list = os.listdir(data_path) 
    train_dataset = BonesPatchDataset(path, file_list, transform=img_transform)
    print('train ({}), test ({}), val ({})'.format(len(train_dataset),0,0))
    
    # dataloader
    bone_train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # networks
    netG = Generator(max_res=max_res, nch=nch, nc=1, bn=bn, ws=ws, pn=pn).to(device)
    netD = Discriminator(max_res=max_res, nch=nch, nc=1, bn=bn, ws=ws).to(device)
    netGs = copy.deepcopy(netG)
    
    # params count
    pytorch_total_params = sum(p.numel() for p in netG.parameters())
    print('Generator has',pytorch_total_params,'parameters')
    pytorch_total_params = sum(p.numel() for p in netD.parameters())
    print('Discriminator has',pytorch_total_params,'parameters')
    
    # gradient penalty
    GP = GradientPenalty(batchSizes[0], lambdaGP, gamma, device=device)
    
    epoch       = 0
    global_step = 0
    total       = len(bone_train_dataloader)
    d_losses    = np.array([])
    d_losses_W  = np.array([])
    g_losses    = np.array([])

    # training progress
    P = Progress(n_iter, max_res, batchSizes)
    P.progress(epoch, 1, total)
    GP.batchSize = P.batchSize
    
    # adam optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=1e-3, betas=(0, 0.99))
    optimizerD = optim.Adam(netD.parameters(), lr=1e-3, betas=(0, 0.99))

    # training loop
    while epoch<num_epochs:
        lossEpochG = []
        lossEpochD = []
        lossEpochD_W = []

        netG.train()
        P.progress(epoch, 1, total)

        for i, batch in enumerate(bone_train_dataloader):
            P.progress(epoch, i + 1, total + 1)
            global_step += 1

            # build mini-batch
            batch = batch.to(device)
            batch_original = batch
            batch = P.resize(batch)

            # ============= Train the discriminator =============#
            # zeroing gradients in D
            netD.zero_grad()

            # compute fake images with G
            z = hypersphere(torch.randn(P.batchSize, nch * 32, 1, 1, 1, device=device))
            with torch.no_grad():
                fake_images = netG(z, P.p)

            # compute scores for real images
            D_real = netD(batch, P.p)
            D_realm = D_real.mean()

            # compute scores for fake images
            D_fake = netD(fake_images, P.p)
            D_fakem = D_fake.mean()

            # compute gradient penalty for WGAN-GP as defined in the article
            gradient_penalty = GP(netD, batch.data, fake_images.data, P.p)

            # prevent D_real from drifting too much from 0
            drift = (D_real ** 2).mean() * e_drift

            # backprop + optimize
            d_loss = D_fakem - D_realm
            d_loss_W = d_loss + gradient_penalty + drift
            d_loss_W.backward()
            optimizerD.step()

            lossEpochD.append(d_loss.item())
            lossEpochD_W.append(d_loss_W.item())

            # =============== Train the generator ===============#
            # zeroing gradients in G
            netG.zero_grad()
            
            # compute fake images with G
            z = hypersphere(torch.randn(P.batchSize, nch * 32, 1, 1, 1, device=device))
            fake_images = netG(z, P.p)
            
            # compute scores with new fake images
            G_fake = netD(fake_images, P.p)
            G_fakem = G_fake.mean()
            
            # no need to compute D_real as it does not affect G
            g_loss = -G_fakem

            # optimize
            g_loss.backward()
            optimizerG.step()

            lossEpochG.append(g_loss.item())

            # update Gs with exponential moving average
            exp_mov_avg(netGs, netG, alpha=0.999, global_step=global_step)
            
        fake_images = netGs(z_fixed)
        showTensorBatch(fake_images.detach().cpu().numpy().squeeze()[:,16,:,:],epoch=epoch,iterat=i)
        print("epoch [{}] - d_loss:{} - d_loss_W: {} - progress:{}".format(epoch,np.mean(lossEpochD),np.mean(lossEpochD_W),P.p))

        # save status 
        if P.p >= P.pmax and not epoch % savemodel:
            torch.save(netG.state_dict(),  os.path.join('model_checkpoints', 'g_nch-{}_epoch-{}.pth'.format(nch,epoch)))
            torch.save(netD.state_dict(),  os.path.join('model_checkpoints', 'd_nch-{}_epoch-{}.pth'.format(nch,epoch)))
            torch.save(netGs.state_dict(), os.path.join('model_checkpoints', 'gs_nch-{}_epoch-{}.pth'.format(nch,epoch))) 

        epoch += 1

if __name__ == '__main__':
    train()