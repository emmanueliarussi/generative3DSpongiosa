"""
Code from MICCAI 2020 https://www.miccai2020.org/en/ paper:
Generative Modelling of 3D in-silico Spongiosa with Controllable Micro-Structural Parameters 
by Emmanuel Iarussi, Felix Thomsen, Claudio Delrieux is licensed under CC BY 4.0. 
To view a copy of this license, visit https://creativecommons.org/licenses/by/4.0

"""
# misc
import os
import copy
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# network Model (PW-GAN GP 3D)
from progressive_model3d import *
from trabecuar_dataset import BonesPatchDataset
from utils3d import *

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim

# torchvision
import torchvision.utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

torch.manual_seed(12)

# arg parser
parser = argparse.ArgumentParser()

# training params
parser.add_argument('--data_path', type=str, default='../data/patches_32x32x32', help='Input data directory')
parser.add_argument('--device', type=str, default='cuda', help='Training device. [cuda|cpu]. Default: cuda')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16, help='Size of the training batch')
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
parser.add_argument('--fix_random_seed', type=int, default=1, help='Fix random seed')
parser.add_argument('--model_checkpoints', type=int, default=5, help='Checkpoints interval')

# PW-GAN GP params
parser.add_argument('--max_res', type=int, default=3, help='Max upsampled resolution (3->32x32x32)')
parser.add_argument('--in_nch', type=int, default=1, help='Number of input channels')
parser.add_argument('--use_batch_norm', type=int, default=0, help='Use BatchNorm')
parser.add_argument('--use_weight_scale', type=int, default=1, help='Use WeightScale')
parser.add_argument('--use_pixel_norm', type=int, default=1, help='Use PixelNorm')

parser.add_argument('--lambda_gradient_penalty', type=float, default=10, help='Lambda for gradient penalty')
parser.add_argument('--gamma_gradient_penalty', type=float, default=1, help='Gamma for gradient penalty')
parser.add_argument('--n_iter', type=int, default=1, help='Number of epochs to train before changing the progress')
parser.add_argument('--epsilon_drift', type=float, default=0.001, help='Epsilon drift for discriminator loss')
parser.add_argument('--batch_sizes', nargs='+', default=[16, 16, 16, 16], help='List of batch sizes during the training')


def saveTensorBatch(aTensor,vals=None,epoch=0,iterat=0):
    fig = plt.figure(figsize=(10,10))
    for i in range(16):
        sub = fig.add_subplot(4, 4, i + 1)
        #sub.set_title(str(i))
        if vals is not None:
            if vals[i]>0:
                sub.set_title("Real")
            else:
                sub.set_title("Fake")
        sub.imshow(aTensor[i,:,:])    
    plt.savefig('../out/epoch_{}_iter_{}.png'.format(str(epoch).zfill(2),str(iterat).zfill(4)),dpi=150)
    
# train
def train(opt):
    
    # options
   
    # data path
    data_path = opt.data_path

    # hyperparameters
    if(opt.device == 'cuda'):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    num_epochs    = opt.epochs
    batch_size    = opt.batch_size
    learning_rate = opt.lr
    savemodel     = opt.model_checkpoints
    max_res = opt.max_res 
    nch     = opt.in_nch                     # number of channels
    bn      = opt.use_batch_norm             # batchnorm
    ws      = opt.use_weight_scale           # weightscale
    pn      = opt.use_pixel_norm             # pixelnorm
    batchSizes = opt.batch_sizes             # list of batch sizes during the training
    lambdaGP   = opt.lambda_gradient_penalty # lambda for gradient penalty
    gamma      = opt.gamma_gradient_penalty  # gamma for gradient penalty
    n_iter = opt.n_iter                      # number of epochs to train before changing the progress
    e_drift = opt.epsilon_drift              # epsilon drift for discriminator loss
    
    # transforms (no rotations yet)
    img_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # dataset
    file_list = os.listdir(data_path) 
    train_dataset = BonesPatchDataset(data_path, file_list, transform=img_transform)
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
    
    # training progress sample
    z_fixed = hypersphere(torch.randn(batch_size, nch * 32, 1, 1, 1, device=device))
    fake_images = netG(z_fixed)
    saveTensorBatch(fake_images.detach().cpu().numpy().squeeze()[:,16,:,:])
    
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

            # compute gradient penalty for WGAN-GP 
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
        saveTensorBatch(fake_images.detach().cpu().numpy().squeeze()[:,16,:,:],epoch=epoch,iterat=i)
        print("epoch [{}] - d_loss:{} - d_loss_W: {} - progress:{}".format(epoch,np.mean(lossEpochD),np.mean(lossEpochD_W),P.p))

        # save status 
        if P.p >= P.pmax and not epoch % savemodel:
            torch.save(netG.state_dict(),  os.path.join('../model_checkpoints', 'g_nch-{}_epoch-{}.pth'.format(nch,epoch)))
            torch.save(netD.state_dict(),  os.path.join('../model_checkpoints', 'd_nch-{}_epoch-{}.pth'.format(nch,epoch)))
            torch.save(netGs.state_dict(), os.path.join('../model_checkpoints', 'gs_nch-{}_epoch-{}.pth'.format(nch,epoch))) 

        epoch += 1

if __name__ == '__main__':
    opt = parser.parse_args()   # get training options
    train(opt)