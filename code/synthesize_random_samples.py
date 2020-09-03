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

# parameters computation
import sp

# arg parser
parser = argparse.ArgumentParser()

# generation params
parser.add_argument('--device', type=str, default='cuda', help='Training device. [cuda|cpu]. Default: cuda')
parser.add_argument('--model_path', type=str, default='../trained_models/g_net.pth', help='Trained model path')
parser.add_argument('--output_dir', type=str, default='../random_synthetic_samples', help='Path to output dir')
parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to synthesize')

# PW-GAN GP params
parser.add_argument('--max_res', type=int, default=3, help='Max upsampled resolution (3->32x32x32)')
parser.add_argument('--in_nch', type=int, default=1, help='Number of input channels')
parser.add_argument('--use_batch_norm', type=int, default=0, help='Use BatchNorm')
parser.add_argument('--use_weight_scale', type=int, default=1, help='Use WeightScale')
parser.add_argument('--use_pixel_norm', type=int, default=1, help='Use PixelNorm')
        
# synthesize bone samples
def synthesize(opt):
    
    print("Starting to synthesize.. ")

    # output dir for synthetic samples
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)      
        
    # hyperparameters
    if(opt.device == 'cuda'):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
        
    # Generator Network
    max_res = opt.max_res 
    nch     = opt.in_nch                     # number of channels
    bn      = opt.use_batch_norm             # batchnorm
    ws      = opt.use_weight_scale           # weightscale
    pn      = opt.use_pixel_norm             # pixelnorm

    # Load pretrained model
    netGs = Generator(max_res=max_res, nch=nch, nc=1, bn=bn, ws=ws, pn=pn).to(device)
    netGs.load_state_dict(torch.load(opt.model_path))
    netGs.eval()
    
    # Generate samples
    for sample_inx in range(0,opt.num_samples):
        # Sample random z
        z_ini = hypersphere(torch.randn(16, nch * 32, 1, 1, 1, device=device))        
        # Generate bone volume
        sample_vol = netGs(z_ini) 
        # Compute its parameters        
        params = sp.VCNoSD(denormalize_volume(sample_vol.detach())).flatten()
        print("Sample {} stats:".format(sample_inx))
        print(" => BMD:{:.4f}".format(params[0]))
        print(" => BV/TV:{:.4f}".format(params[1]))
        print(" => TMD:{:.4f}".format(params[2]))
                    
        out_path = os.path.join(opt.output_dir,"synth_sample_{}.pt".format(sample_inx))
        torch.save(sample_vol.detach().squeeze().cpu().numpy(),out_path)
    
if __name__ == '__main__':
    opt = parser.parse_args()   # get training options
    synthesize(opt)
