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
import sp

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim

# torchvision
import torchvision.utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Fix the random seed 
torch.manual_seed(12)

# arg parser
parser = argparse.ArgumentParser()

parser.add_argument('--target_path', type=str, default='../data/patches_32x32x32/P04288.pt', help='Target input')
parser.add_argument('--device', type=str, default='cuda', help='Tensor device. [cuda|cpu]. Default: cuda')
parser.add_argument('--opt_steps', type=int, default=30, help='Number of LBFGS steps')
parser.add_argument('--content_weight', type=float, default=0.000001, help='Content (vs. style) weight')
parser.add_argument('--output_path', type=str, default='../out_volumes', help='Output directory')

parser.add_argument('--target_bmd', type=float, default=115.3, help='BMD - suggested range: [,]')
parser.add_argument('--target_bvtv', type=float, default=28.1, help='BV/TV - suggested range: [,]')
parser.add_argument('--target_tmd', type=float, default=324.4, help='TMD - suggested range: [,]')

# PW-GAN GP params
parser.add_argument('--max_res', type=int, default=3, help='Max upsampled resolution (3->32x32x32)')
parser.add_argument('--in_nch', type=int, default=1, help='Number of input channels')
parser.add_argument('--use_batch_norm', type=int, default=0, help='Use BatchNorm')
parser.add_argument('--use_weight_scale', type=int, default=1, help='Use WeightScale')
parser.add_argument('--use_pixel_norm', type=int, default=1, help='Use PixelNorm')
parser.add_argument('--pretrained_model_path', type=str, default='../trained_models/g_net.pth', help='Generator model path')


# Sample optimizer
def optimizeSample(z_ini,target_params_pc,target_vol,opt):
    
    # LBFGS Optimizer
    optimizer = torch.optim.LBFGS([z_ini],line_search_fn='strong_wolfe')       
    steps = opt.opt_steps
    content_weight = opt.content_weight
    crit_l2 = nn.MSELoss()
    
    for i in range(steps):
        
        def closure():
            z_ini.data.clamp_(-1, 1)
            optimizer.zero_grad()

            # Generate some volume
            output = netGs(z_ini)

            # Compute its parameters
            output_params = SP2.ReducedVC(denorm_volume(output)).flatten()

            # Content loss
            loss_cont = crit_l2(denorm_volume(output),target_vol)

            # Parameters loss
            loss_para = crit_l2(output_params,target_params_pc)

            loss = content_weight * loss_cont + (1.-content_weight)* loss_para 

            loss.backward()
            if(i%10==0): print("[{}] - Content Loss: {:.4f} - Parameter Loss: {:.4f} - Total Loss {:.4f}".format(i,content_weight *loss_cont.item(),(1.-content_weight)* loss_para.item(),loss.item()))
            return loss
        
        optimizer.step(closure)   
        z_ini.data.clamp_(-1, 1)

    output = netGs(z_ini)
    return output, z_ini

# synthesize
def synthesize(opt):
    # options
    
    # device
    if(opt.device == 'cuda'):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
        
    # hyperparameters
    max_res = opt.max_res 
    nch     = opt.in_nch                     # number of channels
    bn      = opt.use_batch_norm             # batchnorm
    ws      = opt.use_weight_scale           # weightscale
    pn      = opt.use_pixel_norm             # pixelnorm
    
    # load generator model
    netGs = Generator(max_res=max_res, nch=nch, nc=1, bn=bn, ws=ws, pn=pn).to(device)
    netGs.load_state_dict(torch.load(opt.pretrained_model_path))
    netGs.eval()
    
    # sample random initial z (starting optimization point)
    z_ini = hypersphere(torch.randn(1, nch * 32, 1, 1, 1, device=device)).requires_grad_()
    random_ini_vol = netGs(z_ini)

    # load target volume 
    target_vol = torch.load(opt.target_path) 
    target_vol_params = sp.VCNoSD(target_vol).flatten()
    print("Input vol params BMD:{:.4f}, BV/TV:{:.4f}, TMD:{:.4f}".format(target_vol_params[0],target_vol_params[1],target_vol_params[2]))
    
    # target params to optimize
    target_opt_params = torch.Tensor([opt.target_bmd,opt.target_bvtv/100.,opt.target_tmd]).to(device)
    
    # optim volume
    output_volume, z = optimizeSample(random_ini_vol,target_opt_params,target_vol,opt)
    
    # print output info
    out_params = SP2.VCNoSD(denorm_volume(output_volume)).flatten()
print("out_params BMD:{:.4f}, BV/TV:{:.4f}, TMD:{:.4f}".format(out_params[0],out_params[1],out_params[2]))

    # save
    torch.save(output_volume.detach().cpu().numpy(),os.path.join(opt.output_path,"volume"))
    torch.save(z.detach().cpu().numpy(),os.path.join("generator_z"))

if __name__ == '__main__':
    opt = parser.parse_args()   # get training options
    synthesize(opt)