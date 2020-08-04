#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Code from MICCAI 2020 https://www.miccai2020.org/en/ paper:
Generative Modelling of 3D in-silico Spongiosa with Controllable Micro-Structural Parameters 
by Emmanuel Iarussi, Felix Thomsen, Claudio Delrieux is licensed under CC BY 4.0. 
To view a copy of this license, visit https://creativecommons.org/licenses/by/4.0

"""

import torch
import os
from matplotlib import pyplot as plt

def _mean(data,conv_size=(16,16,16),stride=(16,16,16)):
    kernel = (torch.ones((1)).expand((1,1,*conv_size))/torch.tensor(conv_size).prod()).float()
    kernel = kernel.type(torch.cuda.FloatTensor)
    return torch.nn.functional.conv3d(data,kernel,stride = stride)

def _var(data,m_x,conv_size=(16,16,16),stride=(16,16,16)): 
    m_xx = _mean(data**2,conv_size,stride)
    v2 = 1/(conv_size[0]*conv_size[1]*conv_size[2])
    return (m_xx - m_x*m_x)/(1-v2) 

# map data to [0,inf]: y = np.max(x,0)
def _softplus2(x,epsilon=1.0):
    return torch.nn.functional.softplus(x-epsilon,1./epsilon)+epsilon

###############################################################################
###### local parameters of voxel-counting #####################################
###### BMD, BV/TV, TMD, SD ####################################################
###############################################################################
# BMD, SD, BV/TV_1,TMD_1,...,BV/TV_n, TMD_n
def VC(data,si=32,st=32,threshold=[225.],sigma=10.,epsilon=0.0001):
    bmd = _mean(data,(si,si,si),(st,st,st))
    sd = torch.sqrt(_var(data,bmd,(si,si,si),(st,st,st)))
    ret = torch.cat((bmd,sd),dim=1)
    for i,t in enumerate(threshold):
        pw_bvtv = torch.sigmoid((data-t)/sigma)
        bvtv = _mean(pw_bvtv,(si,si,si),(st,st,st))
        pw_tmd = torch.mul(pw_bvtv,data)
        smooth_tmd = _mean(pw_tmd,(si,si,si),(st,st,st))
        tmd = smooth_tmd / _softplus2(bvtv,epsilon)
        ret = torch.cat((ret,bvtv,tmd),dim=1)
    return ret


# BMD,  BV/TV_1,TMD_1,...,BV/TV_n, TMD_n
def VCNoSD(data,si=32,st=32,threshold=[225.],sigma=10.,epsilon=0.0001):
    bmd = _mean(data,(si,si,si),(st,st,st))
    ret = bmd
    for i,t in enumerate(threshold):
        pw_bvtv = torch.sigmoid((data-t)/sigma)
        bvtv = _mean(pw_bvtv,(si,si,si),(st,st,st))
        pw_tmd = torch.mul(pw_bvtv,data)
        smooth_tmd = _mean(pw_tmd,(si,si,si),(st,st,st))
        tmd = smooth_tmd / _softplus2(bvtv,epsilon)
        ret = torch.cat((ret,bvtv,tmd),dim=1)
    return ret


# apply linear-combination with fixed weights, extracted by a PCA from
# real patches: reduction to two variables could be done with keeping 
# 99% of variance. Variables have 0 mean but different standard-deviations
# according to the weighting (similar to paper)
# they are computed over not normalized volumes!
def ReducedVC(data):
    vc = VC(data,threshold = [175.,225.,275.])
    means = [1.2358242e+02, 1.2596702e+02, 2.5884491e-01, 3.0270697e+02,
       1.8897282e-01, 3.4193488e+02, 1.3487895e-01, 3.8006693e+02]
    weights = [2.11633135e-02, 3.33693302e-02, 1.74624375e+01, 5.31215568e-02,
       1.52390445e+01, 3.50745131e-02, 2.60980711e+01, 4.72287886e-02]
    for i in range(8):
        vc[:,i,...] = (vc[:,i,...]-means[i])*weights[i]
    # 99% eigenvalues:
    e0 = torch.tensor([0.2088901 , 0.23005097, 0.3977575 , 0.43240823, 0.30314409,
       0.30034651, 0.45425018, 0.41207521],dtype =vc.dtype,
    device=vc.device).unsqueeze(0).unsqueeze(2).unsqueeze(2).unsqueeze(2).expand(vc.size())
    e1 = torch.tensor([ 0.21908473, -0.08819675,  0.52374055, -0.4046202 ,  0.30235876,
       -0.31234922,  0.30484268, -0.47359226],dtype =vc.dtype,
    device=vc.device).unsqueeze(0).unsqueeze(2).unsqueeze(2).unsqueeze(2).expand(vc.size())
    
    ret = torch.cat(((vc*e0).sum(dim=1,keepdim=True),(vc*e1).sum(dim=1,keepdim=True)),dim=1)
    return ret
