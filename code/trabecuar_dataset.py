"""
Code from MICCAI 2020 https://www.miccai2020.org/en/ paper:
Generative Modelling of 3D in-silico Spongiosa with Controllable Micro-Structural Parameters 
by Emmanuel Iarussi, Felix Thomsen, Claudio Delrieux is licensed under CC BY 4.0. 
To view a copy of this license, visit https://creativecommons.org/licenses/by/4.0

"""

import os
import torch
from torch.utils.data import Dataset
from utils3d import *

# load and normalize data
class BonesPatchDataset(Dataset):
    """Bones Patch dataset."""

    def __init__(self, root_dir, file_list, transform=None):
        self.root_dir  = root_dir
        self.file_list = file_list
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # load volume
        vol_name = os.path.join(self.root_dir,self.file_list[idx])
        vol = torch.load(vol_name)           
        
        # normalize [-350,1100] h-units to [-1,1] range
        vol = normalize_volume(vol)
        
        # transform to tensor
        if self.transform:
            vol = self.transform(vol)
            
        # clamp in range -1,1 (just in case some outlier is out of the range)
        vol = torch.clamp(vol, -1., 1.)         
        vol = vol.unsqueeze(0)
        vol = vol.type(torch.FloatTensor)
        return vol
