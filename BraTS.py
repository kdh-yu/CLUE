import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import os
import json

class BraTSDataset(Dataset):
    def __init__(self, 
                 base_dir, 
                 sequence, 
                 view, 
                 train=True, 
                 transform=None):
        super().__init__()
        self.meta = os.path.join(base_dir, f'meta_{sequence}_{view}.json')
        self.view = view
        self.transform = transform
        
        with open(self.meta, 'r') as f:
            if train:
                self.data = json.load(f)['train']
            else:
                self.data = json.load(f)['test']
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_info = self.data[idx]
        image = nib.load(data_info['image']).get_fdata()
        mask = nib.load(data_info['mask']).get_fdata()
        s = data_info['slice']
        
        if self.view == 'axial':
            image = image[:, :, s]
            mask = mask[:, :, s]
        elif self.view == 'sagittal':
            image = image[:, s, :]
            mask = mask[:, s, :]
        elif self.view == 'coronal':
            image = image[s, :, :]
            mask = mask[s, :, :]
            
        if self.transform:
            image = self.transform[0](image)
            mask = self.transform[1](mask)
            
        mask = np.where(mask!=0, 1, 0)
            
        return image, mask