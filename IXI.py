import torch
import torch.nn as nn
from torch.utils.data import Dataset
import nibabel as nib
import os
import json

class IXIDataset(Dataset):
    def __init__(self, meta_dir, transform, modality):
        super().__init__()
        with open(meta_dir, 'r') as f:
            self.data = json.load(f)
        self.view = [i for i in ['axial', 'sagittal', 'coronal'] if i in meta_dir][0]
        self.transform = transform
        self.modality = modality
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = nib.load(self.data[idx][self.modality]).get_fdata()
        sl = self.data[idx]['slice']
        if self.view == 'axial':
            image = image[:, :, sl]
        elif self.view == 'sagittal':
            image = image[:, sl, :]
        elif self.view == 'coronal':
            image = image[sl, :, :]
            
        if self.transform:
            image = self.transform(image)
            
        return image