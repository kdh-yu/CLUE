import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
from PIL import Image
import os
import json

class RESCDataset(Dataset):
    def __init__(self, 
                 base_dir, 
                 train=False, 
                 transform=None):
        super().__init__()
        self.meta = os.path.join(base_dir, f'meta_RESC.json')
        self.transform = transform
        
        with open(self.meta, 'r') as f:
            if train:
                self.data = json.load(f)['train']
            else:
                self.data = json.load(f)['test']
            
    def __len__(self):
        return len(self.data)
    
    def _convert(self, img: torch.Tensor):
        return img.repeat(3, 1, 1)
    
    def _transform(self, img):
        tf = transforms.Compose([
            transforms.ToTensor(),
            self._convert,
            transforms.Resize((224, 224)),
            transforms.Normalize(0.5, 0.5),
        ])
        return tf(img)
    
    def _transform_mask(self, img):
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])
        return tf(img)
    
    def __getitem__(self, idx):
        data_info = self.data[idx]
        image = Image.open(data_info['image'])
        image = self._transform(image)
        if data_info['mask'] == -1:
            mask = np.zeros(shape=(1, 224, 224))
        else:
            mask = self._transform_mask(Image.open(data_info['mask']).convert("L"))
        label = data_info['label']
        
        mask = np.where(mask!=0, 1, 0)
        return image, mask