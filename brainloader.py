import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
from PIL import Image
import os
import json

class ChestDataset(Dataset):
    def __init__(self, 
                 base_dir, 
                 train=False, 
                 transform=None):
        super().__init__()
        self.meta = os.path.join(base_dir, f'meta_chest.json')
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
            transforms.Resize(224),
            transforms.Normalize(0.5, 0.5),
        ])
        return tf(img)
    
    def __getitem__(self, idx):
        data_info = self.data[idx]
        image = Image.open(data_info['image'])
        image = self._transform(image)
        #mask = torch.zeros_like(image)
        label = data_info['label']
            
        return image, label