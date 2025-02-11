import json
import numpy as np
import os
import sys
import nibabel as nib
from tqdm import tqdm

########## Data Setting ########## 
val_path = '/home/kdh/code/CLUE/datasets/BMAD/BraTS2021_slice/valid/'
test_path = '/home/kdh/code/CLUE/datasets/BMAD/BraTS2021_slice/test/'

dataset = []
for label in ['good', 'Ungood']:
    datalist = os.listdir(os.path.join(val_path, label, 'img'))
    masklist = os.listdir(os.path.join(val_path, 'Ungood', 'label'))
    for data in datalist:
        dataset.append({
            'image' : os.path.join(val_path, label, 'img', data),
            'mask' : os.path.join(val_path, label, 'img', data) if label=='Ungood' else -1,
            'label' : int(label=='Ungood')
        })
        
    datalist = os.listdir(os.path.join(test_path, label, 'img'))
    masklist = os.listdir(os.path.join(test_path, 'Ungood', 'label'))
    for data in datalist:
        dataset.append({
            'image' : os.path.join(test_path, label, 'img', data),
            'mask' : os.path.join(test_path, label, 'img', data) if label=='Ungood' else -1,
            'label' : int(label=='Ungood')
        })

########## Write ########## 
with open(f'/home/kdh/code/CLUE/metadata/meta_brain.json', 'w') as f:
    json.dump({'test' : dataset}, f, indent=4)
