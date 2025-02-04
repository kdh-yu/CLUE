import json
import numpy as np
import os
import sys
import nibabel as nib
from tqdm import tqdm

########## Data Setting ########## 
val_path = '/home/kdh/code/CLUE/datasets/RESC/Val/'
test_path = '/home/kdh/code/CLUE/datasets/RESC/Test/'

dataset = []
for label in ['good', 'Ungood']:
    datalist = os.listdir(os.path.join(val_path, 'val', label))
    masklist = os.listdir(os.path.join(val_path, 'val_label', 'Ungood'))
    for data in datalist:
        dataset.append({
            'image' : os.path.join(val_path, 'val', label, data),
            'mask' : os.path.join(val_path, 'val_label', label, data) if label=='Ungood' else -1,
            'label' : int(label=='Ungood')
        })
        
    datalist = os.listdir(os.path.join(test_path, 'test', label))
    masklist = os.listdir(os.path.join(test_path, 'test_label', 'Ungood'))
    for data in datalist:
        dataset.append({
            'image' : os.path.join(test_path, 'test', label, data),
            'mask' : os.path.join(test_path, 'test_label', label, data) if label=='Ungood' else -1,
            'label' : int(label=='Ungood')
        })

########## Write ########## 
with open(f'/home/kdh/code/CLUE/metadata/meta_RESC.json', 'w') as f:
    json.dump({'test' : dataset}, f, indent=4)
