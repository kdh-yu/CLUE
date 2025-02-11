import json
import numpy as np
import os
import sys
import nibabel as nib
from tqdm import tqdm

########## Data Setting ########## 
val_path = '/home/kdh/code/CLUE/datasets/Chest-RSNA/val/'
test_path = '/home/kdh/code/CLUE/datasets/Chest-RSNA/test/'

dataset = []
for label in ['good', 'Ungood']:
    #datalist = os.listdir(os.path.join(val_path, label))
    #for data in datalist:
    #    dataset.append({
    #        'image' : os.path.join(val_path, label, data),
    #        'label' : int(label=='Ungood')
    #    })
        
    datalist = os.listdir(os.path.join(test_path, label))
    for data in datalist:
        dataset.append({
            'image' : os.path.join(test_path, label, data),
            'label' : int(label=='Ungood')
        })

########## Write ########## 
with open(f'/home/kdh/code/CLUE/meta/meta_chest.json', 'w') as f:
    json.dump({'test' : dataset}, f, indent=4)
