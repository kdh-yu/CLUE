import json
import numpy as np
import os
import sys
import nibabel as nib
from tqdm import tqdm

np.random.seed(2025)

########## Data Setting ########## 
data_path = '/home/kdh/code/CLUE/datasets/BraTS/BraTS2021_Training_Data'
sequence = sys.argv[1].lower()
view = sys.argv[2]
train_num = 400

data_list = np.array([i for i in os.listdir(data_path) if 'BraTS' in i])

train_idx = np.random.choice(np.arange(len(data_list)), train_num, replace=False)
test_idx = np.setdiff1d(np.arange(len(data_list)), train_idx)

########## TRAIN ########## 
train = []
for data in tqdm(data_list[train_idx]):
    image_path = os.path.join(data_path, data, f"{data}_{sequence}.nii.gz")
    mask_path = os.path.join(data_path, data, f"{data}_seg.nii.gz")
    
    image = nib.load(image_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()
    
    if view == 'axial':
        dim = 2
    elif view == 'sagittal':
        dim = 1
    elif view == 'coronal':
        dim = 0
        
    sl_min = mask.shape[dim]
    sl_max = 0
    for s in range(mask.shape[dim]):
        if view == 'axial':
            brain_slice = mask[:, :, s]
        elif view == 'sagittal':
            brain_slice = mask[:, s, :]
        elif view == 'coronal':
            brain_slice = mask[s, :, :]
            
        if np.any(brain_slice != 0):
            sl_min = min(sl_min, s)
            sl_max = max(sl_max, s)
            
    sl_start = (sl_min + sl_max) // 2 - 10
    
    for s in range(20):
        train.append({
            'image' : image_path,
            'mask' : mask_path,
            'slice' : sl_start+s
        })
print('Train Data Finished!\n')
        
########## TEST ########## 
test = []
for data in tqdm(data_list[test_idx]):
    image_path = os.path.join(data_path, data, f"{data}_{sequence}.nii.gz")
    mask_path = os.path.join(data_path, data, f"{data}_seg.nii.gz")
    
    image = nib.load(image_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()
    
    if view == 'axial':
        dim = 2
    elif view == 'sagittal':
        dim = 1
    elif view == 'coronal':
        dim = 0
        
    sl_min = mask.shape[dim]
    sl_max = 0
    for s in range(mask.shape[dim]):
        if view == 'axial':
            brain_slice = mask[:, :, s]
        elif view == 'sagittal':
            brain_slice = mask[:, s, :]
        elif view == 'coronal':
            brain_slice = mask[s, :, :]
            
        if np.any(brain_slice != 0):
            sl_min = min(sl_min, s)
            sl_max = max(sl_max, s)
            
    sl_start = (sl_min + sl_max) // 2 - 10
    
    for s in range(20):
        test.append({
            'image' : image_path,
            'mask' : mask_path,
            'slice' : sl_start+s
        })
print('Test Data Finished!\n')

########## Write ########## 
with open(f'metadata/meta_{sequence}_{view}.json', 'w') as f:
    json.dump({'train' : train, 'test' : test}, f, indent=4)