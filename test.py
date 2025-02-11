########## Import ##########
# Torch Libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Models & Modules
from CLIP import clip
from utils.transform import transform, transform_mask
import modules
from loss import Loss

# Other Utils
from tqdm import tqdm
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from BraTS import BraTSDataset
from Chest import ChestDataset
from RESC import RESCDataset

from IXI import IXIDataset
import nibabel as nib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc
import time
from datetime import datetime
import sys
from utils.metrics import dice_score  # dice score 계산을 위한 함수 추가
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize

########## Define Model ##########
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)
clue = modules.CLUE().cuda()
clue.load_state_dict(torch.load(f'/home/kdh/code/CLUE/checkpoints/CLUE_axial_t1.pt', weights_only=True))

am = modules.AnomalyModule().cuda()

########## Test setting ##########
DATASET = sys.argv[1]

if DATASET == 'chest':
    testset = ChestDataset(
        base_dir='/home/kdh/code/CLUE/meta',
        train=False,
    )
elif DATASET == 'retina':
    testset = RESCDataset(
        base_dir='/home/kdh/code/CLUE/metadata',
        train=False
    )
elif DATASET == 'brain':
    testset = BraTSDataset(
        base_dir='/home/kdh/code/CLUE/meta',
        sequence=sys.argv[2],
        view=sys.argv[3],
        train=False,
        transform=[transform, transform_mask]
    )
else:
    raise Exception("You should choose among chest, retina, and brain")

testloader = DataLoader(testset, batch_size=8, shuffle=True)
if DATASET == 'brain':
    log = open(f'./log/test_{DATASET}_{sys.argv[2]}_{sys.argv[3]}.txt', 'w')
else:
    log = open(f'./log/test_{DATASET}.txt', 'w')
    


cls_name = 'lung' if DATASET == 'chest' else DATASET

########## Test ##########
image_preds = []
image_gts = []

pixel_preds = []
pixel_gts = []

clue.eval()
with torch.no_grad():
    
    for i, data in tqdm(enumerate(testloader), total=len(testloader)):
        image, label = data
        image = image.to(device)    # (B, 3, 224, 224)
        #label = label.to(device)    # (B,)
        
        anomaly_score, anomaly_map, anomaly_map_patch = model(image, cls_name, CLUE=clue, am=am)
        
        anomaly_score = torch.trunc(anomaly_score * 1e5) / 1e5
        anomaly_map = torch.trunc(anomaly_map * 1e5) / 1e5
    
        batch_GT_image = (label.view(label.shape[0], -1).sum(dim=1) > 0).long()
        batch_pred_image = anomaly_score.argmax(dim=1).cpu()
        image_preds.extend(batch_pred_image)
        image_gts.extend(batch_GT_image)
        
        if DATASET != 'chest':
            for b in range(image.shape[0]):
                downsampled = torch.nn.functional.interpolate(label[b].float().unsqueeze(0), size=(16, 16), mode='bilinear', align_corners=False).int().squeeze(0).cpu().numpy()
                pixel_preds.append(anomaly_map_patch[b].detach().cpu().numpy())
                #pixel_gts.append(label[b].numpy())
                pixel_gts.append(downsampled)
        
        if i % 10 == 0:
            if DATASET == 'chest':    
                idx = np.random.randint(1, 8)
                    
                plt.figure(figsize=(25, 5), dpi=100)
                plt.subplot(1, 4, 1)
                plt.imshow(image[idx, 0].cpu(), cmap='gray')
                plt.title('Input')
                plt.axis('off')
                
                plt.subplot(1, 4, 2)
                plt.imshow(anomaly_map_patch[idx].detach().cpu(), cmap='jet', vmax=1, vmin=0)
                plt.title('Anomaly Map (Patches)')
                plt.axis('off')
                
                plt.subplot(1, 4, 3)
                plt.imshow(anomaly_map[idx].detach().cpu(), cmap='jet', vmax=1, vmin=0)
                plt.title(f'Anomaly Map, Predicted: {anomaly_score.argmax(dim=1)[idx]}, Label: {batch_GT_image[idx]}')
                plt.axis('off')
                
                plt.subplot(1, 4, 4)
                plt.imshow(image[idx, 0].cpu(), cmap='gray')
                plt.imshow(anomaly_map[idx].detach().cpu(), cmap='jet', alpha=0.5, vmax=1, vmin=0)
                plt.title('Input + Anomaly Map')
                plt.axis('off')
                
                plt.tight_layout()
            else:
                idx = np.random.randint(1, 8)
                
                plt.figure(figsize=(25, 5), dpi=100)
                plt.subplot(1, 6, 1)
                plt.imshow(image[idx, 0].cpu(), cmap='gray')
                plt.title('Input')
                plt.axis('off')
                
                plt.subplot(1, 6, 2)
                plt.imshow(label[idx, 0].cpu(), cmap='gray')
                plt.title('GT')
                plt.axis('off')
                
                plt.subplot(1, 6, 3)
                plt.imshow(image[idx, 0].cpu(), cmap='gray')
                plt.imshow(label[idx, 0].cpu(), cmap='jet', alpha=0.5, vmax=1, vmin=0)
                plt.title('Input + GT')
                plt.axis('off')
                
                plt.subplot(1, 6, 4)
                plt.imshow(anomaly_map_patch[idx].detach().cpu(), cmap='jet', vmax=1, vmin=0)
                plt.title('Anomaly Map (Patches)')
                plt.axis('off')
                
                plt.subplot(1, 6, 5)
                plt.imshow(anomaly_map[idx].detach().cpu(), cmap='jet', vmax=1, vmin=0)
                plt.title(f'Anomaly Map, Predicted: {anomaly_score.argmax(dim=1)[idx]}, Label: {batch_GT_image[idx]}')
                plt.axis('off')
                
                plt.subplot(1, 6, 6)
                plt.imshow(image[idx, 0].cpu(), cmap='gray')
                plt.imshow(anomaly_map[idx].detach().cpu(), cmap='jet', alpha=0.5, vmax=1, vmin=0)
                plt.title('Input + Anomaly Map')
                plt.axis('off')
                
                plt.tight_layout()
            
            if DATASET == 'brain':
                plt.savefig(f'./pictures/test_{DATASET}_{sys.argv[2]}_{sys.argv[3]}_{i}.png')
            else:
                plt.savefig(f'./pictures/test_{DATASET}_{i}.png')
            plt.close()
           
    img_auroc = roc_auc_score(image_gts, image_preds)
    img_accuracy = accuracy_score(image_gts, image_preds)
    print(img_auroc)
    img_valid = len(image_gts)
    img_total = len(image_preds)

    if DATASET != 'chest':
        pixel_preds_np = np.concatenate([p.flatten() for p in pixel_preds], axis=0)
        pixel_gts_np = np.concatenate([g.flatten() for g in pixel_gts], axis=0)
        pix_auroc = roc_auc_score(pixel_gts_np, pixel_preds_np)

        log.write(f"\n=== SUMMARY ===\n"
                f"Average Image-AUROC: {100 * img_auroc:.4f}%\n"
                f"Average Image-Accuracy: {100 * img_accuracy:.4f}%\n"
                f"Average Pixel-AUROC: {100 * pix_auroc:.4f}%\n"
                f"============================\n\n")

        print(f"\n=== SUMMARY ===\n",
            f"Average Image-AUROC: {100 * img_auroc:.4f}%\n"
            f"Average Image-Accuracy: {100 * img_accuracy:.4f}%\n"
            f"Average Pixel-AUROC: {100 * pix_auroc:.4f}%\n"
            f"============================\n\n")

    else:
        log.write(f"\n=== SUMMARY ===\n"
                f"Average Image-AUROC: {100 * img_auroc:.4f}%\n"
                f"Average Image-Accuracy: {100 * img_accuracy:.4f}%\n"
                f"============================\n\n")

        print(f"\n=== SUMMARY ===\n",
            f"Average Image-AUROC: {100 * img_auroc:.4f}%\n"
            f"Average Image-Accuracy: {100 * img_accuracy:.4f}%\n"
            f"============================\n\n")


log.close()